/*
 * wine_dma_infer.c  —  AXI DMA inference test (Ultra96-v2, wine MLP)
 *
 * Build on Ultra96:   make        (or: gcc -O2 -o wine_dma_infer wine_dma_infer.c)
 * Run as root:        ./wine_dma_infer [sample_index | all]   (default: 0)
 *
 * Load overlay first: ./load_overlay.sh wine_dma_int4
 *
 * C port of wine_dma_infer.py — same register-level protocol, same test
 * dataset (embedded from wine_test_data.json via wine_test_data.h, see
 * gen_wine_test_data.py). For a full register-level hardware diagnosis,
 * use dma_diag.py instead — this program assumes the DMA is already known
 * to work and just runs inference.
 */

#define _DEFAULT_SOURCE /* pread, clock_gettime, MAP_ANONYMOUS, MAP_LOCKED under -std=c11 */

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "wine_test_data.h"

/* ---------- AXI DMA 7.1 register map ---------- */
#define MM2S_CR   0x00
#define MM2S_SR   0x04
#define MM2S_SA   0x18
#define MM2S_SA_H 0x1C
#define MM2S_LEN  0x28
#define S2MM_CR   0x30
#define S2MM_SR   0x34
#define S2MM_DA   0x48
#define S2MM_DA_H 0x4C
#define S2MM_LEN  0x58

#define CR_RS    (1u << 0)
#define CR_RESET (1u << 2)
#define SR_HALT  (1u << 0)
#define SR_ERR   (7u << 4)
#define SR_IOC   (1u << 12)

#define DMA_BASE 0xa0000000ULL

#define N_IN      WINE_N_FEATURES
#define N_OUT     3
#define IN_BYTES  (N_IN  * (int)sizeof(float))
#define OUT_BYTES (N_OUT * (int)sizeof(float))

/* Cortex-A53 L2=512KB; 2MB > 4x L2 guarantees full eviction of dirty lines
 * to DDR. The DMA is not cache-coherent, so this must run after writing
 * in_buf and before arming the transfer, or the IP can read stale input /
 * we can read stale output back from cache. */
#define CACHE_FLUSH_SIZE (2 * 1024 * 1024)

static const float SENTINEL[N_OUT] = {1111.0f, 2222.0f, 3333.0f};
static const char *CLASS_NAMES[3] = {"low", "medium", "high"};

static volatile uint32_t *dma;
#define RD(off)      (dma[(off) >> 2])
#define WR(off, v)   (dma[(off) >> 2] = (uint32_t)(v))

static uint8_t *in_buf, *out_buf;
static uint64_t in_phys, out_phys;
static int uio_fd;

/* ---------------------------------------------------------------------- */

static int find_uio(uint64_t target) {
    DIR *dir = opendir("/sys/class/uio");
    if (!dir) { perror("opendir /sys/class/uio"); return -1; }
    struct dirent *ent;
    while ((ent = readdir(dir))) {
        if (strncmp(ent->d_name, "uio", 3) != 0) continue;
        char path[256];
        snprintf(path, sizeof(path), "/sys/class/uio/%s/maps/map0/addr", ent->d_name);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        uint64_t addr = 0;
        fscanf(f, "0x%lx", &addr);
        fclose(f);
        if (addr == target) {
            closedir(dir);
            snprintf(path, sizeof(path), "/dev/%s", ent->d_name);
            printf("DMA UIO: %s\n", path);
            int fd = open(path, O_RDWR | O_SYNC);
            if (fd < 0) perror("open uio");
            return fd;
        }
    }
    closedir(dir);
    fprintf(stderr, "No UIO device at 0x%08lX — load the overlay first: "
                     "./load_overlay.sh wine_dma_int4\n", target);
    return -1;
}

static uint64_t virt_to_phys(void *v) {
    long pgsz = sysconf(_SC_PAGE_SIZE);
    uint64_t vpn = (uint64_t)(uintptr_t)v / pgsz;
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) { perror("pagemap"); return 0; }
    uint64_t entry = 0;
    pread(fd, &entry, 8, (off_t)(vpn * 8));
    close(fd);
    if (!(entry >> 63)) {
        fprintf(stderr, "page not present — mlockall may have failed\n");
        return 0;
    }
    return (entry & ((1ULL << 55) - 1)) * pgsz + ((uint64_t)(uintptr_t)v % pgsz);
}

static double mono_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void flush_cpu_cache(void) {
    static uint8_t buf[CACHE_FLUSH_SIZE];
    for (int i = 0; i < CACHE_FLUSH_SIZE; i += 64) buf[i] ^= 1;
}

/* Returns raw SR value (positive) or negative on error/timeout */
static int wait_ch(int off, const char *lbl, double timeout) {
    double t0 = mono_s();
    while (1) {
        uint32_t sr = RD(off);
        if (sr & SR_ERR) {
            fprintf(stderr, "%s error SR=0x%08X\n", lbl, sr);
            return -(int)sr;
        }
        if (sr & SR_IOC) return (int)sr;
        if (mono_s() - t0 > timeout) {
            fprintf(stderr, "%s timeout SR=0x%08X\n", lbl, sr);
            return -1;
        }
    }
}

/* ---------------------------------------------------------------------- */

/* Opens the DMA UIO, resets both channels and allocates the in/out DMA
 * buffers. Exits the process on any unrecoverable setup failure. */
static void init_dma(void) {
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0)
        perror("mlockall (non-fatal)");

    uio_fd = find_uio(DMA_BASE);
    if (uio_fd < 0) exit(1);

    dma = mmap(NULL, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, uio_fd, 0);
    if (dma == MAP_FAILED) { perror("mmap dma"); exit(1); }

    /* Liveness probe: write RS then read it back. All-zero means AXI-Lite
     * isn't wired up (peripheral_aresetn=0, wrong bitstream, overlay not
     * loaded) — run dma_diag.py for a full register-level diagnosis. Any
     * nonzero readback is fine: fixed IP config bits (e.g. IRQ_Threshold)
     * stay set regardless of what we write. */
    WR(MM2S_CR, CR_RS);
    if (RD(MM2S_CR) == 0) {
        fprintf(stderr, "AXI-Lite not responding — run dma_diag.py\n");
        munmap((void *)dma, 0x10000);
        close(uio_fd);
        exit(2);
    }

    WR(MM2S_CR, CR_RESET);
    WR(S2MM_CR, CR_RESET);
    int lim = 200000;
    while ((RD(MM2S_CR) & CR_RESET) && lim--);
    lim = 200000;
    while ((RD(S2MM_CR) & CR_RESET) && lim--);
    WR(MM2S_CR, CR_RS);
    WR(S2MM_CR, CR_RS);
    if ((RD(MM2S_SR) & SR_HALT) || (RD(S2MM_SR) & SR_HALT))
        fprintf(stderr, "warning: channel HALTED after reset — peripheral_aresetn may be 0\n");

    in_buf  = mmap(NULL, 4096, PROT_READ|PROT_WRITE,
                   MAP_SHARED|MAP_ANONYMOUS|MAP_LOCKED, -1, 0);
    out_buf = mmap(NULL, 4096, PROT_READ|PROT_WRITE,
                   MAP_SHARED|MAP_ANONYMOUS|MAP_LOCKED, -1, 0);
    if (in_buf == MAP_FAILED || out_buf == MAP_FAILED) { perror("mmap buf"); exit(1); }

    volatile uint8_t touch;
    for (int i = 0; i < 4096; i += 64) { touch = in_buf[i]; touch = out_buf[i]; }
    (void)touch;

    in_phys  = virt_to_phys(in_buf);
    out_phys = virt_to_phys(out_buf);
    if (!in_phys || !out_phys) { fprintf(stderr, "virt_to_phys failed\n"); exit(1); }
}

/* Runs one inference: packs features into in_buf, fires the DMA transfer,
 * unpacks the 3 output scores. Returns the number of output beats that
 * differed from the sentinel (3 = full success). */
static int infer(const float *features, float scores_out[N_OUT],
                  int *sr_mm2s_out, int *sr_s2mm_out) {
    memcpy(in_buf,  features, IN_BYTES);
    memcpy(out_buf, SENTINEL, OUT_BYTES);
    flush_cpu_cache();

    /* Clear IOC bit from previous run (W1C) and re-assert RS. */
    WR(MM2S_SR, SR_IOC);
    WR(S2MM_SR, SR_IOC);
    WR(MM2S_CR, CR_RS);
    WR(S2MM_CR, CR_RS);

    /* Arm S2MM (destination) before triggering MM2S (source) — the IP is
     * free-running, so S2MM must already be ready to receive when MM2S
     * feeds it. */
    WR(S2MM_DA,   (uint32_t)(out_phys & 0xFFFFFFFF));
    WR(S2MM_DA_H, (uint32_t)(out_phys >> 32));
    WR(S2MM_LEN,  OUT_BYTES);

    WR(MM2S_SA,   (uint32_t)(in_phys & 0xFFFFFFFF));
    WR(MM2S_SA_H, (uint32_t)(in_phys >> 32));
    WR(MM2S_LEN,  IN_BYTES);  /* starts the transfer */

    int sr_mm2s = wait_ch(MM2S_SR, "MM2S", 3.0);
    int sr_s2mm = wait_ch(S2MM_SR, "S2MM", 3.0);
    *sr_mm2s_out = sr_mm2s;
    *sr_s2mm_out = sr_s2mm;

    float *res = (float *)out_buf;
    int beats = 0;
    for (int i = 0; i < N_OUT; i++) {
        scores_out[i] = res[i];
        if (res[i] != SENTINEL[i]) beats++;
    }
    return beats;
}

static int argmax3(const float *v) {
    int best = 0;
    for (int i = 1; i < N_OUT; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* Runs one sample and prints a one-line result (Python run_sample equivalent). */
static int run_sample(int idx) {
    float scores[N_OUT];
    int sr_mm2s, sr_s2mm;
    int beats = infer(WINE_X[idx], scores, &sr_mm2s, &sr_s2mm);
    int label = WINE_Y[idx];

    if (beats < N_OUT) {
        printf("[%3d] gt=%d  FAILED — only %d/%d beats written (MM2S=0x%08X S2MM=0x%08X)\n",
               idx, label, beats, N_OUT, (uint32_t)sr_mm2s, (uint32_t)sr_s2mm);
        return 0;
    }

    int pred = argmax3(scores);
    int ok = (pred == label);
    printf("[%3d] gt=%d (%-6s)  scores=[%+.4f %+.4f %+.4f]  pred=%d (%-6s)  %s\n",
           idx, label, CLASS_NAMES[label], scores[0], scores[1], scores[2],
           pred, CLASS_NAMES[pred], ok ? "OK" : "WRONG");
    return ok;
}

/* ---------------------------------------------------------------------- */
int main(int argc, char *argv[]) {
    const char *arg = argc > 1 ? argv[1] : "0";

    init_dma();

    if (strcmp(arg, "all") == 0) {
        int correct = 0;
        for (int i = 0; i < WINE_N_SAMPLES; i++)
            correct += run_sample(i);
        printf("\nAccuracy: %d/%d = %.1f%%\n",
               correct, WINE_N_SAMPLES, 100.0 * correct / WINE_N_SAMPLES);
    } else {
        int idx = atoi(arg);
        if (idx < 0 || idx >= WINE_N_SAMPLES) {
            fprintf(stderr, "sample index out of range: 0..%d\n", WINE_N_SAMPLES - 1);
            return 1;
        }
        run_sample(idx);
    }

    munmap((void *)dma, 0x10000);
    munmap(in_buf,  4096);
    munmap(out_buf, 4096);
    close(uio_fd);
    return 0;
}
