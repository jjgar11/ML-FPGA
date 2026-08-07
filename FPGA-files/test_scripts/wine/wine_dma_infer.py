#!/usr/bin/env python3
"""Wine quality MLP inference via AXI DMA on Ultra96-v2.
Load the overlay first:  ./load_overlay.sh wine_axi_dma

Usage:
  python3 wine_dma_infer.py [sample_index]   (default: 0)
  python3 wine_dma_infer.py all              (run all samples, print accuracy)
"""

import ctypes
import glob
import mmap
import os
import struct
import sys
import time
import json

# ---- AXI DMA 7.1 register offsets (simple/direct mode) ----
# Prefix:  MM2S = Memory-Map to Stream  (DDR → IP input,  "read from RAM")
#          S2MM = Stream to Memory-Map  (IP output → DDR, "write to RAM")
# Each channel has its own CR/SR/address/length registers at separate offsets.
# Full reference: see AXI_DMA_REGISTERS.md in this directory.

MM2S_CR   = 0x00   # MM2S Control Register  — RS (bit0), Reset (bit2)
MM2S_SR   = 0x04   # MM2S Status Register   — Halted(0), Idle(1), errors(4-6), IOC(12)
MM2S_SA   = 0x18   # MM2S Source Addr low   — physical address in DDR (bits 31:0)
MM2S_SA_H = 0x1C   # MM2S Source Addr high  — physical address in DDR (bits 63:32)
MM2S_LEN  = 0x28   # MM2S Length — bytes to read; WRITING HERE STARTS THE TRANSFER

S2MM_CR   = 0x30   # S2MM Control Register  — RS (bit0), Reset (bit2)
S2MM_SR   = 0x34   # S2MM Status Register   — Halted(0), Idle(1), errors(4-6), IOC(12)
S2MM_DA   = 0x48   # S2MM Dest Addr low     — physical address in DDR (bits 31:0)
S2MM_DA_H = 0x4C   # S2MM Dest Addr high    — physical address in DDR (bits 63:32)
S2MM_LEN  = 0x58   # S2MM Length — max bytes to write; WRITING HERE STARTS THE TRANSFER

# Control Register (CR) bit masks
CR_RS     = 0b1        # bit 0: Run/Stop — 1=run, 0=stop/halt
CR_RESET  = 0b1 << 2   # bit 2: Soft reset (self-clearing); also flushes internal FIFO

# Status Register (SR) bit masks  (most error/irq bits are W1C — write 1 to clear)
SR_HALTED  = 0b1            # bit  0: channel halted (RS=0 or post-error)
SR_IDLE    = 0b1 << 1       # bit  1: channel idle (no active transfer)
SR_ERR     = 0b111 << 4     # bits 4-6: DMAIntErr | DMASlvErr | DMADecErr
SR_IOC_IRQ = 0b1 << 12      # bit 12: IOC — transfer complete (W1C)

N_IN      = 13
N_OUT     = 3
IN_BYTES  = N_IN  * 4
OUT_BYTES = N_OUT * 4

SENTINEL = [1111.0, 2222.0, 3333.0]

_CACHE_FLUSH_SIZE = 2 * 1024 * 1024

libc = ctypes.CDLL(None)


def _mlockall():
    libc.mlockall(3)  # MCL_CURRENT | MCL_FUTURE


def _flush_cpu_cache():
    buf = bytearray(_CACHE_FLUSH_SIZE)
    for i in range(0, _CACHE_FLUSH_SIZE, 64):
        buf[i] ^= 1
    del buf


def _virt_to_phys(vaddr):
    page = os.sysconf('SC_PAGE_SIZE')
    with open('/proc/self/pagemap', 'rb') as f:
        f.seek((vaddr // page) * 8)
        entry = struct.unpack('Q', f.read(8))[0]
    if not (entry >> 63):
        raise RuntimeError(f"Page 0x{vaddr:X} not resident — mlockall failed?")
    pfn = entry & ((1 << 55) - 1)
    return pfn * page + (vaddr % page)


def _find_uio(target=0xa0000000):
    for d in sorted(glob.glob('/sys/class/uio/uio*')):
        try:
            addr = int(open(f'{d}/maps/map0/addr').read(), 16)
            if addr == target:
                return '/dev/' + os.path.basename(d)
        except OSError:
            pass
    raise RuntimeError(
        f"No UIO device at 0x{target:08X} — run: ./load_overlay.sh wine_axi_dma"
    )


class WineDMAInfer:
    def __init__(self):
        _mlockall()

        uio = _find_uio()
        print(f"DMA registers: {uio}")
        self._fd = open(uio, 'r+b', buffering=0)
        self._regs = mmap.mmap(self._fd.fileno(), 0x10000,
                               mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

        self._in_mm  = mmap.mmap(-1, 4096)
        self._out_mm = mmap.mmap(-1, 4096)
        self._in_mm.write(b'\x00' * 4096)
        self._in_mm.seek(0)
        self._out_mm.write(b'\x00' * 4096)
        self._out_mm.seek(0)

        self._in_c  = (ctypes.c_char * 4096).from_buffer(self._in_mm)
        self._out_c = (ctypes.c_char * 4096).from_buffer(self._out_mm)

        self._in_phys  = _virt_to_phys(ctypes.addressof(self._in_c))
        self._out_phys = _virt_to_phys(ctypes.addressof(self._out_c))
        print(f"in_buf  phys=0x{self._in_phys:010X}")
        print(f"out_buf phys=0x{self._out_phys:010X}")

        self._reset()

    def _wr(self, off, val):
        self._regs.seek(off)
        self._regs.write(struct.pack('<I', val & 0xFFFFFFFF))

    def _rd(self, off):
        self._regs.seek(off)
        return struct.unpack('<I', self._regs.read(4))[0]

    def _reset(self):
        self._wr(MM2S_CR, CR_RESET)
        self._wr(S2MM_CR, CR_RESET)

        while self._rd(MM2S_CR) & CR_RESET:
            pass
        while self._rd(S2MM_CR) & CR_RESET:
            pass

        self._wr(MM2S_CR, CR_RS)
        self._wr(S2MM_CR, CR_RS)

    def _wait(self, sr_off, label, timeout=2.0):
        t0 = time.time()
        while True:
            sr = self._rd(sr_off)
            if sr & SR_ERR:
                raise RuntimeError(f"{label} error SR=0x{sr:08X}")
            if sr & SR_IOC_IRQ:
                return sr
            if time.time() - t0 > timeout:
                raise TimeoutError(f"{label} timeout SR=0x{sr:08X}")

    def infer(self, features):
        assert len(features) == N_IN

        struct.pack_into(f'<{N_IN}f', self._in_c, 0, *[float(x) for x in features])
        struct.pack_into('<3f', self._out_c, 0, *SENTINEL)
        _flush_cpu_cache()

        # Clear IOC_Irq from previous transfer (W1C) and re-assert RS.
        self._wr(MM2S_SR, SR_IOC_IRQ)
        self._wr(S2MM_SR, SR_IOC_IRQ)
        self._wr(MM2S_CR, CR_RS)
        self._wr(S2MM_CR, CR_RS)

        # Arm S2MM (destination) before triggering MM2S (source) — the IP is
        # free-running, so S2MM must already be ready to receive when MM2S feeds it.
        self._wr(S2MM_DA,   self._out_phys & 0xFFFFFFFF)
        self._wr(S2MM_DA_H, (self._out_phys >> 32) & 0xFFFFFFFF)
        self._wr(S2MM_LEN,  OUT_BYTES)

        self._wr(MM2S_SA,   self._in_phys & 0xFFFFFFFF)
        self._wr(MM2S_SA_H, (self._in_phys >> 32) & 0xFFFFFFFF)
        self._wr(MM2S_LEN,  IN_BYTES)  # starts the transfer

        sr_mm2s = self._wait(MM2S_SR, 'MM2S')
        sr_s2mm = self._wait(S2MM_SR, 'S2MM')

        scores = list(struct.unpack_from('<3f', self._out_c, 0))
        beats_written = sum(1 for i, s in enumerate(scores) if s != SENTINEL[i])

        return scores, beats_written, sr_mm2s, sr_s2mm

    def close(self):
        del self._in_c
        del self._out_c
        self._regs.close()
        self._fd.close()
        self._in_mm.close()
        self._out_mm.close()


def run_sample(dma, features, label, idx):
    scores, beats_written, sr_mm2s, sr_s2mm = dma.infer(features)
    pred   = scores.index(max(scores))
    labels = {0: 'low', 1: 'medium', 2: 'high'}
    ok     = pred == label

    print(f"[sample {idx:3d}] gt={label} ({labels[label]:6s})  "
          f"scores={[f'{s:+.4f}' for s in scores]}  "
          f"pred={pred} ({labels[pred]:6s})  {'OK' if ok else 'WRONG'}")

    if beats_written < N_OUT:
        print(f"  !! solo {beats_written}/{N_OUT} beats escritos "
              f"(SR: MM2S=0x{sr_mm2s:08X} S2MM=0x{sr_s2mm:08X})")

    return ok


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    with open('/home/root/fpga/test_scripts/wine/wine_test_data.json') as f:
        data = json.load(f)
    X, y = data['X'], data['y']

    dma = WineDMAInfer()

    if arg == "all":
        correct = 0
        for i, (feat, label) in enumerate(zip(X, y)):
            ok = run_sample(dma, feat, label, i)
            if ok:
                correct += 1
        print(f"\nAccuracy: {correct}/{len(X)} = {correct/len(X)*100:.1f}%")
    else:
        idx = int(arg)
        run_sample(dma, X[idx], y[idx], idx)

    dma.close()


if __name__ == '__main__':
    main()
