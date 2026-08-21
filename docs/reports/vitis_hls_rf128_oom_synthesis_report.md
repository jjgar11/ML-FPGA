# Vitis HLS Synthesis Run Report — `gtsrb_gap` int4, ReuseFactor=128

**Date:** 2026-08-20 → 2026-08-21
**Project:** `gtsrb_gap_hls4ml_pytorch_int4_vivadoaccelerator_rf128`
**Toolchain:** hls4ml 1.2.0, Vitis HLS 2020.1, `VivadoAccelerator` backend, `io_stream`,
target Ultra96-v2 (`xczu3eg-sbva484-1-i`)
**VM:** Ubuntu 18.04, 19 GB RAM, 61 GB swap (4 swapfiles), `systemd-run --scope -p MemoryMax=32G`
**Outcome:** Linux OOM-killer terminated `vitis_hls` during the "code transformations"
phase after ~7h40m of wall-clock synthesis time.

This report documents a real-time monitoring session of a `vitis_hls -f build_prj.tcl`
synthesis run, capturing the tool's own progress markers plus external process/memory
telemetry gathered via `ps`, `free -h`, and `dmesg` over VirtualBox `guestcontrol`, from
shortly after the run entered its LTO codegen step through to the kernel's OOM-kill.

This run was retesting a bug in `convert_model.py`'s `make_hls_config()`:
`ReuseFactor`/`Strategy` were previously set only at the Model level, but
`hls4ml.utils.config_from_pytorch_model(..., granularity="name")` pre-populates every
layer with its own `ReuseFactor: 1`, which takes precedence — so every `--reuse` value
passed on the CLI was silently ignored, and a prior ~33-hour run requesting `--reuse 64`
actually synthesized fully unrolled (RF=1) and got OOM-killed at ~64 GB. That bug is now
fixed (per-layer `ReuseFactor`/`Strategy` are set explicitly in the same loop that sets
per-layer `Precision`); this run used a genuine `ReuseFactor=128` (hls4ml clamped it
per-layer to valid divisors — 108/128/144 — confirmed by its own
`WARNING: Invalid ReuseFactor=128 ... Using ReuseFactor=108 instead` messages in the log).

---

## 1. Why this run matters

The previous (buggy) RF=1 run spent **~29h44m of CPU time in the "Linking" phase alone**
before eventually OOM-killing at ~64 GB during a later phase. With the fix applied, this
run was a genuine test of whether `ReuseFactor=128` — i.e., 128x less unrolled hardware
than RF=1 — would keep both compile time and peak memory within what this VM can offer.

**Headline result:** Linking dropped from ~29h44m to **~5h08m**, a ~6x improvement, and
peaked at only 6.2 GB instead of tens of GB. However, the very next phase
("code transformations") still grew unbounded and exhausted all 61 GB of swap, ending in
an OOM-kill. RF=128 is a large improvement but still not sufficient for this model/phase
on this toolchain/VM combination.

---

## 2. Monitoring methodology

Vitis HLS spawns a `clang`/`clang -cc1` child process tree for its LTO/codegen step. An
earlier session mistakenly diagnosed a *healthy* run as deadlocked by checking only the
top-level `vitis_hls` PID's CPU time (which legitimately idles in `waitpid()` while a
child does the real work). This run's monitoring instead always inspected the **full
process tree**:

```
ps -e -o pid,ppid,stat,pcpu,time,rss,cmd --forest | grep -E 'vitis|clang'
```

alongside `free -h` for system-wide RAM/swap, and `tail`/`grep` on `vitis_hls.log` for
tool-reported phase transitions and errors. Checks were run repeatedly via
`VBoxManage guestcontrol <vm> run`, with the interval tightened from ~25 minutes down to
as little as 1–2 minutes as the run approached the swap ceiling.

---

## 3. Timeline

### 3.1 — LTO / codegen step (`a.g.ld.5.gdce.bc` → `a.g.lto.bc`)

At the start of this monitoring window, `vitis_hls` (PID 7237) had already spawned
`clang` (PID 7277) → `clang -cc1` (PID 7278) to do LLVM-side LTO/codegen for the
`myproject_axi` top function. This single compiler invocation ran for **over 4.5 hours**,
far longer than any Windows reproduction of this project ever ran before crashing at the
same step — confirming that step duration alone is not evidence of a hang on this
platform.

Representative process tree during this phase:

```
7178  5935 S+    0.0 00:00:00      \_ sudo systemd-run --scope -p MemoryMax=32G bash -c ... vitis_hls -f build_prj.tcl ...
7179  7178 S+    0.0 00:00:00          \_ /bin/bash .../vitis_hls -f build_prj.tcl ...
7213  7179 S+    0.0 00:00:00              \_ /bin/bash .../loader -exec vitis_hls -f build_prj.tcl ...
7237  7213 Sl+   0.0 00:00:02              |   \_ .../unwrapped/lnx64.o/vitis_hls -f build_prj.tcl ...
7277  7237 S+    0.0 00:00:00              |       \_ .../clang-3.9-csynth/bin/clang -fsave-optimization-record ...
                                                        -fhls -mllvm -hls-top-function-name=myproject_axi ...
                                                        -x ir .../a.g.ld.5.gdce.bc -o .../a.g.lto.bc
7278  7277 R+    100 02:19:33              |           \_ .../clang-3.9-csynth/bin/clang -cc1 -triple fpga64-xilinx-none
                                                            -emit-llvm-bc ... -hls-top-function-name=myproject_axi ...
7214  7179 S+    0.0 00:00:00              \_ tee vitis_hls.log
```

CPU time on the `clang -cc1` grandchild (PID 7278) advanced steadily and consistently
1:1 with wall-clock time across the whole step — a clean signal that it was genuinely
computing, not stuck:

| Check | `clang -cc1` CPU TIME |
|---|---|
| 1 | 02:15:40 |
| 2 | 02:16:41 |
| 3 | 02:19:18 |
| 4 | 02:19:33 |
| 5 | 02:45:36 |
| 6 | 03:11:35 |
| 7 | 03:37:35 |
| 8 | 04:03:39 |
| 9 | 04:29:36 |

Memory during this whole step stayed essentially flat and unremarkable — a slow creep
from ~874 MB to ~1.1 GB system-wide RAM used, 0 swap, e.g.:

```
              total        used        free      shared  buff/cache   available
Mem:            19G         874M        17G        1,1M       1,6G          18G
Swap:           61G           0B        61G
```

The `vitis_hls.log` tail was silent through this entire window (as expected — the LTO
sub-step doesn't emit log lines until it finishes):

```
INFO: [HLS 200-10] Analyzing design file 'firmware/myproject_axi.cpp' ...
INFO: [HLS 200-777] Using interface defaults for 'Vivado' target.
```

(The only recurring line in `grep -E 'ERROR|...'` output the whole time was the known-harmless
`ERROR: [HLS 200-642] The 'config_array_partition -maximum_size' command is not supported.`)

### 3.2 — LTO finishes; `vitis_hls` resumes internal work

The `clang`/`clang -cc1` children eventually disappeared from the process tree — the LTO
step had completed — and `vitis_hls` (PID 7237) itself started consuming CPU directly:

```
7237  7213 Sl+   2.6 00:07:54              |   \_ .../unwrapped/lnx64.o/vitis_hls -f build_prj.tcl ...
```

(TIME here resets to a new internal counter — 00:07:54, not a continuation of the LTO
child's clock.) A follow-up check ~1 minute later confirmed real progress:
`00:07:54 → 00:08:17`.

### 3.3 — Linking finishes; "code transformations" begins

Shortly after, the log printed the phase-completion marker that is the direct point of
comparison against the previous (buggy) run:

```
INFO: [HLS 200-111] Finished Linking Time (s): cpu = 05:07:53 ; elapsed = 05:08:17 . Memory (MB): peak = 6207.172 ; gain = 5770.605 ; free physical = 12756 ; free virtual = 77715
INFO: [HLS 200-111] Finished Checking Pragmas Time (s): cpu = 05:07:53 ; elapsed = 05:08:17 . Memory (MB): peak = 6207.172 ; gain = 5770.605 ; free physical = 12756 ; free virtual = 77715
INFO: [HLS 200-10] Starting code transformations ...
```

**Linking: ~5h08m elapsed, peak 6.2 GB.** The previous RF=1-bug run spent **~29h44m of
CPU time** in this same phase before eventually failing later — a roughly 6x wall-clock
improvement from the ReuseFactor fix, on top of a dramatically smaller peak memory
footprint. This confirmed the fix was working as intended.

### 3.4 — Code transformations: RAM saturates, then swap climbs steadily

This is the same phase where the previous buggy run's memory eventually exploded to
~64 GB and got OOM-killed. System RAM (19 GB) filled within the first ~15 minutes of
this phase:

```
RSS 15,902,244 kB (~15.2 GB) — Mem: 19G total, 15G used, 3.7G available, 0 swap
RSS 19,998,460 kB (~19.05 GB) — Mem: 19G total, 19G used, 13M available, 0 swap
```

Swap then began climbing, and continued to climb at a remarkably steady rate for the
next ~70 minutes:

| Swap used | Δ vs. previous check | Interval |
|---|---|---|
| 4.1 GB | — | — |
| 8.2 GB | +4.1 GB | ~5 min |
| 12 GB | +3.8 GB | ~5 min |
| 16 GB | +4 GB | ~5 min |
| 21 GB | +5 GB | ~5 min |
| 26 GB | +5 GB | ~5 min |
| 31 GB | +5 GB | ~5 min |
| 35 GB | +4 GB | ~5 min |
| 38 GB | +3 GB | ~5 min (CPU throughput dipped to ~52s/5min here — thrashing) |
| 38 GB | +0 GB | ~5 min (brief plateau) |
| 39 GB | +1 GB | ~5 min (CPU throughput recovered to ~4m26s/5min) |
| 47 GB | +8 GB | ~10.5 min (plateau was temporary, not real stabilization) |
| 51 GB | +4 GB | ~4 min |
| 54 GB | +3 GB | ~4 min |
| 57 GB | +3 GB | ~3.3 min |
| 59 GB | +2 GB | ~3.5 min |
| 60 GB | +1 GB | ~1.5 min |
| — | exhausted | < 1.5 min |

Representative snapshot near the middle of this climb:

```
              total        used        free      shared  buff/cache   available
Mem:            19G         19G        148M          8K         52M         5,1M
Swap:           61G         12G         49G
```

and near the ceiling:

```
              total        used        free      shared  buff/cache   available
Mem:            19G         19G        164M          8K         61M         9,5M
Swap:           61G         59G        2,6G
```

`vitis_hls.log` produced no new phase markers during this entire window — only a long,
repetitive stream of `INFO: [XFORM 203-603] Inlining function ...` lines (function
inlining across the CNN's `nnet::` template instantiations), e.g.:

```
INFO: [XFORM 203-603] Inlining function 'nnet::conv_2d_buffer_cl<nnet::array<ap_fixed<12, 4, ...>, 32u>,
  nnet::array<ap_fixed<12, 4, ...>, 64u>, config6>' into 'nnet::conv_2d_cl<...>' (firmware/nnet_utils/nnet_conv2d_stream.h:109).
INFO: [XFORM 203-603] Inlining function 'nnet::reduce<ap_fixed<20, 8, ...>, 4, nnet::Op_add<ap_fixed<20, 8, ...>>>'
  into 'nnet::pooling2d_cl<...>' (firmware/nnet_utils/nnet_common.h:46).
```

**A note on throughput vs. crash risk:** partway through this climb, `vitis_hls`'s own
CPU-time-to-wallclock ratio degraded sharply (from near-1:1 down to ~52 seconds of CPU
time per 5 minutes of wall clock) — a classic swap-thrashing signature (the process
spending most of its time on page faults / swap I/O rather than computing). It then
*recovered* to near-1:1 for a couple of checks, which briefly looked like a real
stabilization (swap held flat at 38 GB for one check). It was not: swap resumed climbing
at close to its earlier rate immediately after. **Lesson for future monitoring:** a single
flat reading, or a single recovered-throughput reading, is not sufficient evidence of a
real plateau — only sustained trends across several consecutive checks are meaningful.

### 3.5 — Swap exhaustion and OOM-kill

The process (PID 7237) remained alive and computing right up to the final check before
it disappeared from `ps`:

```
7237  7213 Sl+  27.1 01:51:33  19810136  \_ .../unwrapped/lnx64.o/vitis_hls -f build_prj.tcl ...
```
```
Mem:            19G         19G        190M          8K         49M         29M
Swap:           61G         60G        1,1G
```

The very next check found no `vitis_hls`/`clang` processes at all. `dmesg | tail -30`
confirmed the kernel's OOM-killer had acted:

```
[33308.556395] Out of memory: Kill process 7237 (vitis_hls) score 962 or sacrifice child
[33308.559847] Killed process 7237 (vitis_hls) total-vm:105290944kB, anon-rss:19844012kB, file-rss:0kB, shmem-rss:0kB
[33309.555133] oom_reaper: reaped process 7237 (vitis_hls), now anon-rss:0kB, file-rss:0kB, shmem-rss:0kB
```

The `vitis_hls` wrapper script logged the same event:

```
/home/vivado/opt/Xilinx/Vitis/2020.1/bin/loader: line 286:  7237 Killed                  "$RDI_PROG" "$@"
```

`total-vm:105290944kB` (~100.4 GB) is the process's combined virtual memory footprint at
the moment of the kill — i.e., its resident memory (`anon-rss:19844012kB`, ~19.84 GB) plus
everything it had pushed out to swap. `score 962` (out of a possible ~1000) reflects that
the kernel identified `vitis_hls` as, by far, the single largest memory consumer on the
system when it needed to reclaim.

Memory was fully reclaimed within about a second of the kill:

```
              total        used        free      shared  buff/cache   available
Mem:            19G        223M         19G         20K        101M          19G
Swap:           61G        117M         61G
```

---

## 4. Comparison against the previous (`ReuseFactor=1`-bug) run

| Metric | RF=1 (bug, previous run) | RF=128 (fixed, this run) |
|---|---|---|
| Requested `--reuse` | 64 | 128 |
| Actually applied per-layer | **1** (bug: per-layer config silently overrode Model-level) | 108 / 128 / 144 (clamped correctly per layer) |
| Linking phase duration | ~29h44m CPU | **~5h08m elapsed** |
| Linking peak memory | not recorded (run continued past it) | **6.2 GB** |
| Failure phase | "code transformations" | "code transformations" (same phase) |
| Failure mode | OOM-killed at ~64 GB total demand | OOM-killed at ~100 GB total demand (19 GB RAM + ~81 GB* effective swap use before reclaim) |
| Total swap available | 41 GB | 61 GB |
| Total wall-clock before failure | ~33 hours | ~7h40m |

*\*`total-vm` includes address space that may not all be swapped/resident simultaneously;
the swap-used figures tracked live topped out at 60 GB of the 61 GB available.*

The ReuseFactor fix produced a large, real improvement (6x faster Linking, dramatically
lower peak memory in that phase, and the run reached a materially later failure point in
roughly a quarter of the wall-clock time). It did not, however, fix the underlying issue:
"code transformations" for this CNN's io_stream/DATAFLOW structure appears to have a
memory requirement that scales with something other than (or in addition to)
`ReuseFactor`, since RF=128 — 128x less unrolled than RF=1 — still exhausted 61 GB of
combined RAM+swap in the same phase.

---

## 5. Conclusions and next steps

1. **The `ReuseFactor`/`Strategy` per-layer bug fix in `convert_model.py` is confirmed
   correct and effective** — it is not, by itself, sufficient to make this model
   synthesizable on this VM/toolchain.
2. **`ReuseFactor=128` is still too low** for the "code transformations" phase to fit in
   61 GB of combined memory for this CNN. Options going forward, not yet decided:
   - Retry with a much more aggressive `ReuseFactor` (256, 512+), trading further FPGA
     latency/throughput for a smaller synthesis memory footprint.
   - Revisit the parked decision to modernize the toolchain — `hls4ml backend='Vitis'`
     with a newer Vitis HLS (2022.2+) on a newer Ubuntu VM (20.04/22.04+). This was
     explicitly deferred pending this run's outcome; a newer HLS codegen/compiler
     implementation may not carry the same memory scaling in this phase.
   - Give the VM significantly more RAM/swap headroom, if that's practical, and
     re-attempt RF=128 as a control to see whether it's a hard memory requirement or just
     needs more room than 61 GB.
3. **Monitoring methodology takeaway:** always inspect the full process tree
   (`ps --forest`) rather than a single PID's CPU time before declaring a hang, and treat
   any apparent memory plateau as provisional until confirmed across several consecutive
   checks — a single flat reading during heavy swap use is not reliable evidence of
   stabilization.
