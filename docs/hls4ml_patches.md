# Patches on hls4ml-generated code

All of these live in [`dev/hls4ml/convert_model.py`](../dev/hls4ml/convert_model.py)
and are applied automatically when running
`convert_model.py --backend VivadoAccelerator` — nothing needs to be touched by
hand after the project is generated.

**Why they exist:** the AXI-Stream wrapper hls4ml generates natively for
`VivadoAccelerator`, and part of the `myproject.cpp` it generates for CNNs under
`io_stream`, do not compile/synthesize as-is on the Vitis HLS versions we're
working with (2020.1 on the Linux VM, 2021.1 on Windows — same failures on
both). These patches fix that without touching the trained model or the
quantization — only the interface wrapper and a few codegen details.

Reference toolchain: hls4ml 1.2.0, Vitis HLS 2020.1 / 2021.1, Ultra96-v2
(`xczu3eg-sbva484-1-i`). If hls4ml is ever upgraded and one of these stops being
necessary, each function prints an `[INFO]` line when it applies its patch —
compare against a freshly generated project to confirm whether the underlying
issue is still there before removing anything.

---

## 1. `_hls_input_shape` — 4D input shape for CNNs

**Applies to:** all conv models (`gtsrb_gap`, `nav_cnn`, etc.), `io_parallel`.

**Problem:** the model registry (`mlfpga/models/registry.py`) is inconsistent —
some CNN specs carry the batch dimension in their input shape (`1, C, H, W`),
others don't. Passing that leading `1` through to hls4ml breaks the
channels-last transpose optimizer in the `VivadoAccelerator + io_parallel`
flow: it miscomputes the shape (`[3,64,64,1] → [3,64,1]`) and asserts.

**Fix:** normalize `(1, C, H, W) → (C, H, W)` only for 4D conv inputs, before
handing the shape to hls4ml. MLP shapes are left untouched — `(13,)` is already
a flat vector, and `(1, 784)` is required by models whose first layer is
`Flatten(start_dim=1)` (dropping the leading dim there breaks the flatten).

**References:** no public report matching this exact assert was found —
diagnosed directly by reproducing it. Background on hls4ml's channels-last
handling (relevant, not the same bug): [VivadoAccelerator Backend — hls4ml
docs](https://fastmachinelearning.org/hls4ml/backend/accelerator.html), which
notes that channels-last transposition is not supported at all for `io_stream`
and must be done by hand — a related but distinct wrinkle from the one fixed
here.

---

## 2. `_inject_custom_axi_wrapper` — AXI-Stream wrapper for MLPs (`io_parallel`)

**Applies to:** MLPs (`wine`, `mnist`, `face_mlp`), `io_parallel`.

**Problem:** hls4ml's native `VivadoAccelerator` wrapper template
(`hls4ml/templates/vivado_accelerator/myproject_axi.cpp`) declares the
interface **always** as an array of a `{float data; ap_uint<1> last;}` struct
(`input_axi_t in[N_IN]`), regardless of `io_type`. That array-of-struct:

- Does not compile in Vitis 2020.1 with `#pragma HLS INTERFACE axis`
  (`ERROR: Array of user defined type... Use hls::stream<'in'> instead` — HLS
  214-126), **or**
- If naively converted to `hls::stream<input_axi_t>`, the `last` field ends up
  packed *inside* `TDATA` (64 bits, no real `TLAST` pin) — the AXI DMA cannot
  latch onto that at all. This was the root cause of this session's original
  TLAST bug hunt with wine.

**Fix:** a custom wrapper using `ap_axiu<32,0,0,0>` (32-bit TDATA with
`TLAST`/`TKEEP`/`TSTRB` as real, separate signals). It reads `N_IN` float32
beats into an array, calls `myproject(in_buf, out_buf)` (the array signature
`io_parallel` generates), and writes `N_OUT` beats with `TLAST` on the last one.

**References:** same HLS error code, different context (Vitis Vision Library,
not hls4ml, but the identical "array of user-defined type on an axis port"
class of failure): [AMD Adaptive Support — ERROR: [HLS 214-126] Vitis Vision
Library with stream
function](https://adaptivesupport.amd.com/s/question/0D52E00006ljIcpSAE/error-hls-214126-vitis-vision-library-with-stream-function?language=en_US).
Background on why struct-typed AXI4-Stream ports get split/aggregated the way
they do: [AMD UG1399 — AXI4-Stream
Interfaces](https://docs.amd.com/r/2023.1-English/ug1399-vitis-hls/AXI4-Stream-Interfaces).

---

## 3. `_patch_dataflow_weights` — weights outside the DATAFLOW region (`io_stream`)

**Applies to:** CNNs (`gtsrb_gap`, `nav_cnn`), `io_stream`.

**Problem:** hls4ml declares its weight arrays (`w2`, `b2`, ...) via
`#include "weights/w2.h"` at **file scope** in `parameters.h`, outside
`myproject()`. Vitis HLS's canonical dataflow-form check (`HLS 214-113`,
*"Either use an argument of the function or declare the variable inside the
dataflow loop body"*) rejects using those globals as arguments to the
`nnet::conv_2d_cl(..., w2, b2)` calls inside `myproject()`'s
`#pragma HLS DATAFLOW` region. A multi-layer CNN produces dozens of these
214-113 warnings, which — despite being printed as `WARNING:` — abort
pre-synthesis without a final explicit `ERROR:` in the log.

MLPs (`io_parallel`) never hit this — their `myproject()` doesn't use
`#pragma HLS DATAFLOW`.

**Fix:** move the `#include "weights/*.h"` lines from `parameters.h` to
**inside** `myproject()`, right after `#pragma HLS DATAFLOW`.

**References:** this is the confirmed, documented fix — found in hls4ml's own
issue tracker: [Vitis HLS backend PR #629 —
fastmachinelearning/hls4ml](https://github.com/fastmachinelearning/hls4ml/pull/629)
("moving the weights inside the function body right after `#pragma HLS
DATAFLOW` resolved the warning"). Official background on the canonical
dataflow rule being violated: [AMD/Xilinx — Dataflow Canonical Forms (HLS
200-471)](https://www.xilinx.com/htmldocs/xilinx2020_1/hls-guidance/200-471.html)
and [Dataflow Canonical Rules – 2 (HLS
214-114)](https://www.xilinx.com/htmldocs/xilinx2021_2/hls-guidance/214-114.html)
— the sibling check to 214-113, same underlying rule (see #4 below).

---

## 4. `_streamify_axi_wrapper` — AXI-Stream wrapper for CNNs (`io_stream`)

**Applies to:** CNNs (`gtsrb_gap`, `nav_cnn`), `io_stream`.

**Problem:** the same broken native wrapper from #2 (array-of-struct), but here
`myproject()` expects packed streams (`hls::stream<nnet::array<T,N>>`), not
arrays — the MLP wrapper doesn't apply as-is. A first attempt at
"streamifying" the native wrapper (same enqueue/dequeue loops, with an
`is_last` scalar shared between the input loop and the output loop) hit the
canonical dataflow-form check again (`HLS 214-114`): a scalar read and written
by two different "processes" inside the same `#pragma HLS DATAFLOW` region is
not valid.

**Fix:** a canonical three-stage wrapper, each stage its own function
(`axi_to_stream` → `myproject` → `stream_to_axi`), so the DATAFLOW region
contains only declarations and function calls. No `is_last`: `TLAST` is
asserted unconditionally on the last output beat. Packing matches exactly what
hls4ml generates (grouped by `input_t::size` / `result_t::size`), and the
internal stream depths are the same ones hls4ml computed — extracted from the
native wrapper before replacing it, not guessed.

**References:** [AMD/Xilinx — Dataflow Canonical Rules – 2 (HLS
214-114)](https://www.xilinx.com/htmldocs/xilinx2021_2/hls-guidance/214-114.html)
(official docs, exact error code); [Dataflow Canonical Forms (HLS
200-471)](https://www.xilinx.com/htmldocs/xilinx2020_1/hls-guidance/200-471.html)
— the summary check that reported "Dataflow form checks found N issue(s)" in
our logs.

---

## 5. `_patch_design_tcl` — DMA routing and burst size

**Applies to:** all `VivadoAccelerator` builds (both `io_parallel` and
`io_stream`).

**Problem:** not a compile error but a functional bug in the AXI DMA's
DataMover — multi-beat S2MM bursts (`AWLEN>0`) only commit the first 4 bytes
to DDR despite reporting success (IOC set).

**Fix (three changes to `design.tcl`):**

- `-jobs 6/4 → -jobs 2`: caps synthesis parallelism (also relevant to not
  running the VM out of RAM — see the note below).
- Enables `S_AXI_GP1` (HPC1) at 32 bits and redirects the DMA's `M_AXI_S2MM`
  there, through a SmartConnect dedicated to it, separate from MM2S (HPC0).
- Forces `c_s2mm_burst_size` to 2 (was 256) — with a small AWLEN the
  DataMover stops truncating bursts.

**References:** no public report matching the exact "only first beat commits"
symptom was found — diagnosed directly this session via a custom AXI snooper
IP capturing the DataMover's AW/W transactions (see
[dev/fpga/rtl/axi_s2mm_snooper/](../dev/fpga/rtl/axi_s2mm_snooper/)). Closest
relevant official thread, about controlling this same parameter: [AMD Adaptive
Support — How to control AXI DMA S2MM burst write
length](https://adaptivesupport.amd.com/s/question/0D52E00006hpjy5SAA/how-to-control-axi-dma-s2mm-burst-write-length?language=en_US).

---

## Separate issue, not a code patch: synthesis OOM (VM RAM)

Not a `convert_model.py` patch, but the other recurring failure mode: Vitis
HLS synthesis + Technology Mapping for a `ReuseFactor=1` model can ask for more
RAM than the VM has (hit this with wine int8 — it dies during Technology
Mapping, killed by the Linux OOM killer — look for `Killed` or `Segmentation
fault (core dumped)` in the log, not a Vitis `ERROR:`). No specific public
report was found for this one either — it's a well-known general class of
resource-exhaustion issue, not tied to a particular bug report. Fixes, cheapest
first: give the VM more RAM, or raise `--reuse` in `convert_model.py` (less
parallel hardware, lighter synthesis, somewhat higher latency — irrelevant for
these small models).
