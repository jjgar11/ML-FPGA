# ML-FPGA Project — Status Report

**Date:** 2026-06-10  
**Platform:** Ultra96-v2 (Zynq UltraScale+ xczu3eg-sbva484-1-i)  
**Scope:** Real-time face recognition using hls4ml-synthesized MLP on FPGA

---

## 1. Toolchain — hls4ml Pipeline

| Component | Current State | Expected State |
|---|---|---|
| PyTorch model training | Done — FaceMLP (50→32→16→2), 100% float accuracy | Same |
| ONNX export | Done | Same |
| hls4ml HLS synthesis | Done — `ap_fixed<16,6>`, `io_parallel` | Same |
| Vivado HLS IP generation | Done — AXI-Lite slave IP | Same |
| Vivado block design | Done — IP integrated with PS, AXI Interconnect | Same |
| Bitstream generation | Done — `face_mlp_wrapper.bit.bin` | Same |
| Accuracy after quantization | **62.5%** (down from 100% float) | >80% with retraining or higher precision |

**Problem encountered:** hls4ml with `ap_fixed<16,6>` introduces significant quantization error for a network this small. The 16-bit fixed-point representation with only 6 integer bits clips values outside [-32, 32), causing accuracy loss that can only be recovered by retraining with quantization-aware training or using wider types.

---

## 2. FPGA Hardware Integration

### 2a. AXI-Lite / UIO Driver

| Component | Current State | Expected State |
|---|---|---|
| UIO driver loading | Done — `uio_pdrv_genirq of_id=generic-uio` | Same |
| IP physical address | `0xa0000000`, confirmed via `/sys/bus/platform/devices` | Same |
| UIO device assignment | `/dev/uio4` | Same |
| Automated overlay script | Done — `load_overlay.sh` | Same |

**Problem encountered — `rmmod` missing from script:** The UIO driver `uio_pdrv_genirq` was already loaded in memory without the `of_id=generic-uio` parameter from a previous boot. Calling `modprobe` on an already-loaded module is a no-op — the parameter is ignored and the driver never claims `compatible = "generic-uio"` nodes. The IP appeared in the device tree but `/dev/uio4` was never created. Fix: always `rmmod uio_pdrv_genirq` before `modprobe`.

### 2b. Device Tree Overlay

| Component | Current State | Expected State |
|---|---|---|
| DTS structure | Two-fragment overlay: `/fpga-full` (firmware-name) + `/amba` (peripheral) | Same |
| Overlay application | Via configfs — `/sys/kernel/config/device-tree/overlays/` | Same |
| Bitstream loading | Triggered automatically by `firmware-name` in DTS | Same |
| Peripheral registration | `a0000000.face_mlp` appears in `/sys/bus/platform/devices` | Same |

**Problem encountered — overlay label conflict:** An automated script created a new overlay slot with a different label than the existing one. This triggered a second bitstream reload, resetting the PL and orphaning the peripheral registered under the old overlay label. The device tree entry was still "applied" in configfs but the UIO device disappeared. Fix: remove all existing overlays before applying a new one.

---

## 3. AXI-Lite Inference — Register Map & Handshake

| Component | Current State | Expected State |
|---|---|---|
| ap_ctrl handshake | Working — `ap_start` written to offset `0x00` | Same |
| Input register base | `0x20` — 50 inputs, 2× `ap_fixed<16,6>` packed per 32-bit word | Same |
| Output register base | `0x74` — 2 logits packed in one 32-bit word | Same |
| Inference latency | ~1 ms (AXI-Lite polling, 1 ms sleep) | Sub-ms with interrupt-driven read |

**Problem encountered — ap_rst_n unconnected:** The HLS IP computation core has a separate active-low reset signal (`ap_rst_n`) that is distinct from the AXI-Lite slave reset. In the initial Vivado block design, `ap_rst_n` was left unconnected (floating low), which held the computation core in perpetual reset. The AXI-Lite slave continued to accept reads and writes normally, making the IP appear functional. All output registers read as zero, causing the softmax to produce exactly 50%/50% confidence — the symptom of a dead computation core.

**Problem encountered — Constant(1) on ap_start:** A `Constant` block driving `ap_start` externally bypassed the AXI-Lite control register. The IP started computation continuously, but the `ap_done` bit in the AXI-Lite control register (`0x00`) was never updated because the hardware FSM was not being driven through the AXI-Lite path. Output registers were never refreshed. Fix: remove the Constant block and let the Python script write `ap_start` via AXI-Lite.

**Problem encountered — wrong register map:** Initial assumptions placed inputs at `0x10` and outputs at `0x10`. Empirical testing with known inputs confirmed the correct layout: inputs at `0x20`, outputs at `0x74`.

---

## 4. Face Detection Pipeline

| Component | Current State | Expected State |
|---|---|---|
| Detector | Haar cascade frontal face — CPU (PS) | DNN-based detector (more robust) |
| Input resolution for detection | 640×360 (downscaled from 1920×1080) | Same or native lower-res capture |
| Detection rate | Inconsistent — fails on tilted/angled faces | >80% detection rate |
| False positive rate | Moderate — background objects detected occasionally | Low |
| Cascade file | `haarcascade_frontalface_default.xml` | Same or `alt2` |

**Problem encountered — cascade path case sensitivity:** The cascade XML was located at `/usr/share/OpenCV/haarcascades/` (capital `O` in OpenCV). All initial paths in the script used lowercase (`opencv4`, `opencv`). The `os.path.exists()` check failed silently, `detector` was `None`, and every frame used the full frame as input — producing random predictions regardless of what was in front of the camera.

**Problem encountered — 1920×1080 resolution:** The camera captures at Full HD. Running `detectMultiScale` on a 1920×1080 grayscale image on a Cortex-A53 takes several seconds per frame. The cascade appeared to "never detect" because inference was too slow to keep up with the camera buffer. Fix: downscale to 640×360 before detection (~9× fewer pixels), then scale the bounding box back to original resolution for cropping.

**Problem encountered — empty classifier not handled:** When `CascadeClassifier` loads a file successfully but the XML is malformed or wrong format, `detector.empty()` returns `True`. The code printed a warning but did not set `detector = None`, so `detectMultiScale` was still called on an empty classifier — always returning zero detections. `last_bbox` was never updated and the system was permanently stuck on NO FACE.

---

## 5. Inference Quality

| Component | Current State | Expected State |
|---|---|---|
| Test set accuracy (FPGA) | 62.5% | >85% |
| Confidence on correct predictions | ~0.62–0.65 | >0.75 |
| UNKNOWN threshold | 0.60 | 0.70+ once accuracy improves |
| Training dataset size | Small (few hundred images per person) | 1000+ images, varied conditions |
| Input resolution | 32×32 (1024 pixels) | 64×64 or 112×112 |
| Face alignment | None — raw Haar crop | Landmark-based alignment |

**Root cause of low accuracy:** Three compounding factors:
1. **Small dataset** — the model has not seen enough variation in lighting, angle, and distance.
2. **32×32 resolution** — 1024 pixels contain very limited discriminative information; eyes, nose, and mouth are barely distinguishable after PCA compression.
3. **Quantization loss** — `ap_fixed<16,6>` clips values outside ±32 and truncates 6 fractional bits, introducing error at every layer of the network.

---

## 6. Role of Simpler Models in Validating the Workflow

Deploying the face recognition model directly would have made debugging impossible — a failure could come from the HLS synthesis, the block design, the device tree, the driver, or the inference script. Simpler models were used to confirm each layer works before adding the next.

### Wine Classifier — first end-to-end deployment

- 13 chemical features → 3 wine classes. Inputs are known numbers, so the correct output can be verified manually without any image processing.
- Revealed the correct device tree structure: the Ultra96 OOB 2020.1 kernel requires **two separate DTS fragments** — one under `/fpga-full` for `firmware-name` only, and one under `/amba` for the AXI peripheral. Putting the peripheral inside `/fpga-full` silently fails: the overlay reports `applied` but no device appears.
- Confirmed the driver load order: `uio_pdrv_genirq of_id=generic-uio` must be loaded **before** the overlay is applied, otherwise the kernel has no driver to bind and `/dev/uioN` is never created.
- By the time the face model arrived, bitstream loading, DTS, and AXI-Lite communication were all proven. The `ap_rst_n` and `ap_start` bugs could be diagnosed as hardware-only problems, not toolchain issues.

### MNIST — first image input on FPGA

- 28×28 grayscale digits → 784 inputs → 10 classes. First model with image data as input rather than tabular features.
- Validated the preprocessing pipeline: normalize pixel values, flatten to a 1D array, pack into AXI-Lite registers. The same pattern is used by the face model.
- With 784 inputs, synthesis and place & route took noticeably longer than Wine — this was the first indication that input size directly impacts Vivado iteration time, relevant for planning the face model with 1024 inputs.
- Confirmed that hls4ml correctly handles multi-layer MLPs with ReLU activations at this scale before moving to image data with real semantic content.

### Wine (Brevitas/FINN) — alternative quantization path

- Trained with explicit 4-bit weights using Brevitas, exported to QONNX, compiled with FINN.
- Validates the FINN toolchain as an alternative to hls4ml when aggressive quantization is needed — relevant for future models where `ap_fixed<16,6>` accuracy loss is not acceptable.

---

## 7. Structural Limitations and Workflow Constraints

### Vivado Toolchain — Resource and Iteration Cost

The most significant constraint in this project is not software but infrastructure. Vivado (Xilinx design suite) requires a powerful workstation: 16–32 GB of RAM and multiple CPU cores. It cannot run on the Ultra96 itself. Each iteration of the hardware design follows this cycle:

| Step | Approximate duration |
|---|---|
| hls4ml HLS synthesis (C++ → RTL) | 5–15 minutes |
| Vivado synthesis (RTL → netlist) | 10–20 minutes |
| Vivado implementation (place & route) | 20–60 minutes |
| Bitstream generation | 5–10 minutes |
| Transfer to Ultra96 + overlay application | 1–2 minutes |

**Total per hardware change: 40–90 minutes.** This means that any bug found in the Vivado block design — such as the `ap_rst_n` and `ap_start` issues described in Section 3 — cannot be quickly patched. Each fix requires a full resynthesis cycle. This severely limits the number of hardware iterations that can be completed in a development session and makes hardware-level bugs disproportionately costly compared to software bugs.

### AXI-Lite Register Map Is Not Documented at Runtime

hls4ml does not expose the AXI-Lite register map in a machine-readable format at runtime. The register offsets for inputs and outputs must be read from the generated C++ source (`myproject.cpp`) or inferred empirically by writing known values and scanning offsets for the expected output. For the face model, this required testing multiple offset combinations before confirming `reg_in=0x20` and `reg_out=0x74`.

### ap_rst_n — Silent Failure in Hardware

The HLS IP exposes two independent reset domains: one for the AXI-Lite slave and one for the computation core (`ap_rst_n`, active low). If `ap_rst_n` is unconnected in the block design, it floats low and holds the computation core in perpetual reset. The AXI-Lite slave continues to respond to reads and writes normally — registers can be written and read back — but no inference is ever computed and all output registers return zero. From the software side this is indistinguishable from a model that always predicts equal probabilities, which in turn looks like an undertrained classifier. Diagnosing this required ruling out the software, the register map, and the model before suspecting the block design.

### io_parallel Limits Input Size

hls4ml's `io_parallel` interface places all input values in AXI-Lite registers simultaneously. This works for small inputs (50 values for the face MLP) but becomes impractical for larger models. A 64×64 grayscale image has 4096 inputs; a 112×112 image has 12544 inputs. Each value requires a dedicated register and the AXI-Lite address space and synthesis complexity grow linearly. Scaling the input resolution beyond 32×32 without switching to `io_stream` + AXI DMA is not feasible with the current architecture.

---

## 8. Current Compromises and the Path to a Robust System

### Where each task runs today — and what it would take to move it to the FPGA

| Task | Runs on | What is needed to move to FPGA |
| --- | --- | --- |
| Camera capture | CPU | Always on CPU — hardware interface to PS |
| Grayscale conversion + resize | CPU | Trivial; could run on FPGA but no benefit worth the effort |
| Face detection (Haar cascade) | CPU | Replace with a DNN detector synthesized as an FPGA IP, or keep on CPU and accept the limitation |
| Crop + resize to 32×32, flatten | CPU | Preprocessing block in HLS; only worth it once detection is on FPGA |
| PCA: 1024 → 50 | CPU | Integrate as a frozen `nn.Linear` in the PyTorch model — hls4ml synthesizes it like any dense layer |
| **MLP: 50 → 2** | **FPGA** | Already there |
| Softmax + label | CPU | Trivial float operation; no benefit moving it |

The FPGA currently accelerates only the last dense layer. Moving PCA to the FPGA is the highest-value change: it requires no new hardware IP, only retraining and re-synthesis, and it enables switching to AXI DMA — which in turn unlocks larger inputs and higher resolution.

---

v1 proves the full toolchain end-to-end. Every limitation below is a deliberate compromise, not a dead end — each has a clear unlock that builds on what already works.

---

**FPGA only runs the 50→2 MLP. The CPU does Haar cascade + PCA — the heavier parts.**

- *Why:* `io_parallel` maps each input to an AXI-Lite register. 1024 inputs is not feasible this way. PCA was also kept outside the model at training time.
- *Unlock:* Integrate PCA as a frozen layer in PyTorch → re-synthesize with `io_stream` → add AXI DMA to the Vivado block design.
- *Result:* CPU sends raw pixels. FPGA runs the full pipeline. The architecture finally matches the intent.

---

**Input resolution is 32×32 — only 1024 pixels per face.**

- *Why:* Directly tied to the AXI-Lite / `io_parallel` constraint above. More pixels = more registers = not feasible.
- *Unlock:* AXI DMA removes the register count limit. Resolution becomes a training decision, not a hardware constraint.
- *Result:* 64×64 or 112×112 inputs. Faces are actually distinguishable. Expected large accuracy gain.

---

**62.5% FPGA accuracy. Confidence barely clears the UNKNOWN threshold.**

- *Why:* Two compounding causes — small dataset (few hundred images per person, limited conditions) and `ap_fixed<16,6>` quantization (clips values outside ±32, loses precision at every layer).
- *Unlock:* More training data with varied lighting and angles + retrain with quantization-aware training or wider types (`ap_fixed<16,8>`).
- *Result:* Accuracy approaches the float baseline. Confidence rises above 0.75. Recognition becomes reliable enough to use.

---

**Hardware changes take 40–90 minutes per iteration.**

- *Why:* Every change to the HLS source or block design requires a full Vivado synthesis + implementation + bitstream generation cycle. Vivado needs 16–32 GB RAM and cannot run on the Ultra96 itself.
- *Unlock:* Access to a more powerful workstation or remote Vivado server. Out-of-context synthesis can reuse unchanged IP blocks.
- *Result:* More experiments per session. Block design bugs (like `ap_rst_n`) go from day-long investigations to quick fix-and-verify cycles.

---

**Haar cascade fails on tilted faces and occasionally detects backgrounds instead.**

- *Why:* Haar is the only practical CPU-only detector on the Cortex-A53. DNN-based detectors are more robust but require a model file and more compute.
- *Unlock:* OpenCV DNN module with a pre-trained SSD face model — runs on CPU, no GPU needed.
- *Result:* Detection works at any angle and lighting. False positives drop. Bounding boxes cover the actual face, which is a hard prerequisite for the classifier to produce meaningful results.

---

## 9. Current System Architecture (v1)

```
[USB Camera — 1920×1080]
         ↓ BGR frame
[CPU — PS Cortex-A53]
         ↓ grayscale + resize to 640×360
         ↓ Haar cascade detectMultiScale() → face bounding box
         ↓ crop + resize to 32×32 → flatten → 1024 floats
         ↓ PCA: 1024 → 50  (sklearn, precomputed matrix)
         ↓ quantize to ap_fixed<16,6>
[FPGA — PL via AXI-Lite + UIO]
         ↓ FaceMLP: 50 → 32 → 16 → 2  (hls4ml, io_parallel)
[CPU]
         ↓ softmax → label (juan / felipe / UNKNOWN)
```

---

## 9. Next Steps — Unlock Chain

Each item below is a prerequisite for the ones that follow. Completing it unblocks progress toward the final goal: a robot car with real-time vision inference on the FPGA.

| Step | What needs to be done | What it unlocks |
|---|---|---|
| **1. Improve face detection** | Replace Haar cascade with a DNN-based detector (e.g. OpenCV DNN + SSD face model) | Reliable bounding boxes at any angle and lighting → makes all subsequent accuracy measurements meaningful |
| **2. Expand training dataset** | Collect 1000+ images per person with varied lighting, distance, and angle using the improved detector | Eliminates dataset size as a variable; allows fair evaluation of model and quantization quality |
| **3. Increase input resolution** | Move from 32×32 to 64×64 or 112×112 | More discriminative features available to PCA and the MLP; expected large accuracy gain |
| **4. Integrate PCA into the model** | Add a frozen `nn.Linear(1024, 50)` layer at the start of the PyTorch model with PCA weights | Allows the entire feature extraction + classification pipeline to run on the FPGA; enables AXI DMA use |
| **5. Retrain with quantization-aware training** | Use QAT or wider fixed-point types (`ap_fixed<16,8>`) to recover quantization accuracy loss | Brings FPGA accuracy closer to float baseline; raises confidence scores above the UNKNOWN threshold reliably |
| **6. Switch to `io_stream` + AXI DMA** | Re-synthesize with hls4ml `io_stream`, add AXI DMA to Vivado block design | Handles 1024-input model efficiently; eliminates per-register AXI-Lite bottleneck; scales to larger CNNs |
| **7. Integrate navigation CNN** | Synthesize and deploy the navigation CNN on the same FPGA fabric | Face recognition + navigation inference running concurrently on the PL |
| **8. Robot car integration** | Connect Ultra96 to motor controller; wire camera and inference output to steering/speed decisions | Final project goal — autonomous robot car with real-time FPGA-accelerated vision |

## 10. Planned Architecture (v2)

| Change | v1 | v2 |
|---|---|---|
| PCA location | CPU (sklearn) | FPGA (frozen `nn.Linear`) |
| Input to FPGA | 50 values via AXI-Lite | 1024 values via AXI DMA |
| Interface | AXI-Lite (register-by-register) | AXI DMA (stream) |
| hls4ml io type | `io_parallel` | `io_stream` |
| Input resolution | 32×32 | 64×64 or 112×112 |
| Face detector | Haar cascade (CPU) | DNN-based (CPU) |
| FPGA workload | FaceMLP only (50→2) | PCA + FaceMLP (1024→2) |
