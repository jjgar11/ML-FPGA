# ML-FPGA Project — Technical Status Report

**Date:** 2026-06-10  
**Platform:** Ultra96-v2 (Zynq UltraScale+ xczu3eg-sbva484-1-i)  
**Previous scope:** Real-time face recognition using an hls4ml-synthesized MLP on FPGA  
**Current note:** After the discussion on 2026-06-10, the ML target changed. This document summarizes the validated workflow, technical progress, issues found, and reusable components for the next project direction.

---

## 1. Context After the Scope Change

The previous implementation focused on a face-recognition pipeline running on the Ultra96-v2, with the final MLP inference step accelerated on the FPGA. Although the application target has changed, most of the engineering work remains directly relevant:

- PyTorch to ONNX to hls4ml conversion flow
- HLS IP generation and Vivado block design integration
- Bitstream generation and deployment through FPGA Manager
- Device-tree overlay workflow on the Ultra96-v2
- UIO-based AXI-Lite communication from Linux
- Register-map validation and debugging methodology
- Identification of architectural limits of AXI-Lite / `io_parallel`
- Preparation for a future `io_stream` + AXI DMA design

The main result is that the complete toolchain from model training to FPGA inference has been validated end-to-end on the Ultra96-v2.

---

## 2. hls4ml Pipeline Status

| Component | Current State | Notes |
|---|---|---|
| PyTorch model training | Done — FaceMLP (50→32→16→2), 100% float accuracy | Previous face-recognition target |
| ONNX export | Done | Export path validated |
| hls4ml HLS synthesis | Done — `ap_fixed<16,6>`, `io_parallel` | Synthesizes successfully |
| Vivado HLS IP generation | Done — AXI-Lite slave IP | IP generated and integrated |
| Vivado block design | Done — IP connected to PS through AXI Interconnect | Several reset/control issues debugged |
| Bitstream generation | Done — `face_mlp_wrapper.bit.bin` | Deployable on Ultra96-v2 |
| Accuracy after quantization | 62.5% FPGA accuracy | Needs QAT, wider fixed-point type, or better input pipeline |

### Quantization issue

The float model reached 100% accuracy on the small test set, but the FPGA version with `ap_fixed<16,6>` dropped to 62.5% accuracy. In `ap_fixed<16,6>`, there are 16 total bits and 6 integer bits, leaving 10 fractional bits. The representable range is approximately `[-32, 32)`. Values outside this range can be clipped, and each layer introduces fixed-point rounding/quantization effects. For this small model and dataset, these effects were enough to reduce classification quality significantly.

Possible solutions:

- retrain with quantization-aware training,
- use a wider type such as `ap_fixed<16,8>` or `ap_fixed<18,8>`,
- normalize or rescale intermediate activations more carefully,
- increase the dataset size and improve preprocessing before quantization.

---

## 3. FPGA Hardware Integration

### 3.1 AXI-Lite / UIO Driver

| Component | Current State | Notes |
|---|---|---|
| UIO driver loading | Done — `uio_pdrv_genirq of_id=generic-uio` | Must be loaded with the correct parameter |
| IP physical address | `0xa0000000` | Confirmed through `/sys/bus/platform/devices` |
| UIO device assignment | `/dev/uio4` | Created after correct overlay and driver binding |
| Automated overlay script | Done — `load_overlay.sh` | Automates bitstream and overlay loading |

### Issue found: UIO driver already loaded without parameter

The UIO driver `uio_pdrv_genirq` had previously been loaded without the `of_id=generic-uio` parameter. Calling `modprobe` again while the module was already loaded did not update the parameter, so the driver did not bind to nodes with `compatible = "generic-uio"`. As a result, the device appeared in the device tree but `/dev/uio4` was not created.

**Fix:** the overlay loading script should remove and reload the driver explicitly:

```bash
rmmod uio_pdrv_genirq
modprobe uio
modprobe uio_pdrv_genirq of_id=generic-uio
```

---

### 3.2 Device-Tree Overlay

| Component | Current State | Notes |
|---|---|---|
| DTS structure | Two-fragment overlay | `/fpga-full` for firmware, `/amba` for peripheral |
| Overlay application | Via configfs | `/sys/kernel/config/device-tree/overlays/` |
| Bitstream loading | Triggered by `firmware-name` | Uses FPGA Manager |
| Peripheral registration | `a0000000.face_mlp` | Appears under `/sys/bus/platform/devices` |

### Issue found: overlay label conflict

An automated script created a new overlay slot with a different label while an older overlay was still present. This triggered a second PL reload, resetting the programmable logic and orphaning the peripheral registered by the previous overlay. The old overlay still appeared as applied in configfs, but the UIO device disappeared.

**Fix:** remove all existing overlays before applying a new one.

---

## 4. AXI-Lite Inference: Register Map and Handshake

| Component | Current State | Notes |
|---|---|---|
| `ap_ctrl` handshake | Working | `ap_start` written to offset `0x00` |
| Input register base | `0x20` | 50 inputs, two `ap_fixed<16,6>` values packed per 32-bit word |
| Output register base | `0x74` | Two logits packed in one 32-bit word |
| Inference latency | ~1 ms | Current script uses polling and a 1 ms sleep |

### Issue found: `ap_rst_n` left unconnected

The HLS IP exposes a separate active-low reset signal, `ap_rst_n`, for the computation core. This signal is distinct from the AXI-Lite slave reset. In the initial Vivado block design, `ap_rst_n` was left unconnected, which effectively held the computation core in reset. The AXI-Lite slave still accepted reads and writes, so the IP looked functional from software, but all output registers stayed at zero.

The visible symptom was a constant 50% / 50% softmax output, which initially looked like a model-quality issue rather than a hardware reset issue.

**Fix:** connect `ap_rst_n` correctly in the Vivado block design.

### Issue found: external Constant block driving `ap_start`

A Constant block was initially connected to `ap_start`, which bypassed the AXI-Lite control register. The model appeared to run continuously, but the AXI-Lite control register did not update `ap_done` as expected and the outputs were not refreshed correctly.

**Fix:** remove the Constant block and start inference by writing `ap_start` through the AXI-Lite control register.

### Issue found: wrong initial register-map assumption

The first software tests assumed inputs and outputs started at `0x10`. Empirical tests with known inputs confirmed the correct layout:

- inputs: `0x20`
- outputs: `0x74`

---

## 5. Face Detection and Preprocessing Pipeline

| Component | Current State | Limitation |
|---|---|---|
| Detector | Haar cascade frontal face — CPU | Inconsistent on tilted or angled faces |
| Detection input resolution | 640×360 | Downscaled from 1920×1080 for speed |
| False positives | Moderate | Some background objects detected |
| Classifier file | `haarcascade_frontalface_default.xml` | Path and loading must be checked carefully |

### Issue found: cascade path case sensitivity

The cascade XML file was located at `/usr/share/OpenCV/haarcascades/` with a capital `O`. Initial script paths used lowercase variants such as `opencv4` or `opencv`. The file-existence check failed, the detector was not initialized correctly, and the full frame was used as input, causing random predictions.

### Issue found: Full HD detection too slow

Running `detectMultiScale()` directly on 1920×1080 grayscale frames on the Cortex-A53 was too slow for practical use. The detector seemed to fail because each frame took too long to process.

**Fix:** downscale frames to 640×360 before detection and scale the bounding box back to the original resolution for cropping.

### Issue found: empty classifier not handled

When `CascadeClassifier` loads an invalid or incompatible XML file, `detector.empty()` returns `True`. The code printed a warning but continued using the empty detector. This caused zero detections and left the system permanently in the `NO FACE` state.

**Fix:** if `detector.empty()` is true, set `detector = None` and handle that case explicitly.

---

## 6. Inference Quality

| Component | Current State | Target for a Robust Version |
|---|---|---|
| FPGA test accuracy | 62.5% | >85% |
| Confidence on correct predictions | ~0.62–0.65 | >0.75 |
| UNKNOWN threshold | 0.60 | 0.70+ after accuracy improves |
| Training dataset size | Few hundred images per person | 1000+ images per person |
| Input resolution | 32×32 | 64×64 or 112×112 |
| Face alignment | None | Landmark-based or more stable crop |

The low accuracy is likely caused by three main factors:

1. **Small dataset:** the model has limited examples across lighting, angle, and distance.
2. **Low input resolution:** 32×32 faces contain limited discriminative information.
3. **Quantization effects:** fixed-point conversion introduces clipping and rounding error across layers.

---

## 7. Validation Through Simpler Models

Deploying the face-recognition model directly would have made debugging difficult, because failures could come from the model, HLS conversion, Vivado block design, device tree, Linux driver, or Python inference script. Simpler models were therefore used to validate the workflow step by step.

### 7.1 Wine classifier — first end-to-end deployment

The Wine classifier used 13 chemical features and produced 3 output classes. Since the inputs are known numerical values, the output can be checked manually without any image-processing dependency.

This model helped validate:

- the two-fragment device-tree overlay structure required by the Ultra96 OOB 2020.1 image,
- the correct placement of `firmware-name` under `/fpga-full`,
- peripheral registration under `/amba`,
- UIO driver loading before overlay application,
- AXI-Lite communication from Python.

By the time the face model was deployed, bitstream loading, DTS structure, and basic AXI-Lite communication had already been validated.

### 7.2 MNIST — first image-input model on FPGA

The MNIST model used 28×28 grayscale images, flattened into 784 inputs and classified into 10 classes. It validated the basic image preprocessing pattern:

- normalize pixels,
- flatten image to a one-dimensional array,
- pack values into AXI-Lite registers,
- read logits back from the FPGA.

MNIST also showed that larger input vectors increase synthesis and place-and-route time, which is relevant for future image models.

### 7.3 Wine with Brevitas/FINN — alternative quantization path

A Wine model was also trained with explicit low-bit quantization using Brevitas, exported to QONNX, and prepared for the FINN toolchain. This validates FINN as a possible alternative to hls4ml for future models that need more aggressive quantization.

---

## 8. Structural Limitations and Workflow Constraints

### 8.1 Vivado iteration cost

The main development constraint is the cost of hardware iterations. Vivado cannot run directly on the Ultra96-v2 and requires a capable workstation. Each hardware change follows this approximate cycle:

| Step | Approximate Duration |
|---|---|
| hls4ml HLS synthesis | 5–15 minutes |
| Vivado synthesis | 10–20 minutes |
| Vivado implementation | 20–60 minutes |
| Bitstream generation | 5–10 minutes |
| Transfer to Ultra96-v2 and overlay application | 1–2 minutes |

A single hardware change can therefore take approximately 40–90 minutes. Bugs in the Vivado block design, such as the `ap_rst_n` and `ap_start` issues, are much more expensive to debug than software-only issues.

### 8.2 AXI-Lite register map not available at runtime

The hls4ml AXI-Lite register map is not exposed in a machine-readable runtime format. Offsets must be read from the generated HLS files or inferred experimentally. For the face model, the correct input and output offsets were confirmed empirically.

### 8.3 `io_parallel` does not scale well for image inputs

The current hls4ml design uses `io_parallel`, which maps each input value to an AXI-Lite register. The face model currently receives 50 values (PCA output) — small enough for this interface. However, the v2 architecture requires sending raw pixels directly to the FPGA, and that input size does not scale:

- current (PCA output): **50 values** — feasible with `io_parallel`
- 32×32 grayscale image: 1024 values
- 64×64 grayscale image: 4096 values
- 112×112 grayscale image: 12544 values

Integrating PCA into the FPGA model requires moving to `io_stream` and AXI DMA, since the register count for raw pixel inputs makes `io_parallel` impractical.

---

## 9. Current Compromises and the Path to a Robust System

Every architectural decision in v1 is a deliberate compromise under time and resource constraints. Each has a specific unlock that builds on what already works.

---

**FPGA only runs the 50→2 MLP. The CPU handles Haar cascade + PCA — the heavier parts.**

- *Why:* `io_parallel` maps each input to an AXI-Lite register. 1024 inputs is not feasible this way. PCA was also kept outside the model at training time.
- *Unlock:* Integrate PCA as a frozen `nn.Linear` layer in PyTorch → re-synthesize with `io_stream` → add AXI DMA to the Vivado block design.
- *Result:* CPU sends raw pixels. FPGA runs the full pipeline. The architecture matches the original intent.

---

**Input resolution is 32×32 — only 1024 pixels per face.**

- *Why:* Directly tied to the `io_parallel` constraint above. More pixels means more registers, which is not feasible.
- *Unlock:* AXI DMA removes the register count limit. Resolution becomes a training decision, not a hardware constraint.
- *Result:* 64×64 or 112×112 inputs. Faces contain enough information to be distinguished reliably.

---

**62.5% FPGA accuracy. Confidence barely clears the UNKNOWN threshold.**

- *Why:* Small dataset (few hundred images per person, limited conditions) and `ap_fixed<16,6>` quantization clipping values outside ±32 with precision loss at every layer.
- *Unlock:* More training data with varied conditions + retrain with quantization-aware training or wider types (`ap_fixed<16,8>`).
- *Result:* Accuracy approaches the float baseline. Confidence rises above 0.75 and recognition becomes reliable.

---

**Hardware changes take 40–90 minutes per iteration.**

- *Why:* Every block design or HLS change requires a full Vivado synthesis + implementation + bitstream cycle. Vivado needs 16–32 GB RAM and cannot run on the Ultra96 itself.
- *Unlock:* Access to a more powerful workstation or remote Vivado server.
- *Result:* More experiments per session. Block design bugs go from day-long investigations to quick fix-and-verify cycles.

---

**Haar cascade fails on tilted faces and occasionally detects backgrounds.**

- *Why:* Haar is the only practical CPU-only detector on the Cortex-A53. DNN-based detectors are more robust but heavier.
- *Unlock:* OpenCV DNN module with a pre-trained SSD face model — runs on CPU, no GPU needed.
- *Result:* Detection works at any angle and lighting. False positives drop. Bounding boxes cover the actual face, which is a prerequisite for the classifier to produce meaningful results.

---

## 10. Current System Architecture (v1)

```text
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

| Task | Current Location | Notes |
|---|---|---|
| Camera capture | CPU | Camera interface handled by Linux/PS |
| Grayscale conversion and resize | CPU | Simple preprocessing |
| Face detection | CPU | Haar cascade; limited robustness |
| Crop, resize, flatten | CPU | Prepares model input |
| PCA: 1024 → 50 | CPU | Could be integrated as frozen `nn.Linear` |
| MLP: 50 → 2 | FPGA | Already deployed |
| Softmax and label selection | CPU | Low computational cost |

---

## 11. Planned Architecture (v2)

| Change | v1 | v2 |
|---|---|---|
| PCA location | CPU (`sklearn`) | FPGA as frozen `nn.Linear` layer |
| Input to FPGA | 50 values via AXI-Lite | 1024+ values via AXI DMA |
| Interface | AXI-Lite | AXI DMA / AXI Stream |
| hls4ml I/O type | `io_parallel` | `io_stream` |
| Input resolution | 32×32 | 64×64 or 112×112 |
| Face detector | Haar cascade on CPU | DNN-based detector on CPU |
| FPGA workload | MLP only | PCA + MLP |

The next architectural step is to integrate PCA into the PyTorch model as a frozen linear layer, then re-synthesize the model using `io_stream` and connect it through AXI DMA. This would remove the AXI-Lite register bottleneck and make larger input sizes feasible.

---

## 12. Next Steps

| Step | Task | Expected Result |
|---|---|---|
| 1 | Replace Haar cascade with a more robust DNN-based detector | More stable bounding boxes and fewer false positives |
| 2 | Expand the training dataset | Better generalization across lighting, distance, and angle |
| 3 | Increase input resolution | More useful visual information for recognition |
| 4 | Integrate PCA as a frozen PyTorch layer | Move feature extraction into the synthesized model |
| 5 | Retrain with QAT or wider fixed-point types | Recover accuracy after quantization |
| 6 | Switch from `io_parallel` to `io_stream` | Prepare the design for AXI DMA |
| 7 | Add AXI DMA to the Vivado block design | Support larger inputs efficiently |
| 8 | Reuse the validated workflow for the new ML target | Apply the established toolchain to the updated project objective |

---

## 13. Summary

The project has validated the complete ML-to-FPGA workflow on the Ultra96-v2:

1. model training in PyTorch,
2. ONNX export,
3. hls4ml conversion,
4. HLS synthesis,
5. Vivado IP integration,
6. bitstream generation,
7. device-tree overlay deployment,
8. Linux UIO access,
9. Python-based AXI-Lite inference.

The current face-recognition model is not yet robust enough for practical use, mainly due to dataset size, detection quality, low input resolution, and quantization effects. However, the toolchain and hardware/software integration path are working. This provides a reusable foundation for the updated ML objective discussed on 2026-06-10.
