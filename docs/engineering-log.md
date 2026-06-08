# Engineering Log — ML to FPGA Project

This document records technical experiments, decisions, and findings during
the exploration of ML-to-FPGA toolchains. Its purpose is to document both
successful and unsuccessful approaches, providing traceability for design
decisions.

---

## January 2026 — Summary of Work (High Level)

- Baseline float models (MNIST, Wine) trained and validated.
- PyTorch PTQ (qnnpack) validated in software; quantized `.pth` created.
- ONNX-based hls4ml conversion explored and found fragile/unsupported for key ops.
- hls4ml pipeline stabilized by switching to the PyTorch frontend (torch.fx-based).
- HLS project generation works; Vivado HLS toolchain required for build/synthesis.
- FINN pipeline started using Brevitas + QONNX with QAT-ready quantized models.

---

## [DATE] — Initial ML Baseline Models

Goal:
Train simple ML models suitable for FPGA deployment.

Actions:

- Implemented MLP-based classifiers for:
    - MNIST digit classification
    - Wine dataset classification
- Training performed using PyTorch.
- Achieved >96% accuracy on MNIST and >97% on Wine dataset.

Outcome:
Baseline floating-point models validated and saved for further experiments.

---

## [DATE] — Post-Training Quantization (PTQ) in PyTorch (Software Validation)

Goal:
Evaluate 8-bit integer inference using Post-Training Quantization.

Approach:

- Applied static PTQ using PyTorch quantization APIs.
- Used `qnnpack` backend.
- Inserted QuantStub / DeQuantStub in model definitions.
- Calibrated models using validation datasets.

Results:

- Quantized models achieved similar accuracy to floating-point models.
- Model parameters successfully converted to `torch.qint8`.
- Quantized models saved as `.pth`.

Conclusion:
PTQ in PyTorch is effective for 8-bit inference at the software level.

---

## [DATE] — Quantized Model Export to ONNX (Investigation)

Goal:
Export quantized PyTorch models to ONNX format for FPGA toolchains.

Approach:

- Exported quantized PyTorch models (`int8`) to ONNX.
- Verified inference correctness post-export.
- Observed reduction in model file size.

Results:

- ONNX export succeeded.
- ONNX graphs contained explicit quantization operators
  (QuantizeLinear / DequantizeLinear / QLinear ops).

---

## [DATE] — ONNX to hls4ml Conversion Attempt (Findings)

Goal:
Convert ONNX models to HLS using hls4ml.

Approach:

- Attempted ONNX frontend with `config_from_onnx_model` / `convert_from_onnx_model`.
- Targeted Vivado backend and Ultra96-v2.

Findings:

- ONNX frontend showed limitations/fragility across multiple issues:
    - Unsupported ops encountered (e.g., Constant, Gemm) depending on graph form
    - Shape inference issues arising from reshape/flatten patterns

Conclusion:
The pipeline:
PyTorch PTQ → Quantized ONNX → hls4ml
is not a robust path with the current tool versions. This is a toolchain
compatibility limitation rather than a model implementation issue.

---

## [DATE] — Toolchain Strategy Revision

Decision:
Split the project into two complementary pipelines:

1) hls4ml pipeline using floating-point models and quantization defined
   at the HLS configuration level.

2) FINN pipeline using explicit low-bit quantization (4-bit / 8-bit) via
   quantization-aware tooling.

Rationale:

- hls4ml quantization is most reliable when set in the hls4ml config, not via
  ONNX quantization operators.
- FINN is designed for explicit low-bit quantized graphs and FPGA accelerators.

---

## [DATE] — hls4ml Pipeline Stabilization (PyTorch Frontend)

Goal:
Establish a stable hls4ml flow without relying on ONNX parsing.

Approach:

- Switched to hls4ml PyTorch frontend (torch.fx-based):
    - `config_from_pytorch_model`
    - `convert_from_pytorch_model`
- Added fixed-point precision policies for int8/int4 using `ap_fixed`.

Results:

- MNIST model successfully interpreted and converted to an HLS project.
- Build step fails locally due to missing Vivado HLS executable in PATH.

Blocking Issue:

- Vivado HLS installation not present/available in current environment.
  The project generation works, but C-sim / synth requires Vivado HLS.

---

## [DATE] — hls4ml Wine Conversion Issue (Input Shape)

Goal:
Convert Wine model with the same PyTorch-to-hls4ml flow.

Issue:

- hls4ml interpreted Wine input as 3D shape ([[None, 1, 13]]) and triggered an
  im2col-related Vivado pass, ending in an array split error.

Resolution Plan:

- Adjust Wine `input_shape_for_hls4ml` to a 1D feature vector ((13,)) so the
  network is treated as Dense-only, avoiding unnecessary convolution-oriented passes.

---

## [DATE] — FINN Pipeline Kickoff (Brevitas + QONNX)

Goal:
Start FINN-compatible quantized training and export.

Approach:

- Adopted Brevitas for QAT-style quantized models.
- Confirmed Brevitas version: 0.12.1.
- Enumerated available quantizers to select compatible 8-bit and 4-bit options.

Findings:

- 8-bit fixed-point quantizers available:
    - `Int8WeightPerTensorFixedPoint`, `Int8ActPerTensorFixedPoint`
- 4-bit weight quantizer available (activations not fixed-point 4-bit in this version):
    - `Int4WeightPerTensorFloatDecoupled`

Next Steps:

- Implement and train:
    - Wine 8b/8b model (weights 8-bit, activations 8-bit)
    - Wine 4b/8b model (weights 4-bit, activations 8-bit)
- Export to QONNX for FINN toolchain inside Docker.
