# Project Roadmap — ML on FPGA

This document outlines the planned steps for completing the ML-to-FPGA project.

---

## Phase 1 — Baseline ML Models (Completed)

- Implement and train MLP models (MNIST, Wine).
- Validate inference accuracy.
- Save floating-point models.

Status: ✅ Completed

---

## Phase 2 — Post-Training Quantization Evaluation (Completed)

- Apply PyTorch PTQ (int8).
- Validate accuracy preservation.
- Export quantized models.

Status: ✅ Completed

---

## Phase 3 — hls4ml Pipeline (Completed)

- Export floating-point models to ONNX.
- Convert ONNX models using hls4ml.
- Define 8-bit and 4-bit fixed-point arithmetic in HLS configuration.
- Perform C-simulation and synthesis.
- Deploy on Ultra96-v2.

Status: ✅ Completed

---

## Phase 4 — FINN Pipeline (Planned)

- Implement quantized models using Brevitas.
- Train using QAT or PTQ.
- Export to QONNX.
- Apply FINN transformations.
- Generate FPGA accelerator.

Status: ⏳ Planned

---

## Phase 5 — Evaluation & Comparison

- Compare accuracy, resource usage, and performance:
    - Floating-point vs quantized
    - hls4ml vs FINN
- Document findings and trade-offs.

Status: ⏳ Planned
