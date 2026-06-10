# ML-to-FPGA Architecture Overview

This document describes the evaluated ML-to-FPGA toolchains and the rationale
behind architectural decisions.

---

## Project Goal

Explore the use of HLS tools to deploy AI/ML models on FPGA platforms
(Ultra96-v2), with support for integer and quantized arithmetic
(e.g., 4-bit and 8-bit).

---

## Toolchains Evaluated

### 1. hls4ml-Based Pipeline

#### Workflow

- Model training: PyTorch (floating-point)
- Export format: ONNX (floating-point)
- Quantization: Defined in hls4ml configuration
- Backend: Vivado HLS
- Target: Ultra96-v2 FPGA

#### Characteristics

- Quantization is applied during HLS generation using fixed-point types
  (`ap_fixed`, `ap_int`).
- ONNX graph must be free of explicit quantization operators.

#### Advantages

- Simple and fast iteration.
- Well suited for baseline FPGA inference.
- Easy integration with Vivado HLS.

#### Limitations

- Limited support for ONNX graphs with explicit quantization (Q/DQ ops).
- Not ideal for aggressive low-bit (e.g., 4-bit) quantization.

---

### 2. FINN-Based Pipeline

#### Workflow

- Model training: Quantized training (Brevitas / QAT or PTQ)
- Export format: QONNX
- Quantization: Explicit (4-bit, 8-bit)
- Backend: FINN compiler
- Target: FPGA accelerator

#### Characteristics

- Quantization is part of the computational graph.
- Designed specifically for FPGA-friendly inference.

#### Advantages

- Native support for low-bit inference (4-bit, 8-bit).
- FPGA-oriented architecture optimizations.

#### Limitations

- Higher setup complexity.
- Longer development cycle.

---

## Architectural Decision

Both pipelines are relevant and complementary:

- hls4ml is used for rapid prototyping and validation.
- FINN is used for advanced low-bit quantized inference.

This approach aligns with the project requirement to explore multiple
HLS-based ML deployment strategies.

---

## Face Recognition Demo — Current State and Planned Evolution

### Current Architecture (v1 — working demo)

```text
[USB Camera]
     ↓ BGR frame (e.g. 640×480)
[CPU — PS Cortex-A53]
     ↓ grayscale conversion
     ↓ Haar cascade detectMultiScale() → face bounding box
     ↓ crop + resize to 32×32 → flatten → 1024 floats
     ↓ PCA (50 components): 1024 → 50  (precomputed 50×1024 matrix)
     ↓ quantize to ap_fixed<16,6>
[FPGA — PL, via AXI-Lite + UIO]
     ↓ FaceMLP: 50 → 32 → 16 → 2  (hls4ml, io_parallel)
[CPU]
     ↓ softmax → label (juan / felipe / UNKNOWN)
```

**Known limitations of v1:**

- Face detection (Haar cascade) runs on CPU — the heaviest part of the pipeline.
- PCA also runs on CPU, even though it is mathematically part of the ML model.
- The FPGA only accelerates the 50→2 MLP, which is computationally trivial.
- 32×32 is very low resolution; with only 1024 pixels the network has limited discriminative information.
- AXI-Lite interface (UIO): per-register access latency, inefficient for larger input volumes.
- Limited precision: quantization to ap_fixed<16,6> introduces error; ~62% accuracy vs 100% float.

### Target Architecture (v2 — full FPGA integration)

PCA is a linear operation (`embedding = (x - mean) @ components.T`) equivalent to a
`nn.Linear` layer without activation. It can be integrated directly into the PyTorch model
as a frozen layer, and hls4ml converts it the same as any other dense layer.

Unified model:

```text
input: 1024 pixels (32×32 grayscale, normalized)
  → Linear(1024, 50, bias=True)  [weights = PCA components, bias = -components @ mean]
  → Linear(50, 32) → ReLU
  → Linear(32, 16) → ReLU
  → Linear(16, 2)
output: 2 logits
```

Target pipeline:

```text
[USB Camera]
     ↓ frame
[CPU — PS]
     ↓ Haar cascade → crop → resize 32×32 → 1024 floats
     ↓ AXI DMA (stream of 1024 × 16-bit values)
[FPGA — PL, io_stream + AXI DMA]
     ↓ PCA + full FaceMLP in hardware
     ↓ AXI DMA (2 logits back)
[CPU]
     ↓ softmax → label
```

**Changes required for v2:**

1. Retrain with PCA integrated into the PyTorch model (frozen `nn.Linear`).
2. Increase input resolution (64×64 or 112×112 instead of 32×32).
3. Re-synthesize with hls4ml using `io_stream` and more inputs.
4. Redesign Vivado BD with AXI DMA instead of bare AXI-Lite.
5. Optionally replace Haar cascade with a more robust face detector (e.g. DNN-based) on CPU.
