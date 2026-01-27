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
