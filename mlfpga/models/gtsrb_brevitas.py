"""
Brevitas QAT version of TinyCNN_GAP for GTSRB — FINN-compatible.

Architecture matches TinyCNN_GAP exactly except:
  - AvgPool2d(8) → MaxPool2d(8): misma reducción 8×8→1×1, sin división (FINN-friendly)
  - Dropout eliminado (inference-only)
  - Todas las capas con pesos/activaciones cuantizadas (Brevitas)

Input: (B, 3, 64, 64)
Spatial path: 64 → 32 → 16 → 8 → 1  (3×MaxPool2d(2) + MaxPool2d(8))
Features en FC: 128 → 256 → 43

FINN export pattern:
  BN se coloca entre QuantConv2d y QuantReLU. Durante export_qonnx,
  FINN absorbe BN en los pesos de Conv (BatchNorm folding) → solo queda Conv cuantizado.
"""

import torch
import torch.nn as nn
from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import (
    Int8ActPerTensorFloat,
    Int8WeightPerTensorFloat,
    Int4WeightPerTensorFloatDecoupled,
)


class TinyCNN_GAP_Brevitas(nn.Module):
    """
    FINN-compatible Brevitas CNN for GTSRB, 64x64 input.
    ~29 K parámetros (igual que TinyCNN_GAP).
    """

    def __init__(self, num_classes: int = 43, weight_bits: int = 4, act_bits: int = 8):
        super().__init__()

        WQ = Int4WeightPerTensorFloatDecoupled if weight_bits == 4 else Int8WeightPerTensorFloat
        AQ = Int8ActPerTensorFloat

        # Cuantiza la imagen de entrada [0,1] float a INT8
        self.q_in = QuantIdentity(act_quant=AQ, return_quant_tensor=True)

        # Block 1: 3→32, 64×64 → 32×32
        self.conv1 = QuantConv2d(3, 32, kernel_size=3, padding=1,
                                 weight_quant=WQ, return_quant_tensor=True)
        self.bn1   = nn.BatchNorm2d(32)
        self.act1  = QuantReLU(act_quant=AQ, return_quant_tensor=True)
        self.pool1 = nn.MaxPool2d(2)

        # Block 2: 32→64, 32×32 → 16×16
        self.conv2 = QuantConv2d(32, 64, kernel_size=3, padding=1,
                                 weight_quant=WQ, return_quant_tensor=True)
        self.bn2   = nn.BatchNorm2d(64)
        self.act2  = QuantReLU(act_quant=AQ, return_quant_tensor=True)
        self.pool2 = nn.MaxPool2d(2)

        # Block 3: 64→128, 16×16 → 8×8
        self.conv3 = QuantConv2d(64, 128, kernel_size=3, padding=1,
                                 weight_quant=WQ, return_quant_tensor=True)
        self.bn3   = nn.BatchNorm2d(128)
        self.act3  = QuantReLU(act_quant=AQ, return_quant_tensor=True)
        self.pool3 = nn.MaxPool2d(2)

        # QuantIdentity después de cada MaxPool re-ancla la cuantización.
        # MaxPool2d rompe la cadena QuantTensor (no es una capa Brevitas);
        # el QuantIdentity siguiente re-cuantiza la salida float y devuelve QuantTensor.
        # FINN exporta estos nodos como MultiThreshold → hardware nativo.
        self.q2 = QuantIdentity(act_quant=AQ, return_quant_tensor=True)
        self.q3 = QuantIdentity(act_quant=AQ, return_quant_tensor=True)

        # MaxPool2d(8) reemplaza AvgPool2d(8): 8×8 → 1×1, 128 features.
        # MaxPool es pass-through en hardware — FINN lo mapea a una FIFO comparadora.
        # AvgPool requeriría división entera, cara en FPGA dataflow.
        self.pool_final = nn.MaxPool2d(8)
        self.q_final = QuantIdentity(act_quant=AQ, return_quant_tensor=True)

        self.flatten = nn.Flatten()

        # Classifier: 128 → 256 → 43
        self.fc1  = QuantLinear(128, 256, bias=True,
                                weight_quant=WQ, return_quant_tensor=True)
        self.act4 = QuantReLU(act_quant=AQ, return_quant_tensor=True)
        self.fc2  = QuantLinear(256, num_classes, bias=True,
                                weight_quant=WQ, return_quant_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.q_in(x)

        # BN recibe .value (float) porque BN no es capa Brevitas.
        # QuantReLU re-entra en el mundo cuantizado después del BN.
        # MaxPool recibe el float de x.value — el QuantIdentity siguiente lo re-cuantiza.
        x = self.conv1(x)
        x = self.bn1(x.value)
        x = self.act1(x)
        x = self.pool1(x.value)

        x = self.q2(x)
        x = self.conv2(x)
        x = self.bn2(x.value)
        x = self.act2(x)
        x = self.pool2(x.value)

        x = self.q3(x)
        x = self.conv3(x)
        x = self.bn3(x.value)
        x = self.act3(x)
        x = self.pool3(x.value)

        x = self.pool_final(x)
        x = self.q_final(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        return x

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
