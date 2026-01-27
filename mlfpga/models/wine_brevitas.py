import torch
import torch.nn as nn

from brevitas.nn import QuantLinear, QuantReLU, QuantIdentity
from brevitas.quant import Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint

from brevitas.quant import Int4WeightPerTensorFloatDecoupled

class WineBrevitasMLP_8b8b(nn.Module):
    """Weights 8-bit fixed-point, activations 8-bit fixed-point."""

    def __init__(self):
        super().__init__()
        WQ = Int8WeightPerTensorFixedPoint
        AQ = Int8ActPerTensorFixedPoint

        self.q_in = QuantIdentity(act_quant=AQ, return_quant_tensor=True)

        self.fc1 = QuantLinear(13, 32, weight_quant=WQ, bias=True, return_quant_tensor=True)
        self.act1 = QuantReLU(act_quant=AQ, return_quant_tensor=True)

        self.fc2 = QuantLinear(32, 16, weight_quant=WQ, bias=True, return_quant_tensor=True)
        self.act2 = QuantReLU(act_quant=AQ, return_quant_tensor=True)

        self.fc3 = QuantLinear(16, 3, weight_quant=WQ, bias=True, return_quant_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.q_in(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x

class WineBrevitasMLP_4b8b(nn.Module):
    """
    Weights 4-bit (float-decoupled), activations 8-bit fixed-point.
    Good compromise for early 4-bit experiments in 0.12.1.
    """

    def __init__(self):
        super().__init__()
        WQ = Int4WeightPerTensorFloatDecoupled
        AQ = Int8ActPerTensorFixedPoint

        self.q_in = QuantIdentity(act_quant=AQ, return_quant_tensor=True)

        self.fc1 = QuantLinear(13, 32, weight_quant=WQ, bias=True, return_quant_tensor=True)
        self.act1 = QuantReLU(act_quant=AQ, return_quant_tensor=True)

        self.fc2 = QuantLinear(32, 16, weight_quant=WQ, bias=True, return_quant_tensor=True)
        self.act2 = QuantReLU(act_quant=AQ, return_quant_tensor=True)

        self.fc3 = QuantLinear(16, 3, weight_quant=WQ, bias=True, return_quant_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.q_in(x)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x
