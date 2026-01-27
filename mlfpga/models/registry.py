from dataclasses import dataclass
from typing import Callable, Tuple
import torch
import torch.nn as nn

from .digit_classification import DigitClassificationNN
from .wine_classification import WineNet

@dataclass(frozen=True)
class ModelSpec:
    name: str
    float_cls: Callable[[], nn.Module]
    pth_filename: str
    onnx_float_filename: str
    dummy_input: Callable[[], torch.Tensor]
    input_shape_for_hls4ml: Tuple[int, ...]  # if needed later

MODEL_REGISTRY = {
    "mnist": ModelSpec(
        name="mnist",
        float_cls=DigitClassificationNN,
        pth_filename="mnist.pth",
        onnx_float_filename="mnist_float.onnx",
        dummy_input=lambda: torch.randn(1, 1, 28, 28),
        input_shape_for_hls4ml=(1, 28 * 28),
    ),
    "wine": ModelSpec(
        name="wine",
        float_cls=WineNet,
        pth_filename="wine.pth",
        onnx_float_filename="wine_float.onnx",
        dummy_input=lambda: torch.randn(1, 13),
        input_shape_for_hls4ml=(13,),
    ),
}

def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]
