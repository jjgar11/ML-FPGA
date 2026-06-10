from dataclasses import dataclass, field
from typing import Callable, Tuple
import torch
import torch.nn as nn

from .digit_classification import DigitClassificationNN
from .wine_classification import WineNet
from .gtsrb import TinyCNN as GtsrbCNN
from .nav_cnn import NavCNN
from .face_mlp import FaceMLP


@dataclass(frozen=True)
class ModelSpec:
    name: str
    float_cls: Callable[[], nn.Module]
    pth_filename: str
    onnx_float_filename: str
    dummy_input: Callable[[], torch.Tensor]
    input_shape_for_hls4ml: Tuple[int, ...]
    # io_stream is mandatory for spatial (CNN) inputs; io_parallel for flat MLPs.
    default_io_type: str = "io_parallel"


MODEL_REGISTRY: dict = {
    "mnist": ModelSpec(
        name="mnist",
        float_cls=DigitClassificationNN,
        pth_filename="mnist.pth",
        onnx_float_filename="mnist_float.onnx",
        dummy_input=lambda: torch.randn(1, 1, 28, 28),
        input_shape_for_hls4ml=(1, 28 * 28),
        default_io_type="io_parallel",
    ),
    "wine": ModelSpec(
        name="wine",
        float_cls=WineNet,
        pth_filename="wine.pth",
        onnx_float_filename="wine_float.onnx",
        dummy_input=lambda: torch.randn(1, 13),
        input_shape_for_hls4ml=(13,),
        default_io_type="io_parallel",
    ),
    "gtsrb": ModelSpec(
        name="gtsrb",
        float_cls=GtsrbCNN,
        pth_filename="gtsrb.pth",
        onnx_float_filename="gtsrb_float.onnx",
        dummy_input=lambda: torch.randn(1, 3, 32, 32),
        input_shape_for_hls4ml=(1, 3, 32, 32),
        default_io_type="io_stream",
    ),
    "nav_cnn": ModelSpec(
        name="nav_cnn",
        float_cls=NavCNN,
        pth_filename="nav_cnn.pth",
        onnx_float_filename="nav_cnn_float.onnx",
        dummy_input=lambda: torch.randn(1, 1, 64, 64),
        input_shape_for_hls4ml=(1, 1, 64, 64),
        default_io_type="io_stream",
    ),
    # num_classes is set at training time; registry uses a default for hls4ml conversion.
    # Override by passing float_cls=lambda: FaceMLP(num_classes=N) before convert.
    "face_mlp": ModelSpec(
        name="face_mlp",
        float_cls=lambda: FaceMLP(n_components=50, num_classes=2),
        pth_filename="face_mlp.pth",
        onnx_float_filename="face_mlp_float.onnx",
        dummy_input=lambda: torch.randn(1, 50),
        input_shape_for_hls4ml=(50,),
        default_io_type="io_parallel",
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]
