"""
Smoke test for NavCNN: registry, model forward pass, param count,
and synthetic data generator.

Usage:
    python dev/test_nav_cnn.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from mlfpga.models.registry import MODEL_REGISTRY, get_model_spec
from mlfpga.models.nav_cnn import NavCNN

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "gen_nav_synthetic",
    os.path.join(os.path.dirname(__file__), "data", "gen_nav_synthetic.py")
)
_gen = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_gen)


def test_registry():
    assert "nav_cnn" in MODEL_REGISTRY, "nav_cnn missing from registry"
    assert "gtsrb"   in MODEL_REGISTRY, "gtsrb missing from registry"
    spec = get_model_spec("nav_cnn")
    assert spec.default_io_type == "io_stream", "nav_cnn must default to io_stream"
    assert spec.input_shape_for_hls4ml == (1, 1, 64, 64)
    print("[OK] registry")


def test_forward():
    model = NavCNN()
    model.eval()
    x = torch.zeros(2, 1, 64, 64)
    y = model(x)
    assert y.shape == (2, 4), f"Expected (2,4), got {y.shape}"
    total = sum(p.numel() for p in model.parameters())
    print(f"[OK] forward pass  — output {y.shape}  params={total:,}")


def test_generator():
    imgs, lbls = _gen.generate_dataset(n_per_class=20, seed=0)
    assert imgs.shape == (80, 64, 64), f"Unexpected shape {imgs.shape}"
    assert lbls.shape == (80,)
    assert imgs.min() >= 0.0 and imgs.max() <= 1.0
    counts = {c: (lbls == c).sum() for c in range(4)}
    assert all(v == 20 for v in counts.values()), f"Uneven class counts: {counts}"
    print(f"[OK] generator     — {imgs.shape}  classes={counts}")


def test_onnx_export():
    import tempfile
    model = NavCNN()
    model.eval()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    torch.onnx.export(model, torch.zeros(1, 1, 64, 64), path,
                      input_names=["input"], output_names=["logits"],
                      opset_version=11)
    assert os.path.getsize(path) > 0
    os.unlink(path)
    print("[OK] ONNX export")


if __name__ == "__main__":
    test_registry()
    test_forward()
    test_generator()
    test_onnx_export()
    print("\nAll tests passed.")
