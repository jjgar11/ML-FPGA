"""
Export a model to float ONNX for CPU deployment.

Usage:
    # Registry default weights → registry ONNX path:
    python dev/export_onnx.py --model gtsrb_gap

    # Custom weights (e.g. mixed model) → custom output path:
    python dev/export_onnx.py --model gtsrb_gap \
        --pth data/models/gtsrb_gap_mixed.pth \
        --out data/models/gtsrb_gap_mixed.onnx
"""

import argparse
import os
import torch

from mlfpga.models.registry import get_model_spec
from mlfpga.config import MODELS_ROOT


def export_float_onnx(model_name: str, pth: str = None, out: str = None, opset: int = 13):
    spec = get_model_spec(model_name)

    pth_path  = pth or os.path.join(MODELS_ROOT, spec.pth_filename)
    onnx_path = out or os.path.join(MODELS_ROOT, spec.onnx_float_filename)

    model = spec.float_cls()
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy = spec.dummy_input()

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"[OK] {pth_path} → {onnx_path}  ({size_kb:.0f} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["mnist", "wine", "gtsrb", "gtsrb_gap", "gtsrb_gap_s", "nav_cnn", "face_mlp"])
    ap.add_argument("--pth",   default=None,
                    help="Path to .pth weights (default: registry path)")
    ap.add_argument("--out",   default=None,
                    help="Output .onnx path (default: registry path)")
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()
    export_float_onnx(args.model, args.pth, args.out, args.opset)


if __name__ == "__main__":
    main()
