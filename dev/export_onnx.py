import argparse
import os
import torch

from mlfpga.models.registry import get_model_spec
from mlfpga.config import MODELS_ROOT

def export_float_onnx(model_name: str, opset: int = 13):
    spec = get_model_spec(model_name)

    model = spec.float_cls()
    pth_path = os.path.join(MODELS_ROOT, spec.pth_filename)
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy = spec.dummy_input()
    onnx_path = os.path.join(MODELS_ROOT, spec.onnx_float_filename)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    print(f"[OK] Exported float ONNX: {onnx_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mnist", "wine"])
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()
    export_float_onnx(args.model, args.opset)

if __name__ == "__main__":
    main()
