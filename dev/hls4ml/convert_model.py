import argparse
import os
import torch
import hls4ml

from mlfpga.config import MODELS_ROOT, HLS4ML_ROOT
from mlfpga.models.registry import get_model_spec


def make_hls_config(model, input_shape, mode):
    cfg = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=input_shape,
        granularity="name",
        backend="Vivado",
    )

    # Precision policy
    if mode == "int8":
        model_prec = "ap_fixed<16,6>"
        w_prec = "ap_fixed<8,2>"
        r_prec = "ap_fixed<16,6>"
        a_prec = "ap_fixed<24,10>"
    elif mode == "int4":
        model_prec = "ap_fixed<16,6>"
        w_prec = "ap_fixed<4,1>"
        r_prec = "ap_fixed<12,4>"
        a_prec = "ap_fixed<20,8>"
    else:
        raise ValueError("mode must be int8 or int4")

    cfg["Model"]["Precision"] = model_prec
    cfg["Model"]["ReuseFactor"] = 1

    for _, layer_cfg in cfg["LayerName"].items():
        layer_cfg["Precision"] = {
            "result": r_prec,
            "weight": w_prec,
            "bias": r_prec,
            "accum": a_prec,
        }

    return cfg


def convert(model_name: str, mode: str, part: str, csim: bool, synth: bool):
    # 1) Get model spec
    spec = get_model_spec(model_name)

    # 2) Load PyTorch float model
    model = spec.float_cls()
    pth_path = os.path.join(MODELS_ROOT, spec.pth_filename)
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.eval()

    # 3) Build hls4ml config FROM PYTORCH (no ONNX)
    cfg = make_hls_config(model, spec.input_shape_for_hls4ml, mode)

    # 4) Convert directly from PyTorch
    out_dir = os.path.join(HLS4ML_ROOT, f"{model_name}_hls4ml_pytorch_{mode}")

    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape=spec.input_shape_for_hls4ml,
        hls_config=cfg,
        output_dir=out_dir,
        part=part,
        backend="Vivado",
    )

    # 5) Build
    hls_model.compile()
    # hls_model.build(csim=csim, synth=synth, vsynth=False)

    print(f"[OK] hls4ml project generated at: {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mnist", "wine"])
    ap.add_argument("--mode", required=True, choices=["int8", "int4"])
    ap.add_argument("--part", default="xczu3eg-sbva484-1-e")
    ap.add_argument("--csim", action="store_true")
    ap.add_argument("--synth", action="store_true")
    args = ap.parse_args()

    convert(args.model, args.mode, args.part, args.csim, args.synth)


if __name__ == "__main__":
    main()
