"""
Sanity check + single-image inference for GTSRB classifiers.
Run on dev machine after training to confirm the model is ready for hls4ml.

Usage:
    python dev/test_gtsrb.py                          # smoke tests only
    python dev/test_gtsrb.py --arch gtsrb_gap         # test GAP variant
    python dev/test_gtsrb.py --image path/to/crop.png # classify one image
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlfpga.config import DATA_ROOT, MODELS_ROOT
from mlfpga.models.registry import MODEL_REGISTRY, get_model_spec

MEAN = np.array([0.3337, 0.3064, 0.3171], dtype=np.float32)
STD  = np.array([0.2672, 0.2564, 0.2629], dtype=np.float32)

CLASS_NAMES_PATH = os.path.join(DATA_ROOT, "gtsrb_class_names.json")


def load_class_names():
    with open(CLASS_NAMES_PATH) as f:
        raw = json.load(f)
    return [raw[str(i)] for i in range(43)]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def preprocess_file(path, img_size=64):
    """BGR image file → (1, 3, H, W) normalised float32 tensor."""
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (img_size, img_size))
    arr  = rgb.astype(np.float32) / 255.0
    arr  = (arr - MEAN) / STD
    blob = arr.transpose(2, 0, 1)[np.newaxis]
    return torch.from_numpy(blob)


def test_registry(arch):
    assert arch in MODEL_REGISTRY, f"{arch} missing from registry"
    spec = get_model_spec(arch)
    assert spec.default_io_type == "io_stream", f"{arch} must default to io_stream"
    assert spec.input_shape_for_hls4ml[0:2] == (1, 3), f"Expected 3-channel input, got {spec.input_shape_for_hls4ml}"
    print(f"[OK] registry   {arch}  io_type={spec.default_io_type}")


def test_forward(arch):
    spec  = get_model_spec(arch)
    model = spec.float_cls()
    model.eval()
    _, C, H, W = spec.input_shape_for_hls4ml
    x = torch.zeros(2, C, H, W)
    y = model(x)
    assert y.shape == (2, 43), f"Expected (2,43), got {y.shape}"
    n = sum(p.numel() for p in model.parameters())
    print(f"[OK] forward    {arch}  input=({C},{H},{W})  output={tuple(y.shape)}  params={n:,}")


def test_onnx(arch):
    import tempfile
    spec  = get_model_spec(arch)
    model = spec.float_cls()
    model.eval()
    dummy = spec.dummy_input()
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        path = f.name
    torch.onnx.export(model, dummy, path,
                      input_names=["input"], output_names=["logits"],
                      opset_version=11)
    assert os.path.getsize(path) > 0
    os.unlink(path)
    print(f"[OK] ONNX       {arch}")


def test_class_names():
    names = load_class_names()
    assert len(names) == 43, f"Expected 43 names, got {len(names)}"
    print(f"[OK] class_names  43 clases — ej. clase 14 = '{names[14]}'")


def infer_image(path, arch):
    spec     = get_model_spec(arch)
    pth_path = os.path.join(MODELS_ROOT, spec.pth_filename)
    if not os.path.exists(pth_path):
        print(f"\n[SKIP] pesos no encontrados: {pth_path}")
        print(f"       Entrena primero:  python dev/train/gtsrb.py --arch {arch}")
        return

    names = load_class_names()
    model = spec.float_cls()
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.eval()

    x = preprocess_file(path)
    with torch.no_grad():
        logits = model(x).squeeze(0).numpy()

    probs = softmax(logits)
    top5  = np.argsort(probs)[::-1][:5]

    print(f"\nImagen : {path}")
    print(f"Modelo : {arch}  ({sum(p.numel() for p in model.parameters()):,} params)")
    print("Top-5  :")
    for rank, cls in enumerate(top5):
        bar = "#" * int(probs[cls] * 30)
        print(f"  {rank+1}. [{cls:2d}] {names[cls]:<38} {probs[cls]*100:5.1f}%  {bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch",  choices=["gtsrb", "gtsrb_gap"], default="gtsrb_gap")
    ap.add_argument("--image", default=None, help="Ruta a un recorte de señal")
    args = ap.parse_args()

    test_registry(args.arch)
    test_forward(args.arch)
    test_onnx(args.arch)
    test_class_names()

    if args.image:
        infer_image(args.image, args.arch)
    else:
        pth = os.path.join(MODELS_ROOT, get_model_spec(args.arch).pth_filename)
        if os.path.exists(pth):
            print(f"\n[INFO] Pesos encontrados: {pth}")
            print(f"       Usa --image path/crop.png para clasificar una señal.")
        else:
            print(f"\n[INFO] Sin pesos todavía. Entrena con:")
            print(f"       python dev/train/gtsrb.py --arch {args.arch} --epochs 20")

    print("\nSmoke tests OK.")


if __name__ == "__main__":
    main()
