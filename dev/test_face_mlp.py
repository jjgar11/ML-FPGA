"""
Sanity check: loads test embeddings and runs CPU inference via PyTorch.
Run on dev machine before committing to hls4ml synthesis.

Usage:
    python dev/test_face_mlp.py
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mlfpga.models.face_mlp import FaceMLP
from mlfpga.config import FPGA_FILES_ROOT, MODELS_ROOT

LABELS_PATH    = os.path.join(MODELS_ROOT, "face_labels.json")
PTH_PATH       = os.path.join(MODELS_ROOT, "face_mlp.pth")
TEST_DATA_PATH = os.path.join(FPGA_FILES_ROOT, "test_scripts", "face", "face_test_data.json")
UNKNOWN_THR    = 0.75


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def main():
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    state        = torch.load(PTH_PATH, map_location="cpu", weights_only=True)
    n_components = state["net.0.weight"].shape[1]
    num_classes  = state["net.4.weight"].shape[0]
    model        = FaceMLP(n_components=n_components, num_classes=num_classes)
    model.load_state_dict(state)
    model.eval()
    print(f"FaceMLP({n_components} → 32 → 16 → {num_classes})   labels={labels}")

    with open(TEST_DATA_PATH) as f:
        data = json.load(f)
    X     = np.array(data["X"], dtype=np.float32)
    y     = np.array(data["y"], dtype=np.int64)
    names = data["names"]

    with torch.no_grad():
        logits = model(torch.from_numpy(X)).numpy()

    preds = logits.argmax(axis=1)
    acc   = (preds == y).mean()
    print(f"\nCPU accuracy: {acc*100:.1f}%  ({(preds==y).sum()}/{len(y)})")

    for cls in range(num_classes):
        mask = y == cls
        if not mask.any():
            continue
        cls_acc = (preds[mask] == cls).mean()
        print(f"  {names[cls]:<12}: {cls_acc*100:.1f}%  ({(preds[mask]==cls).sum()}/{mask.sum()})")

    print("\n--- First 5 samples ---")
    for i in range(min(5, len(y))):
        probs  = softmax(logits[i])
        pred   = int(preds[i])
        conf   = float(probs[pred])
        decision = f"CONOCIDO ({names[pred]})" if conf >= UNKNOWN_THR else "DESCONOCIDO"
        status   = "OK" if pred == y[i] else "FAIL"
        print(f"  [{i}] real={names[y[i]]:<10} → {decision:<22} conf={conf:.2f}  {status}")

    if acc < 0.80:
        print("\nWARNING: accuracy < 80% — captura más imágenes o revisa la calidad del crop.")
    else:
        print(f"\nOK — listo para hls4ml conversion.")


if __name__ == "__main__":
    main()
