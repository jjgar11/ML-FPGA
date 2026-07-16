"""
Extrae los pesos de gtsrb_gap.pth a un .npz para inferencia numpy en la Ultra96.
No requiere PyTorch en el destino — solo numpy.

Uso:
    python dev/export_weights_numpy.py
    python dev/export_weights_numpy.py --pth data/models/gtsrb_gap_mixed.pth
Salida:
    data/models/gtsrb_gap_weights.npz  (o <mismo nombre>_weights.npz si se usa --pth)
"""

import argparse
import os
import torch
import numpy as np

from mlfpga.config import MODELS_ROOT
from mlfpga.models.gtsrb import TinyCNN_GAP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", default=None,
                    help="Ruta al .pth (default: data/models/gtsrb_gap.pth)")
    args = ap.parse_args()

    PTH_PATH = args.pth or os.path.join(MODELS_ROOT, "gtsrb_gap.pth")
    NPZ_PATH = os.path.splitext(PTH_PATH)[0] + "_weights.npz"
    model = TinyCNN_GAP(num_classes=43)
    model.load_state_dict(torch.load(PTH_PATH, map_location="cpu", weights_only=True))
    model.eval()

    sd = model.state_dict()

    weights = {
        # Conv1 (3→32) + BN1
        "conv1_w":  sd["features.0.weight"].numpy(),
        "conv1_b":  sd["features.0.bias"].numpy(),
        "bn1_w":    sd["features.1.weight"].numpy(),
        "bn1_b":    sd["features.1.bias"].numpy(),
        "bn1_mean": sd["features.1.running_mean"].numpy(),
        "bn1_var":  sd["features.1.running_var"].numpy(),
        # Conv2 (32→64) + BN2
        "conv2_w":  sd["features.4.weight"].numpy(),
        "conv2_b":  sd["features.4.bias"].numpy(),
        "bn2_w":    sd["features.5.weight"].numpy(),
        "bn2_b":    sd["features.5.bias"].numpy(),
        "bn2_mean": sd["features.5.running_mean"].numpy(),
        "bn2_var":  sd["features.5.running_var"].numpy(),
        # Conv3 (64→128) + BN3
        "conv3_w":  sd["features.8.weight"].numpy(),
        "conv3_b":  sd["features.8.bias"].numpy(),
        "bn3_w":    sd["features.9.weight"].numpy(),
        "bn3_b":    sd["features.9.bias"].numpy(),
        "bn3_mean": sd["features.9.running_mean"].numpy(),
        "bn3_var":  sd["features.9.running_var"].numpy(),
        # FC head: Linear(128→256) + Linear(256→43)
        "fc1_w":    sd["head.0.weight"].numpy(),
        "fc1_b":    sd["head.0.bias"].numpy(),
        "fc2_w":    sd["head.3.weight"].numpy(),
        "fc2_b":    sd["head.3.bias"].numpy(),
    }

    np.savez(NPZ_PATH, **weights)
    size_kb = os.path.getsize(NPZ_PATH) / 1024
    print(f"[OK] Guardado: {NPZ_PATH}  ({size_kb:.0f} KB)")

    # Verificación rápida: forward pass numpy vs pytorch deben coincidir
    w = np.load(NPZ_PATH)

    def conv2d_np(x, W, b, padding=1):
        C_in, H, Wi = x.shape
        C_out, _, kH, kW = W.shape
        if padding:
            x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
        H_out, W_out = H, Wi
        cols = np.lib.stride_tricks.as_strided(
            x, shape=(C_in, kH, kW, H_out, W_out),
            strides=(x.strides[0], x.strides[1], x.strides[2], x.strides[1], x.strides[2]),
        ).reshape(C_in * kH * kW, H_out * W_out)
        return (W.reshape(C_out, -1) @ cols + b[:, None]).reshape(C_out, H_out, W_out)

    def bn_np(x, ww, b, mean, var, eps=1e-5):
        return ww[:, None, None] * (x - mean[:, None, None]) / np.sqrt(var[:, None, None] + eps) + b[:, None, None]

    def relu_np(x): return np.maximum(0.0, x)
    def pool_np(x):
        C, H, W = x.shape
        return x[:, :H//2*2, :W//2*2].reshape(C, H//2, 2, W//2, 2).max(axis=(2, 4))

    def forward_np(x):
        x = relu_np(bn_np(conv2d_np(x, w["conv1_w"], w["conv1_b"]), w["bn1_w"], w["bn1_b"], w["bn1_mean"], w["bn1_var"]))
        x = pool_np(x)
        x = relu_np(bn_np(conv2d_np(x, w["conv2_w"], w["conv2_b"]), w["bn2_w"], w["bn2_b"], w["bn2_mean"], w["bn2_var"]))
        x = pool_np(x)
        x = relu_np(bn_np(conv2d_np(x, w["conv3_w"], w["conv3_b"]), w["bn3_w"], w["bn3_b"], w["bn3_mean"], w["bn3_var"]))
        x = pool_np(x)
        x = x.mean(axis=(1, 2))
        x = relu_np(w["fc1_w"] @ x + w["fc1_b"])
        return w["fc2_w"] @ x + w["fc2_b"]

    x_pt = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out_pt = model(x_pt).numpy()[0]
    out_np = forward_np(x_pt.numpy()[0])

    max_diff = np.abs(out_pt - out_np).max()
    print(f"[OK] Max diferencia PyTorch vs numpy: {max_diff:.6f}  (debe ser < 0.01)")
    assert max_diff < 0.01, f"Diferencia demasiado grande: {max_diff}"
    print("[OK] Pesos verificados.")


if __name__ == "__main__":
    main()
