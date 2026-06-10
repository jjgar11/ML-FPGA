"""
Train NavCNN on synthetic navigation dataset.

Usage:
    python dev/train/nav_cnn.py [--epochs 30] [--batch 64] [--n-per-class 2000]

Output:
    data/models/nav_cnn.pth        — best weights (PyTorch)
    data/models/nav_cnn_float.onnx — float ONNX (for CPU inference via OpenCV DNN)
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Allow running from project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mlfpga.config import MODELS_ROOT
from mlfpga.models.nav_cnn import NavCNN

# Import generator relative to project root (not a package)
_DATA_GEN = os.path.join(os.path.dirname(__file__), "..", "data", "gen_nav_synthetic.py")
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("gen_nav_synthetic", _DATA_GEN)
_gen_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_gen_mod)
generate_dataset = _gen_mod.generate_dataset

CLASSES = ["forward", "left", "right", "stop"]


class NavDataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images  = images   # (N, 64, 64) float32  [0,1]
        self.labels  = labels   # (N,) int64
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img   = self.images[idx].copy()
        label = int(self.labels[idx])

        if self.augment:
            # Brightness / contrast jitter
            img = img * random.uniform(0.8, 1.2) + random.uniform(-0.05, 0.05)
            img = np.clip(img, 0.0, 1.0)
            # Extra noise
            img += np.random.normal(0, 0.02, img.shape).astype(np.float32)
            img = np.clip(img, 0.0, 1.0)
            # Vertical shift (±4 px) — does NOT change label
            shift = random.randint(-4, 4)
            if shift > 0:
                img = np.pad(img, ((shift, 0), (0, 0)))[:64, :]
            elif shift < 0:
                img = np.pad(img, ((0, -shift), (0, 0)))[(-shift):(-shift)+64, :]

        x = torch.from_numpy(img).unsqueeze(0)  # (1, 64, 64)
        return x, label


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct    += (out.argmax(1) == y).sum().item()
        total      += x.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            total      += x.size(0)
    return total_loss / total, correct / total


def export_onnx(model, out_path):
    model.eval()
    dummy = torch.zeros(1, 1, 64, 64)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=11,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"  → ONNX saved ({out_path})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--batch",        type=int, default=64)
    parser.add_argument("--n-per-class",  type=int, default=2000,
                        help="Training images per class (val uses 20%% of this)")
    parser.add_argument("--lr",           type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    n_train = args.n_per_class
    n_val   = max(200, n_train // 5)

    print(f"Generating train set ({n_train}/class)…")
    tr_imgs, tr_lbs = generate_dataset(n_per_class=n_train, seed=42)
    print(f"Generating val set   ({n_val}/class)…")
    vl_imgs, vl_lbs = generate_dataset(n_per_class=n_val, seed=99)

    tr_ds = NavDataset(tr_imgs, tr_lbs, augment=True)
    vl_ds = NavDataset(vl_imgs, vl_lbs, augment=False)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    vl_dl = DataLoader(vl_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    print(f"Train: {len(tr_ds)} | Val: {len(vl_ds)}")

    model     = NavCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    os.makedirs(MODELS_ROOT, exist_ok=True)
    pth_path  = os.path.join(MODELS_ROOT, "nav_cnn.pth")
    onnx_path = os.path.join(MODELS_ROOT, "nav_cnn_float.onnx")

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_dl, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model, vl_dl, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train {tr_acc:.3f} ({tr_loss:.4f}) | "
              f"val {vl_acc:.3f} ({vl_loss:.4f})")
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), pth_path)
            print(f"  → saved ({pth_path})")

    print(f"\nBest val accuracy: {best_acc:.4f}")

    # Load best weights and export to ONNX for CPU inference on Ultra96
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    export_onnx(model, onnx_path)

    print("\nClass map:  0=forward  1=left  2=right  3=stop")
    print("Copy to Ultra96:")
    print(f"  scp {onnx_path} root@192.168.0.103:/home/root/")


if __name__ == "__main__":
    main()
