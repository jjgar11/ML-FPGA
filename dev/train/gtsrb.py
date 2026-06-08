"""
Train TinyCNN on GTSRB (German Traffic Sign Recognition Benchmark).
Dataset downloads automatically via torchvision (~300 MB).

Usage:
    python dev/train/gtsrb.py [--epochs 15] [--batch 64]

Output:
    data/models/gtsrb.pth
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mlfpga.config import DATA_ROOT, MODELS_ROOT
from mlfpga.models.gtsrb import TinyCNN

GTSRB_ROOT = os.path.join(DATA_ROOT, "gtsrb")

TRAIN_TF = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
])

VAL_TF = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629)),
])


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            total_loss += criterion(out, labels).item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch",  type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = datasets.GTSRB(GTSRB_ROOT, split="train", download=True, transform=TRAIN_TF)
    val_ds   = datasets.GTSRB(GTSRB_ROOT, split="test",  download=True, transform=VAL_TF)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model     = TinyCNN(num_classes=43).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model, val_dl,   criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train {tr_acc:.3f} | val {vl_acc:.3f}")
        if vl_acc > best_acc:
            best_acc = vl_acc
            out_path = os.path.join(MODELS_ROOT, "gtsrb.pth")
            torch.save(model.state_dict(), out_path)
            print(f"  → saved ({out_path})")

    print(f"Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
