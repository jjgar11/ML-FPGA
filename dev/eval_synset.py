"""
Evaluate a gtsrb_gap model on Synset Signset Germany and/or GTSRB test set.

Basic usage (all Synset images):
    python dev/eval_synset.py --synset-dir data/synset/SynsetSignsetGermany/Cycles

Official val split from the paper (4300 imgs):
    python dev/eval_synset.py --synset-dir data/synset/SynsetSignsetGermany --official-split

With alternative weights:
    python dev/eval_synset.py --synset-dir data/synset/SynsetSignsetGermany --official-split \
        --pth data/models/gtsrb_gap_synset.pth
"""

import argparse
import json
import os
import sys

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlfpga.config import DATA_ROOT, MODELS_ROOT
from mlfpga.models.registry import get_model_spec
from dev.train.gtsrb import SynsetDataset, ARCH_DEFAULT_SIZE, MEAN, STD


def make_val_tf(arch):
    img_size = ARCH_DEFAULT_SIZE[arch]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    per_class_correct = [0] * 43
    per_class_total   = [0] * 43
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            for p, l in zip(preds.cpu(), labels.cpu()):
                per_class_total[l]   += 1
                per_class_correct[l] += int(p == l)
    return correct / total, per_class_correct, per_class_total


def print_results(label, acc, per_cls_ok, per_cls_tot, names):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  Top-1 accuracy: {acc*100:.2f}%  ({sum(per_cls_ok)}/{sum(per_cls_tot)})")
    print(f"{'='*55}")
    print(f"{'Cls':>3}  {'Acc':>6}  {'OK/Tot':>9}  Nombre")
    print("-" * 55)
    for c in range(43):
        n  = per_cls_tot[c]
        ok = per_cls_ok[c]
        a  = ok / n if n else 0
        print(f"{c:>3}  {a*100:>5.1f}%  {ok:>4}/{n:<4}  {names[c]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synset-dir", required=True,
                    help="Normal mode: path to Cycles/. "
                         "With --official-split: path to SynsetSignsetGermany/")
    ap.add_argument("--official-split", action="store_true",
                    help="Use only the 4300 images from the official val split (CSV)")
    ap.add_argument("--arch",  default="gtsrb_gap")
    ap.add_argument("--pth",   default=None,
                    help="Explicit path to .pth weights (defaults to registry path)")
    ap.add_argument("--gtsrb", action="store_true",
                    help="Also evaluate on GTSRB real-photo test set")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec  = get_model_spec(args.arch)
    pth   = args.pth or os.path.join(MODELS_ROOT, spec.pth_filename)
    model = spec.float_cls()
    model.load_state_dict(torch.load(pth, map_location="cpu", weights_only=True))
    model.to(device)
    print(f"Model: {args.arch}  |  weights: {pth}")

    val_tf = make_val_tf(args.arch)

    names_path = os.path.join(DATA_ROOT, "gtsrb_class_names.json")
    with open(names_path) as f:
        raw = json.load(f)
    names = [raw[str(i)] for i in range(43)]

    # --- Synset ---
    if args.official_split:
        csv_dir = os.path.join(DATA_ROOT, "synset")
        val_csv = os.path.join(csv_dir, "cyclesGTSRB_val.csv")
        ds = SynsetDataset(args.synset_dir, transform=val_tf, csv_file=val_csv)
        split_label = "official val split (4300 imgs)"
    else:
        ds = SynsetDataset(args.synset_dir, transform=val_tf)
        split_label = "all images"

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)
    acc, ok, tot = evaluate(model, loader, device)
    print_results(f"Synset Cycles — {split_label}", acc, ok, tot, names)
    print("\n  Paper reference (trained on GTSRB, tested on Synset Cycles): ~89.4%")

    # --- GTSRB (optional) ---
    if args.gtsrb:
        gtsrb_ds = datasets.GTSRB(os.path.join(DATA_ROOT, "gtsrb"),
                                   split="test", download=False, transform=val_tf)
        loader2  = DataLoader(gtsrb_ds, batch_size=args.batch, shuffle=False, num_workers=4)
        acc2, ok2, tot2 = evaluate(model, loader2, device)
        print_results("GTSRB test set (real photos)", acc2, ok2, tot2, names)
        print("\n  Baseline reference (GTSRB-trained model): 90.05%")


if __name__ == "__main__":
    main()
