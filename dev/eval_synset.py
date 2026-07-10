"""
Evalúa gtsrb_gap sobre las primeras 43 clases de Synset Signset Germany.

El paper (Tabla II) reporta ~89% cuando se entrena en GTSRB y se evalúa
en Synset Cycles, y ~87% en Synset OGRE.

Descarga: synset.de/datasets/synset-signset-ger/
Extrae y pasa la ruta al renderer elegido (Cycles u OGRE):
    SynsetSignsetGermany/
        Cycles/
            00000/  ← clase 0
            ...
            00042/  ← clase 42  (solo estas 43 nos interesan)

Uso:
    python dev/eval_synset.py --synset-dir /ruta/SynsetSignsetGermany/Cycles
    python dev/eval_synset.py --synset-dir /ruta/SynsetSignsetGermany/OGRE
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlfpga.config import MODELS_ROOT
from mlfpga.models.registry import get_model_spec
from dev.train.gtsrb import SynsetDataset, IMG_SIZE, MEAN, STD

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synset-dir", required=True,
                    help="Ruta a las 43 clases de Synset (ej. .../Cycles o .../OGRE)")
    ap.add_argument("--arch",  default="gtsrb_gap")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec    = get_model_spec(args.arch)
    pth     = os.path.join(MODELS_ROOT, spec.pth_filename)
    model   = spec.float_cls()
    model.load_state_dict(torch.load(pth, map_location="cpu", weights_only=True))
    model.to(device)

    dataset = SynsetDataset(args.synset_dir, transform=VAL_TF)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=False, num_workers=4)

    print(f"Evaluando {args.arch} sobre {len(dataset)} imágenes de Synset...")
    acc, per_cls_ok, per_cls_tot = evaluate(model, loader, device)

    print(f"\nTop-1 accuracy en Synset (43 clases): {acc*100:.2f}%")
    print(f"Referencia del paper (entrenado en GTSRB → Synset Cycles): ~89.4%")
    print(f"Referencia del paper (entrenado en GTSRB → Synset OGRE):   ~87.4%")

    # Clases con peor accuracy
    per_cls_acc = [(per_cls_ok[i] / max(per_cls_tot[i], 1), i)
                   for i in range(43) if per_cls_tot[i] > 0]
    per_cls_acc.sort()
    print("\nPeores 10 clases:")
    for acc_c, cls_id in per_cls_acc[:10]:
        print(f"  Clase {cls_id:2d}: {acc_c*100:5.1f}%  ({per_cls_ok[cls_id]}/{per_cls_tot[cls_id]})")


if __name__ == "__main__":
    main()
