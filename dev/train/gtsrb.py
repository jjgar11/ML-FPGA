"""
Train TinyCNN_GAP on GTSRB + Synset Signset Germany (opcional).

Synset Signset Germany (Fraunhofer IOSB/IPA, ITSC 2024) es un dataset
sintético de alta calidad que mejora la generalización del modelo al
entrenar con señales perfectamente nítidas y diversas condiciones.
Las clases 0-42 de Synset corresponden exactamente a las 43 de GTSRB.

Dataset Synset: synset.de/datasets/synset-signset-ger/
Descarga el zip, extráelo y pasa la ruta con --synset-dir.

Uso:
    # Solo GTSRB filtrado (sin Synset):
    python dev/train/gtsrb.py --arch gtsrb_gap --epochs 30 --min-size 32

    # GTSRB + Synset Signset (recomendado):
    python dev/train/gtsrb.py --arch gtsrb_gap --epochs 30 --min-size 32 \
        --synset-dir /ruta/a/SynsetSignsetGermany/Cycles

Salida:
    data/models/gtsrb_gap.pth
"""

import argparse
import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Dataset

from mlfpga.config import DATA_ROOT, MODELS_ROOT
from mlfpga.models.gtsrb import TinyCNN, TinyCNN_GAP

ARCH_MAP = {"gtsrb": TinyCNN, "gtsrb_gap": TinyCNN_GAP}

GTSRB_ROOT = os.path.join(DATA_ROOT, "gtsrb")

IMG_SIZE = 64   # 64×64 — mejor distinción de dígitos en señales de velocidad

MEAN = (0.3337, 0.3064, 0.3171)
STD  = (0.2672, 0.2564, 0.2629)

TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE + 8, IMG_SIZE + 8)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# ---------------------------------------------------------------------------
# GTSRB con filtro de tamaño mínimo
# ---------------------------------------------------------------------------

class FilteredGTSRB(datasets.GTSRB):
    """GTSRB eliminando imágenes con lado mínimo < min_side px.

    Las imágenes más pequeñas son borrosas y enseñan al modelo a clasificar
    manchas sin forma en lugar de señales con estructura real.
    """
    def __init__(self, *args, min_side=32, **kwargs):
        super().__init__(*args, **kwargs)
        original = len(self._samples)
        valid = []
        for path, label in self._samples:
            w, h = Image.open(path).size
            if min(w, h) >= min_side:
                valid.append((path, label))
        self._samples = valid
        print(f"[FilteredGTSRB] min_side={min_side}px: "
              f"{original} → {len(valid)} imágenes "
              f"({original - len(valid)} eliminadas)")


# ---------------------------------------------------------------------------
# Synset Signset Germany — primeras 43 clases (= GTSRB)
# ---------------------------------------------------------------------------

class SynsetDataset(Dataset):
    """
    Carga las primeras 43 clases de Synset Signset Germany.

    Estructura esperada del directorio (apunta al renderer elegido):
        <root>/
            00000/   ← clase 0
                *.png
            00001/   ← clase 1
            ...
            00042/   ← clase 42

    Ejemplo:
        SynsetSignsetGermany/
            Cycles/      ← pasa esta ruta como --synset-dir
                00000/
                ...
            OGRE/
                ...
    """
    EXTS = {".png", ".jpg", ".jpeg", ".ppm"}

    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples   = []   # list of (path, class_id)

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Synset dir no encontrado: {root}")

        found_classes = 0
        for class_id in range(43):
            # Support both zero-padded (00000) and named (0_Geschwindigkeit20) formats
            folder = os.path.join(root, f"{class_id:05d}")
            if not os.path.isdir(folder):
                import glob as _glob
                matches = _glob.glob(os.path.join(root, f"{class_id}_*"))
                folder = next((m for m in matches if os.path.isdir(m)), None)
                if folder is None:
                    continue
            found_classes += 1
            for fname in os.listdir(folder):
                if os.path.splitext(fname)[1].lower() in self.EXTS:
                    self.samples.append((os.path.join(folder, fname), class_id))

        print(f"[SynsetDataset] {root}")
        print(f"  Clases encontradas: {found_classes}/43  |  Imágenes: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            total_loss += criterion(out, labels).item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",     choices=list(ARCH_MAP), default="gtsrb_gap",
                        help="gtsrb=TinyCNN (baseline), gtsrb_gap=FPGA-optimised")
    parser.add_argument("--epochs",   type=int, default=50)
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--min-size", type=int, default=32,
                        help="Descartar imágenes GTSRB con lado < N px (default 32)")
    parser.add_argument("--synset-dir", default=None,
                        help="Ruta a las 43 clases de Synset Signset Germany "
                             "(ej. .../SynsetSignsetGermany/Cycles)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Arch: {args.arch}  IMG_SIZE: {IMG_SIZE}")

    # --- Datasets ---
    gtsrb_train = FilteredGTSRB(GTSRB_ROOT, split="train", download=True,
                                 transform=TRAIN_TF, min_side=args.min_size)
    val_ds = datasets.GTSRB(GTSRB_ROOT, split="test", download=True, transform=VAL_TF)

    if args.synset_dir:
        synset_ds  = SynsetDataset(args.synset_dir, transform=TRAIN_TF)
        train_ds   = ConcatDataset([gtsrb_train, synset_ds])
        print(f"[Mix] GTSRB {len(gtsrb_train)} + Synset {len(synset_ds)} "
              f"= {len(train_ds)} imágenes de entrenamiento")
    else:
        train_ds = gtsrb_train
        print(f"[Solo GTSRB] {len(train_ds)} imágenes de entrenamiento")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)
    print(f"Val: {len(val_ds)} imágenes")

    # --- Modelo ---
    model_cls = ARCH_MAP[args.arch]
    model     = model_cls(num_classes=43).to(device)
    n_params  = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = os.path.join(MODELS_ROOT, f"{args.arch}.pth")
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_dl, criterion, optimizer, device)
        vl_loss, vl_acc = eval_epoch(model, val_dl,   criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{args.epochs} | train {tr_acc:.3f} | val {vl_acc:.3f}")
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), out_path)
            print(f"  → guardado ({out_path})")

    print(f"Mejor val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
