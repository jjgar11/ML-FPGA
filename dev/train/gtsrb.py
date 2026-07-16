"""
Train TinyCNN_GAP on GTSRB, Synset Signset Germany, or both.

Synset Signset Germany (Fraunhofer IOSB/IPA, ITSC 2024): dataset sintético
de alta calidad. Las primeras 43 clases corresponden exactamente a GTSRB.
Estructura esperada: SynsetSignsetGermany/Cycles/{N}_{Name}/*.png
Split oficial en CSV: SynsetSignsetGermany/../cyclesGTSRB_{train,val}.csv

Uso:
    # Solo GTSRB:
    python dev/train/gtsrb.py --arch gtsrb_gap --epochs 30

    # Solo Synset (split oficial, para comparar con el paper):
    python dev/train/gtsrb.py --arch gtsrb_gap --epochs 30 --synset-only \
        --synset-dir data/synset/SynsetSignsetGermany

    # GTSRB + Synset mezclados:
    python dev/train/gtsrb.py --arch gtsrb_gap --epochs 30 \
        --synset-dir data/synset/SynsetSignsetGermany/Cycles

    # Fine-tune sobre GTSRB a partir de pesos Synset:
    python dev/train/gtsrb.py --arch gtsrb_gap --epochs 15 \
        --finetune data/models/gtsrb_gap_synset.pth --lr 1e-4

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
from mlfpga.models.gtsrb import TinyCNN, TinyCNN_GAP, TinyCNN_GAP_S

ARCH_MAP = {
    "gtsrb":              TinyCNN,
    "gtsrb_gap":          TinyCNN_GAP,
    "gtsrb_gap_s":        TinyCNN_GAP_S,
}

# Default IMG_SIZE per arch (can be overridden with --img-size)
ARCH_DEFAULT_SIZE = {"gtsrb": 32, "gtsrb_gap": 64, "gtsrb_gap_s": 32}

GTSRB_ROOT = os.path.join(DATA_ROOT, "gtsrb")

MEAN = (0.3337, 0.3064, 0.3171)
STD  = (0.2672, 0.2564, 0.2629)


def make_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size + 8, img_size + 8)),
        transforms.RandomCrop(img_size),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


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

    Modo scan (sin csv_file):
        root apunta a Cycles/; escanea todas las imágenes de clases 0-42.
        Soporta nombres con prefijo numérico (0_Geschwindigkeit20) y zero-padded (00000).

    Modo CSV (con csv_file):
        root apunta a SynsetSignsetGermany/ (un nivel arriba de Cycles/).
        csv_file lista paths relativos tipo 'Cycles\\0_Geschwindigkeit20\\0_cycles.png'.
        Usa el split oficial train/val del paper.
    """
    EXTS = {".png", ".jpg", ".jpeg", ".ppm"}

    def __init__(self, root, transform=None, csv_file=None):
        self.transform = transform
        self.samples   = []

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Synset dir no encontrado: {root}")

        if csv_file:
            self._load_from_csv(root, csv_file)
        else:
            self._scan_dirs(root)

    def _load_from_csv(self, root, csv_file):
        with open(csv_file) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        loaded = skipped = 0
        for rel in lines:
            rel_unix = rel.replace("\\", "/")
            # class id = number prefix of the subdirectory (e.g. "Cycles/0_Geschw../file.png" → 0)
            parts = rel_unix.split("/")
            if len(parts) < 3:
                continue
            try:
                class_id = int(parts[1].split("_")[0])
            except ValueError:
                continue
            if class_id > 42:
                continue
            if os.path.splitext(parts[-1])[1].lower() not in self.EXTS:
                skipped += 1
                continue
            full = os.path.join(root, rel_unix)
            if os.path.isfile(full):
                self.samples.append((full, class_id))
                loaded += 1
        print(f"[SynsetDataset] {csv_file}")
        print(f"  Loaded images: {loaded}  |  skipped (json/missing): {skipped}")

    def _scan_dirs(self, root):
        found_classes = 0
        for class_id in range(43):
            folder = os.path.join(root, f"{class_id:05d}")
            if not os.path.isdir(folder):
                matches = glob.glob(os.path.join(root, f"{class_id}_*"))
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
    parser.add_argument("--arch",       choices=list(ARCH_MAP), default="gtsrb_gap")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch",      type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--min-size",   type=int, default=32,
                        help="Descartar imágenes GTSRB con lado < N px")
    parser.add_argument("--synset-dir", default=None,
                        help="Modo scan: ruta a Cycles/. "
                             "Modo --synset-only: ruta a SynsetSignsetGermany/")
    parser.add_argument("--synset-only", action="store_true",
                        help="Entrenar y validar solo en Synset (split CSV oficial). "
                             "--synset-dir debe apuntar a SynsetSignsetGermany/")
    parser.add_argument("--finetune",   default=None, metavar="PTH",
                        help="Cargar pesos de este .pth antes de entrenar (para fine-tune)")
    parser.add_argument("--out-suffix", default="",
                        help="Sufijo para el nombre del .pth de salida (ej. '_synset')")
    parser.add_argument("--img-size",  type=int, default=None,
                        help="Tamaño de imagen (default: 32 para gtsrb/gtsrb_gap_s, 64 para gtsrb_gap)")
    args = parser.parse_args()

    img_size = args.img_size or ARCH_DEFAULT_SIZE[args.arch]
    train_tf, val_tf = make_transforms(img_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Arch: {args.arch}  IMG_SIZE: {img_size}")

    # --- Datasets ---
    if args.synset_only:
        if not args.synset_dir:
            raise ValueError("--synset-only requiere --synset-dir apuntando a SynsetSignsetGermany/")
        csv_dir  = os.path.join(DATA_ROOT, "synset")
        train_ds = SynsetDataset(args.synset_dir, transform=train_tf,
                                 csv_file=os.path.join(csv_dir, "cyclesGTSRB_train.csv"))
        val_ds   = SynsetDataset(args.synset_dir, transform=val_tf,
                                 csv_file=os.path.join(csv_dir, "cyclesGTSRB_val.csv"))
        print(f"[Synset-only] train={len(train_ds)}  val={len(val_ds)}")
    else:
        gtsrb_train = FilteredGTSRB(GTSRB_ROOT, split="train", download=True,
                                     transform=train_tf, min_side=args.min_size)
        val_ds = datasets.GTSRB(GTSRB_ROOT, split="test", download=True, transform=val_tf)
        if args.synset_dir:
            synset_ds = SynsetDataset(args.synset_dir, transform=train_tf)
            train_ds  = ConcatDataset([gtsrb_train, synset_ds])
            print(f"[Mix] GTSRB {len(gtsrb_train)} + Synset {len(synset_ds)} = {len(train_ds)}")
        else:
            train_ds = gtsrb_train
            print(f"[Solo GTSRB] {len(train_ds)} imágenes")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)

    # --- Modelo ---
    model_cls = ARCH_MAP[args.arch]
    model     = model_cls(num_classes=43).to(device)
    if args.finetune:
        model.load_state_dict(torch.load(args.finetune, map_location="cpu", weights_only=True))
        print(f"[Fine-tune] pesos cargados desde {args.finetune}")
    print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_path = os.path.join(MODELS_ROOT, f"{args.arch}{args.out_suffix}.pth")
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        _, tr_acc = train_epoch(model, train_dl, criterion, optimizer, device)
        _, vl_acc = eval_epoch(model, val_dl,   criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{args.epochs} | train {tr_acc:.3f} | val {vl_acc:.3f}")
        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), out_path)
            print(f"  → guardado ({out_path})")

    print(f"\nMejor val accuracy: {best_acc:.4f}  →  {out_path}")


if __name__ == "__main__":
    main()
