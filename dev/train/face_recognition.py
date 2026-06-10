"""
Train PCA + FaceMLP on collected face images.
Run on dev machine (requires sklearn, torch, opencv-python).

Expects face images in:
  data/faces/<person_name>/*.jpg   (32x32 grayscale, from collect_faces.py)

Usage:
  python dev/train/face_recognition.py [--faces-dir data/faces] [--components 50]

Outputs (all in data/models/):
  face_pca_mean.npy        — mean face vector (1024,)
  face_pca_components.npy  — eigenvectors (50, 1024)
  face_labels.json         — {index: person_name}
  face_mlp.pth             — MLP weights
  face_mlp_float.onnx      — ONNX for CPU inference (backup / validation)
  face_test_data.json      — scaled embeddings for FPGA test (like wine test_data.json)
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mlfpga.config import MODELS_ROOT
from mlfpga.models.face_mlp import FaceMLP

FACE_SIZE   = 32
N_PIXELS    = FACE_SIZE * FACE_SIZE   # 1024
UNKNOWN_THR = 0.75                    # softmax confidence below this → "unknown"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_faces(faces_dir):
    """
    Returns flat images (N, 1024) float32 [0,1] and integer labels,
    plus a dict {idx: person_name}.
    """
    if not os.path.isdir(faces_dir):
        raise FileNotFoundError(f"Faces directory not found: {faces_dir}")

    person_dirs = sorted([
        d for d in os.listdir(faces_dir)
        if os.path.isdir(os.path.join(faces_dir, d))
    ])
    if not person_dirs:
        raise ValueError(f"No person sub-folders found in {faces_dir}")

    label_map = {i: name for i, name in enumerate(person_dirs)}
    print(f"Found {len(person_dirs)} people: {person_dirs}")

    images, labels = [], []
    for idx, name in label_map.items():
        person_dir = os.path.join(faces_dir, name)
        files = [f for f in os.listdir(person_dir)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not files:
            print(f"  WARNING: no images for {name}, skipping")
            continue
        for fname in files:
            path = os.path.join(person_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (FACE_SIZE, FACE_SIZE))
            images.append(img.flatten().astype(np.float32) / 255.0)
            labels.append(idx)

        print(f"  {name}: {len(files)} images")

    return np.array(images), np.array(labels, dtype=np.int64), label_map


# ── PCA ───────────────────────────────────────────────────────────────────────

def fit_pca(images, n_components=50):
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(images)
    print(f"PCA: {n_components} components, "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% variance explained")
    return pca


def transform(pca, images):
    return pca.transform(images).astype(np.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def augment_embeddings(X, y, n_aug=8, noise_std=0.15):
    """Embed-space augmentation: add Gaussian noise to each sample."""
    Xa, ya = [X], [y]
    for _ in range(n_aug):
        Xa.append(X + np.random.normal(0, noise_std, X.shape).astype(np.float32))
        ya.append(y)
    return np.concatenate(Xa), np.concatenate(ya)


def train_mlp(X_train, y_train, X_val, y_val, num_classes, n_components, epochs=60):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | classes: {num_classes} | components: {n_components}")

    # Augment training set
    X_aug, y_aug = augment_embeddings(X_train, y_train)
    print(f"Training samples: {len(X_train)} → {len(X_aug)} (after augmentation)")

    tr_ds = TensorDataset(torch.from_numpy(X_aug),
                          torch.from_numpy(y_aug))
    vl_ds = TensorDataset(torch.from_numpy(X_val),
                          torch.from_numpy(y_val))
    tr_dl = DataLoader(tr_ds, batch_size=64, shuffle=True)
    vl_dl = DataLoader(vl_ds, batch_size=64)

    model     = FaceMLP(n_components=n_components, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in vl_dl:
                out = model(xb.to(device))
                correct += (out.argmax(1) == yb.to(device)).sum().item()
                total   += len(yb)
        acc = correct / total
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{epochs}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc  = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Best val accuracy: {best_acc:.4f}")
    model.load_state_dict(best_state)
    return model.cpu()


# ── Export ────────────────────────────────────────────────────────────────────

def export_onnx(model, n_components, path):
    model.eval()
    dummy = torch.zeros(1, n_components)
    torch.onnx.export(model, dummy, path,
                      input_names=["embedding"], output_names=["logits"],
                      opset_version=11)
    print(f"ONNX: {path}")


def save_test_data(X_scaled, y, label_map, path):
    """JSON file for FPGA inference (mirrors generate_test_data.py for Wine)."""
    data = {
        "X":     X_scaled.tolist(),
        "y":     y.tolist(),
        "names": [label_map[i] for i in sorted(label_map)],
    }
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Test data: {path}  ({len(y)} samples)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--faces-dir",  default="data/faces",
                        help="Root folder with one sub-dir per person")
    parser.add_argument("--components", type=int, default=50,
                        help="Number of PCA components (= MLP input size)")
    parser.add_argument("--epochs",     type=int, default=60)
    args = parser.parse_args()

    os.makedirs(MODELS_ROOT, exist_ok=True)

    # 1. Load images
    images, labels, label_map = load_faces(args.faces_dir)
    num_classes = len(label_map)
    print(f"\nTotal images: {len(images)}  classes: {num_classes}")

    # 2. Fit PCA on training split images (no leakage)
    idx_tr, idx_vl = train_test_split(
        np.arange(len(images)), test_size=0.2, stratify=labels, random_state=42
    )
    pca = fit_pca(images[idx_tr], n_components=args.components)

    # 3. Transform all images
    X_all = transform(pca, images)
    X_tr, X_vl = X_all[idx_tr], X_all[idx_vl]
    y_tr, y_vl = labels[idx_tr], labels[idx_vl]

    # 4. Save PCA artifacts (needed on Ultra96 for inference)
    mean_path  = os.path.join(MODELS_ROOT, "face_pca_mean.npy")
    comp_path  = os.path.join(MODELS_ROOT, "face_pca_components.npy")
    label_path = os.path.join(MODELS_ROOT, "face_labels.json")
    np.save(mean_path, pca.mean_.astype(np.float32))
    np.save(comp_path, pca.components_.astype(np.float32))
    with open(label_path, "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    print(f"\nPCA mean:   {mean_path}")
    print(f"PCA comps:  {comp_path}")
    print(f"Labels:     {label_path}")

    # 5. Train MLP
    print("\n── Training MLP ──")
    model = train_mlp(X_tr, y_tr, X_vl, y_vl,
                      num_classes=num_classes,
                      n_components=args.components,
                      epochs=args.epochs)

    # 6. Save weights and ONNX
    pth_path  = os.path.join(MODELS_ROOT, "face_mlp.pth")
    onnx_path = os.path.join(MODELS_ROOT, "face_mlp_float.onnx")
    torch.save(model.state_dict(), pth_path)
    export_onnx(model, args.components, onnx_path)
    print(f"Weights:    {pth_path}")

    # 7. Test data JSON for FPGA validation (mirrors Wine pattern)
    test_path = os.path.join(MODELS_ROOT, "..", "face_test_data.json")
    save_test_data(X_vl, y_vl, label_map, test_path)

    print("\n── Files to copy to Ultra96 ──")
    files = [mean_path, comp_path, label_path, pth_path, onnx_path]
    for f in files:
        print(f"  scp {f} root@192.168.0.103:/home/root/face/")

    print(f"\n── hls4ml conversion (run in VM) ──")
    print(f"  python dev/hls4ml/convert_model.py "
          f"--model face_mlp --mode int8 --backend Vivado --reuse 1 --synth")
    print(f"  NOTE: after generate, update myproject.cpp pragmas like Wine MLP.")
    print(f"        Input array: x[{args.components}], output: last layer out[{num_classes}]")


if __name__ == "__main__":
    main()
