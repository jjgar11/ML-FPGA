"""
Face recognition — CPU inference on Ultra96 (fallback, no FPGA needed).
Uses OpenCV DNN with the ONNX model + PCA computed in numpy on PS.

Requirements on Ultra96  (copy with scp):
  /home/root/face/face_pca_mean.npy
  /home/root/face/face_pca_components.npy
  /home/root/face/face_labels.json
  /home/root/face/face_mlp_float.onnx

Usage:
  python infer_face_cpu.py [--cam 0] [--save]
  Ctrl+C to stop.
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

CASCADE_PATHS = [
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
]

FACE_DIR    = "/home/root/face"
FACE_SIZE   = 32
UNKNOWN_THR = 0.75


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def load_pca(face_dir):
    mean  = np.load(os.path.join(face_dir, "face_pca_mean.npy"))        # (1024,)
    comps = np.load(os.path.join(face_dir, "face_pca_components.npy"))  # (50, 1024)
    return mean, comps


def pca_transform(img_gray, mean, comps):
    """Gray frame (any size) → (50,) float32 embedding."""
    resized  = cv2.resize(img_gray, (FACE_SIZE, FACE_SIZE))
    flat     = resized.flatten().astype(np.float32) / 255.0
    centered = flat - mean
    return (comps @ centered).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam",      type=int, default=0)
    parser.add_argument("--face-dir", default=FACE_DIR)
    parser.add_argument("--save",     action="store_true",
                        help="Save annotated frames to ./face_frames/")
    args = parser.parse_args()

    # Load PCA artifacts
    pca_mean, pca_comps = load_pca(args.face_dir)
    with open(os.path.join(args.face_dir, "face_labels.json")) as f:
        labels = json.load(f)
    n_components = pca_comps.shape[0]
    print(f"PCA: {pca_comps.shape}   labels: {labels}")

    # Load ONNX model via OpenCV DNN
    onnx_path = os.path.join(args.face_dir, "face_mlp_float.onnx")
    if not os.path.exists(onnx_path):
        print(f"ONNX model not found: {onnx_path}")
        print("Copy from dev machine:")
        print("  scp data/models/face_mlp_float.onnx root@192.168.0.103:/home/root/face/")
        sys.exit(1)
    net = cv2.dnn.readNetFromONNX(onnx_path)
    print(f"Model loaded: {onnx_path}")

    # Face detector
    cascade_path = next((p for p in CASCADE_PATHS if os.path.exists(p)), None)
    detector     = cv2.CascadeClassifier(cascade_path) if cascade_path else None
    if not detector:
        print("WARNING: Haar cascade not found — using full frame as face region")

    # Camera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Cannot open camera {args.cam}")
        sys.exit(1)
    print(f"Camera /dev/video{args.cam} open. Ctrl+C to stop.\n")

    if args.save:
        os.makedirs("face_frames", exist_ok=True)

    frame_idx = 0
    t_prev    = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection
            face_img = gray
            if detector is not None:
                faces = detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                if len(faces) > 0:
                    x, y_, w, h = sorted(faces, key=lambda f: f[2]*f[3])[-1]
                    margin = int(0.1 * min(w, h))
                    x1 = max(0, x - margin)
                    y1 = max(0, y_ - margin)
                    x2 = min(gray.shape[1], x + w + margin)
                    y2 = min(gray.shape[0], y_ + h + margin)
                    face_img = gray[y1:y2, x1:x2]
                else:
                    t_now  = time.time()
                    t_prev = t_now
                    frame_idx += 1
                    continue   # skip frame if no face found

            # PCA transform (PS numpy — microseconds)
            embedding = pca_transform(face_img, pca_mean, pca_comps)

            # OpenCV DNN inference — blob shape (1, 50)
            blob = embedding.reshape(1, n_components)
            net.setInput(blob)
            logits = net.forward()[0]     # (num_classes,)
            probs  = softmax(logits)
            cls    = int(np.argmax(probs))
            conf   = float(probs[cls])

            # Decision
            if conf >= UNKNOWN_THR:
                name     = labels.get(str(cls), f"person_{cls}")
                decision = f"CONOCIDO ({name})"
            else:
                decision = "DESCONOCIDO"

            t_now = time.time()
            fps   = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            print(f"[{frame_idx:05d}] {decision:<22}  conf={conf:.2f}  fps={fps:.1f}  "
                  f"probs={[f'{p:.2f}' for p in probs]}")

            if args.save:
                small = cv2.resize(face_img, (FACE_SIZE * 4, FACE_SIZE * 4))
                small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
                cv2.putText(small_bgr, decision, (2, 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 255, 0) if "CONOCIDO" in decision else (0, 0, 255), 1)
                cv2.imwrite(f"face_frames/frame_{frame_idx:05d}.jpg", small_bgr)

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
