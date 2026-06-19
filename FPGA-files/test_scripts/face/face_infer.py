"""
Face recognition inference on Ultra96 — PCA on PS, MLP on PL via UIO.

Requires in /home/root/face/:
  face_pca_mean.npy        — from dev/train/face_recognition.py
  face_pca_components.npy  — from dev/train/face_recognition.py
  face_labels.json         — from dev/train/face_recognition.py
  face_test_data.json      — (optional) for batch accuracy test

Usage — live camera:
  python face_infer.py --mode live [--cam 0]

Usage — batch accuracy test (no camera needed):
  python face_infer.py --mode test --data face_test_data.json

Register map (fill in after checking generated myproject.cpp / hardware.h):
  --reg-out  offset of first output word   (default 0x10, same as Wine)
  --reg-in   offset of first input word    (check synthesis report)
  --n-out    number of output values       (= num_classes)
  --n-in     number of input values        (= n_components = 50)
  --prec     fixed-point fractional bits   (10 for ap_fixed<16,6>)
  --bits     fixed-point total bits        (16 for ap_fixed<16,6>)
"""

import argparse
import json
import mmap
import os
import struct
import sys
import time

import numpy as np

CASCADE_PATHS = [
    "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml",
    "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml",
    "/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
]

FACE_SIZE    = 32
FACE_DIR     = "/home/root/fpga/test_scripts/face"
UIO_DEV      = "/dev/uio4"
MAP_SIZE     = 0x10000
UNKNOWN_THR  = 0.60
BBOX_MAX_AGE = 20


# ── Fixed-point helpers (mirror wine_infer.py pattern) ────────────────────────

def make_fixed(frac_bits, total_bits):
    max_val =  (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))
    mask    =  (1 << total_bits) - 1
    half    =   1 << (total_bits - 1)

    def to_fixed(val):
        raw = int(round(val * (1 << frac_bits)))
        return max(min_val, min(max_val, raw)) & mask

    def from_fixed(raw):
        raw &= mask
        if raw >= half:
            raw -= (1 << total_bits)
        return raw / (1 << frac_bits)

    return to_fixed, from_fixed


def read32(mm, off):
    mm.seek(off)
    return struct.unpack('<I', mm.read(4))[0]


def write32(mm, off, val):
    mm.seek(off)
    mm.write(struct.pack('<I', val))


def write_inputs(mm, values, reg_in, n_in, to_fixed, bits):
    """Pack two fixed-point values per 32-bit AXI-Lite word."""
    n_words = (n_in + 1) // 2
    for w in range(n_words):
        lo = to_fixed(values[2 * w])     if 2 * w     < n_in else 0
        hi = to_fixed(values[2 * w + 1]) if 2 * w + 1 < n_in else 0
        word = ((hi & ((1 << bits) - 1)) << bits) | (lo & ((1 << bits) - 1))
        write32(mm, reg_in + w * 4, word)


def read_outputs(mm, reg_out, n_out, from_fixed, bits):
    """Unpack two fixed-point values per 32-bit AXI-Lite word."""
    results = []
    n_words = (n_out + 1) // 2
    for w in range(n_words):
        word = read32(mm, reg_out + w * 4)
        results.append(from_fixed(word & ((1 << bits) - 1)))
        if 2 * w + 1 < n_out:
            results.append(from_fixed((word >> bits) & ((1 << bits) - 1)))
    return results[:n_out]


# ── PCA transform (PS side) ────────────────────────────────────────────────────

def load_pca(face_dir):
    mean = np.load(os.path.join(face_dir, "face_pca_mean.npy"))       # (1024,)
    comps = np.load(os.path.join(face_dir, "face_pca_components.npy")) # (50, 1024)
    return mean, comps


def pca_transform(img_flat, mean, comps):
    """img_flat: (1024,) float32 [0,1] → embedding: (50,) float32"""
    centered = img_flat - mean
    return (comps @ centered).astype(np.float32)


def preprocess_face(img_gray):
    """Crop / resize gray frame → (1024,) float32 [0,1]"""
    resized = __import__("cv2").resize(img_gray, (FACE_SIZE, FACE_SIZE))
    return resized.flatten().astype(np.float32) / 255.0


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ── FPGA inference ────────────────────────────────────────────────────────────

def fpga_predict(mm, embedding, reg_in, reg_out, n_in, n_out, to_fixed, from_fixed, bits):
    write_inputs(mm, embedding, reg_in, n_in, to_fixed, bits)
    write32(mm, 0x00, 0x01)   # ap_start
    time.sleep(0.001)
    logits = read_outputs(mm, reg_out, n_out, from_fixed, bits)
    probs  = softmax(np.array(logits))
    cls    = int(np.argmax(probs))
    conf   = float(probs[cls])
    return cls, conf, probs


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_test(mm, test_data_path, pca_mean, pca_comps,
             reg_in, reg_out, n_in, n_out, to_fixed, from_fixed, bits, labels):
    with open(test_data_path) as f:
        data = json.load(f)
    X, y = data["X"], data["y"]
    correct = 0
    for i in range(len(X)):
        emb = np.array(X[i], dtype=np.float32)
        cls, conf, _ = fpga_predict(mm, emb, reg_in, reg_out,
                                     n_in, n_out, to_fixed, from_fixed, bits)
        if cls == y[i]:
            correct += 1
    print(f"FPGA accuracy: {correct}/{len(y)} = {correct/len(y)*100:.1f}%")

    print("\n--- First 5 samples ---")
    for i in range(min(5, len(X))):
        emb = np.array(X[i], dtype=np.float32)
        cls, conf, probs = fpga_predict(mm, emb, reg_in, reg_out,
                                         n_in, n_out, to_fixed, from_fixed, bits)
        status = "OK" if cls == y[i] else "FAIL"
        pred_name = labels.get(str(cls), f"cls{cls}")
        real_name = labels.get(str(y[i]), f"cls{y[i]}")
        print(f"[{i}] real={real_name:<12} pred={pred_name:<12} "
              f"conf={conf:.2f} {status}")


def _cap_read(cap, result):
    result[0], result[1] = cap.read()


def run_live(mm, pca_mean, pca_comps,
             reg_in, reg_out, n_in, n_out, to_fixed, from_fixed, bits, labels, cam,
             save_dir=None):
    import cv2
    import threading

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    cascade_path = next((p for p in CASCADE_PATHS if os.path.exists(p)), None)
    detector = cv2.CascadeClassifier(cascade_path) if cascade_path else None
    if detector is None or detector.empty():
        print("WARNING: No Haar cascade found — skipping frames without face")
        detector = None
    else:
        print(f"Cascade loaded: {cascade_path}")

    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"Cannot open camera {cam}")
        sys.exit(1)

    print("Live face recognition (Ctrl+C to stop)\n")
    frame_idx     = 0
    last_bbox     = None
    last_bbox_age = BBOX_MAX_AGE
    try:
        while True:
            result = [False, None]
            t = threading.Thread(target=_cap_read, args=(cap, result), daemon=True)
            t.start()
            t.join(timeout=3.0)
            if t.is_alive():
                print("Camera read timeout — exiting")
                break
            ret, frame = result
            if not ret:
                time.sleep(0.05)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (640, 360))
            scale_x = gray.shape[1] / 640
            scale_y = gray.shape[0] / 360

            face_img = None
            bbox = None
            if detector is not None and not detector.empty():
                faces = detector.detectMultiScale(
                    small, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
                )
                if len(faces) > 0:
                    sx, sy, sw, sh = sorted(faces, key=lambda f: f[2]*f[3])[-1]
                    x  = int(sx * scale_x)
                    y_ = int(sy * scale_y)
                    w  = int(sw * scale_x)
                    h  = int(sh * scale_y)
                    H_img, W_img = gray.shape
                    pad_x   = int(w * 0.20)
                    pad_top = int(h * 0.60)
                    pad_bot = int(h * 0.20)
                    x  = max(0, x - pad_x)
                    y_ = max(0, y_ - pad_top)
                    w  = min(W_img - x, w + 2 * pad_x)
                    h  = min(H_img - y_, h + pad_top + pad_bot)
                    last_bbox     = (x, y_, w, h)
                    last_bbox_age = 0

            if last_bbox is not None and last_bbox_age < BBOX_MAX_AGE:
                bbox = last_bbox
                last_bbox_age += 1
                x, y_, w, h = bbox
                face_img = gray[y_:y_+h, x:x+w]
            else:
                print(f"[{frame_idx:05d}] NO FACE")
                frame_idx += 1
                continue

            flat   = preprocess_face(face_img)
            emb    = pca_transform(flat, pca_mean, pca_comps)
            cls, conf, probs = fpga_predict(mm, emb, reg_in, reg_out,
                                             n_in, n_out, to_fixed, from_fixed, bits)

            if conf < UNKNOWN_THR:
                decision = "UNKNOWN"
            else:
                decision = labels.get(str(cls), f"person_{cls}").upper()

            print(f"[{frame_idx:05d}] {decision:<14}  conf={conf:.2f}  "
                  f"probs={[f'{p:.2f}' for p in probs]}")

            if save_dir:
                vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if bbox:
                    x, y_, w, h = bbox
                    cv2.rectangle(vis, (x, y_), (x+w, y_+h), (0, 255, 0), 2)
                else:
                    h_f, w_f = gray.shape
                    cv2.rectangle(vis, (0, 0), (w_f-1, h_f-1), (0, 165, 255), 2)
                cv2.putText(vis, f"{decision} {conf:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(save_dir, f"{frame_idx:05d}_{decision}.jpg"), vis)

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["live", "test"], default="live")
    parser.add_argument("--cam",     type=int, default=0)
    parser.add_argument("--save-frames", metavar="DIR", default=None,
                        help="Guardar frames clasificados (no UNKNOWN) en este directorio")
    parser.add_argument("--data",    default=os.path.join(FACE_DIR, "face_test_data.json"))
    parser.add_argument("--face-dir", default=FACE_DIR)
    parser.add_argument("--uio",     default=UIO_DEV)
    # Register map — check generated myproject.cpp after hls4ml synthesis
    parser.add_argument("--reg-out", default="0x74",
                        help="AXI-Lite offset of first output word (hex)")
    parser.add_argument("--reg-in",  default="0x20",
                        help="AXI-Lite offset of first input word (hex)")
    parser.add_argument("--n-out",   type=int, default=2,
                        help="Number of output classes")
    parser.add_argument("--n-in",    type=int, default=50,
                        help="Number of PCA components (input size)")
    parser.add_argument("--prec",    type=int, default=10,
                        help="Fractional bits (10 for ap_fixed<16,6>)")
    parser.add_argument("--bits",    type=int, default=16,
                        help="Total bits (16 for ap_fixed<16,6>)")
    args = parser.parse_args()

    reg_out = int(args.reg_out, 16)
    reg_in  = int(args.reg_in,  16)

    # Load PCA
    pca_mean, pca_comps = load_pca(args.face_dir)
    with open(os.path.join(args.face_dir, "face_labels.json")) as f:
        labels = json.load(f)
    print(f"Loaded PCA: {pca_comps.shape}  labels: {labels}")

    to_fixed, from_fixed = make_fixed(args.prec, args.bits)

    # Open UIO
    if not os.path.exists(args.uio):
        print(f"UIO device not found: {args.uio}")
        print("Apply DT overlay first:  mkdir /sys/kernel/config/device-tree/overlays/face")
        print("  echo -n face_mlp.dtbo > /sys/kernel/config/device-tree/overlays/face/path")
        sys.exit(1)

    with open(args.uio, 'r+b', buffering=0) as f:
        mm = mmap.mmap(f.fileno(), MAP_SIZE)

        if args.mode == "test":
            run_test(mm, args.data, pca_mean, pca_comps,
                     reg_in, reg_out, args.n_in, args.n_out,
                     to_fixed, from_fixed, args.bits, labels)
        else:
            run_live(mm, pca_mean, pca_comps,
                     reg_in, reg_out, args.n_in, args.n_out,
                     to_fixed, from_fixed, args.bits, labels, args.cam,
                     save_dir=args.save_frames)

        mm.close()


if __name__ == "__main__":
    main()
