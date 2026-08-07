"""
GTSRB classifier on Ultra96 — CPU inference via OpenCV DNN (no onnxruntime).
Compatible with Python 3.7.6 + OpenCV 3.4.3 from the Avnet 2020.1 OOB.

Two modes:
  --image crop.jpg   Classifies an already-extracted crop (Phase 1).
  (no --image)       Live camera: HSV detector + classifier (Phase 2).

Preparation on the dev machine:
    python dev/train/gtsrb.py --arch gtsrb_gap --epochs 20
    python dev/export_onnx.py --model gtsrb_gap
    scp data/models/gtsrb_gap_float.onnx root@192.168.0.103:/home/root/
    scp data/gtsrb_class_names.json      root@192.168.0.103:/home/root/

Usage on the Ultra96:
    python infer_gtsrb_cpu.py --image my_sign.jpg
    python infer_gtsrb_cpu.py --cam 0
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

IMG_SIZE   = 48
MEAN       = np.array([0.3337 * 255, 0.3064 * 255, 0.3171 * 255], dtype=np.float32)
STD        = np.array([0.2672,       0.2564,        0.2629      ], dtype=np.float32)
CONF_THR   = 0.50   # show detection only if conf >= threshold


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(bgr_crop):
    """BGR crop → (1, 3, 32, 32) float32 blob ready for cv2.dnn."""
    rgb  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    resz = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    # blobFromImage: scalefactor=1, mean subtracts the GTSRB mean (* 255)
    blob = cv2.dnn.blobFromImage(resz, scalefactor=1.0,
                                  size=(IMG_SIZE, IMG_SIZE),
                                  mean=MEAN, swapRB=False, crop=False)
    # blob shape: (1, 3, 32, 32) — divide by std channel by channel
    blob[:, 0] /= STD[0]
    blob[:, 1] /= STD[1]
    blob[:, 2] /= STD[2]
    return blob


# ---------------------------------------------------------------------------
# Color-based sign detector (Phase 2)
# ---------------------------------------------------------------------------

def detect_sign_rois(frame):
    """
    Returns a list of (x, y, w, h) with candidate traffic signs.
    Looks for red regions (speed limits, STOP, yield) and blue ones (mandatory).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red: H wraps around 0, need to join two ranges
    r1 = cv2.inRange(hsv, (0,   80, 80), (12,  255, 255))
    r2 = cv2.inRange(hsv, (168, 80, 80), (180, 255, 255))
    red_mask  = cv2.bitwise_or(r1, r2)

    # Blue: H ~100-130
    blue_mask = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))

    mask = cv2.bitwise_or(red_mask, blue_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,   kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fh, fw = frame.shape[:2]
    boxes  = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:          # too small
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if not (0.35 < aspect < 2.8):   # reasonably square shape
            continue
        # 15% padding
        pad = int(max(w, h) * 0.15)
        x  = max(0, x - pad)
        y  = max(0, y - pad)
        w  = min(fw - x, w + 2 * pad)
        h  = min(fh - y, h + 2 * pad)
        boxes.append((x, y, w, h))
    return boxes


# ---------------------------------------------------------------------------
# Softmax helper
# ---------------------------------------------------------------------------

def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def classify_crop(net, bgr_crop):
    """Returns (class_id, confidence, probs)."""
    blob   = preprocess(bgr_crop)
    net.setInput(blob)
    logits = net.forward()[0]   # (43,)
    probs  = softmax(logits)
    cls    = int(np.argmax(probs))
    return cls, float(probs[cls]), probs


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_image(net, path, names):
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"Could not open: {path}")
        sys.exit(1)

    cls, conf, probs = classify_crop(net, bgr)
    top5 = np.argsort(probs)[::-1][:5]

    print(f"Image  : {path}")
    print(f"Top-5  :")
    for rank, c in enumerate(top5):
        bar = "#" * int(probs[c] * 30)
        print(f"  {rank+1}. [{c:2d}] {names[c]:<38} {probs[c]*100:5.1f}%  {bar}")


def run_camera(net, cam_idx, names, save):
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Could not open /dev/video{cam_idx}")
        sys.exit(1)
    print(f"Camera /dev/video{cam_idx} opened. Ctrl+C to exit.\n")

    if save:
        os.makedirs("gtsrb_frames", exist_ok=True)

    frame_idx = 0
    t_prev    = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")
                break

            t_now = time.time()
            fps   = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            boxes = detect_sign_rois(frame)

            if not boxes:
                print(f"[{frame_idx:05d}] fps={fps:.1f}  no detections")
            else:
                for i, (x, y, w, h) in enumerate(boxes):
                    crop = frame[y:y+h, x:x+w]
                    if crop.size == 0:
                        continue
                    cls, conf, _ = classify_crop(net, crop)
                    status = names[cls] if conf >= CONF_THR else f"? (conf={conf:.2f})"
                    print(f"[{frame_idx:05d}] fps={fps:.1f}  ROI#{i}  [{cls:2d}] {status:<38} conf={conf:.2f}")

                    if save:
                        cv2.imwrite(f"gtsrb_frames/frame{frame_idx:05d}_roi{i}.jpg", crop)

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="/home/root/test_inference/road_signs/gtsrb_gap_float.onnx",
                    help="Path to the ONNX (gtsrb_gap_float.onnx recommended)")
    ap.add_argument("--names",
                    default="/home/root/test_inference/road_signs/gtsrb_class_names.json")
    ap.add_argument("--image", default=None,
                    help="Classifies this sign crop and exits")
    ap.add_argument("--cam",   type=int, default=0,
                    help="Camera index for live mode")
    ap.add_argument("--save",  action="store_true",
                    help="Saves the detected crops to ./gtsrb_frames/")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("On the dev machine:")
        print("  python dev/train/gtsrb.py --arch gtsrb_gap --epochs 20")
        print("  python dev/export_onnx.py --model gtsrb_gap")
        print("  scp data/models/gtsrb_gap_float.onnx root@192.168.0.103:/home/root/")
        sys.exit(1)

    if not os.path.exists(args.names):
        print(f"class_names not found: {args.names}")
        print("  scp data/gtsrb_class_names.json root@192.168.0.103:/home/root/")
        sys.exit(1)

    with open(args.names) as f:
        raw = json.load(f)
    names = [raw[str(i)] for i in range(43)]

    print(f"Loading model: {args.model}")
    net = cv2.dnn.readNet(args.model)
    print("Model loaded.")

    if args.image:
        run_image(net, args.image, names)
    else:
        run_camera(net, args.cam, names, args.save)


if __name__ == "__main__":
    main()