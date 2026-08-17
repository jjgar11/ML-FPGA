"""
GTSRB inference in pure numpy — compatible with Python 3.7 + OpenCV 3.4.3 on Ultra96.
Does not require onnxruntime or PyTorch. Only numpy and opencv (for image/camera).

Preparation on the dev machine:
    python dev/export_weights_numpy.py
    scp data/models/gtsrb_gap_weights.npz  root@10.164.145.147:/home/root/
    scp assets/gtsrb_class_names.json       root@10.164.145.147:/home/root/
    scp dev/camera/infer_gtsrb_numpy.py     root@10.164.145.147:/home/root/

Usage on the Ultra96:
    python3 infer_gtsrb_numpy.py --image sign.ppm
    python3 infer_gtsrb_numpy.py --cam 0 [--conf 0.80] [--confirm 3]
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

IMG_SIZE = 64
MEAN = np.array([0.3337, 0.3064, 0.3171], dtype=np.float32)
STD  = np.array([0.2672, 0.2564, 0.2629], dtype=np.float32)

MIN_ROI_PX   = 40    # minimum bounding box side — discards small noise
MAX_ROI_FRAC = 0.30  # discards ROIs occupying > 30% of the frame (colored backgrounds)


# ---------------------------------------------------------------------------
# Numpy operations for the forward pass
# ---------------------------------------------------------------------------

def conv2d(x, W, b, padding=1):
    """x:(C_in,H,W)  W:(C_out,C_in,kH,kW) → (C_out,H,W), same-padding, stride 1."""
    C_in, H, Wi = x.shape
    C_out, _, kH, kW = W.shape
    if padding:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))
    H_out, W_out = H, Wi
    cols = np.lib.stride_tricks.as_strided(
        x,
        shape=(C_in, kH, kW, H_out, W_out),
        strides=(x.strides[0], x.strides[1], x.strides[2],
                 x.strides[1], x.strides[2]),
    ).reshape(C_in * kH * kW, H_out * W_out)
    return (W.reshape(C_out, -1) @ cols + b[:, None]).reshape(C_out, H_out, W_out)


def batchnorm(x, w, b, mean, var, eps=1e-5):
    return w[:, None, None] * (x - mean[:, None, None]) / np.sqrt(var[:, None, None] + eps) + b[:, None, None]


def maxpool2(x):
    C, H, W = x.shape
    return x[:, :H//2*2, :W//2*2].reshape(C, H//2, 2, W//2, 2).max(axis=(2, 4))


def relu(x):
    return np.maximum(0.0, x)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GtsrbNumpyModel:
    def __init__(self, npz_path):
        w = np.load(npz_path)
        self.conv1_w = w["conv1_w"].astype(np.float32)
        self.conv1_b = w["conv1_b"].astype(np.float32)
        self.bn1     = (w["bn1_w"], w["bn1_b"], w["bn1_mean"], w["bn1_var"])
        self.conv2_w = w["conv2_w"].astype(np.float32)
        self.conv2_b = w["conv2_b"].astype(np.float32)
        self.bn2     = (w["bn2_w"], w["bn2_b"], w["bn2_mean"], w["bn2_var"])
        self.conv3_w = w["conv3_w"].astype(np.float32)
        self.conv3_b = w["conv3_b"].astype(np.float32)
        self.bn3     = (w["bn3_w"], w["bn3_b"], w["bn3_mean"], w["bn3_var"])
        self.fc1_w   = w["fc1_w"].astype(np.float32)
        self.fc1_b   = w["fc1_b"].astype(np.float32)
        self.fc2_w   = w["fc2_w"].astype(np.float32)
        self.fc2_b   = w["fc2_b"].astype(np.float32)

    def forward(self, x):
        """x: (3, H, W) normalized float32 → logits (43,)  [any resolution: GAP]"""
        x = relu(batchnorm(conv2d(x, self.conv1_w, self.conv1_b), *self.bn1))
        x = maxpool2(x)
        x = relu(batchnorm(conv2d(x, self.conv2_w, self.conv2_b), *self.bn2))
        x = maxpool2(x)
        x = relu(batchnorm(conv2d(x, self.conv3_w, self.conv3_b), *self.bn3))
        x = maxpool2(x)
        x = x.mean(axis=(1, 2))
        x = relu(self.fc1_w @ x + self.fc1_b)
        return self.fc2_w @ x + self.fc2_b


# ---------------------------------------------------------------------------
# Temporal filter — avoids single-frame false detections
# ---------------------------------------------------------------------------

class TemporalFilter:
    """
    Confirms a class only if it appears with conf >= conf_thr during
    `confirm` consecutive frames. Resets after `clear_after` frames
    with no valid detection.

    Typical usage:
        filt = TemporalFilter(conf_thr=0.80, confirm=3)
        result = filt.update([(cls, conf), ...])   # None or (cls, conf)
    """
    def __init__(self, conf_thr=0.80, confirm=3, clear_after=5):
        self.conf_thr    = conf_thr
        self.confirm     = confirm
        self.clear_after = clear_after
        self._candidate  = None   # (class_id, conf)
        self._streak     = 0      # consecutive frames with the same candidate
        self._no_det     = 0      # frames with no valid detection

    def update(self, detections):
        """
        detections: list of (class_id, conf) sorted by conf desc.
        Returns (class_id, conf) if there is a confirmed detection, None otherwise.
        """
        valid = [(c, p) for c, p in detections if p >= self.conf_thr]

        if not valid:
            self._no_det += 1
            if self._no_det >= self.clear_after:
                self._candidate = None
                self._streak    = 0
            return None

        self._no_det = 0
        best = valid[0]   # highest confidence

        if self._candidate is not None and best[0] == self._candidate[0]:
            self._streak   += 1
            self._candidate = best
        else:
            self._candidate = best
            self._streak    = 1

        if self._streak >= self.confirm:
            return self._candidate
        return None

    def reset(self):
        self._candidate = None
        self._streak    = 0
        self._no_det    = 0


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def is_sharp(bgr_crop, min_var=50):
    """Discards crops without well-defined edges (backgrounds, uniform color blobs)."""
    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= min_var


def preprocess(bgr_crop):
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)


# ---------------------------------------------------------------------------
# Sign detector by HSV color
# ---------------------------------------------------------------------------

def detect_sign_rois(frame, min_px=MIN_ROI_PX, max_frac=MAX_ROI_FRAC):
    """
    Returns a list of (x, y, w, h, area) sorted by descending area.
    Filters:
      - min_px: minimum side in pixels
      - max_frac: discards ROIs occupying more than max_frac of the frame (backgrounds)
      - square aspect (signs are circular or triangular ~1:1)
    """
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    r1     = cv2.inRange(hsv, (  0, 100, 80), ( 12, 255, 255))  # warm red
    r2     = cv2.inRange(hsv, (168, 100, 80), (180, 255, 255))  # cool red
    blue   = cv2.inRange(hsv, (100,  80, 80), (130, 255, 255))  # blue signs (mandatory)
    yellow = cv2.inRange(hsv, ( 15, 120, 80), ( 38, 255, 255))  # yellow signs (S>=120 avoids cables/screens)
    mask   = r1 | r2 | blue | yellow

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    ret = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = ret[-2]
    fh, fw   = frame.shape[:2]
    max_bbox = fw * fh * max_frac
    boxes    = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1200:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > max_bbox:                        # bbox too large
            continue
        fill = area / max(w * h, 1)
        if fill < 0.25:                             # sparse blob (edge, gradient)
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * 3.14159 * area / (perimeter * perimeter)
            if circularity < 0.35:                  # signs are compact; wall stripes are not
                continue
        if w < min_px or h < min_px:
            continue
        if not (0.45 < w / max(h, 1) < 2.2):
            continue
        # Exclude ROIs that touch the frame border — backgrounds always reach the border
        border = 4
        if x <= border or y <= border or (x + w) >= fw - border or (y + h) >= fh - border:
            continue
        pad = int(max(w, h) * 0.12)
        x = max(0, x - pad);  y = max(0, y - pad)
        w = min(fw - x, w + 2 * pad);  h = min(fh - y, h + 2 * pad)
        boxes.append((x, y, w, h, area))
    # Sort by descending area — the real sign is usually the largest object
    boxes.sort(key=lambda b: b[4], reverse=True)
    candidates = [(x, y, w, h) for x, y, w, h, _ in boxes]
    return _nms(candidates)


def _nms(boxes, iou_thr=0.4):
    """Removes overlapping boxes (>iou_thr IoU). Keeps the one with the largest area."""
    if len(boxes) <= 1:
        return boxes
    suppressed = [False] * len(boxes)
    areas = [w * h for x, y, w, h in boxes]
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        xi, yi, wi, hi = boxes[i]
        for j in range(i + 1, len(boxes)):
            if suppressed[j]:
                continue
            xj, yj, wj, hj = boxes[j]
            ix  = max(xi, xj);         iy  = max(yi, yj)
            ix2 = min(xi + wi, xj + wj); iy2 = min(yi + hi, yj + hj)
            if ix2 <= ix or iy2 <= iy:
                continue
            inter = (ix2 - ix) * (iy2 - iy)
            iou   = inter / max(areas[i] + areas[j] - inter, 1)
            if iou > iou_thr:
                suppressed[j] = True
    return [boxes[i] for i in range(len(boxes)) if not suppressed[i]]


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_image(model, path, names, conf_thr):
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"Could not open: {path}"); sys.exit(1)
    t0     = time.time()
    logits = model.forward(preprocess(bgr))
    ms     = (time.time() - t0) * 1000
    probs  = softmax(logits)
    top5   = np.argsort(probs)[::-1][:5]
    print(f"Image  : {path}   ({ms:.1f} ms)")
    print(f"Threshold: {conf_thr:.0%}   (would only be reported if conf >= threshold)")
    for rank, c in enumerate(top5):
        bar    = "#" * int(probs[c] * 30)
        flag   = " ✓" if probs[c] >= conf_thr else ""
        print(f"  {rank+1}. [{c:2d}] {names[c]:<38} {probs[c]*100:5.1f}%  {bar}{flag}")


def run_camera(model, cam_idx, names, conf_thr, confirm, save, debug, flip, top1=False):
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Could not open /dev/video{cam_idx}"); sys.exit(1)
    if debug:
        os.makedirs("gtsrb_debug", exist_ok=True)
    if save:
        os.makedirs("gtsrb_frames", exist_ok=True)

    use_filter = confirm > 1
    filt       = TemporalFilter(conf_thr=conf_thr, confirm=confirm) if use_filter else None
    frame_idx  = 0
    t_prev     = time.time()

    print(f"cam={cam_idx}  conf>={conf_thr:.0%}  confirm={confirm}"
          + ("  flip" if flip else "") + ("  debug" if debug else ""))
    print("─" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if flip:
                frame = cv2.flip(frame, 1)

            t_now  = time.time()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            boxes = detect_sign_rois(frame)

            detections = []
            for x, y, w, h in boxes:
                crop = frame[y:y+h, x:x+w]
                if crop.size == 0:
                    continue
                if not is_sharp(crop):
                    continue
                logits = model.forward(preprocess(crop))
                probs  = softmax(logits)
                cls    = int(np.argmax(probs))
                conf   = float(probs[cls])
                detections.append((cls, conf, x, y, w, h))
                if save:
                    cv2.imwrite(f"gtsrb_frames/f{frame_idx:05d}_r{len(detections)-1}.jpg", crop)

            detections.sort(key=lambda d: d[1], reverse=True)
            if top1 and detections:
                detections = detections[:1]

            # --- Print each frame ---
            if not detections:
                print(f"[{frame_idx:04d}] {fps:4.1f}fps  no ROIs")
            else:
                for cls, conf, *_ in detections:
                    flag = "OK" if conf >= conf_thr else "  "
                    print(f"[{frame_idx:04d}] {fps:4.1f}fps  {flag}  [{cls:2d}] {names[cls]:<35} {conf:.2f}")

            # --- Optional temporal filter (--confirm > 1) ---
            if use_filter:
                det_kv = [(c, p) for c, p, *_ in detections]
                result = filt.update(det_kv)
                if result:
                    print(f"  >>> CONFIRMED: [{result[0]:2d}] {names[result[0]]}  conf={result[1]:.2f}")

            # --- Debug: save each frame ---
            if debug:
                ann = frame.copy()
                for cls, conf, x, y, w, h in detections:
                    color = (0, 255, 0) if conf >= conf_thr else (0, 165, 255)
                    cv2.rectangle(ann, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(ann, f"{names[cls][:16]} {conf:.2f}",
                                (x, max(y - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                if not detections:
                    cv2.putText(ann, "no ROIs", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite(f"gtsrb_debug/f{frame_idx:05d}.jpg", ann)

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
    ap.add_argument("--weights", default="./gtsrb_gap_weights.npz")
    ap.add_argument("--names",   default="./gtsrb_class_names.json")
    ap.add_argument("--image",   default=None,  help="Classifies this crop and exits")
    ap.add_argument("--cam",     type=int, default=0)
    ap.add_argument("--conf",    type=float, default=0.80,
                    help="Minimum confidence to report (default 0.80)")
    ap.add_argument("--confirm", type=int, default=3,
                    help="Consecutive frames needed to confirm (default 3)")
    ap.add_argument("--save",    action="store_true",
                    help="Saves detected crops to ./gtsrb_frames/")
    ap.add_argument("--debug",   action="store_true",
                    help="Saves annotated frames with ROIs to ./gtsrb_debug/ (every 15 frames)")
    ap.add_argument("--flip",    action="store_true",
                    help="Flips the image horizontally (corrects a mirrored camera)")
    ap.add_argument("--top1",   action="store_true",
                    help="Returns only the highest-confidence detection per frame")
    args = ap.parse_args()

    for path, label in [(args.weights, "weights"), (args.names, "names")]:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(1)

    with open(args.names) as f:
        raw = json.load(f)
    names = [raw[str(i)] for i in range(43)]

    print(f"Loading weights: {args.weights}")
    model = GtsrbNumpyModel(args.weights)
    print("Model loaded.")

    if args.image:
        run_image(model, args.image, names, args.conf)
    else:
        run_camera(model, args.cam, names, args.conf, args.confirm, args.save, args.debug, args.flip, args.top1)


if __name__ == "__main__":
    main()