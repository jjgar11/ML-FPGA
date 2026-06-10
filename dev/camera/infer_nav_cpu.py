"""
NavCNN CPU inference on Ultra96 — uses OpenCV DNN (no onnxruntime needed).
Runs the ONNX float model on frames from the robot camera and prints
the navigation command.

Requirements on Ultra96:
  /home/root/nav_cnn_float.onnx   (copy with scp from dev machine)

Usage:
  python infer_nav_cpu.py [--model nav_cnn_float.onnx] [--cam 0] [--save]

Controls:  Ctrl+C to stop.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

CLASSES   = ["forward", "left", "right", "stop"]
IMG_SIZE  = 64


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def preprocess(frame):
    """BGR frame → (1, 1, 64, 64) float32 blob normalised to [0, 1]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blobFromImage on a single-channel image produces (1, 1, H, W)
    blob = cv2.dnn.blobFromImage(gray, scalefactor=1.0 / 255.0,
                                 size=(IMG_SIZE, IMG_SIZE),
                                 mean=0, swapRB=False, crop=False)
    return blob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/root/nav_cnn_float.onnx")
    parser.add_argument("--cam",   type=int, default=0)
    parser.add_argument("--save",  action="store_true",
                        help="Save annotated JPEG frames to ./nav_frames/")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        print("Copy from dev machine:  scp data/models/nav_cnn_float.onnx root@192.168.0.103:/home/root/")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    net = cv2.dnn.readNetFromONNX(args.model)
    print("Model loaded.")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Cannot open camera {args.cam}")
        sys.exit(1)
    print(f"Camera /dev/video{args.cam} opened. Press Ctrl+C to stop.\n")

    if args.save:
        os.makedirs("nav_frames", exist_ok=True)

    frame_idx = 0
    t_prev    = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed")
                break

            blob = preprocess(frame)
            net.setInput(blob)
            logits = net.forward()[0]      # (4,) raw logits
            probs  = softmax(logits)
            cls    = int(np.argmax(probs))
            conf   = float(probs[cls])

            t_now = time.time()
            fps   = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            print(f"[{frame_idx:05d}] {CLASSES[cls]:>8s}  conf={conf:.2f}  "
                  f"fps={fps:.1f}  "
                  f"[fwd={probs[0]:.2f} lft={probs[1]:.2f} rgt={probs[2]:.2f} stp={probs[3]:.2f}]")

            if args.save:
                # Annotate a 64×64 crop for inspection
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                   (IMG_SIZE, IMG_SIZE))
                small_bgr = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
                cv2.putText(small_bgr, CLASSES[cls], (2, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.imwrite(f"nav_frames/frame_{frame_idx:05d}.jpg", small_bgr)

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
