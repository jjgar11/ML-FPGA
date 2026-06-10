"""
Capture face images for training — runs on Ultra96.
Saves cropped 32x32 grayscale face images for one person at a time.

Usage (run once per person):
  python collect_faces.py --person juan  --n 60 --cam 0
  python collect_faces.py --person maria --n 60 --cam 0

Output:
  data/faces/juan/  → 60 .jpg files
  data/faces/maria/ → 60 .jpg files

Then scp the entire data/faces/ folder to your dev machine for training.
"""

import argparse
import os
import sys
import time

import cv2

CASCADE_PATHS = [
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
]

FACE_SIZE = 32


def find_cascade():
    for p in CASCADE_PATHS:
        if os.path.exists(p):
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True, help="Person name (folder label)")
    parser.add_argument("--n",      type=int, default=60,
                        help="Number of face crops to capture")
    parser.add_argument("--cam",    type=int, default=0)
    parser.add_argument("--out",    default="data/faces")
    parser.add_argument("--delay",  type=float, default=0.15,
                        help="Seconds between captures (avoid near-duplicate frames)")
    args = parser.parse_args()

    out_dir = os.path.join(args.out, args.person)
    os.makedirs(out_dir, exist_ok=True)

    cascade_path = find_cascade()
    if cascade_path:
        detector = cv2.CascadeClassifier(cascade_path)
        print(f"Face detector: {cascade_path}")
    else:
        detector = None
        print("WARNING: Haar cascade not found — saving full frames (crop manually later)")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Cannot open camera {args.cam}")
        sys.exit(1)

    print(f"\nCapturing {args.n} images for '{args.person}'")
    print("Position your face in front of the camera.")
    print("Ctrl+C to stop early.\n")

    saved = 0
    skipped = 0

    try:
        while saved < args.n:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed, retrying...")
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if detector is not None:
                faces = detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                if len(faces) == 0:
                    skipped += 1
                    if skipped % 20 == 0:
                        print(f"  No face detected ({skipped} skipped)...")
                    time.sleep(0.05)
                    continue

                # Take the largest face
                x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3])[-1]
                # Add 10% margin
                margin = int(0.1 * min(w, h))
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(gray.shape[1], x + w + margin)
                y2 = min(gray.shape[0], y + h + margin)
                crop = gray[y1:y2, x1:x2]
            else:
                crop = gray

            face_resized = cv2.resize(crop, (FACE_SIZE, FACE_SIZE))
            path = os.path.join(out_dir, f"{saved:04d}.jpg")
            cv2.imwrite(path, face_resized)
            saved += 1
            print(f"  [{saved:03d}/{args.n}] saved {path}")
            time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\nStopped early.")

    finally:
        cap.release()

    print(f"\nDone: {saved} images saved to {out_dir}/")
    if saved > 0:
        print(f"\nNext: scp -r {args.out} <dev-machine>:<project>/data/")


if __name__ == "__main__":
    main()
