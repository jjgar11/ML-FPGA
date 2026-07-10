"""
GTSRB classifier on Ultra96 — CPU inference via OpenCV DNN (no onnxruntime).
Compatible con Python 3.7.6 + OpenCV 3.4.3 del OOB Avnet 2020.1.

Dos modos:
  --image crop.jpg   Clasifica un recorte ya extraído (Fase 1).
  (sin --image)      Cámara en vivo: detector HSV + clasificador (Fase 2).

Preparación en la dev machine:
    python dev/train/gtsrb.py --arch gtsrb_gap --epochs 20
    python dev/export_onnx.py --model gtsrb_gap
    scp data/models/gtsrb_gap_float.onnx root@192.168.0.103:/home/root/
    scp data/gtsrb_class_names.json      root@192.168.0.103:/home/root/

Uso en la Ultra96:
    python infer_gtsrb_cpu.py --image mi_señal.jpg
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
CONF_THR   = 0.50   # mostrar detección solo si conf >= umbral


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(bgr_crop):
    """BGR crop → (1, 3, 32, 32) float32 blob listo para cv2.dnn."""
    rgb  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    resz = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
    # blobFromImage: scalefactor=1, mean sustrae la media GTSRB (* 255)
    blob = cv2.dnn.blobFromImage(resz, scalefactor=1.0,
                                  size=(IMG_SIZE, IMG_SIZE),
                                  mean=MEAN, swapRB=False, crop=False)
    # blob shape: (1, 3, 32, 32) — divide por std canal a canal
    blob[:, 0] /= STD[0]
    blob[:, 1] /= STD[1]
    blob[:, 2] /= STD[2]
    return blob


# ---------------------------------------------------------------------------
# Color-based sign detector (Fase 2)
# ---------------------------------------------------------------------------

def detect_sign_rois(frame):
    """
    Devuelve lista de (x, y, w, h) con candidatos a señal de tránsito.
    Busca regiones rojas (límites de velocidad, STOP, ceda) y azules (obligatorias).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rojo: H envuelve alrededor de 0, hay que unir dos rangos
    r1 = cv2.inRange(hsv, (0,   80, 80), (12,  255, 255))
    r2 = cv2.inRange(hsv, (168, 80, 80), (180, 255, 255))
    red_mask  = cv2.bitwise_or(r1, r2)

    # Azul: H ~100-130
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
        if area < 500:          # demasiado pequeño
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        if not (0.35 < aspect < 2.8):   # forma razonablemente cuadrada
            continue
        # padding del 15 %
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
    """Devuelve (class_id, confidence, probs)."""
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
        print(f"No se pudo abrir: {path}")
        sys.exit(1)

    cls, conf, probs = classify_crop(net, bgr)
    top5 = np.argsort(probs)[::-1][:5]

    print(f"Imagen : {path}")
    print(f"Top-5  :")
    for rank, c in enumerate(top5):
        bar = "#" * int(probs[c] * 30)
        print(f"  {rank+1}. [{c:2d}] {names[c]:<38} {probs[c]*100:5.1f}%  {bar}")


def run_camera(net, cam_idx, names, save):
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"No se pudo abrir /dev/video{cam_idx}")
        sys.exit(1)
    print(f"Cámara /dev/video{cam_idx} abierta. Ctrl+C para salir.\n")

    if save:
        os.makedirs("gtsrb_frames", exist_ok=True)

    frame_idx = 0
    t_prev    = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error leyendo frame")
                break

            t_now = time.time()
            fps   = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

            boxes = detect_sign_rois(frame)

            if not boxes:
                print(f"[{frame_idx:05d}] fps={fps:.1f}  sin detecciones")
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
        print("\nDetenido.")
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="/home/root/test_inference_/road_signs/gtsrb_gap_float.onnx",
                    help="Ruta al ONNX (gtsrb_gap_float.onnx recomendado)")
    ap.add_argument("--names",
                    default="/home/root/test_inference_/road_signs/gtsrb_class_names.json")
    ap.add_argument("--image", default=None,
                    help="Clasifica este recorte de señal y sale")
    ap.add_argument("--cam",   type=int, default=0,
                    help="Índice de cámara para modo en vivo")
    ap.add_argument("--save",  action="store_true",
                    help="Guarda los crops detectados en ./gtsrb_frames/")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"Modelo no encontrado: {args.model}")
        print("En la dev machine:")
        print("  python dev/train/gtsrb.py --arch gtsrb_gap --epochs 20")
        print("  python dev/export_onnx.py --model gtsrb_gap")
        print("  scp data/models/gtsrb_gap_float.onnx root@192.168.0.103:/home/root/")
        sys.exit(1)

    if not os.path.exists(args.names):
        print(f"class_names no encontrado: {args.names}")
        print("  scp data/gtsrb_class_names.json root@192.168.0.103:/home/root/")
        sys.exit(1)

    with open(args.names) as f:
        raw = json.load(f)
    names = [raw[str(i)] for i in range(43)]

    print(f"Cargando modelo: {args.model}")
    net = cv2.dnn.readNet(args.model)
    print("Modelo cargado.")

    if args.image:
        run_image(net, args.image, names)
    else:
        run_camera(net, args.cam, names, args.save)


if __name__ == "__main__":
    main()
