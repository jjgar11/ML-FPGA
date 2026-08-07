"""
GTSRB AXI-Lite FPGA inference — gtsrb_gap_s (32×32 input, 43 classes).
Mapa de registros: hls4ml io_parallel + Vivado backend.

Preparación:
    1. Cargar bitstream:  sudo ./load_overlay.sh gtsrb_gap_s_wrapper.bit ...
    2. Identificar UIO:   ls -la /sys/class/uio/  (buscar el IP de la CNN)
    3. En la dev machine:
       python dev/export_onnx.py --model gtsrb_gap_s
       scp data/models/gtsrb_gap_s_float.onnx root@192.168.0.103:/home/root/

Uso en Ultra96:
    python3 test_scripts/gtsrb/gtsrb_axilite_infer.py --image mi_señal.jpg
    python3 test_scripts/gtsrb/gtsrb_axilite_infer.py --cam 0
    python3 test_scripts/gtsrb/gtsrb_axilite_infer.py --bench     # latencia FPGA

Nota: el UIO_DEV y los offsets exactos (REG_IN, REG_OUT, REG_CTRL)
los da hls4ml tras síntesis — ver myproject_axi.h en el proyecto HLS.
"""

import argparse
import json
import mmap
import struct
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Parámetros del modelo
# ---------------------------------------------------------------------------
IMG_SIZE   = 32
N_IN       = 3 * 32 * 32          # 3072
N_OUT      = 43
MEAN       = np.array([0.3337, 0.3064, 0.3171], dtype=np.float32)
STD        = np.array([0.2672, 0.2564, 0.2629], dtype=np.float32)
CONF_THR   = 0.40

# ---------------------------------------------------------------------------
# AXI-Lite mapa de registros (completar tras síntesis — ver myproject_axi.h)
# ap_fixed<16,6>: 6 integer bits, 10 fractional bits
# ---------------------------------------------------------------------------
UIO_DEV    = "/dev/uio0"           # actualizar según ls /sys/class/uio/
MAP_SIZE   = 0x100000              # 1 MB — ajustar según síntesis
REG_CTRL   = 0x00                  # ap_start en bit 0, ap_done en bit 1
REG_IN     = 0x100                 # base de los registros de entrada (confirmar)
REG_OUT    = 0x100 + N_IN * 2     # base de los 43 valores de salida (confirmar)

# ap_fixed<16,6>: fracción = 10 bits
FRAC_IN    = 10
FRAC_OUT   = 10
MAX_IN     = (1 << 15) - 1        # 32767 para ap_fixed<16,6> signed


def to_fixed16(val):
    """Float → ap_fixed<16,6> (signed 16-bit, 10 fractional bits)."""
    raw = int(round(val * (1 << FRAC_IN)))
    return max(-32768, min(32767, raw)) & 0xFFFF


def from_fixed16(raw):
    """ap_fixed<16,6> → float."""
    raw &= 0xFFFF
    if raw >= 0x8000:
        raw -= 0x10000
    return raw / (1 << FRAC_OUT)


def write32(mm, off, val):
    mm.seek(off)
    mm.write(struct.pack('<I', val & 0xFFFFFFFF))


def read32(mm, off):
    mm.seek(off)
    return struct.unpack('<I', mm.read(4))[0]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(bgr_crop):
    """BGR crop → flat float32 array de 3072 elementos, normalizado."""
    rgb   = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resz  = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))          # (32, 32, 3)
    # normalización ImageNet-like: (pixel - MEAN) / STD
    resz  = (resz - MEAN) / STD                             # (32, 32, 3)
    # PyTorch channels-first: (3, 32, 32) → flat
    chw   = resz.transpose(2, 0, 1).flatten()               # (3072,)
    return chw


# ---------------------------------------------------------------------------
# FPGA predict via AXI-Lite
# ---------------------------------------------------------------------------

def write_inputs(mm, flat):
    """Escribe 3072 valores ap_fixed<16,6> en pares de 16-bit dentro de words de 32-bit."""
    for i in range(0, N_IN, 2):
        lo = to_fixed16(flat[i])
        hi = to_fixed16(flat[i + 1]) if i + 1 < N_IN else 0
        write32(mm, REG_IN + (i // 2) * 4, (hi << 16) | lo)


def read_outputs(mm):
    """Lee 43 logits ap_fixed<16,6>."""
    logits = []
    for i in range(0, N_OUT, 2):
        word = read32(mm, REG_OUT + (i // 2) * 4)
        logits.append(from_fixed16(word & 0xFFFF))
        if i + 1 < N_OUT:
            logits.append(from_fixed16((word >> 16) & 0xFFFF))
    return np.array(logits[:N_OUT], dtype=np.float32)


def fpga_predict(mm, flat):
    """Escribe entradas, lanza ap_start, espera ap_done, lee salidas."""
    write_inputs(mm, flat)
    write32(mm, REG_CTRL, 0x1)           # ap_start
    t0 = time.time()
    for _ in range(10000):
        status = read32(mm, REG_CTRL)
        if status & 0x2:                  # ap_done
            break
    latency_ms = (time.time() - t0) * 1000
    logits  = read_outputs(mm)
    probs   = softmax(logits)
    cls     = int(np.argmax(probs))
    return cls, float(probs[cls]), probs, latency_ms


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# Detector de señales (HSV)
# ---------------------------------------------------------------------------

def detect_sign_rois(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    r1   = cv2.inRange(hsv, (0,   80, 80), (12,  255, 255))
    r2   = cv2.inRange(hsv, (168, 80, 80), (180, 255, 255))
    blue = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(r1, r2), blue)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern), cv2.MORPH_OPEN, kern)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fh, fw  = frame.shape[:2]
    boxes   = []
    for c in cnts:
        if cv2.contourArea(c) < 500:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (0.35 < w / max(h, 1) < 2.8):
            continue
        pad = int(max(w, h) * 0.15)
        boxes.append((max(0, x-pad), max(0, y-pad),
                      min(fw-x+pad, w+2*pad), min(fh-y+pad, h+2*pad)))
    return boxes


# ---------------------------------------------------------------------------
# Modos de operación
# ---------------------------------------------------------------------------

def run_image(mm, path, names):
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"No se pudo abrir: {path}")
        sys.exit(1)
    flat = preprocess(bgr)
    cls, conf, probs, lat = fpga_predict(mm, flat)
    top5 = np.argsort(probs)[::-1][:5]
    print(f"Imagen : {path}  (FPGA latencia: {lat:.1f} ms)")
    for rank, c in enumerate(top5):
        bar = "#" * int(probs[c] * 30)
        print(f"  {rank+1}. [{c:2d}] {names[c]:<38} {probs[c]*100:5.1f}%  {bar}")


def run_camera(mm, cam_idx, names):
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"No se pudo abrir /dev/video{cam_idx}")
        sys.exit(1)
    print(f"Cámara /dev/video{cam_idx}  Ctrl+C para salir.\n")
    frame_idx = 0
    t_prev    = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t_now  = time.time()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now
            boxes  = detect_sign_rois(frame)
            if not boxes:
                print(f"[{frame_idx:05d}] fps={fps:.1f}  sin detecciones")
            else:
                for i, (x, y, w, h) in enumerate(boxes):
                    crop = frame[y:y+h, x:x+w]
                    if crop.size == 0:
                        continue
                    flat = preprocess(crop)
                    cls, conf, _, lat = fpga_predict(mm, flat)
                    label = names[cls] if conf >= CONF_THR else f"? ({conf:.2f})"
                    print(f"[{frame_idx:05d}] fps={fps:.1f}  ROI#{i}  "
                          f"[{cls:2d}] {label:<38} conf={conf:.2f}  lat={lat:.1f}ms")
            frame_idx += 1
    except KeyboardInterrupt:
        print("\nDetenido.")
    finally:
        cap.release()


def run_bench(mm, n=100):
    """Mide latencia promedio con entrada aleatoria."""
    flat = np.random.randn(N_IN).astype(np.float32) * 0.5
    lats = []
    for _ in range(n):
        _, _, _, lat = fpga_predict(mm, flat)
        lats.append(lat)
    print(f"Latencia FPGA ({n} runs): avg={np.mean(lats):.2f}ms  "
          f"min={np.min(lats):.2f}ms  max={np.max(lats):.2f}ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uio",   default=UIO_DEV,
                    help=f"Dispositivo UIO (default: {UIO_DEV})")
    ap.add_argument("--names", default="/home/root/fpga/gtsrb_class_names.json")
    ap.add_argument("--image", default=None)
    ap.add_argument("--cam",   type=int, default=0)
    ap.add_argument("--bench", action="store_true",
                    help="Medir latencia FPGA con entradas aleatorias")
    args = ap.parse_args()

    with open(args.names) as f:
        raw   = json.load(f)
    names = [raw[str(i)] for i in range(43)]

    print(f"Abriendo {args.uio}  (MAP_SIZE=0x{MAP_SIZE:X})")
    print("AVISO: verificar REG_IN/REG_OUT/REG_CTRL en myproject_axi.h antes de correr")
    with open(args.uio, 'r+b', buffering=0) as f:
        mm = mmap.mmap(f.fileno(), MAP_SIZE)
        if args.bench:
            run_bench(mm)
        elif args.image:
            run_image(mm, args.image, names)
        else:
            run_camera(mm, args.cam, names)
        mm.close()


if __name__ == "__main__":
    main()
