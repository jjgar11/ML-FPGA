"""
GTSRB FPGA inference via AXI-DMA + UIO.

ESTADO: PLANTILLA — requiere bitstream sintetizado.
        Completa las constantes de la sección CONFIG antes de usar.

Diferencia con wine_infer.py (AXI-Lite):
  El modelo GTSRB usa hls4ml io_stream, que expone puertos AXI-Stream.
  El bloque DMA hace de puente entre Linux y el acelerador:
    PS (mmap DMA buffer) → AXI-Stream → HLS4ML CNN → AXI-Stream → PS

Requisitos en la Ultra96:
  1. Bitstream generado por hls4ml VivadoAccelerator backend:
       python dev/hls4ml/convert_model.py --model gtsrb_gap --mode int8 \
              --backend VivadoAccelerator --build --synth
  2. Device Tree Overlay con:
       - xlnx,axi-dma-1.00.a compatible en /amba
       - hls4ml acelerador en /amba
  3. Módulos UIO cargados (igual que wine_infer.py):
       modprobe uio && modprobe uio_pdrv_genirq of_id=generic-uio

Flujo de datos:
  1. Escribir imagen normalizada (3×64×64 = 12288 float32) en buffer TX DMA
  2. Arrancar transferencia S2MM (MM2S → acelerador → S2MM)
  3. Leer 43 int32 logits del buffer RX DMA
  4. Argmax → clase predicha

Referencias:
  - AXI DMA PG021 (Xilinx)
  - hls4ml VivadoAccelerator docs: https://fastmachinelearning.org/hls4ml/
"""

import json
import mmap
import os
import struct
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# CONFIG — rellenar después de generar el bitstream
# ---------------------------------------------------------------------------

# Dirección base del AXI DMA (ver /proc/device-tree o leer el .xsa)
DMA_BASE   = 0x80000000   # placeholder
DMA_SIZE   = 0x10000

# Dirección base del acelerador HLS4ML (ap_ctrl register)
HLS_BASE   = 0xA0010000   # placeholder
HLS_SIZE   = 0x10000

# Buffers DMA contiguos (reservar con /dev/mem o módulo dma-proxy)
TX_BUF_PHYS = 0x1E000000  # placeholder — ajustar según CMA disponible
RX_BUF_PHYS = 0x1E200000  # placeholder

# Número de valores de entrada y salida
N_IN   = 3 * 64 * 64   # 12288 float32
N_OUT  = 43            # logits int32 (o ap_fixed según config hls4ml)

UIO_DMA = "/dev/uio0"   # puede variar — verificar con ls /sys/class/uio/uio*/name

# ---------------------------------------------------------------------------
# Normalización GTSRB (mismos valores que en el entrenamiento)
# ---------------------------------------------------------------------------

MEAN = np.array([0.3337, 0.3064, 0.3171], dtype=np.float32)
STD  = np.array([0.2672, 0.2564, 0.2629], dtype=np.float32)

CLASS_NAMES_PATH = "/home/root/gtsrb_class_names.json"


def preprocess(bgr_crop):
    """BGR crop (cualquier tamaño) → float32 array (3072,) normalizado."""
    import cv2
    rgb  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (64, 64))
    arr  = rgb.astype(np.float32) / 255.0
    arr  = (arr - MEAN) / STD          # (64, 64, 3)
    arr  = arr.transpose(2, 0, 1)      # (3, 64, 64) — channel-first
    return arr.flatten()               # (12288,)


# ---------------------------------------------------------------------------
# AXI DMA helpers (registro offsets según PG021)
# ---------------------------------------------------------------------------

MM2S_DMACR   = 0x00
MM2S_DMASR   = 0x04
MM2S_SA      = 0x18   # source address (32-bit físico)
MM2S_LENGTH  = 0x28

S2MM_DMACR   = 0x30
S2MM_DMASR   = 0x34
S2MM_DA      = 0x48   # dest address
S2MM_LENGTH  = 0x58


def dma_write32(mm, off, val):
    mm.seek(off)
    mm.write(struct.pack('<I', val & 0xFFFFFFFF))


def dma_read32(mm, off):
    mm.seek(off)
    return struct.unpack('<I', mm.read(4))[0]


def dma_wait_idle(mm, status_reg, timeout=2.0):
    """Espera hasta que el bit Idle (bit 1) se ponga a 1."""
    t0 = time.time()
    while True:
        sr = dma_read32(mm, status_reg)
        if sr & 0x2:    # Idle bit
            return True
        if time.time() - t0 > timeout:
            raise TimeoutError(f"DMA timeout  SR=0x{sr:08x}")
        time.sleep(0.0001)


def run_dma_transfer(dma_mm, tx_phys, rx_phys, tx_bytes, rx_bytes):
    """Lanza transferencia MM2S+S2MM y espera que ambas terminen."""
    # Arrancar S2MM primero para que esté listo cuando lleguen los datos
    dma_write32(dma_mm, S2MM_DMACR,  0x1)          # run
    dma_write32(dma_mm, S2MM_DA,     rx_phys)
    dma_write32(dma_mm, S2MM_LENGTH, rx_bytes)

    # Arrancar MM2S
    dma_write32(dma_mm, MM2S_DMACR,  0x1)
    dma_write32(dma_mm, MM2S_SA,     tx_phys)
    dma_write32(dma_mm, MM2S_LENGTH, tx_bytes)

    # Esperar ambas
    dma_wait_idle(dma_mm, MM2S_DMASR)
    dma_wait_idle(dma_mm, S2MM_DMASR)


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

def predict(dma_mm, tx_mem, rx_mem, input_flat):
    """
    input_flat: numpy float32 array (3072,)
    Devuelve (class_id, logits_array)
    """
    # Escribir input en buffer TX (como bytes)
    tx_mem.seek(0)
    tx_mem.write(input_flat.tobytes())

    tx_bytes = N_IN  * 4
    rx_bytes = N_OUT * 4

    run_dma_transfer(dma_mm, TX_BUF_PHYS, RX_BUF_PHYS, tx_bytes, rx_bytes)

    # Leer logits del buffer RX
    rx_mem.seek(0)
    raw = rx_mem.read(rx_bytes)
    logits = np.frombuffer(raw, dtype=np.float32)   # ajustar dtype si hls4ml usa int

    cls = int(np.argmax(logits))
    return cls, logits


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not os.path.exists(UIO_DMA):
        print(f"ERROR: {UIO_DMA} no encontrado.")
        print("Carga el overlay y los módulos UIO primero.")
        sys.exit(1)

    with open(CLASS_NAMES_PATH) as f:
        raw = json.load(f)
    names = [raw[str(i)] for i in range(43)]

    # Mapear registros DMA
    with open(UIO_DMA, 'r+b', buffering=0) as f:
        dma_mm = mmap.mmap(f.fileno(), DMA_SIZE)

    # Mapear buffers DMA (necesita /dev/mem o módulo dma-proxy)
    # Si usas /dev/mem ajusta los offsets según el mapa de memoria de tu bitstream
    mem_fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    tx_mem = mmap.mmap(mem_fd, N_IN  * 4, offset=TX_BUF_PHYS)
    rx_mem = mmap.mmap(mem_fd, N_OUT * 4, offset=RX_BUF_PHYS)

    # --- Demo: usar imagen de prueba desde archivo ---
    import cv2
    test_image = sys.argv[1] if len(sys.argv) > 1 else None
    if test_image is None:
        print("Uso: python gtsrb_infer.py imagen_señal.jpg")
        sys.exit(0)

    bgr = cv2.imread(test_image)
    if bgr is None:
        print(f"No se pudo abrir: {test_image}")
        sys.exit(1)

    x = preprocess(bgr)
    t0 = time.time()
    cls, logits = predict(dma_mm, tx_mem, rx_mem, x)
    dt = (time.time() - t0) * 1000

    softmax_probs = np.exp(logits - logits.max())
    softmax_probs /= softmax_probs.sum()

    print(f"Imagen : {test_image}")
    print(f"Clase  : [{cls:2d}] {names[cls]}")
    print(f"Conf   : {softmax_probs[cls]*100:.1f}%")
    print(f"Tiempo : {dt:.2f} ms")

    dma_mm.close()
    tx_mem.close()
    rx_mem.close()
    os.close(mem_fd)


if __name__ == "__main__":
    main()
