#!/usr/bin/env python3
"""Wine quality MLP inference via AXI DMA on Ultra96-v2.
Load the overlay first:  ./load_overlay.sh wine_axi_dma

Usage:
  python3 wine_dma_infer.py [sample_index]   (default: 0)
  python3 wine_dma_infer.py all              (run all samples, print accuracy)
"""

import ctypes
import glob
import mmap
import os
import struct
import sys
import time
import json

# ---- AXI DMA 7.1 register offsets (simple/direct mode) ----
MM2S_CR   = 0x00
MM2S_SR   = 0x04
MM2S_SA   = 0x18
MM2S_SA_H = 0x1C
MM2S_LEN  = 0x28
S2MM_CR   = 0x30
S2MM_SR   = 0x34
S2MM_DA   = 0x48
S2MM_DA_H = 0x4C
S2MM_LEN  = 0x58

CR_RS      = 0b1
CR_RESET   = 0b1 << 2
SR_IDLE    = 0b1 << 1       # bit  1 — channel idle after transfer
SR_HALTED  = 0b1            # bit  0 — channel halted (RS=0 or post-completion in some HW)
SR_ERR     = 0b111 << 4     # bits 4-6 — DMAIntErr | DMASlvErr | DMADecErr
SR_IOC_IRQ = 0b1 << 12      # bit 12 — transfer complete (set regardless of IrqEn)

N_IN      = 13
N_OUT     = 3
IN_BYTES  = N_IN  * 4
OUT_BYTES = N_OUT * 4

SENTINEL = [99.0, 98.0, 97.0]

_CACHE_FLUSH_SIZE = 2 * 1024 * 1024

libc = ctypes.CDLL(None)


def _mlockall():
    libc.mlockall(3)  # MCL_CURRENT | MCL_FUTURE


def _flush_cpu_cache():
    buf = bytearray(_CACHE_FLUSH_SIZE)
    for i in range(0, _CACHE_FLUSH_SIZE, 64):
        buf[i] ^= 1
    del buf


def _virt_to_phys(vaddr):
    page = os.sysconf('SC_PAGE_SIZE')
    with open('/proc/self/pagemap', 'rb') as f:
        f.seek((vaddr // page) * 8)
        entry = struct.unpack('Q', f.read(8))[0]
    if not (entry >> 63):
        raise RuntimeError(f"Page 0x{vaddr:X} not resident — mlockall failed?")
    pfn = entry & ((1 << 55) - 1)
    return pfn * page + (vaddr % page)


def _find_uio(target=0xa0000000):
    for d in sorted(glob.glob('/sys/class/uio/uio*')):
        try:
            addr = int(open(f'{d}/maps/map0/addr').read(), 16)
            if addr == target:
                return '/dev/' + os.path.basename(d)
        except OSError:
            pass
    raise RuntimeError(
        f"No UIO device at 0x{target:08X} — run: ./load_overlay.sh wine_axi_dma"
    )


class WineDMAInfer:
    def __init__(self):
        _mlockall()

        uio = _find_uio()
        print(f"DMA registers: {uio}")
        self._fd = open(uio, 'r+b', buffering=0)
        self._regs = mmap.mmap(self._fd.fileno(), 0x10000,
                               mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

        self._in_mm  = mmap.mmap(-1, 4096)
        self._out_mm = mmap.mmap(-1, 4096)
        self._in_mm.write(b'\x00' * 4096)
        self._in_mm.seek(0)
        self._out_mm.write(b'\x00' * 4096)
        self._out_mm.seek(0)

        self._in_c  = (ctypes.c_char * 4096).from_buffer(self._in_mm)
        self._out_c = (ctypes.c_char * 4096).from_buffer(self._out_mm)

        self._in_phys  = _virt_to_phys(ctypes.addressof(self._in_c))
        self._out_phys = _virt_to_phys(ctypes.addressof(self._out_c))
        print(f"in_buf  phys=0x{self._in_phys:010X}")
        print(f"out_buf phys=0x{self._out_phys:010X}")

        self._reset()
        sr_mm2s = self._rd(MM2S_SR)
        sr_s2mm = self._rd(S2MM_SR)
        halted = (sr_mm2s & SR_HALTED) and (sr_s2mm & SR_HALTED)
        status = "Halted (RS=1 needed)" if halted else "Running"
        print(f"After reset: MM2S_SR=0x{sr_mm2s:08X}  S2MM_SR=0x{sr_s2mm:08X}  [{status}]")

    def _wr(self, off, val):
        self._regs.seek(off)
        self._regs.write(struct.pack('<I', val & 0xFFFFFFFF))

    def _rd(self, off):
        self._regs.seek(off)
        return struct.unpack('<I', self._regs.read(4))[0]

    def _reset(self):
        self._wr(MM2S_CR, CR_RESET)
        self._wr(S2MM_CR, CR_RESET)

        while self._rd(MM2S_CR) & CR_RESET:
            pass
        while self._rd(S2MM_CR) & CR_RESET:
            pass

        self._wr(MM2S_CR, CR_RS)
        self._wr(S2MM_CR, CR_RS)

    def _wait(self, sr_off, label, timeout=2.0):
        t0 = time.time()
        while True:
            sr = self._rd(sr_off)
            if sr & SR_ERR:
                raise RuntimeError(f"{label} error SR=0x{sr:08X}")
            # Accept Idle (bit 1) or IOC_Irq (bit 12) as completion.
            # AXI DMA 7.1 in simple mode may signal Halted+IOC_Irq instead of Idle.
            if sr & SR_IOC_IRQ:
                return sr
            if time.time() - t0 > timeout:
                raise TimeoutError(f"{label} timeout SR=0x{sr:08X}")

    @staticmethod
    def _sr_str(sr):
        bits = []
        bits.append(f"Halted={'1' if sr & SR_HALTED  else '0'}")
        bits.append(f"Idle={'1'   if sr & SR_IDLE    else '0'}")
        bits.append(f"IntErr={'1' if sr & 0b1 << 4   else '0'}")
        bits.append(f"SlvErr={'1' if sr & 0b1 << 5   else '0'}")
        bits.append(f"DecErr={'1' if sr & 0b1 << 6   else '0'}")
        bits.append(f"IOC={'1'    if sr & SR_IOC_IRQ  else '0'}")
        return f"0x{sr:08X}  [{', '.join(bits)}]"

    def infer(self, features):
        assert len(features) == N_IN

        struct.pack_into(f'<{N_IN}f', self._in_c, 0, *[float(x) for x in features])
        struct.pack_into('<3f', self._out_c, 0, *SENTINEL)
        _flush_cpu_cache()

        # Clear IOC_Irq from previous transfer (write-1-to-clear) and re-assert RS.
        self._wr(MM2S_SR, SR_IOC_IRQ)
        self._wr(S2MM_SR, SR_IOC_IRQ)
        self._wr(MM2S_CR, CR_RS)
        self._wr(S2MM_CR, CR_RS)

        print("  [1] Antes de armar — canales en reposo:")
        print(f"        MM2S_SR = {self._sr_str(self._rd(MM2S_SR))}")
        print(f"        S2MM_SR = {self._sr_str(self._rd(S2MM_SR))}")

        da   = self._out_phys & 0xFFFFFFFF
        da_h = (self._out_phys >> 32) & 0xFFFFFFFF
        print(f"  [addr] out_phys=0x{self._out_phys:010X}  DA=0x{da:08X}  DA_H=0x{da_h:08X}")
        self._wr(S2MM_DA,   da)
        self._wr(S2MM_DA_H, da_h)
        print(f"  [addr] DA  leído de vuelta = 0x{self._rd(S2MM_DA):08X}")
        print(f"  [addr] DA_H leído de vuelta = 0x{self._rd(S2MM_DA_H):08X}")

        print("  [2] S2MM con dirección destino cargada, antes de escribir LEN:")
        print(f"        MM2S_SR = {self._sr_str(self._rd(MM2S_SR))}")
        print(f"        S2MM_SR = {self._sr_str(self._rd(S2MM_SR))}")

        self._wr(S2MM_LEN,  20)  # larger than OUT_BYTES=12 to verify register update

        print("  [3] S2MM armado (LEN escrito) — esperando datos del IP por AXI-Stream:")
        print(f"        MM2S_SR = {self._sr_str(self._rd(MM2S_SR))}")
        print(f"        S2MM_SR = {self._sr_str(self._rd(S2MM_SR))}")

        self._wr(MM2S_SA,   self._in_phys & 0xFFFFFFFF)
        self._wr(MM2S_SA_H, (self._in_phys >> 32) & 0xFFFFFFFF)

        print("  [4] MM2S con dirección origen cargada, antes de escribir LEN:")
        print(f"        MM2S_SR = {self._sr_str(self._rd(MM2S_SR))}")
        print(f"        S2MM_SR = {self._sr_str(self._rd(S2MM_SR))}")

        self._wr(MM2S_LEN,  IN_BYTES)  # dispara la transferencia

        print("  [5] MM2S disparado (LEN escrito) — transferencia en curso:")
        print(f"        MM2S_SR = {self._sr_str(self._rd(MM2S_SR))}")
        print(f"        S2MM_SR = {self._sr_str(self._rd(S2MM_SR))}")

        sr_mm2s = self._wait(MM2S_SR, 'MM2S')
        sr_s2mm = self._wait(S2MM_SR, 'S2MM')

        s2mm_len_actual = self._rd(S2MM_LEN)

        scores = list(struct.unpack_from('<3f', self._out_c, 0))
        raw    = bytes(self._out_c[:OUT_BYTES]).hex()

        # Dump primeros 64 bytes del buffer como floats (stride 4) buscando datos
        # en offsets inesperados — si el bus es 64-bit los beats podrían estar en 0,8,16
        print("  [dump] Primeros 64 bytes de out_buf como float32 (offset: valor):")
        for i in range(0, 64, 4):
            v = struct.unpack_from('<f', self._out_c, i)[0]
            marker = ""
            if i < OUT_BYTES and abs(v - SENTINEL[i // 4]) < 0.001:
                marker = " ← sentinel sin cambiar"
            elif i < OUT_BYTES and v != SENTINEL[i // 4]:
                marker = " ← ESCRITO POR DMA"
            elif i >= OUT_BYTES and abs(v) > 0.001:
                marker = " ← dato fuera del rango esperado"
            print(f"        [{i:2d}] {v:+.4f}{marker}")

        beats_written = sum(1 for i, s in enumerate(scores) if s != SENTINEL[i])

        return scores, beats_written, raw, sr_mm2s, sr_s2mm, s2mm_len_actual

    def close(self):
        del self._in_c
        del self._out_c
        self._regs.close()
        self._fd.close()
        self._in_mm.close()
        self._out_mm.close()


def run_sample(dma, features, label, idx):
    scores, beats_written, raw, sr_mm2s, sr_s2mm, s2mm_len_actual = dma.infer(features)
    pred   = scores.index(max(scores))
    labels = {0: 'low', 1: 'medium', 2: 'high'}
    ok     = pred == label

    print(f"\n[sample {idx}]  ground truth: class {label} ({labels[label]})")
    print(f"  Scores:  {[f'{s:.4f}' for s in scores]}")
    print(f"  raw hex: {raw}")
    print(f"  SR:      MM2S=0x{sr_mm2s:08X}  S2MM=0x{sr_s2mm:08X}")
    print(f"  S2MM_LENGTH actual: {s2mm_len_actual} bytes  (programado: {20})")

    if beats_written < N_OUT:
        remaining = [f'{SENTINEL[i]:.1f}' for i in range(beats_written, N_OUT)]
        print(f"  !! TLAST BUG: only {beats_written}/{N_OUT} beats written — "
              f"scores {list(range(beats_written, N_OUT))} are still sentinels {remaining}")
    else:
        print(f"  All {N_OUT} beats written OK")

    print(f"  Predicted: class {pred} ({labels[pred]})  {'CORRECT' if ok else 'WRONG'}")
    return ok


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    with open('/home/root/fpga/test_scripts/wine/wine_test_data.json') as f:
        data = json.load(f)
    X, y = data['X'], data['y']

    dma = WineDMAInfer()

    if arg == "all":
        correct = 0
        for i, (feat, label) in enumerate(zip(X, y)):
            ok = run_sample(dma, feat, label, i)
            if ok:
                correct += 1
        print(f"\nAccuracy: {correct}/{len(X)} = {correct/len(X)*100:.1f}%")
    else:
        idx = int(arg)
        run_sample(dma, X[idx], y[idx], idx)

    dma.close()


if __name__ == '__main__':
    main()
