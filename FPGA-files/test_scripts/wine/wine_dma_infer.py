#!/usr/bin/env python3
"""Wine quality MLP inference via AXI DMA on Ultra96-v2.
Load the overlay first:  ./load_overlay.sh wine_axi_dma
"""

import ctypes
import glob
import mmap
import os
import struct
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

CR_RS    = 0x00000001
CR_RESET = 0x00000004
SR_IDLE  = 0x00000002
SR_ERR   = 0x00000070  # Internal | Slave | Decode error

N_IN      = 13
N_OUT     = 3
IN_BYTES  = N_IN  * 4
OUT_BYTES = N_OUT * 4

# Cortex-A53: L1=32KB/core, L2=512KB shared. Writing 2MB (4×L2) forces all
# dirty lines — including the DMA input buffer — to be written back to DDR.
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

        # Anonymous mmaps — one page each = physically contiguous
        self._in_mm  = mmap.mmap(-1, 4096)
        self._out_mm = mmap.mmap(-1, 4096)
        self._in_mm.write(b'\x00' * 4096)
        self._in_mm.seek(0)
        self._out_mm.write(b'\x00' * 4096)
        self._out_mm.seek(0)

        # ctypes views — must be deleted before closing mmaps
        self._in_c  = (ctypes.c_char * 4096).from_buffer(self._in_mm)
        self._out_c = (ctypes.c_char * 4096).from_buffer(self._out_mm)

        self._in_phys  = _virt_to_phys(ctypes.addressof(self._in_c))
        self._out_phys = _virt_to_phys(ctypes.addressof(self._out_c))
        print(f"in_buf  phys=0x{self._in_phys:010X}")
        print(f"out_buf phys=0x{self._out_phys:010X}")

        self._reset()
        sr_mm2s = self._rd(MM2S_SR)
        sr_s2mm = self._rd(S2MM_SR)
        print(f"After reset: MM2S_SR=0x{sr_mm2s:08X}  S2MM_SR=0x{sr_s2mm:08X}", end="")
        if sr_mm2s == 0 or sr_s2mm == 0:
            print("  <-- WARNING: should be 0x00000002, register access may be broken")
        else:
            print()

    def _wr(self, off, val):
        self._regs.seek(off)
        self._regs.write(struct.pack('<I', val & 0xFFFFFFFF))

    def _rd(self, off):
        self._regs.seek(off)
        return struct.unpack('<I', self._regs.read(4))[0]

    def _reset(self):
        self._wr(MM2S_CR, CR_RESET)
        self._wr(S2MM_CR, CR_RESET)
        time.sleep(0.05)
        self._wr(MM2S_CR, CR_RS)
        self._wr(S2MM_CR, CR_RS)
        time.sleep(0.01)

    def _wait(self, sr_off, label, timeout=2.0):
        t0 = time.time()
        while True:
            sr = self._rd(sr_off)
            if sr & SR_ERR:
                raise RuntimeError(f"{label} error SR=0x{sr:08X}")
            if sr & SR_IDLE:
                return sr
            if time.time() - t0 > timeout:
                raise TimeoutError(f"{label} timeout SR=0x{sr:08X}")

    def infer(self, features, debug=False):
        assert len(features) == N_IN

        struct.pack_into(f'<{N_IN}f', self._in_c, 0, *[float(x) for x in features])
        # Write sentinel values to out_buf before flush.
        # After DMA: if TLAST arrived on beat 1, bytes 4-11 will still be 98.0/97.0.
        # If TLAST arrived on beat 3, all 12 bytes will be replaced by model output.
        struct.pack_into('<3f', self._out_c, 0, 99.0, 98.0, 97.0)
        # Evict all L1+L2 dirty lines (including in_buf + out_buf sentinels) to DDR.
        # Cortex-A53 L2=512KB; 2MB > 4×L2 guarantees full eviction.
        _flush_cpu_cache()

        if debug:
            check = struct.unpack_from(f'<{N_IN}f', self._in_c, 0)
            print(f"  in_buf packed: {check[:3]}...")

        # Arm S2MM (receiver) before kicking MM2S (sender)
        self._wr(S2MM_DA,   self._out_phys & 0xFFFFFFFF)
        self._wr(S2MM_DA_H, (self._out_phys >> 32) & 0xFFFFFFFF)
        self._wr(S2MM_LEN,  OUT_BYTES)

        if debug:
            print(f"  S2MM_SR after arm: 0x{self._rd(S2MM_SR):08X}")

        self._wr(MM2S_SA,   self._in_phys & 0xFFFFFFFF)
        self._wr(MM2S_SA_H, (self._in_phys >> 32) & 0xFFFFFFFF)
        self._wr(MM2S_LEN,  IN_BYTES)

        if debug:
            print(f"  MM2S_SR after kick: 0x{self._rd(MM2S_SR):08X}")

        sr_mm2s = self._wait(MM2S_SR, 'MM2S')
        sr_s2mm = self._wait(S2MM_SR, 'S2MM')

        if debug:
            print(f"  Done: MM2S_SR=0x{sr_mm2s:08X}  S2MM_SR=0x{sr_s2mm:08X}")
            print(f"  out_buf raw: {bytes(self._out_c[:OUT_BYTES]).hex()}")

        return list(struct.unpack_from('<3f', self._out_c, 0))

    def close(self):
        del self._in_c    # release buffer views before closing mmaps
        del self._out_c
        self._regs.close()
        self._fd.close()
        self._in_mm.close()
        self._out_mm.close()


def main():
    dma = WineDMAInfer()

    with open('/home/root/fpga/test_scripts/wine/wine_test_data.json') as f:
        data = json.load(f)
    sample = data['X'][0]
    print(f"{sample}")

    print("\nRunning inference...")
    scores = dma.infer(sample, debug=True)

    pred   = scores.index(max(scores))
    labels = {0: 'low', 1: 'medium', 2: 'high'}
    print(f"\nScores:    {[f'{s:.4f}' for s in scores]}")
    print(f"Predicted: class {pred} ({labels[pred]} quality)")

    dma.close()


if __name__ == '__main__':
    main()
