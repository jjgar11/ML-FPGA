#!/usr/bin/env python3
"""AXI DMA loopback (polling mode). Direcciones fijas del proyecto de prueba
desde cero (Address Editor): DMA control regs en 0xA0000000, DDR en 0x0-0x7FFFFFFF.
"""
import ctypes, glob, mmap, os, struct, time

MM2S_CR, MM2S_SR, MM2S_SA, MM2S_SA_H, MM2S_LEN = 0x00, 0x04, 0x18, 0x1C, 0x28
S2MM_CR, S2MM_SR, S2MM_DA, S2MM_DA_H, S2MM_LEN = 0x30, 0x34, 0x48, 0x4C, 0x58
CR_RS, CR_RESET = 1, 1 << 2
SR_ERR, SR_IOC = 0b111 << 4, 1 << 12

N = 64
DMA_BASE = 0xA0000000

libc = ctypes.CDLL(None)
libc.mlockall(3)  # MCL_CURRENT | MCL_FUTURE


def virt_to_phys(vaddr):
    page = os.sysconf('SC_PAGE_SIZE')
    with open('/proc/self/pagemap', 'rb') as f:
        f.seek((vaddr // page) * 8)
        entry = struct.unpack('Q', f.read(8))[0]
    return (entry & ((1 << 55) - 1)) * page + (vaddr % page)


def flush_cpu_cache():
    # el DMA lee/escribe DRAM física sin snoop de caché de CPU — hay que forzar
    # el drenaje de la caché antes de disparar la transferencia, si no lee basura.
    buf = bytearray(2 * 1024 * 1024)
    for i in range(0, len(buf), 64):
        buf[i] ^= 1
    del buf


def read_phys_uncached(phys_addr, n_bytes):
    page = os.sysconf('SC_PAGE_SIZE')
    aligned = phys_addr & ~(page - 1)
    delta = phys_addr - aligned
    fd = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
    try:
        m = mmap.mmap(fd, delta + n_bytes, mmap.MAP_SHARED,
                      mmap.PROT_READ | mmap.PROT_WRITE, offset=aligned)
        data = bytes(m[delta:delta + n_bytes])
        m.close()
    finally:
        os.close(fd)
    return data


print("[*] Buscando UIO del DMA...")
uio = None
for d in sorted(glob.glob('/sys/class/uio/uio*')):
    addr = int(open(f'{d}/maps/map0/addr').read(), 16)
    print(f"    {d} @ 0x{addr:08X}")
    if addr == DMA_BASE:
        uio = '/dev/' + os.path.basename(d)
if uio is None:
    raise RuntimeError(f"No UIO device @ 0x{DMA_BASE:08X}")
print(f"[*] DMA en {uio}")

fd = open(uio, 'r+b', buffering=0)
regs = mmap.mmap(fd.fileno(), 0x10000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)


def wr(off, val):
    regs.seek(off)
    regs.write(struct.pack('<I', val & 0xFFFFFFFF))


def rd(off):
    regs.seek(off)
    return struct.unpack('<I', regs.read(4))[0]


# --- buffers ---
in_mm  = mmap.mmap(-1, 4096)
out_mm = mmap.mmap(-1, 4096)
pattern = bytes(range(N))
in_mm.write(pattern + b'\x00' * (4096 - N))
out_mm.write(b'\xAA' * 4096)

in_c  = (ctypes.c_char * 4096).from_buffer(in_mm)
out_c = (ctypes.c_char * 4096).from_buffer(out_mm)
in_phys  = virt_to_phys(ctypes.addressof(in_c))
out_phys = virt_to_phys(ctypes.addressof(out_c))
print(f"[*] in_phys  = 0x{in_phys:X}")
print(f"[*] out_phys = 0x{out_phys:X}")

print("[*] Flush de caché (para que el patrón llegue a DRAM física)...")
flush_cpu_cache()

# --- reset ---
print("[*] Reset MM2S/S2MM...")
wr(MM2S_CR, CR_RESET)
wr(S2MM_CR, CR_RESET)
while rd(MM2S_CR) & CR_RESET:
    pass
while rd(S2MM_CR) & CR_RESET:
    pass
wr(MM2S_CR, CR_RS)
wr(S2MM_CR, CR_RS)
print(f"[*] tras reset: MM2S_SR=0x{rd(MM2S_SR):08X}  S2MM_SR=0x{rd(S2MM_SR):08X}")

# --- armar S2MM (destino) antes de disparar MM2S ---
print("[*] Armando S2MM...")
wr(S2MM_DA, out_phys & 0xFFFFFFFF)
wr(S2MM_DA_H, (out_phys >> 32) & 0xFFFFFFFF)
wr(S2MM_LEN, N)

# --- disparar MM2S (origen) ---
print("[*] Disparando MM2S...")
wr(MM2S_SA, in_phys & 0xFFFFFFFF)
wr(MM2S_SA_H, (in_phys >> 32) & 0xFFFFFFFF)
wr(MM2S_LEN, N)

# --- polling ---
print("[*] Polling...")
t0 = time.time()
while True:
    sr_m, sr_s = rd(MM2S_SR), rd(S2MM_SR)
    print(f"    MM2S_SR=0x{sr_m:08X}  S2MM_SR=0x{sr_s:08X}")
    if (sr_m | sr_s) & SR_ERR:
        print("[!] ERROR bit set")
        break
    if (sr_m & SR_IOC) and (sr_s & SR_IOC):
        print("[+] IOC en ambos canales")
        break
    if time.time() - t0 > 2:
        print("[!] TIMEOUT")
        break
    time.sleep(0.05)

result = read_phys_uncached(out_phys, N)  # bypassea la caché del CPU, lee DRAM real
print(f"[*] esperado = {pattern.hex()}")
print(f"[*] recibido = {result.hex()}")
print("[+] MATCH" if result == pattern else "[!] MISMATCH")
