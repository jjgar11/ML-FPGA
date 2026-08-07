#!/usr/bin/env python3
"""
Diagnóstico del AXI DMA (7.1) y del AXI S2MM Snooper via UIO.

Modos:
  python3 dma_diag.py          — diagnóstico estándar del DMA
  python3 dma_diag.py --snoop  — lee registros del snooper tras una inferencia
  python3 dma_diag.py --clear  — limpia los contadores del snooper
"""
import mmap, struct, time, os, sys, glob

DMA_BASE = 0xa0000000

REGS = {
    'MM2S_DMACR': 0x00,
    'MM2S_DMASR': 0x04,
    'S2MM_DMACR': 0x30,
    'S2MM_DMASR': 0x34,
}

# AXI DMA 7.1 simple-mode reset default:
# DMACR = 0x00010002  (IRQ_Threshold=1 en bits[23:16]=0x01, RS=0, bit1=1)
# DMASR = 0x00000001  (Halted=1, porque RS=0)
DMA_RESET_DMACR = 0x00010002
DMA_RESET_DMASR = 0x00000001


def find_uio(target=DMA_BASE):
    for d in sorted(glob.glob('/sys/class/uio/uio*')):
        try:
            addr = int(open(f'{d}/maps/map0/addr').read(), 16)
            if addr == target:
                name = open(f'{d}/name').read().strip()
                return f'/dev/{os.path.basename(d)}', name
        except OSError:
            pass
    return None, None


def decode_dmacr(v):
    rs      = v & 1
    reset   = (v >> 2) & 1
    irqen   = (v >> 12) & 1
    thresh  = (v >> 16) & 0xFF
    return f"RS={rs} RESET={reset} IOC_IrqEn={irqen} IRQ_Threshold={thresh}"


def decode_dmasr(v):
    halted  = v & 1
    idle    = (v >> 1) & 1
    sg_incl = (v >> 3) & 1
    dma_int = (v >> 4) & 1
    dma_slv = (v >> 5) & 1
    dma_dec = (v >> 6) & 1
    ioc_irq = (v >> 12) & 1
    return f"Halted={halted} Idle={idle} DMAIntErr={dma_int} DMASlvErr={dma_slv} DMADecErr={dma_dec} IOC_Irq={ioc_irq}"


def main():
    uio_dev, uio_name = find_uio()
    if uio_dev is None:
        print(f"ERROR: No se encontró UIO en 0x{DMA_BASE:08x}")
        print("¿Cargaste el overlay?  ./load_overlay.sh wine_axi_dma  (o wine_axi_dma_snooper)")
        sys.exit(1)
    print(f"UIO: {uio_dev}  ({uio_name})  @ 0x{DMA_BASE:08x}")

    f = open(uio_dev, 'r+b', buffering=0)
    m = mmap.mmap(f.fileno(), 65536)

    rd = lambda off: struct.unpack('<I', m[off:off+4])[0]
    wr = lambda off, v: m.__setitem__(slice(off, off+4), struct.pack('<I', v))

    print("\n[1] Registros actuales:")
    for name, off in REGS.items():
        v = rd(off)
        dec = decode_dmacr(v) if 'DMACR' in name else decode_dmasr(v)
        print(f"    {name:15s} = 0x{v:08x}  {dec}")

    print("\n[2] Test escritura->lectura en MM2S_DMACR:")
    # Escribir RESET bit (0x4) — hardware lo auto-limpia y vuelve al default
    wr(0x00, 0x00000004)
    time.sleep(0.05)
    rb = rd(0x00)
    print(f"    Escribí  0x00000004 (RESET bit)")
    print(f"    Leí      0x{rb:08x}")

    if rb == 0x00000000:
        print("    >> FALLO: AXI-Lite no responde (todos ceros). Bitstream roto o FPGA no programada.")
    elif rb == DMA_RESET_DMACR:
        print(f"    >> OK: DMA respondió al reset con el valor default correcto (0x{DMA_RESET_DMACR:08x})")
        print(f"       IRQ_Threshold=1 + bit1=1 es el DMACR estándar tras reset. No es un error.")
    elif rb == 0x00000004:
        print("    >> OK parcial: el reset bit no se auto-limpió todavía (espera más)")
    else:
        print(f"    >> Valor inesperado 0x{rb:08x} — puede ser un estado de transición")

    print("\n[3] Registros tras reset:")
    for name, off in REGS.items():
        v = rd(off)
        dec = decode_dmacr(v) if 'DMACR' in name else decode_dmasr(v)
        status = ""
        if name == 'MM2S_DMASR' or name == 'S2MM_DMASR':
            if v & 0x70:
                status = "  !! ERRORES DE BUS"
        print(f"    {name:15s} = 0x{v:08x}  {dec}{status}")

    print("\n[4] Arrancar DMA (RS=1) y verificar Idle:")
    wr(0x00, 0x00000001)  # MM2S RS=1
    wr(0x30, 0x00000001)  # S2MM RS=1
    time.sleep(0.02)
    mm2s_sr = rd(0x04)
    s2mm_sr = rd(0x34)
    mm2s_idle = (mm2s_sr >> 1) & 1
    s2mm_idle = (s2mm_sr >> 1) & 1
    print(f"    MM2S_DMASR = 0x{mm2s_sr:08x}  Idle={mm2s_idle}")
    print(f"    S2MM_DMASR = 0x{s2mm_sr:08x}  Idle={s2mm_idle}")
    if mm2s_idle and s2mm_idle:
        print("    >> OK: ambos canales Idle. DMA listo para transferencia.")
    elif mm2s_sr == 0 or s2mm_sr == 0:
        print("    >> NORMAL: SR=0 (Halted=0, Idle=0) con RS=1 = DMA corriendo sin transferencia activa.")
        print("       Con ap_ctrl_none la IP espera TVALID en in_r. No es un error.")
    else:
        print("    >> Canales no Idle — puede estar en estado de error o transferencia activa")

    m.close()
    f.close()


# =============================================================================
# AXI S2MM Snooper — leer registros de captura via UIO
# =============================================================================
# Mapa de registros del snooper (offset desde la base del UIO):
#   0x00  AWADDR    — dirección AXI-M que el DataMover usó (debe ser out_phys)
#   0x04  AWLEN     — burst length programado (beats = AWLEN+1)
#   0x08  WDATA0    — dato en el beat 0 (float0 como uint32)
#   0x0C  WDATA1    — dato en el beat 1 (float1 como uint32)
#   0x10  WDATA2    — dato en el beat 2 (float2 como uint32)
#   0x14  WLAST_AT  — en qué beat (0-based) se activó WLAST
#   0x18  AW_CNT    — cuántas transacciones AW emitió el DataMover
#   0x1C  W_CNT     — cuántos beats W se transfirieron en total
#   0x20  B_CNT     — cuántas respuestas B (BRESP) llegaron
#   0x24  BRESP     — último BRESP (0=OKAY, 2=SLVERR, 3=DECERR)
#   0x28  CLEAR     — escribir 1 para resetear todos los contadores

SNOOP_REGS = [
    (0x00, 'AWADDR'),
    (0x04, 'AWLEN'),
    (0x08, 'WDATA0'),
    (0x0C, 'WDATA1'),
    (0x10, 'WDATA2'),
    (0x14, 'WLAST_AT'),
    (0x18, 'AW_CNT'),
    (0x1C, 'W_CNT'),
    (0x20, 'B_CNT'),
    (0x24, 'BRESP'),
]

BRESP_NAMES = {0: 'OKAY', 1: 'EXOKAY', 2: 'SLVERR', 3: 'DECERR'}


def find_uio_by_name(name_fragment):
    """Busca un UIO device cuyo nombre contenga name_fragment."""
    for d in sorted(glob.glob('/sys/class/uio/uio*')):
        try:
            name = open(f'{d}/name').read().strip()
            if name_fragment.lower() in name.lower():
                addr = int(open(f'{d}/maps/map0/addr').read(), 16)
                size = int(open(f'{d}/maps/map0/size').read(), 16)
                return f'/dev/{os.path.basename(d)}', name, addr, size
        except OSError:
            pass
    return None, None, None, None


def bits_to_float(u):
    return struct.unpack('<f', struct.pack('<I', u))[0]


def snoop_read():
    dev, name, addr, size = find_uio_by_name('snooper')
    if dev is None:
        print("ERROR: no se encontró UIO para el snooper.")
        print("  ¿Cargaste el bitstream con el snooper?")
        print("  Busca en:  ls /sys/class/uio/*/name")
        sys.exit(1)

    print(f"Snooper UIO: {dev}  ({name})  @ 0x{addr:08X}  size=0x{size:X}")
    f = open(dev, 'r+b', buffering=0)
    m = mmap.mmap(f.fileno(), max(size, 4096))
    rd = lambda off: struct.unpack('<I', m[off:off+4])[0]
    wr = lambda off, v: m.__setitem__(slice(off, off+4), struct.pack('<I', v))

    print()
    vals = {}
    for off, name_r in SNOOP_REGS:
        v = rd(off)
        vals[name_r] = v

    # Decodificación
    awaddr   = vals['AWADDR']
    awlen    = vals['AWLEN']
    beats    = awlen + 1
    wdata    = [vals['WDATA0'], vals['WDATA1'], vals['WDATA2']]
    wlast_at = vals['WLAST_AT']
    aw_cnt   = vals['AW_CNT']
    w_cnt    = vals['W_CNT']
    b_cnt    = vals['B_CNT']
    bresp    = vals['BRESP']

    print(f"  AWADDR    = 0x{awaddr:08X}   ← dirección que el DataMover usó")
    print(f"  AWLEN     = {awlen}  → burst de {beats} beat(s) programado")
    print()
    for i, d in enumerate(wdata):
        f32 = bits_to_float(d)
        mark = " ← WLAST aquí" if i == wlast_at else ""
        print(f"  WDATA{i}    = 0x{d:08X}  ({f32:+.4f}){mark}")
    print()
    print(f"  WLAST_AT  = beat {wlast_at}  (el DataMover terminó el burst en el beat {wlast_at})")
    print(f"  AW_CNT    = {aw_cnt}  transacciones AW emitidas")
    print(f"  W_CNT     = {w_cnt}  beats W transferidos")
    print(f"  B_CNT     = {b_cnt}  respuestas B recibidas")
    print(f"  BRESP     = {bresp} ({BRESP_NAMES.get(bresp, '?')})")

    print()
    # Diagnóstico
    if aw_cnt == 0:
        print("!! AW_CNT=0: el DataMover NO emitió ninguna transacción AW. Bug antes del bus AXI-M.")
    else:
        print(f"[AW] DataMover emitió {aw_cnt} transacción(es) AW.")
        if aw_cnt == 1:
            print(f"     Programó burst de {beats} beat(s) (AWLEN={awlen}).")
            if beats == 1:
                print("     !! Solo 1 beat por burst → AWLEN=0. El DataMover usa single-beat pese a LEN=12.")
            elif beats == 3:
                print("     OK: burst de 3 beats programado correctamente.")
        else:
            print(f"     {aw_cnt} transacciones para 12 bytes → DataMover divide en bursts pequeños.")

    if w_cnt == 0:
        print("!! W_CNT=0: el canal W no transfirió ningún beat.")
    else:
        print(f"[W]  {w_cnt} beat(s) W transferidos. WLAST en beat {wlast_at}.")
        if wlast_at == 0 and beats > 1:
            print("     !! WLAST en beat 0 pero AWLEN>0 → el canal W cerró el burst prematuramente.")

    if bresp != 0:
        print(f"!! BRESP={bresp} ({BRESP_NAMES.get(bresp,'?')}): el slave reportó error en la escritura.")
    else:
        print(f"[B]  BRESP=OKAY. El slave aceptó la escritura sin error.")

    m.close()
    f.close()


def snoop_clear():
    dev, name, addr, size = find_uio_by_name('snooper')
    if dev is None:
        print("ERROR: no se encontró UIO para el snooper.")
        sys.exit(1)
    f = open(dev, 'r+b', buffering=0)
    m = mmap.mmap(f.fileno(), max(size, 4096))
    m[0x28:0x2C] = struct.pack('<I', 1)
    m.close()
    f.close()
    print("Snooper: contadores limpiados.")


if __name__ == '__main__':
    if '--snoop' in sys.argv:
        snoop_read()
    elif '--clear' in sys.argv:
        snoop_clear()
    else:
        main()
