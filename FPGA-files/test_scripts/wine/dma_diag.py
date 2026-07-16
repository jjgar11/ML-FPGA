#!/usr/bin/env python3
"""Diagnóstico del AXI DMA (7.1) via UIO — para confirmar si el hardware responde."""
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
        print("¿Cargaste el overlay?  ./load_overlay.sh wine_axi_dma")
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


if __name__ == '__main__':
    main()
