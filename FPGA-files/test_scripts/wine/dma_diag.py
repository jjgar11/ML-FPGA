#!/usr/bin/env python3
"""
Diagnostics for the AXI DMA (7.1) and the AXI S2MM Snooper via UIO.

Modes:
  python3 dma_diag.py          — standard DMA diagnostics
  python3 dma_diag.py --snoop  — reads snooper registers after an inference
  python3 dma_diag.py --clear  — clears the snooper counters
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
# DMACR = 0x00010002  (IRQ_Threshold=1 in bits[23:16]=0x01, RS=0, bit1=1)
# DMASR = 0x00000001  (Halted=1, because RS=0)
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
        print(f"ERROR: No UIO found at 0x{DMA_BASE:08x}")
        print("Did you load the overlay?  ./load_overlay.sh wine_axi_dma  (or wine_axi_dma_snooper)")
        sys.exit(1)
    print(f"UIO: {uio_dev}  ({uio_name})  @ 0x{DMA_BASE:08x}")

    f = open(uio_dev, 'r+b', buffering=0)
    m = mmap.mmap(f.fileno(), 65536)

    rd = lambda off: struct.unpack('<I', m[off:off+4])[0]
    wr = lambda off, v: m.__setitem__(slice(off, off+4), struct.pack('<I', v))

    print("\n[1] Current registers:")
    for name, off in REGS.items():
        v = rd(off)
        dec = decode_dmacr(v) if 'DMACR' in name else decode_dmasr(v)
        print(f"    {name:15s} = 0x{v:08x}  {dec}")

    print("\n[2] Write->read test on MM2S_DMACR:")
    # Write RESET bit (0x4) — hardware auto-clears it and returns to default
    wr(0x00, 0x00000004)
    time.sleep(0.05)
    rb = rd(0x00)
    print(f"    Wrote    0x00000004 (RESET bit)")
    print(f"    Read     0x{rb:08x}")

    if rb == 0x00000000:
        print("    >> FAILURE: AXI-Lite is not responding (all zeros). Broken bitstream or FPGA not programmed.")
    elif rb == DMA_RESET_DMACR:
        print(f"    >> OK: DMA responded to reset with the correct default value (0x{DMA_RESET_DMACR:08x})")
        print(f"       IRQ_Threshold=1 + bit1=1 is the standard DMACR after reset. Not an error.")
    elif rb == 0x00000004:
        print("    >> Partial OK: the reset bit hasn't auto-cleared yet (wait longer)")
    else:
        print(f"    >> Unexpected value 0x{rb:08x} — could be a transitional state")

    print("\n[3] Registers after reset:")
    for name, off in REGS.items():
        v = rd(off)
        dec = decode_dmacr(v) if 'DMACR' in name else decode_dmasr(v)
        status = ""
        if name == 'MM2S_DMASR' or name == 'S2MM_DMASR':
            if v & 0x70:
                status = "  !! BUS ERRORS"
        print(f"    {name:15s} = 0x{v:08x}  {dec}{status}")

    print("\n[4] Start DMA (RS=1) and check Idle:")
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
        print("    >> OK: both channels Idle. DMA ready for transfer.")
    elif mm2s_sr == 0 or s2mm_sr == 0:
        print("    >> NORMAL: SR=0 (Halted=0, Idle=0) with RS=1 = DMA running with no active transfer.")
        print("       With ap_ctrl_none the IP waits for TVALID on in_r. Not an error.")
    else:
        print("    >> Channels not Idle — could be in an error state or an active transfer")

    m.close()
    f.close()


# =============================================================================
# AXI S2MM Snooper — read capture registers via UIO
# =============================================================================
# Snooper register map (offset from the UIO base):
#   0x00  AWADDR    — AXI-M address used by the DataMover (should be out_phys)
#   0x04  AWLEN     — programmed burst length (beats = AWLEN+1)
#   0x08  WDATA0    — data on beat 0 (float0 as uint32)
#   0x0C  WDATA1    — data on beat 1 (float1 as uint32)
#   0x10  WDATA2    — data on beat 2 (float2 as uint32)
#   0x14  WLAST_AT  — beat (0-based) on which WLAST was asserted
#   0x18  AW_CNT    — how many AW transactions the DataMover issued
#   0x1C  W_CNT     — how many W beats were transferred in total
#   0x20  B_CNT     — how many B responses (BRESP) arrived
#   0x24  BRESP     — last BRESP (0=OKAY, 2=SLVERR, 3=DECERR)
#   0x28  CLEAR     — write 1 to reset all counters

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
    """Finds a UIO device whose name contains name_fragment."""
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
        print("ERROR: no UIO found for the snooper.")
        print("  Did you load the bitstream with the snooper?")
        print("  Check:  ls /sys/class/uio/*/name")
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

    # Decoding
    awaddr   = vals['AWADDR']
    awlen    = vals['AWLEN']
    beats    = awlen + 1
    wdata    = [vals['WDATA0'], vals['WDATA1'], vals['WDATA2']]
    wlast_at = vals['WLAST_AT']
    aw_cnt   = vals['AW_CNT']
    w_cnt    = vals['W_CNT']
    b_cnt    = vals['B_CNT']
    bresp    = vals['BRESP']

    print(f"  AWADDR    = 0x{awaddr:08X}   ← address used by the DataMover")
    print(f"  AWLEN     = {awlen}  → burst of {beats} beat(s) programmed")
    print()
    for i, d in enumerate(wdata):
        f32 = bits_to_float(d)
        mark = " ← WLAST here" if i == wlast_at else ""
        print(f"  WDATA{i}    = 0x{d:08X}  ({f32:+.4f}){mark}")
    print()
    print(f"  WLAST_AT  = beat {wlast_at}  (the DataMover ended the burst on beat {wlast_at})")
    print(f"  AW_CNT    = {aw_cnt}  AW transactions issued")
    print(f"  W_CNT     = {w_cnt}  W beats transferred")
    print(f"  B_CNT     = {b_cnt}  B responses received")
    print(f"  BRESP     = {bresp} ({BRESP_NAMES.get(bresp, '?')})")

    print()
    # Diagnostics
    if aw_cnt == 0:
        print("!! AW_CNT=0: the DataMover did NOT issue any AW transaction. Bug before the AXI-M bus.")
    else:
        print(f"[AW] DataMover issued {aw_cnt} AW transaction(s).")
        if aw_cnt == 1:
            print(f"     Programmed a burst of {beats} beat(s) (AWLEN={awlen}).")
            if beats == 1:
                print("     !! Only 1 beat per burst → AWLEN=0. The DataMover is using single-beat despite LEN=12.")
            elif beats == 3:
                print("     OK: burst of 3 beats programmed correctly.")
        else:
            print(f"     {aw_cnt} transactions for 12 bytes → DataMover splits into small bursts.")

    if w_cnt == 0:
        print("!! W_CNT=0: the W channel did not transfer any beat.")
    else:
        print(f"[W]  {w_cnt} W beat(s) transferred. WLAST on beat {wlast_at}.")
        if wlast_at == 0 and beats > 1:
            print("     !! WLAST on beat 0 but AWLEN>0 → the W channel closed the burst prematurely.")

    if bresp != 0:
        print(f"!! BRESP={bresp} ({BRESP_NAMES.get(bresp,'?')}): the slave reported an error on the write.")
    else:
        print(f"[B]  BRESP=OKAY. The slave accepted the write without error.")

    m.close()
    f.close()


def snoop_clear():
    dev, name, addr, size = find_uio_by_name('snooper')
    if dev is None:
        print("ERROR: no UIO found for the snooper.")
        sys.exit(1)
    f = open(dev, 'r+b', buffering=0)
    m = mmap.mmap(f.fileno(), max(size, 4096))
    m[0x28:0x2C] = struct.pack('<I', 1)
    m.close()
    f.close()
    print("Snooper: counters cleared.")


if __name__ == '__main__':
    if '--snoop' in sys.argv:
        snoop_read()
    elif '--clear' in sys.argv:
        snoop_clear()
    else:
        main()
