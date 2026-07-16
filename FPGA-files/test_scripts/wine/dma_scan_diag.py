#!/usr/bin/env python3
"""
Page-scan diagnostic: fill the entire 4KB out_phys page with a recognizable
pattern, run one inference, then read back ALL 4096 bytes via /dev/mem and
report every word that changed. This reveals WHERE in physical memory the DMA
actually wrote beats 1 and 2.

Usage: python3 dma_scan_diag.py [sample_index]  (default: 0)
"""

import ctypes, glob, json, mmap, os, struct, sys, time

MM2S_CR=0x00; MM2S_SR=0x04; MM2S_SA=0x18; MM2S_SA_H=0x1C; MM2S_LEN=0x28
S2MM_CR=0x30; S2MM_SR=0x34; S2MM_DA=0x48; S2MM_DA_H=0x4C; S2MM_LEN=0x58
CR_RS=0x1; CR_RESET=0x4; SR_IDLE=0x2; SR_HALTED=0x1
SR_ERR=0x70; SR_IOC_IRQ=0x1000
N_IN=13; N_OUT=3; IN_BYTES=N_IN*4; OUT_BYTES=N_OUT*4
SENTINEL=[99.0,98.0,97.0]
_FLUSH=2*1024*1024
libc=ctypes.CDLL(None)

def mlockall(): libc.mlockall(3)

def flush():
    b=bytearray(_FLUSH)
    for i in range(0,_FLUSH,64): b[i]^=1
    del b

def virt2phys(va):
    page=os.sysconf('SC_PAGE_SIZE')
    with open('/proc/self/pagemap','rb') as f:
        f.seek((va//page)*8)
        e=struct.unpack('Q',f.read(8))[0]
    if not(e>>63): raise RuntimeError('page not resident')
    return (e&((1<<55)-1))*page+(va%page)

def find_uio(addr=0xa0000000):
    for d in sorted(glob.glob('/sys/class/uio/uio*')):
        try:
            if int(open(f'{d}/maps/map0/addr').read(),16)==addr:
                return '/dev/'+os.path.basename(d)
        except OSError: pass
    raise RuntimeError('UIO not found — load overlay first')

def devmem_read(phys, size):
    page=os.sysconf('SC_PAGE_SIZE')
    pg_off=phys&(page-1)
    with open('/dev/mem','rb') as dm:
        mm=mmap.mmap(dm.fileno(),size+pg_off,mmap.MAP_PRIVATE,mmap.PROT_READ,
                     offset=phys&~(page-1))
        data=bytes(mm[pg_off:pg_off+size])
        mm.close()
    return data

def main():
    idx=int(sys.argv[1]) if len(sys.argv)>1 else 0
    with open('/home/root/fpga/test_scripts/wine/wine_test_data.json') as f:
        data=json.load(f)
    features=[float(x) for x in data['X'][idx]]
    label=data['y'][idx]

    mlockall()
    uio=find_uio()
    print(f'UIO: {uio}')
    fd=open(uio,'r+b',buffering=0)
    regs=mmap.mmap(fd.fileno(),0x10000,mmap.MAP_SHARED,mmap.PROT_READ|mmap.PROT_WRITE)

    in_mm=mmap.mmap(-1,4096); out_mm=mmap.mmap(-1,4096)
    in_mm.write(b'\x00'*4096); in_mm.seek(0)
    out_mm.write(b'\x00'*4096); out_mm.seek(0)
    in_c=(ctypes.c_char*4096).from_buffer(in_mm)
    out_c=(ctypes.c_char*4096).from_buffer(out_mm)
    in_phys=virt2phys(ctypes.addressof(in_c))
    out_phys=virt2phys(ctypes.addressof(out_c))
    print(f'in_phys =0x{in_phys:010X}')
    print(f'out_phys=0x{out_phys:010X}')

    def wr(off,v):
        regs.seek(off); regs.write(struct.pack('<I',v&0xFFFFFFFF))
    def rd(off):
        regs.seek(off); return struct.unpack('<I',regs.read(4))[0]
    def wait(sr_off,label,timeout=2.0):
        t0=time.time()
        while True:
            sr=rd(sr_off)
            if sr&SR_ERR: raise RuntimeError(f'{label} ERR SR=0x{sr:08X}')
            if sr&SR_IOC_IRQ: return sr
            if time.time()-t0>timeout: raise TimeoutError(f'{label} timeout SR=0x{sr:08X}')

    # Reset DMA
    wr(MM2S_CR,CR_RESET); wr(S2MM_CR,CR_RESET); time.sleep(0.05)
    wr(MM2S_CR,CR_RS);    wr(S2MM_CR,CR_RS);    time.sleep(0.01)

    # --- FILL ENTIRE 4KB PAGE WITH RECOGNIZABLE PATTERN ---
    FILL_BASE=0xBEEF0000
    for i in range(4096//4):
        struct.pack_into('<I', out_c, i*4, FILL_BASE|i)
    # Overwrite first 12 bytes with normal sentinels
    struct.pack_into('<3f', out_c, 0, *SENTINEL)

    # Pack input
    struct.pack_into(f'<{N_IN}f', in_c, 0, *features)
    flush()  # evict to DDR

    # Snapshot the page in DDR BEFORE transfer
    pre=devmem_read(out_phys,4096)
    print('\nDDR page BEFORE transfer (first 64 bytes / 16 words):')
    for i in range(16):
        v=struct.unpack_from('<I',pre,i*4)[0]
        print(f'  +0x{i*4:04X}: 0x{v:08X}')

    # --- START TRANSFER ---
    wr(MM2S_SR,SR_IOC_IRQ); wr(S2MM_SR,SR_IOC_IRQ)
    wr(MM2S_CR,CR_RS);      wr(S2MM_CR,CR_RS)
    wr(S2MM_DA,  out_phys&0xFFFFFFFF); wr(S2MM_DA_H,(out_phys>>32)&0xFFFFFFFF)
    wr(S2MM_LEN, OUT_BYTES)
    wr(MM2S_SA,  in_phys&0xFFFFFFFF);  wr(MM2S_SA_H,(in_phys>>32)&0xFFFFFFFF)
    wr(MM2S_LEN, IN_BYTES)

    sr_mm2s=wait(MM2S_SR,'MM2S')
    sr_s2mm=wait(S2MM_SR,'S2MM')
    flush()

    # Snapshot the page in DDR AFTER transfer
    post=devmem_read(out_phys,4096)

    print(f'\nSR: MM2S=0x{sr_mm2s:08X}  S2MM=0x{sr_s2mm:08X}')
    print(f'Ground truth: class {label}')

    # --- FIND ALL CHANGED WORDS ---
    changed=[]
    for i in range(4096//4):
        b_pre=struct.unpack_from('<I',pre,i*4)[0]
        b_post=struct.unpack_from('<I',post,i*4)[0]
        if b_pre!=b_post:
            f_val=struct.unpack_from('<f',post,i*4)[0]
            changed.append((i*4,b_pre,b_post,f_val))

    print(f'\nChanged words ({len(changed)} total):')
    if not changed:
        print('  NONE — DMA wrote nothing to this page!')
    for off,pre_v,post_v,fval in changed:
        sent_idx=-1
        for si,sv in enumerate(SENTINEL):
            if abs(struct.unpack('<f',struct.pack('<I',pre_v))[0]-sv)<0.001:
                sent_idx=si
        marker=f'<-- SENTINEL[{sent_idx}]→DMA' if sent_idx>=0 else ''
        print(f'  +0x{off:04X} (byte {off:4d}): 0x{pre_v:08X} -> 0x{post_v:08X}  ({fval:+.6f}) {marker}')

    # Decode as 3 floats from offset 0
    print('\nFirst 12 bytes decoded as 3 floats:')
    for i in range(3):
        v=struct.unpack_from('<f',post,i*4)[0]
        print(f'  score[{i}] = {v:.6f}  ({"SENTINEL" if abs(v-SENTINEL[i])<0.001 else "written"})')

    regs.close(); fd.close(); in_mm.close(); out_mm.close()

if __name__=='__main__':
    main()
