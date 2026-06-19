import mmap
import struct
import time

UIO = "/dev/uio4"
MAP_SIZE = 0x10000

GPIO_DATA = 0x00
GPIO_TRI  = 0x04

def write32(mm, offset, value):
    mm.seek(offset)
    mm.write(struct.pack("<I", value & 0xFFFFFFFF))

def read32(mm, offset):
    mm.seek(offset)
    return struct.unpack("<I", mm.read(4))[0]

with open(UIO, "r+b", buffering=0) as f:
    mm = mmap.mmap(f.fileno(), MAP_SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)

    print("GPIO_TRI before:", hex(read32(mm, GPIO_TRI)))
    print("GPIO_DATA before:", hex(read32(mm, GPIO_DATA)))

    write32(mm, GPIO_TRI, 0x0)

    for value in [0x0, 0x1, 0x2, 0x4, 0x8, 0xF]:
        write32(mm, GPIO_DATA, value)
        print("wrote", hex(value), "read", hex(read32(mm, GPIO_DATA)))
        time.sleep(0.5)

    mm.close()
