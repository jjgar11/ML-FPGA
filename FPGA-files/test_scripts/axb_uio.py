import mmap
import struct
import time

UIO_DEV  = "/dev/uio4"
MAP_SIZE = 0x10000

REG_CTRL = 0x00
REG_A    = 0x10
REG_X    = 0x18
REG_B    = 0x20
REG_Y    = 0x28

def read32(mm, offset):
    mm.seek(offset)
    return struct.unpack('<I', mm.read(4))[0]

def write32(mm, offset, value):
    mm.seek(offset)
    mm.write(struct.pack('<I', value))

with open(UIO_DEV, 'r+b', buffering=0) as f:
    mm = mmap.mmap(f.fileno(), MAP_SIZE)

    print(f"ap_ctrl = 0x{read32(mm, REG_CTRL):08x}")

    for a, x, b in [(3, 4, 5), (10, 2, 7), (0, 99, 1)]:
        expected = a * x + b
        write32(mm, REG_A, a)
        write32(mm, REG_X, x)
        write32(mm, REG_B, b)
        time.sleep(0.001)
        y = read32(mm, REG_Y)
        status = "PASS" if y == expected else "FAIL"
        print(f"a={a:3d} x={x:3d} b={b:3d} → y={y:5d}  (esperado {expected:5d})  {status}")

    mm.close()
