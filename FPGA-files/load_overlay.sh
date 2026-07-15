#!/bin/bash
set -e

# Load a bitstream + device tree overlay on Ultra96 via configfs.
# Must run as root.
#
# Usage:  ./load_overlay.sh <folder>
# Example: ./load_overlay.sh axb

FOLDER="${1:?Usage: $0 <folder>}"
LABEL="$(basename "$FOLDER")"

BITBIN="$(ls "$FOLDER"/*.bit.bin 2>/dev/null | head -n1)"
DTBO="$(ls "$FOLDER"/*.dtbo 2>/dev/null | head -n1)"

[ -n "$BITBIN" ] || { echo "ERROR: no .bit.bin found in $FOLDER"; exit 1; }
[ -n "$DTBO"   ] || { echo "ERROR: no .dtbo found in $FOLDER"; exit 1; }

CONFIGFS="/sys/kernel/config"
OVERLAYS="$CONFIGFS/device-tree/overlays"
SLOT="$OVERLAYS/$LABEL"

[ "$(id -u)" -eq 0 ] || { echo "ERROR: run as root"; exit 1; }
[ -d "$FOLDER" ] || { echo "ERROR: folder not found: $FOLDER"; exit 1; }

FSTYPE=$(stat -f -c '%T' "$CONFIGFS" 2>/dev/null || true)
[ "$FSTYPE" = "configfs" ] || {
    echo "ERROR: $CONFIGFS is not configfs. Current type: $FSTYPE"
    exit 1
}

[ -d "$OVERLAYS" ] || {
    echo "ERROR: overlay configfs path not found: $OVERLAYS"
    exit 1
}

echo "[1/5] Copying firmware files..."
cp "$BITBIN" /lib/firmware/
cp "$DTBO" /lib/firmware/

echo "[2/5] Loading UIO driver..."
CURRENT_ID=$(cat /sys/module/uio_pdrv_genirq/parameters/of_id 2>/dev/null || true)
if [ "$CURRENT_ID" != "generic-uio" ]; then
    rmmod uio_pdrv_genirq 2>/dev/null || true
    modprobe uio
    modprobe uio_pdrv_genirq of_id=generic-uio
fi

echo "[3/5] Removing existing overlays..."
for existing in "$OVERLAYS"/*/; do
    [ -d "$existing" ] || continue
    echo "Removing $(basename "$existing")..."
    rmdir "$existing" || {
        echo "ERROR: could not remove $existing"
        echo "A previous overlay may still be in use. Reboot the board and try again."
        exit 1
    }
done

echo "[4/5] Applying overlay: $LABEL"
mkdir "$SLOT"
if ! echo -n "$(basename "$DTBO")" > "$SLOT/path"; then
    echo "ERROR: failed to write overlay path"
    rmdir "$SLOT" 2>/dev/null || true
    exit 1
fi

echo "[4b/5] Programming FPGA bitstream..."
fpgautil -b "/lib/firmware/$(basename "$BITBIN")"

echo "[5/5] Verifying..."
STATUS=$(cat "$SLOT/status" 2>/dev/null || true)
if [ "$STATUS" != "applied" ]; then
    echo "ERROR: overlay status='$STATUS'"
    dmesg | tail -n 40
    exit 1
fi

echo "OK [$LABEL] applied"

echo
echo "Platform devices:"
find /sys/bus/platform/devices -maxdepth 1 -type l | grep -E 'a0[0-9a-f]{6}' || true

echo
echo "UIO devices:"
for uio in /sys/class/uio/uio*/; do
    [ -d "$uio" ] || continue
    printf "  /dev/%-6s  %-30s  addr=%s  size=%s\n" \
        "$(basename "$uio")" \
        "$(cat "$uio/name" 2>/dev/null)" \
        "$(cat "$uio/maps/map0/addr" 2>/dev/null)" \
        "$(cat "$uio/maps/map0/size" 2>/dev/null)"
done

echo
echo "Recent dmesg:"
dmesg | tail -n 10