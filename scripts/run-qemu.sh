#!/usr/bin/env bash
# ==============================================================================
# TensorOS — QEMU Run Script
# ==============================================================================
# Usage:
#   ./scripts/run-qemu.sh              Normal boot
#   ./scripts/run-qemu.sh --debug      Boot with GDB stub on :1234
#   ./scripts/run-qemu.sh --iso        Boot from ISO
#   ./scripts/run-qemu.sh --monitor    Boot with QEMU monitor on stdio
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
KERNEL="$BUILD_DIR/tensoros.bin"
ISO="$BUILD_DIR/tensoros.iso"
LOG="$BUILD_DIR/qemu.log"

QEMU=qemu-system-x86_64
MEM=4G
CPUS=4

# Check QEMU exists
if ! command -v $QEMU &>/dev/null; then
    echo "ERROR: $QEMU not found. Install QEMU first."
    echo "  Ubuntu/Debian: sudo apt install qemu-system-x86"
    echo "  macOS:         brew install qemu"
    echo "  Windows:       choco install qemu  (or download from qemu.org)"
    exit 1
fi

# Check kernel exists
if [ ! -f "$KERNEL" ] && [ "$1" != "--iso" ]; then
    echo "Kernel not found at $KERNEL. Building..."
    cd "$PROJECT_DIR" && make
fi

# KVM detection
KVM_FLAGS=""
if [ -e /dev/kvm ]; then
    echo "[*] KVM available — hardware acceleration enabled"
    KVM_FLAGS="-enable-kvm -cpu host"
else
    echo "[*] KVM not available — using software emulation"
    KVM_FLAGS="-cpu qemu64,+sse2,+sse4.1,+sse4.2,+avx"
fi

# Common flags
COMMON_FLAGS="-m $MEM -smp $CPUS -serial stdio -no-reboot -no-shutdown"
COMMON_FLAGS="$COMMON_FLAGS -device virtio-gpu-pci"
COMMON_FLAGS="$COMMON_FLAGS -netdev user,id=net0,hostfwd=tcp::8080-:8080"
COMMON_FLAGS="$COMMON_FLAGS -device virtio-net-pci,netdev=net0"
COMMON_FLAGS="$COMMON_FLAGS -D $LOG"

case "${1:-}" in
    --debug)
        echo "================================================================"
        echo "  TensorOS — DEBUG MODE"
        echo "  GDB stub listening on localhost:1234"
        echo "  Connect with: gdb -ex 'target remote :1234' $KERNEL"
        echo "================================================================"
        mkdir -p "$BUILD_DIR"
        $QEMU $COMMON_FLAGS $KVM_FLAGS -kernel "$KERNEL" -s -S
        ;;
    --iso)
        if [ ! -f "$ISO" ]; then
            echo "ISO not found. Building..."
            cd "$PROJECT_DIR" && make iso
        fi
        echo "[*] Booting from ISO: $ISO"
        $QEMU $COMMON_FLAGS $KVM_FLAGS -cdrom "$ISO"
        ;;
    --monitor)
        echo "[*] QEMU monitor on stdio (Ctrl-A c to switch)"
        $QEMU $COMMON_FLAGS $KVM_FLAGS -kernel "$KERNEL" -monitor stdio
        ;;
    ""|--run)
        echo "================================================================"
        echo "  TensorOS — Booting in QEMU"
        echo "  Memory: $MEM | CPUs: $CPUS"
        echo "  Serial console on stdio"
        echo "  HTTP port forwarded: host:8080 → guest:8080"
        echo "================================================================"
        $QEMU $COMMON_FLAGS $KVM_FLAGS -kernel "$KERNEL"
        ;;
    *)
        echo "Usage: $0 [--debug|--iso|--monitor|--run]"
        exit 1
        ;;
esac
