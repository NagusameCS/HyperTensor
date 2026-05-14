#!/usr/bin/env bash
# Build the native libhypercore + geodessical binaries and stage them into
# hypertensor_runtime/bin/<platform>/ so the wheel picks them up.

set -euo pipefail

uname_s=$(uname -s)
uname_m=$(uname -m)

case "$uname_s" in
    Linux*)  plat="linux-${uname_m}";    libext="so"  ; exe=""    ;;
    Darwin*) plat="mac-${uname_m}";      libext="dylib"; exe=""    ;;
    MINGW*|MSYS*|CYGWIN*) plat="win-${uname_m}"; libext="dll"; exe=".exe" ;;
    *) echo "unknown platform: $uname_s"; exit 1 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/hypertensor_runtime/bin/$plat"
mkdir -p "$OUT"

cmake -S "$ROOT" -B "$ROOT/build_release" \
    -DCMAKE_BUILD_TYPE=Release \
    -DHT_BUILD_RUNTIME=ON \
    -DHT_BUILD_TESTS=OFF
cmake --build "$ROOT/build_release" --parallel

# Stage outputs (best-effort copy; not all targets exist on every platform)
find "$ROOT/build_release" -name "libhypercore.${libext}" -exec cp {} "$OUT/" \; || true
find "$ROOT/build_release" -name "hypercore.${libext}"    -exec cp {} "$OUT/" \; || true
find "$ROOT/build_release" -name "geodessical${exe}"      -exec cp {} "$OUT/" \; || true

echo "Staged native binaries in: $OUT"
ls -la "$OUT" || true
