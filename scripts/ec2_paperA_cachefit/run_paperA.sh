#!/usr/bin/env bash
# scripts/ec2_paperA_cachefit/run_paperA.sh
#
# Paper-A cache-fit reproduction on a single GPU.
#
# Required env:
#   GPU_NAME       short tag like "L40S", "A10G", "A100", "H100", "RTX4090"
#   SRC_DIR        path to checked-out HyperTensor source (default /opt/hypertensor)
#   MODEL_PATH     path to Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
#   OUT_DIR        directory to write all artifacts (created)
# Optional:
#   N_DECODE       decode tokens (default 16, matches user spec)
#   PROMPT         prompt string (default "hello")
#   SEED           default 42
#   RANK           default 1024
#   RUNS_BASELINE  default 5
#   RUNS_GRC       default 5
#   SKIP_NCU       set to 1 to skip Nsight Compute pass
#   NCU_BIN        path to ncu binary (auto-detected)
#   SHUTDOWN_AT_END  if set, sudo shutdown -h +1 after results are tarballed
#
# Output artifacts in $OUT_DIR:
#   meta.json
#   nvidia_smi.txt
#   gpu_query.csv
#   paperA_baseline_<GPU>.txt
#   paperA_grc_<GPU>.txt
#   paperA_baseline_<GPU>_run<i>.log     (full per-run logs)
#   paperA_grc_<GPU>_run<i>.log
#   paperA_ncu_baseline_<GPU>.ncu-rep    (if ncu present)
#   paperA_ncu_grc_<GPU>.ncu-rep
#   paperA_ncu_baseline_<GPU>.csv
#   paperA_ncu_grc_<GPU>.csv
#   summary.txt
#   paperA_<GPU>_<ts>.tar.gz
set -uo pipefail

GPU_NAME="${GPU_NAME:?GPU_NAME required}"
SRC_DIR="${SRC_DIR:-/opt/hypertensor}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH required}"
OUT_DIR="${OUT_DIR:?OUT_DIR required}"
N_DECODE="${N_DECODE:-16}"
PROMPT="${PROMPT:-hello}"
SEED="${SEED:-42}"
RANK="${RANK:-1024}"
RUNS_BASELINE="${RUNS_BASELINE:-5}"
RUNS_GRC="${RUNS_GRC:-5}"
SKIP_NCU="${SKIP_NCU:-0}"

mkdir -p "$OUT_DIR"
cd "$SRC_DIR"

# Make CUDA libs visible (DLAMI usually already does this; belt + braces)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$SRC_DIR"
export GD_CUDA_KERNELS_PATH="$SRC_DIR/cuda_kernels.so"

# Sanity: confirm CUDA visible
echo "[paperA] nvidia-smi sanity:"
nvidia-smi --query-gpu=name,memory.free --format=csv 2>&1 | head -n 5 || true
echo "[paperA] cuda_kernels.so: $(ls -la "$GD_CUDA_KERNELS_PATH" 2>&1)"
ldd "$GD_CUDA_KERNELS_PATH" 2>&1 | head -n 20 || true

EXE="$SRC_DIR/geodessical"
if [ ! -x "$EXE" ]; then
  echo "ERROR: $EXE not found / not executable" >&2
  exit 2
fi

# --------------------------------------------------------------------------
# 0. Capture environment
# --------------------------------------------------------------------------
{
  echo "===== uname -a ====="
  uname -a
  echo
  echo "===== /proc/cpuinfo (model) ====="
  grep -m1 'model name' /proc/cpuinfo || true
  grep -c '^processor' /proc/cpuinfo | sed 's/^/cpus=/'
  echo
  echo "===== free -h ====="
  free -h
  echo
  echo "===== nvidia-smi ====="
  nvidia-smi || true
  echo
  echo "===== nvcc --version ====="
  (nvcc --version || /usr/local/cuda/bin/nvcc --version) 2>&1 || true
  echo
  echo "===== geodessical --help (head) ====="
  "$EXE" --help 2>&1 | head -n 40 || true
} > "$OUT_DIR/env.txt" 2>&1

nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,pcie.link.gen.current,pcie.link.width.current \
  --format=csv > "$OUT_DIR/gpu_query.csv" 2>&1 || true

cat > "$OUT_DIR/meta.json" <<EOF
{
  "gpu_name": "$GPU_NAME",
  "model_path": "$MODEL_PATH",
  "n_decode": $N_DECODE,
  "prompt": "$PROMPT",
  "seed": $SEED,
  "rank": $RANK,
  "runs_baseline": $RUNS_BASELINE,
  "runs_grc": $RUNS_GRC,
  "host": "$(hostname)",
  "started_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_rev": "$(git -C "$SRC_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
}
EOF

# --------------------------------------------------------------------------
# Helper: run geodessical and tee output
# --------------------------------------------------------------------------
GREP_THROUGHPUT='decode.*tok/?s|eval time|prompt eval|tokens?/sec|tok/s|axex-pca|axex-weight-pca|axex-compress|GRC|wproj|model loaded|cuda backend|backend init'

# common flags (model is positional first arg; no --seed in CLI, --temp 0 is deterministic)
COMMON_ARGS=("$MODEL_PATH" -p "$PROMPT" -n "$N_DECODE" --temp 0)

# GRC arms — using actual flags (see host/main.c):
#   --axex-compress              enable
#   --axex-weight-pca            calibration-free path (memory note: actaware path crashes)
#   --axex-compress-rank N       rank
#   --axex-attn-only             attention only
#   --axex-skip-o                skip O-projection
#   --axiom-skip-geodesic        belt + braces (avoid extra geodesic init)
GRC_ARGS=(
  --axex-compress
  --axex-weight-pca
  --axex-compress-rank "$RANK"
  --axex-attn-only
  --axex-skip-o
  --axiom-skip-geodesic
)

# --------------------------------------------------------------------------
# 1. Baseline decode throughput x RUNS_BASELINE
# --------------------------------------------------------------------------
echo "[paperA] baseline decode runs (n=$RUNS_BASELINE)..."
> "$OUT_DIR/paperA_baseline_${GPU_NAME}.txt"
for i in $(seq 1 "$RUNS_BASELINE"); do
  log="$OUT_DIR/paperA_baseline_${GPU_NAME}_run${i}.log"
  echo "  - run $i -> $log"
  /usr/bin/time -v "$EXE" "${COMMON_ARGS[@]}" >"$log" 2>&1 || echo "  ! exit $?"
  {
    echo "===== run $i ====="
    grep -Ei "$GREP_THROUGHPUT" "$log" || true
    echo
  } >> "$OUT_DIR/paperA_baseline_${GPU_NAME}.txt"
done

# --------------------------------------------------------------------------
# 2. GRC k=$RANK attn-only (cold cache populate, then RUNS_GRC warm runs)
# --------------------------------------------------------------------------
echo "[paperA] GRC cold cache populate..."
cold_log="$OUT_DIR/paperA_grc_${GPU_NAME}_cold.log"
/usr/bin/time -v "$EXE" "${COMMON_ARGS[@]}" "${GRC_ARGS[@]}" >"$cold_log" 2>&1 || echo "  ! cold exit $?"
echo "  cold log: $cold_log"

echo "[paperA] GRC decode runs (n=$RUNS_GRC)..."
> "$OUT_DIR/paperA_grc_${GPU_NAME}.txt"
for i in $(seq 1 "$RUNS_GRC"); do
  log="$OUT_DIR/paperA_grc_${GPU_NAME}_run${i}.log"
  echo "  - run $i -> $log"
  /usr/bin/time -v "$EXE" "${COMMON_ARGS[@]}" "${GRC_ARGS[@]}" >"$log" 2>&1 || echo "  ! exit $?"
  {
    echo "===== run $i ====="
    grep -Ei "$GREP_THROUGHPUT" "$log" || true
    echo
  } >> "$OUT_DIR/paperA_grc_${GPU_NAME}.txt"
done

# --------------------------------------------------------------------------
# 3. NCU L2 trace pair on kernel_gemv_q4_k (optional)
# --------------------------------------------------------------------------
if [ "$SKIP_NCU" != "1" ]; then
  NCU_BIN="${NCU_BIN:-}"
  if [ -z "$NCU_BIN" ]; then
    for c in /usr/local/cuda/bin/ncu /usr/local/cuda/nsight-compute*/ncu /opt/nvidia/nsight-compute/*/ncu $(command -v ncu 2>/dev/null); do
      [ -x "$c" ] && NCU_BIN="$c" && break
    done
  fi
  if [ -n "$NCU_BIN" ] && [ -x "$NCU_BIN" ]; then
    echo "[paperA] using NCU at $NCU_BIN"
    "$NCU_BIN" --version 2>&1 | head -n2 >> "$OUT_DIR/env.txt" || true

    METRICS="lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_miss.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
    NCU_COMMON=( --metrics "$METRICS" --launch-skip 200 --launch-count 100 --kernel-name "kernel_gemv_q4_k" --target-processes all )

    echo "[paperA] NCU baseline trace..."
    base_rep="$OUT_DIR/paperA_ncu_baseline_${GPU_NAME}.ncu-rep"
    "$NCU_BIN" "${NCU_COMMON[@]}" -o "${base_rep%.ncu-rep}" \
      "$EXE" "${COMMON_ARGS[@]}" \
      >"$OUT_DIR/paperA_ncu_baseline_${GPU_NAME}.stdout" \
      2>"$OUT_DIR/paperA_ncu_baseline_${GPU_NAME}.stderr" || echo "  ! ncu baseline exit $?"

    echo "[paperA] NCU GRC trace..."
    grc_rep="$OUT_DIR/paperA_ncu_grc_${GPU_NAME}.ncu-rep"
    "$NCU_BIN" "${NCU_COMMON[@]}" -o "${grc_rep%.ncu-rep}" \
      "$EXE" "${COMMON_ARGS[@]}" "${GRC_ARGS[@]}" \
      >"$OUT_DIR/paperA_ncu_grc_${GPU_NAME}.stdout" \
      2>"$OUT_DIR/paperA_ncu_grc_${GPU_NAME}.stderr" || echo "  ! ncu grc exit $?"

    # Export CSVs (raw page = full per-launch metrics)
    [ -f "$base_rep" ] && "$NCU_BIN" --import "$base_rep" --page raw --csv > "$OUT_DIR/paperA_ncu_baseline_${GPU_NAME}.csv" 2>>"$OUT_DIR/paperA_ncu_baseline_${GPU_NAME}.stderr" || true
    [ -f "$grc_rep"  ] && "$NCU_BIN" --import "$grc_rep"  --page raw --csv > "$OUT_DIR/paperA_ncu_grc_${GPU_NAME}.csv"      2>>"$OUT_DIR/paperA_ncu_grc_${GPU_NAME}.stderr"      || true
  else
    echo "[paperA] NCU not found, skipping" | tee "$OUT_DIR/ncu_skipped.txt"
  fi
else
  echo "[paperA] SKIP_NCU=1, skipping NCU pass" | tee "$OUT_DIR/ncu_skipped.txt"
fi

# --------------------------------------------------------------------------
# 4. Summary
# --------------------------------------------------------------------------
{
  echo "===== Paper-A cache-fit results — $GPU_NAME ====="
  date -u
  echo
  echo "----- baseline tok/s extract -----"
  cat "$OUT_DIR/paperA_baseline_${GPU_NAME}.txt"
  echo
  echo "----- GRC tok/s extract -----"
  cat "$OUT_DIR/paperA_grc_${GPU_NAME}.txt"
  echo
  echo "----- NCU csv heads -----"
  for f in "$OUT_DIR"/paperA_ncu_*.csv; do
    [ -f "$f" ] || continue
    echo "## $f"
    head -n 5 "$f"
    echo
  done
} > "$OUT_DIR/summary.txt"
cat "$OUT_DIR/summary.txt"

# Update meta.json end time
python3 - <<PY 2>/dev/null || true
import json, os, datetime
p = os.path.join(os.environ["OUT_DIR"], "meta.json")
with open(p) as f: d = json.load(f)
d["finished_utc"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
with open(p, "w") as f: json.dump(d, f, indent=2)
PY

# --------------------------------------------------------------------------
# 5. Tarball
# --------------------------------------------------------------------------
ts="$(date -u +%Y%m%dT%H%M%SZ)"
TAR_PATH="$OUT_DIR/../paperA_${GPU_NAME}_${ts}.tar.gz"
tar czf "$TAR_PATH" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
echo "[paperA] tarball: $TAR_PATH ($(du -h "$TAR_PATH" | awk '{print $1}'))"
echo "$TAR_PATH" > "$OUT_DIR/tarball_path.txt"

if [ -n "${SHUTDOWN_AT_END:-}" ]; then
  echo "[paperA] requesting shutdown -h +1"
  sudo shutdown -h +1 || true
fi

echo "[paperA] DONE"
