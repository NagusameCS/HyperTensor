#!/usr/bin/env bash
# Watch for "[paperA] GRC decode runs" in /tmp/paperA_remote.log; when seen, kill warm loop processes.
# Then run NCU baseline manually.

set -x
LOG=/tmp/paperA_remote.log
SRC=/opt/hypertensor
OUT=/opt/hypertensor/results_L40S
EXE=$SRC/geodessical
MODEL=$SRC/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Wait for the warm-loop log line (or cold-failed). At that point kill the orchestrator
# and any warm-run geodessical process.
while true; do
  if grep -q "GRC decode runs" "$LOG" 2>/dev/null; then
    echo "[watch] warm loop detected; killing orchestrator + warm runs"
    pkill -9 -f run_paperA.sh
    # Kill the warm geodessical (started after cold). Cold has already exited by now.
    pkill -9 -f "axex-compress" || true
    sleep 5
    break
  fi
  if ! pgrep -f run_paperA.sh >/dev/null; then
    echo "[watch] run_paperA.sh exited"
    break
  fi
  sleep 10
done

# Now run NCU baseline only (no GRC NCU --- too expensive).
NCU_BIN=""
for c in /usr/local/cuda/bin/ncu /usr/local/cuda-12.9/bin/ncu /opt/nvidia/nsight-compute/*/ncu $(command -v ncu 2>/dev/null); do
  [ -x "$c" ] && NCU_BIN="$c" && break
done
echo "[watch] NCU=$NCU_BIN"

if [ -n "$NCU_BIN" ] && [ -x "$NCU_BIN" ]; then
  cd "$SRC"
  export GD_CUDA_KERNELS_PATH="$SRC/cuda_kernels.so"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$SRC"

  METRICS="lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_miss.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"

  echo "[watch] NCU baseline trace..."
  base_rep="$OUT/paperA_ncu_baseline_L40S.ncu-rep"
  "$NCU_BIN" --metrics "$METRICS" --launch-skip 200 --launch-count 100 \
    --kernel-name "kernel_gemv_q4_k" --target-processes all \
    -o "${base_rep%.ncu-rep}" \
    "$EXE" "$MODEL" -p hello -n 16 --temp 0 \
    >"$OUT/paperA_ncu_baseline_L40S.stdout" \
    2>"$OUT/paperA_ncu_baseline_L40S.stderr" || echo "[watch] ncu baseline exit $?"

  [ -f "$base_rep" ] && "$NCU_BIN" --import "$base_rep" --page raw --csv \
    > "$OUT/paperA_ncu_baseline_L40S.csv" 2>>"$OUT/paperA_ncu_baseline_L40S.stderr" || true
fi

# Build summary + tarball
cd "$OUT"
{
  echo "===== Paper-A cache-fit results --- L40S (GPU, partial) ====="
  date -u
  echo
  echo "----- baseline tok/s extract -----"
  cat paperA_baseline_L40S.txt
  echo
  echo "----- GRC cold tok/s extract -----"
  grep -Ei "decode|tok/?s|prefill|TpF|GRC|axex|model loaded|backend" paperA_grc_L40S_cold.log | tail -n 30 || true
  echo
  echo "----- NCU csv heads -----"
  for f in paperA_ncu_*.csv; do
    [ -f "$f" ] || continue
    echo "## $f"
    head -n 5 "$f"
    echo
  done
} > "$OUT/summary.txt"

ts="$(date -u +%Y%m%dT%H%M%SZ)"
TAR_PATH="$SRC/paperA_L40S_${ts}.tar.gz"
tar czf "$TAR_PATH" -C "$SRC" results_L40S
echo "[watch] tarball: $TAR_PATH ($(du -h "$TAR_PATH" | awk '{print $1}'))"
echo "$TAR_PATH" > "$OUT/tarball_path.txt"
echo "[watch] DONE"
