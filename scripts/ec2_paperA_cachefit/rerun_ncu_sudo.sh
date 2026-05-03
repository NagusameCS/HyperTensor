#!/usr/bin/env bash
set -x
SRC=/opt/hypertensor
OUT=$SRC/results_L40S
EXE=$SRC/geodessical
MODEL=$SRC/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
NCU=/usr/local/cuda-12.9/bin/ncu

# Try to enable counters without reboot via modprobe option file then reload
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-counters.conf
# Reload nvidia modules (likely fails because UVM/driver in use); fallback to sudo run.
sudo rmmod nvidia_uvm 2>&1 | head -n 2 || true
sudo rmmod nvidia_drm nvidia_modeset 2>&1 | head -n 2 || true
sudo rmmod nvidia 2>&1 | head -n 2 || true
sudo modprobe nvidia 2>&1 | head -n 2 || true
sudo modprobe nvidia_uvm 2>&1 | head -n 2 || true

cd "$SRC"
export GD_CUDA_KERNELS_PATH="$SRC/cuda_kernels.so"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$SRC"

METRICS="lts__t_sectors_op_read_lookup_hit.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_read_lookup_miss.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"

base_rep="$OUT/paperA_ncu_baseline_L40S"
sudo -E "$NCU" --metrics "$METRICS" --launch-skip 200 --launch-count 100 \
  --kernel-name "kernel_gemv_q4_k" --target-processes all \
  -o "$base_rep" \
  "$EXE" "$MODEL" -p hello -n 16 --temp 0 \
  >"$OUT/paperA_ncu_baseline_L40S.stdout" \
  2>"$OUT/paperA_ncu_baseline_L40S.stderr"
echo "ncu exit $?"

ls -la "$base_rep".ncu-rep 2>&1 || true

if [ -f "$base_rep.ncu-rep" ]; then
  sudo chown ubuntu:ubuntu "$base_rep.ncu-rep"
  "$NCU" --import "$base_rep.ncu-rep" --page raw --csv \
    > "$OUT/paperA_ncu_baseline_L40S.csv" 2>>"$OUT/paperA_ncu_baseline_L40S.stderr" || true
  echo "csv lines: $(wc -l < $OUT/paperA_ncu_baseline_L40S.csv)"
  head -n 3 "$OUT/paperA_ncu_baseline_L40S.csv"
fi

# Refresh tarball
ts="$(date -u +%Y%m%dT%H%M%SZ)"
TAR="$SRC/paperA_L40S_${ts}.tar.gz"
tar czf "$TAR" -C "$SRC" results_L40S
echo "$TAR" > "$OUT/tarball_path.txt"
ls -la "$TAR"
