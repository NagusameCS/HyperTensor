#!/usr/bin/env bash
# scripts/paperA_proof/expB_a100_cuda_limit.sh
# -----------------------------------------------------------------------------
# Experiment B (P2) on A100/H100: clean L2 carveout via cudaLimitPersistingL2CacheSize.
#
# This is the "definitive" version of the L2 manipulation experiment:
# instead of an external thrash co-tenant (which adds SM-cycle confounds),
# we use the CUDA driver API to artificially shrink the persistent L2 carveout
# available to the geodessical process.
#
# Requires: GPU with sm_80+ (A100, H100, RTX 4090 - Ada exposes the API but
# the RTX-class enforcement is best-effort).
#
# This is a *driver-side* request: we modify geodessical's startup to set
# cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, <bytes>) before any
# inference work begins. See host/main.c:axex_init_runtime() for the hook.
#
# Without code modification, an externally-injected env var read at startup
# is the cleanest path. We provide a wrapper here that exports the env var
# and runs geodessical; the runtime reads AXEX_L2_PERSISTING_BYTES and calls
# the API if present.
# -----------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-/opt/hypertensor/models/Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
EXE="${EXE:-/opt/hypertensor/build/geodessical}"
OUT_DIR="${OUT_DIR:-/opt/hypertensor/results_paperA_expB_a100}"
N_REPS="${N_REPS:-3}"
TOKENS="${TOKENS:-16}"
PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog.}"

# Persistent L2 carveout sweep, in MB. 0 = full L2 available.
DELTAS=(0 8 16 24 32)

mkdir -p "$OUT_DIR"

run_one() {
    local label="$1" delta="$2"
    shift 2
    local out_csv="$OUT_DIR/${label}.ncu.csv"
    local out_log="$OUT_DIR/${label}.stdout.log"
    if [[ -f "$out_csv" ]]; then
        echo "[skip] $label"
        return
    fi
    # Convert MB to bytes; 0 means "do not call setLimit" (default).
    local persist_bytes=0
    if [[ "$delta" -gt 0 ]]; then
        persist_bytes=$(( delta * 1024 * 1024 ))
    fi
    echo "[expB-a100] $label persist_bytes=$persist_bytes"
    AXEX_L2_PERSISTING_BYTES="$persist_bytes" \
    ncu --metrics lts__t_sector_hit_rate.pct,lts__t_sectors_op_read.sum,dram__bytes_read.sum,gpu__time_duration.sum \
        --target-processes all \
        --launch-skip 50 --launch-count 200 \
        --kernel-name regex:gemv \
        --csv --log-file "$out_csv" \
        -- "$EXE" "$MODEL" -p "$PROMPT" -n "$TOKENS" --temp 0 "$@" \
        > "$out_log" 2>&1
}

for delta in "${DELTAS[@]}"; do
    for cond in baseline grc_k1024; do
        for r in $(seq 1 "$N_REPS"); do
            label="delta${delta}_${cond}_rep${r}"
            if [[ "$cond" == "baseline" ]]; then
                run_one "$label" "$delta"
            else
                run_one "$label" "$delta" \
                    --axex-compress --axex-attn-only --axex-weight-pca \
                    --axiom-skip-geodesic --axex-skip-o --axex-compress-rank 1024
            fi
            sleep 10
        done
    done
done

echo "[expB-a100] complete; results in $OUT_DIR"
