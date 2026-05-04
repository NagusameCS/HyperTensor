#!/usr/bin/env bash
# rerun_ppl_fixed.sh
# PPL + throughput rerun with correct flags:
#   - --axex-weight-pca-only alone (sets compress=1, skips Axiom survey, uses weight-gram PCA)
#   - NO --axex-compress (which would trigger the full Axiom beta survey)
#   - Default axex_attn_only=1 preserved -> FFN not compressed
#
# Arms: baseline + compress + compress_gauge + compress_online + compress_spec
set -uo pipefail

GPU_NAME="${GPU_NAME:-A10G}"
SRC_DIR="${SRC_DIR:-/opt/hypertensor}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH required}"
OUT_DIR="${OUT_DIR:?OUT_DIR required}"
N_DECODE="${N_DECODE:-32}"
PROMPT="${PROMPT:-Hello, the quick brown fox jumps over the lazy dog and runs into the night}"
RANK="${RANK:-1024}"
PPL_RUNS="${PPL_RUNS:-1}"
THR_RUNS="${THR_RUNS:-3}"

mkdir -p "$OUT_DIR"
cd "$SRC_DIR"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$SRC_DIR"
export GD_CUDA_KERNELS_PATH="$SRC_DIR/cuda_kernels.so"

EXE="$SRC_DIR/geodessical"
[ -x "$EXE" ] || { echo "ERROR: $EXE not found"; exit 2; }

# --axex-weight-pca-only: sets compress=1, weight_pca_only=1, axiom_skip_geodesic=1
# Does NOT set axiom_beta_run -> no Axiom survey -> direct per-layer weight-PCA (~1-2 min)
# Default axex_attn_only=1 -> only Q/K/V compressed, FFN preserved -> no NaN PPL
COMMON=("$MODEL_PATH" --temp 0)
ARM_BASELINE=()
ARM_COMPRESS=(  --axex-weight-pca-only --axex-compress-rank "$RANK" --axex-attn-only --axex-skip-o )
ARM_GAUGE=(     --axex-weight-pca-only --axex-compress-rank "$RANK" --axex-attn-only --axex-skip-o --axex-gauge )
ARM_ONLINE=(    --axex-weight-pca-only --axex-compress-rank "$RANK" --axex-attn-only --axex-skip-o --axex-online-basis --ott-speculative )
ARM_SPEC=(      --axex-weight-pca-only --axex-compress-rank "$RANK" --axex-attn-only --axex-skip-o --ott-speculative )

THR_GREP='Decode-only|tok/?s|TpF|model loaded|backend|AXEX-GAUGE|ONB\]'

run_arm() {
    local arm="$1"
    shift
    local extra_flags=("$@")
    echo
    echo "[paperB] ===== ARM: $arm ====="
    echo "[paperB]  flags: ${extra_flags[*]:-<none>}"

    local ppl_log="$OUT_DIR/${arm}_ppl.log"
    echo "[paperB]  PPL run -> $ppl_log"
    /usr/bin/time -v "$EXE" "${COMMON[@]}" "${extra_flags[@]}" --ppl-eval \
        >"$ppl_log" 2>&1 && echo "[paperB]  ppl exit 0" || echo "[paperB]  ppl exit $?"
    grep -E '\[PPL-JSON\]|\[PPL\]|\[GD\] Model loaded|backend|\[ONB\]|\[AXEX-GAUGE\]' "$ppl_log" | head -n 10 || true

    > "$OUT_DIR/${arm}_thr.txt"
    for i in $(seq 1 "$THR_RUNS"); do
        local thr_log="$OUT_DIR/${arm}_thr_run${i}.log"
        echo "[paperB]  thr run $i -> $thr_log"
        /usr/bin/time -v "$EXE" "${COMMON[@]}" -p "$PROMPT" -n "$N_DECODE" "${extra_flags[@]}" \
            >"$thr_log" 2>&1 && echo "[paperB]  thr exit 0" || echo "[paperB]  thr exit $?"
        {
            echo "===== run $i ====="
            grep -Ei "$THR_GREP" "$thr_log" || true
            echo
        } >> "$OUT_DIR/${arm}_thr.txt"
    done
}

echo "[paperB] === Fixed rerun: weight-pca-only, no Axiom survey ==="
run_arm baseline    "${ARM_BASELINE[@]}"
run_arm compress    "${ARM_COMPRESS[@]}"
run_arm compress_gauge  "${ARM_GAUGE[@]}"
run_arm compress_online "${ARM_ONLINE[@]}"
run_arm compress_spec   "${ARM_SPEC[@]}"

echo "[paperB] === Fixed rerun complete $(date -u) ==="
