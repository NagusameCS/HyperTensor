#!/usr/bin/env bash
# rerun_compress_arms.sh
# Re-run ONLY the failing compress arms on A10G (baseline already captured).
# Fixed flags: --axex-weight-pca (alias for -only), --axex-skip-o (accepted),
# --ppl-eval (restored), --axex-gauge, --axex-online-basis all work now.
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

COMMON=("$MODEL_PATH" --temp 0)
ARM_COMPRESS=( --axex-compress --axex-weight-pca --axex-compress-rank "$RANK" --axex-attn-only --axex-skip-o --axiom-skip-geodesic )
ARM_GAUGE=( "${ARM_COMPRESS[@]}" --axex-gauge )
ARM_ONLINE=( "${ARM_COMPRESS[@]}" --axex-online-basis --ott-speculative )
ARM_SPEC=( "${ARM_COMPRESS[@]}" --ott-speculative )

THR_GREP='Decode-only|tok/?s|prefill|TpF|model loaded|backend|AXEX-GAUGE|ONB\]'

run_arm() {
    local arm="$1"
    shift
    local extra_flags=("$@")
    echo
    echo "[paperB] ===== ARM: $arm ====="
    echo "[paperB]  flags: ${extra_flags[*]}"

    # PPL eval
    local ppl_log="$OUT_DIR/${arm}_ppl.log"
    echo "[paperB]  PPL run -> $ppl_log"
    /usr/bin/time -v "$EXE" "${COMMON[@]}" "${extra_flags[@]}" --ppl-eval \
        >"$ppl_log" 2>&1 && echo "[paperB]  ppl exit 0" || echo "[paperB]  ppl exit $?"
    grep -E '\[PPL-JSON\]|\[PPL\]|\[GD\] Model loaded|backend' "$ppl_log" | head -n 10 || true

    # Throughput runs
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

run_arm compress    "${ARM_COMPRESS[@]}"
run_arm compress_gauge  "${ARM_GAUGE[@]}"
run_arm compress_online "${ARM_ONLINE[@]}"
run_arm compress_spec   "${ARM_SPEC[@]}"

echo "[paperB] === compress rerun complete ==="
