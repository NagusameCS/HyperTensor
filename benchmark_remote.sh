#!/usr/bin/env bash
# benchmark_remote.sh — HyperTensor benchmark for the opencs server (Arch Linux)
# Runs Geodessical (CPU+GPU) and Ollama (CPU+GPU) benchmarks, outputs CSV.
#
# Usage:
#   bash benchmark_remote.sh [--geo-only | --ollama-only] [--skip-build] [--trials N]
#
# Outputs:
#   /tmp/benchmark_remote_results.csv
#   /tmp/benchmark_remote_results.md
#
# Expects:
#   - Ollama installed and models pulled (gemma4:e2b, qwen2.5:7b, llama3.2)
#   - Either geodessical binary already at ~/geodessical, or zig available to build
#   - GGUF models at ~/models/ (will download smollm2-135m if absent)

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
TRIALS=3
WARMUPS=1
GEO_BIN="${HOME}/geodessical"
MODEL_DIR="${HOME}/models"
REPO_DIR="${HOME}/HyperTensor"
OLL_URL="http://localhost:11434/api/generate"
OUT_CSV="/tmp/benchmark_remote_results.csv"
OUT_MD="/tmp/benchmark_remote_results.md"

GEO_ONLY=0
OLL_ONLY=0
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --geo-only)    GEO_ONLY=1 ;;
        --ollama-only) OLL_ONLY=1 ;;
        --skip-build)  SKIP_BUILD=1 ;;
        --trials)      TRIALS="$2"; shift ;;
    esac
    shift
done

PROMPTS_SHORT="The quick brown fox jumps"
PROMPTS_MEDIUM="Explain what a transformer neural network is."
PROMPTS_LONG="Explain transformer attention mechanisms in detail, including queries, keys, values, and multi-head attention. Discuss layer normalization and its training benefits."

NS=(40 128 512 1024)

# Machine info
CPU_INFO=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
CPU_CORES=$(nproc)
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "no GPU")
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
OS_INFO=$(uname -r)

echo ""
echo "  HyperTensor Remote Benchmark (opencs)"
echo "  CPU:  ${CPU_INFO} (${CPU_CORES} cores)"
echo "  GPU:  ${GPU_INFO} (${GPU_VRAM} MB VRAM)"
echo "  RAM:  ${RAM_GB} GB"
echo "  OS:   Linux ${OS_INFO}"
echo ""

mkdir -p "$MODEL_DIR"

# ─── Build Geodessical ───────────────────────────────────────────────────────
build_geodessical() {
    if [[ -x "$GEO_BIN" ]] && [[ "$SKIP_BUILD" -eq 1 ]]; then
        echo "[build] Using existing $GEO_BIN"
        return 0
    fi

    echo "[build] Building Geodessical for Linux..."

    if [[ ! -d "$REPO_DIR" ]]; then
        echo "[build] ERROR: HyperTensor repo not found at $REPO_DIR"
        return 1
    fi

    cd "$REPO_DIR"

    SOURCES=(
        "host/hal.c"
        "host/main.c"
        "host/api_server.c"
        "host/gd_daemon.c"
        "host/mcp_server.c"
        "runtime/nn/llm.c"
        "runtime/nn/gguf.c"
        "runtime/nn/backend.c"
        "runtime/nn/model_meta.c"
        "runtime/nn/tensor_bridge.c"
        "runtime/nn/mod_package.c"
        "runtime/nn/token_comm.c"
        "runtime/nn/hf_download.c"
        "runtime/nn/flash_attn.c"
        "runtime/nn/axiom_linalg.c"
        "runtime/nn/axiom_geo.c"
        "runtime/nn/axiom_beta.c"
        "runtime/nn/axiom_exploit.c"
        "runtime/nn/axiom_vis.c"
        "runtime/jit/x86_jit.c"
        "runtime/jit/llm_jit.c"
    )

    CFLAGS=(
        "-std=gnu11" "-O2"
        "-msse2" "-mavx2" "-mfma"
        "-DGEODESSICAL_HOSTED=1"
        "-DGEODESSICAL_LINUX=1"
        "-D_GNU_SOURCE"
        "-Ihost/shims" "-I." "-Ihost"
        "-Wno-unused-function" "-Wno-unused-variable" "-Wno-format"
        "-Wno-incompatible-pointer-types" "-Wno-int-conversion"
        "-Wno-sign-compare" "-Wno-missing-field-initializers"
        "-Wno-unused-parameter" "-Wno-implicit-function-declaration"
    )
    LDFLAGS=("-lm" "-lpthread" "-ldl" "-lcblas")

    # Try with CUDA first (libcuda.so, not nvcc)
    CUDA_LIB_DIR=""
    for d in /opt/cuda/targets/x86_64-linux/lib /usr/local/cuda/lib64 /usr/lib /usr/lib64; do
        if [[ -f "$d/libcudart.so" ]] || [[ -f "$d/libcuda.so" ]]; then
            CUDA_LIB_DIR="$d"; break
        fi
    done
    CUDA_INC=""
    for d in /opt/cuda/targets/x86_64-linux/include /usr/local/cuda/include /opt/cuda/include; do
        if [[ -f "$d/cuda_runtime_api.h" ]] || [[ -f "$d/cuda_runtime.h" ]]; then
            CUDA_INC="$d"; break
        fi
    done

    if [[ -n "$CUDA_INC" ]] && [[ -n "$CUDA_LIB_DIR" ]]; then
        echo "[build] CUDA detected at $CUDA_INC, building with GPU support..."
        set +e
        gcc "${CFLAGS[@]}" \
            -DENABLE_CUDA \
            -I"$CUDA_INC" \
            "${SOURCES[@]}" runtime/nn/backend_cuda.c \
            -o "$GEO_BIN" \
            "${LDFLAGS[@]}" -L"$CUDA_LIB_DIR" -lcudart -Wl,-rpath,"$CUDA_LIB_DIR" 2>&1
        BUILD_RC=$?
        set -e
        if [[ $BUILD_RC -eq 0 ]]; then
            echo "[build] SUCCESS (CUDA)"
            return 0
        fi
        echo "[build] CUDA build failed, falling back to CPU-only..."
    fi

    # CPU-only build with gcc
    set +e
    gcc "${CFLAGS[@]}" \
        "${SOURCES[@]}" \
        -o "$GEO_BIN" \
        "${LDFLAGS[@]}" 2>&1
    BUILD_RC=$?
    set -e

    if [[ $BUILD_RC -eq 0 ]]; then
        echo "[build] SUCCESS (CPU-only, gcc)"
    else
        echo "[build] FAILED — Geodessical benchmarks will be skipped"
        return 1
    fi
}

# ─── Download models ──────────────────────────────────────────────────────────
ensure_model() {
    local path="$1" url="$2" name="$3"
    if [[ -f "$path" ]]; then
        echo "[model] $name already present ($(du -h "$path" | cut -f1))"
        return 0
    fi
    echo "[model] Downloading $name..."
    mkdir -p "$(dirname "$path")"
    if ! curl -L --progress-bar "$url" -o "${path}.tmp"; then
        echo "[model] Download failed for $name"
        return 1
    fi
    mv "${path}.tmp" "$path"
    echo "[model] $name ready ($(du -h "$path" | cut -f1))"
}

# ─── nvidia-smi sampler ───────────────────────────────────────────────────────
MON_FILE="/tmp/geo_remote_monitor.csv"
TS_LOG="/tmp/benchmark_timeseries.csv"
STAGE_FILE="/tmp/geo_remote_stage.txt"
MON_PID=""

set_stage() { echo -n "$1" > "$STAGE_FILE"; }

start_monitor() {
    : > "$MON_FILE"
    (while true; do
        GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw \
            --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        # estimate RAM from any running geodessical or ollama process
        RAM=$(ps aux 2>/dev/null | awk '/geodessical|ollama_llama/{sum+=int($6/1024)} END{print sum+0}')
        echo "${GPU:-0,0,0},${RAM}" >> "$MON_FILE"
        # Persistent time-series log with stage label
        STAGE=$(cat "$STAGE_FILE" 2>/dev/null || echo 'idle')
        TS=$(date +%s%3N)  # milliseconds
        echo "${TS},${GPU:-0,0,0},${RAM},${STAGE}" >> "$TS_LOG"
        sleep 0.4
    done) &
    MON_PID=$!
}

stop_monitor() {
    if [[ -n "$MON_PID" ]]; then
        kill "$MON_PID" 2>/dev/null || true
        MON_PID=""
    fi
}

read_hw_stats() {
    if [[ ! -s "$MON_FILE" ]]; then
        echo "0,0,0,0"; return
    fi
    python3 - <<'EOF'
import sys
rows = []
for line in open("/tmp/geo_remote_monitor.csv"):
    p = line.strip().split(',')
    if len(p) >= 4:
        try: rows.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
        except: pass
if not rows:
    print("0,0,0,0")
else:
    gpu_avg = round(sum(r[0] for r in rows)/len(rows),1)
    vram_max = int(max(r[1] for r in rows))
    power_avg = round(sum(r[2] for r in rows)/len(rows),1)
    ram_max = int(max(r[3] for r in rows))
    print(f"{gpu_avg},{vram_max},{power_avg},{ram_max}")
EOF
}

# ─── Geodessical runner ───────────────────────────────────────────────────────
run_geo() {
    local gguf="$1" text="$2" n="$3"
    local raw
    raw=$("$GEO_BIN" "$gguf" -p "$text" -n "$n" 2>&1) || true
    # Parse: [GD] N tokens in MS ms (T tok/s)
    DECODE_TS=0; PREFILL_TS=0; PREFILL_MS=0; GEN_MS=0; TOTAL_MS=0; NGEN=0; GEO_ERR=""
    if echo "$raw" | grep -q '\[GD\] [0-9]* tokens in [0-9]* ms'; then
        NGEN=$(echo "$raw"      | grep -oP '\[GD\] \K[0-9]+(?= tokens in)')
        GEN_MS=$(echo "$raw"    | grep -oP '\[GD\] [0-9]+ tokens in \K[0-9]+(?= ms)')
        DECODE_TS=$(echo "$raw" | grep -oP '\[GD\] [0-9]+ tokens in [0-9]+ ms \(\K[\d.]+(?= tok/s\))')
        if echo "$raw" | grep -q 'Decode-only: prefill'; then
            PREFILL_MS=$(echo "$raw"  | grep -oP 'prefill \K[\d.]+(?= ms)')
            PREFILL_TS=$(echo "$raw"  | grep -oP 'prefill [\d.]+ ms, \K[\d.]+(?= tok/s)')
        fi
        TOTAL_MS=$(echo "$n $GEN_MS $PREFILL_MS" | awk '{print int($2 + $3)}')
    else
        GEO_ERR="no output pattern matched"
    fi
}

# ─── Ollama runner ────────────────────────────────────────────────────────────
run_oll() {
    local model="$1" text="$2" n="$3" gpu="$4"
    local body resp
    if [[ "$gpu" -eq 0 ]]; then
        body=$(printf '{"model":"%s","prompt":%s,"stream":false,"options":{"num_predict":%d,"num_gpu":0}}' \
            "$model" "$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$text")" "$n")
    else
        body=$(printf '{"model":"%s","prompt":%s,"stream":false,"options":{"num_predict":%d}}' \
            "$model" "$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$text")" "$n")
    fi

    DECODE_TS=0; PREFILL_TS=0; PREFILL_MS=0; GEN_MS=0; TOTAL_MS=0; NGEN=0; OLL_ERR=""
    resp=$(curl -s -m 900 -X POST "$OLL_URL" -H 'Content-Type: application/json' -d "$body" 2>&1) || {
        OLL_ERR="curl failed"; return
    }
    if ! echo "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); \
        ed=d.get('eval_duration',0); pd=d.get('prompt_eval_duration',0); \
        ec=d.get('eval_count',0); pc=d.get('prompt_eval_count',0); \
        dec=round(ec/(ed/1e9),1) if ed>0 else 0; \
        prt=round(pc/(pd/1e9),1) if pd>0 else 0; \
        gms=round(ed/1e6,0); pms=round(pd/1e6,1); \
        print(f'{ec},{dec},{prt},{pms},{gms},{int(gms+pms)}')" 2>/dev/null; then
        OLL_ERR="parse failed"
        return
    fi
    read NGEN DECODE_TS PREFILL_TS PREFILL_MS GEN_MS TOTAL_MS < <(echo "$resp" | python3 -c "
import json,sys
d=json.load(sys.stdin)
ed=d.get('eval_duration',0); pd=d.get('prompt_eval_duration',0)
ec=d.get('eval_count',0);    pc=d.get('prompt_eval_count',0)
dec=round(ec/(ed/1e9),1) if ed>0 else 0
prt=round(pc/(pd/1e9),1) if pd>0 else 0
gms=round(ed/1e6,0); pms=round(pd/1e6,1)
print(ec, dec, prt, pms, gms, int(gms+pms))
")
}

# ─── CSV helpers ─────────────────────────────────────────────────────────────
CSV_ROWS=()
add_row() {
    local machine="remote" rt="$1" be="$2" model="$3" quant="$4" prompt="$5" n="$6" trial="$7"
    local ngen="$8" dec="$9" pre="${10}" pms="${11}" gms="${12}" tms="${13}"
    local vram="${14}" ram="${15}" gpu_pct="${16}" power="${17}" err="${18}"
    CSV_ROWS+=("${machine},${rt},${be},${model},${quant},${prompt},${n},${trial},${ngen},${dec},${pre},${pms},${gms},${tms},${vram},${ram},${gpu_pct},${power},${err}")
    printf "  [%s/%s] %-14s %-8s n=%-5s t=%s  dec=%-8s pre=%-8s ttft=%-6s gen=%-6s vram=%s\n" \
        "$rt" "$be" "$model" "$prompt" "$n" "$trial" \
        "${dec} t/s" "${pre} t/s" "${pms}ms" "${gms}ms" "${vram}MB"
}

# Initialize time-series log
echo "Timestamp,GpuPct,VramMB,PowerW,RamMB,Stage" > "$TS_LOG"

# ─── Main benchmark loops ─────────────────────────────────────────────────────

declare -A PROMPTS_MAP
PROMPTS_MAP["short"]="$PROMPTS_SHORT"
PROMPTS_MAP["medium"]="$PROMPTS_MEDIUM"
PROMPTS_MAP["long"]="$PROMPTS_LONG"

GEO_MODELS=()
GEO_GGUF_PATHS=()
GEO_QUANTS=()

# ─── Geodessical benchmarks ───────────────────────────────────────────────────
if [[ "$OLL_ONLY" -eq 0 ]]; then
    if build_geodessical; then
        # Download smollm2 (tiny, fast to transfer)
        SMOL_PATH="$MODEL_DIR/smollm2-135m-instruct-q8_0.gguf"
        GEMMA_PATH="$MODEL_DIR/google_gemma-4-E2B-it-Q4_0.gguf"
        PHI_PATH="$MODEL_DIR/Phi-3.5-mini-instruct-Q4_0.gguf"

        # smollm2-135m is ~138MB, pull from HuggingFace if absent
        ensure_model "$SMOL_PATH" \
            "https://huggingface.co/bartowski/smollm2-135m-instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf" \
            "smollm2-135m Q8_0" || true

        # Build model list from what's actually present
        if [[ -f "$SMOL_PATH"  ]]; then GEO_MODELS+=("smollm2-135m");  GEO_GGUF_PATHS+=("$SMOL_PATH");  GEO_QUANTS+=("Q8_0"); fi
        if [[ -f "$PHI_PATH"   ]]; then GEO_MODELS+=("phi35-mini");    GEO_GGUF_PATHS+=("$PHI_PATH");   GEO_QUANTS+=("Q4_0"); fi
        if [[ -f "$GEMMA_PATH" ]]; then GEO_MODELS+=("gemma4-2b");     GEO_GGUF_PATHS+=("$GEMMA_PATH"); GEO_QUANTS+=("Q4_0"); fi

        # Determine if we built with CUDA
        GEO_BACKENDS=("CPU")
        if "$GEO_BIN" --version 2>&1 | grep -qi cuda || \
           strings "$GEO_BIN" 2>/dev/null | grep -qi cudart; then
            GEO_BACKENDS=("CPU" "GPU")
        fi

        for be in "${GEO_BACKENDS[@]}"; do
            echo ""
            echo "=== Geodessical $be ==="

            for i in "${!GEO_MODELS[@]}"; do
                model="${GEO_MODELS[$i]}"
                gguf="${GEO_GGUF_PATHS[$i]}"
                quant="${GEO_QUANTS[$i]}"
                echo "  Model: $model ($quant)"

                for pname in short medium long; do
                    ptext="${PROMPTS_MAP[$pname]}"
                    for n in "${NS[@]}"; do
                        # Warmup
                        for ((w=1; w<=WARMUPS; w++)); do
                            echo "    warmup $w/$WARMUPS $pname n=$n..."
                            run_geo "$gguf" "$ptext" "$n" &>/dev/null || true
                        done
                        # Measured trials
                        for ((t=1; t<=TRIALS; t++)); do
                            set_stage "Geo${be}|${model}|${pname}|n${n}|t${t}"
                            : > "$MON_FILE"
                            start_monitor
                            run_geo "$gguf" "$ptext" "$n"
                            stop_monitor
                            HW=$(read_hw_stats)
                            IFS=',' read GPU_PCT VRAM POWER RAM <<< "$HW"
                            if [[ -n "$GEO_ERR" ]]; then
                                add_row "Geodessical" "$be" "$model" "$quant" "$pname" "$n" "$t" \
                                    0 0 0 0 0 0 "$VRAM" "$RAM" "$GPU_PCT" "$POWER" "$GEO_ERR"
                            else
                                add_row "Geodessical" "$be" "$model" "$quant" "$pname" "$n" "$t" \
                                    "$NGEN" "$DECODE_TS" "$PREFILL_TS" "$PREFILL_MS" "$GEN_MS" "$TOTAL_MS" \
                                    "$VRAM" "$RAM" "$GPU_PCT" "$POWER" ""
                            fi
                        done
                    done
                done
            done
        done
    else
        echo "[geo] Build failed — skipping Geodessical benchmarks"
    fi
fi

# ─── Ollama benchmarks ────────────────────────────────────────────────────────
if [[ "$GEO_ONLY" -eq 0 ]]; then
    # Check ollama is running
    if ! curl -s --max-time 3 "http://localhost:11434" &>/dev/null; then
        echo "[ollama] Starting ollama serve..."
        nohup ollama serve &>/tmp/ollama.log &
        sleep 5
    fi

    # Get list of actually-pulled models
    OLL_MODELS=()
    OLL_NAMES=()
    AVAILABLE=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}')
    for tag in gemma4:e2b qwen2.5:7b llama3.2:latest smollm2:135m phi3.5; do
        base=$(echo "$tag" | cut -d: -f1)
        if echo "$AVAILABLE" | grep -qi "$base"; then
            actual=$(echo "$AVAILABLE" | grep -i "$base" | head -1 | awk '{print $1}')
            short_name=$(echo "$tag" | tr ':' '-' | tr '.' '-')
            OLL_MODELS+=("$actual")
            OLL_NAMES+=("$short_name")
            echo "[ollama] Found model: $actual → $short_name"
        fi
    done

    for be in GPU CPU; do
        echo ""
        echo "=== Ollama $be ==="
        for i in "${!OLL_MODELS[@]}"; do
            mdl="${OLL_MODELS[$i]}"
            name="${OLL_NAMES[$i]}"
            use_gpu=1; [[ "$be" == "CPU" ]] && use_gpu=0
            echo "  Model: $name ($mdl)"

            for pname in short medium long; do
                ptext="${PROMPTS_MAP[$pname]}"
                for n in "${NS[@]}"; do
                    # Warmup
                    for ((w=1; w<=WARMUPS; w++)); do
                        echo "    warmup $w/$WARMUPS $pname n=$n..."
                        run_oll "$mdl" "$ptext" "$n" "$use_gpu" || true
                    done
                    # Measured trials
                    for ((t=1; t<=TRIALS; t++)); do
                        set_stage "Oll${be}|${name}|${pname}|n${n}|t${t}"
                        : > "$MON_FILE"
                        start_monitor
                        run_oll "$mdl" "$ptext" "$n" "$use_gpu"
                        stop_monitor
                        HW=$(read_hw_stats)
                        IFS=',' read GPU_PCT VRAM POWER RAM <<< "$HW"
                        if [[ -n "$OLL_ERR" ]]; then
                            add_row "Ollama" "$be" "$name" "ollama" "$pname" "$n" "$t" \
                                0 0 0 0 0 0 "$VRAM" "$RAM" "$GPU_PCT" "$POWER" "$OLL_ERR"
                        else
                            add_row "Ollama" "$be" "$name" "ollama" "$pname" "$n" "$t" \
                                "$NGEN" "$DECODE_TS" "$PREFILL_TS" "$PREFILL_MS" "$GEN_MS" "$TOTAL_MS" \
                                "$VRAM" "$RAM" "$GPU_PCT" "$POWER" ""
                        fi
                    done
                done
            done
        done
    done
fi

# ─── Write CSV ────────────────────────────────────────────────────────────────
echo ""
echo "=== Writing Results ==="
{
    echo "Machine,Runtime,Backend,Model,Quant,Prompt,N,Trial,NGen,DecodeTS,PrefillTS,PrefillMs,GenMs,TotalMs,VramMB,RamMB,GpuPctAvg,PowerW,Err"
    printf '%s\n' "${CSV_ROWS[@]}"
} > "$OUT_CSV"
echo "  CSV: $OUT_CSV (${#CSV_ROWS[@]} rows)"

# ─── Write Markdown (quick summary) ─────────────────────────────────────────
python3 - "$OUT_CSV" "$OUT_MD" <<'PYEOF'
import sys, csv, statistics
from collections import defaultdict

csv_path, md_path = sys.argv[1], sys.argv[2]

rows = []
with open(csv_path) as f:
    for row in csv.DictReader(f):
        if row['Err']: continue
        row['DecodeTS'] = float(row['DecodeTS']) if row['DecodeTS'] else 0
        row['PrefillTS'] = float(row['PrefillTS']) if row['PrefillTS'] else 0
        row['PrefillMs'] = float(row['PrefillMs']) if row['PrefillMs'] else 0
        row['VramMB'] = int(row['VramMB']) if row['VramMB'] else 0
        row['PowerW'] = float(row['PowerW']) if row['PowerW'] else 0
        rows.append(row)

import subprocess
cpu_info  = open('/proc/cpuinfo').read().split('\n')
cpu_name  = next((l.split(':')[1].strip() for l in cpu_info if 'model name' in l), 'unknown')
cpu_cores = subprocess.check_output(['nproc']).decode().strip()
try:
    gpu_name = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],stderr=subprocess.DEVNULL).decode().strip().split('\n')[0]
    gpu_vram = subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],stderr=subprocess.DEVNULL).decode().strip().split('\n')[0]
except: gpu_name='unknown'; gpu_vram='0'
import os; ram_gb = round(os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')/1e9)

lines = []
lines.append("# HyperTensor Remote Benchmark (opencs)")
lines.append("")
from datetime import datetime
lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
lines.append("")
lines.append(f"**Machine:** opencs (remote, Arch Linux)")
lines.append(f"**CPU:** {cpu_name} ({cpu_cores} cores)")
lines.append(f"**GPU:** {gpu_name} ({gpu_vram} MiB VRAM)")
lines.append(f"**RAM:** {ram_gb} GB")
lines.append("")
lines.append("---")
lines.append("")
lines.append("## Summary — Averaged Across All Conditions")
lines.append("")
lines.append("| Runtime | Backend | Model | Decode t/s | Prefill t/s | TTFT ms | Peak VRAM MB | Avg Power W |")
lines.append("|---------|---------|-------|:----------:|:-----------:|:-------:|:------------:|:-----------:|")

combos = {}
for r in rows:
    key = (r['Runtime'], r['Backend'], r['Model'])
    combos.setdefault(key, []).append(r)

for key in sorted(combos):
    rs = combos[key]
    dec   = round(statistics.mean(x['DecodeTS']  for x in rs if x['DecodeTS']>0),1) if any(x['DecodeTS']>0 for x in rs) else '--'
    pre   = round(statistics.mean(x['PrefillTS'] for x in rs if x['PrefillTS']>0),1) if any(x['PrefillTS']>0 for x in rs) else '--'
    ttft  = round(statistics.mean(x['PrefillMs'] for x in rs if x['PrefillMs']>0),1) if any(x['PrefillMs']>0 for x in rs) else '--'
    vram  = max(x['VramMB'] for x in rs)
    pwr   = round(statistics.mean(x['PowerW'] for x in rs if x['PowerW']>0),1) if any(x['PowerW']>0 for x in rs) else '--'
    lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {dec} | {pre} | {ttft} | {vram} | {pwr} |")

lines.append("")
lines.append("---")
lines.append("")
lines.append("## Raw Results")
lines.append("")
lines.append("| Runtime | Backend | Model | Prompt | N | Trial | N gen | Decode t/s | TTFT ms | Gen ms |")
lines.append("|---------|---------|-------|--------|--:|------:|------:|:----------:|:-------:|:------:|")
for r in sorted(rows, key=lambda x:(x['Runtime'],x['Backend'],x['Model'],x['Prompt'],int(x['N']),int(x['Trial']))):
    lines.append(f"| {r['Runtime']} | {r['Backend']} | {r['Model']} | {r['Prompt']} | {r['N']} | {r['Trial']} | {r['NGen']} | {r['DecodeTS']} | {r['PrefillMs']} | {r['GenMs']} |")

lines.append("")
with open(md_path,'w') as f:
    f.write('\n'.join(lines))
print(f"  MD:  {md_path}")
PYEOF

echo ""
echo "  Done. ${#CSV_ROWS[@]} result rows written."
echo "  CSV: $OUT_CSV"
echo "  MD:  $OUT_MD"
echo "  TS:  $TS_LOG"
echo ""

# ─── Generate graphs ────────────────────────────────────────────────
GRAPH_SCRIPT="$(dirname "$0")/benchmark_graph.py"
if [[ -f "$GRAPH_SCRIPT" ]] && command -v python3 &>/dev/null; then
    # Copy CSV to script dir and set TS path so graph script finds them
    cp "$OUT_CSV" "$(dirname "$GRAPH_SCRIPT")/benchmark_extended.csv"
    cp "$TS_LOG"  "$(dirname "$GRAPH_SCRIPT")/benchmark_timeseries.csv"
    echo "  Generating resource graphs..."
    python3 "$GRAPH_SCRIPT" && echo "  Graphs saved to: $(dirname "$GRAPH_SCRIPT")/benchmark_graphs/"
else
    echo "  (benchmark_graph.py not found or python3 unavailable — copy CSV locally and run manually)"
fi
echo ""
