#!/bin/bash
# 70B Benchmark Suite: baseline vs compression quality/speed analysis
# Usage: bash bench_70b.sh [model_path] [output_dir]
# Goal: determine if geodesic SVD compression lobotomizes the 70B model

MODEL="${1:-/root/models/llama31-70b-iq2xs.gguf}"
OUT="${2:-/tmp/bench_70b_results}"
GEO="${GEO_BIN:-/tmp/geodessical.batched}"
KERN="${GD_CUDA_KERNELS_PATH:-/tmp/cuda_kernels_batched.so}"

mkdir -p "$OUT"
LOG="$OUT/bench_log.txt"
echo "=== 70B Benchmark Suite $(date) ===" | tee "$LOG"
echo "Model: $MODEL" | tee -a "$LOG"
echo "Binary: $GEO" | tee -a "$LOG"

if [ ! -f "$MODEL" ]; then echo "ERROR: model not found: $MODEL"; exit 1; fi
if [ ! -f "$GEO" ]; then echo "ERROR: binary not found: $GEO"; exit 1; fi

export GD_CUDA_KERNELS_PATH="$KERN"
export OPENBLAS_NUM_THREADS=8

# ── Helper ────────────────────────────────────────────────────────────────────
run_test() {
    local name="$1"; local prompt="$2"; local tokens="$3"; local extra="$4"
    local outfile="$OUT/${name}.txt"
    echo "" | tee -a "$LOG"
    echo ">>> [$name] prompt: ${prompt:0:60}..." | tee -a "$LOG"
    timeout 600 "$GEO" "$MODEL" --axiom-skip-geodesic $extra \
        -p "$prompt" -n "$tokens" --temp 0 > "$outfile" 2>&1
    local rc=$?
    # Extract speed
    local toks=$(grep 'tok/s' "$outfile" | grep -oP '[0-9]+\.[0-9]+ tok/s' | tail -1)
    # Extract generated text (strip metadata lines)
    local text=$(grep -v '^\[' "$outfile" | grep -v '^$' | tail -6 | head -4)
    echo "  Speed: $toks  RC=$rc" | tee -a "$LOG"
    echo "  Output: $text" | tee -a "$LOG"
}

# ── Section 1: Baseline (no compression) ──────────────────────────────────────
echo "" | tee -a "$LOG"
echo "══════════════════════════════════════════════" | tee -a "$LOG"
echo " SECTION 1: BASELINE (no compression)"          | tee -a "$LOG"
echo "══════════════════════════════════════════════" | tee -a "$LOG"

# Sanity / coherence checks
run_test "b01_capital"    "The capital of France is"                            30  ""
run_test "b02_math_basic" "What is 17 multiplied by 24?"                        40  ""
run_test "b03_story"      "Once upon a time in a galaxy far away, a scientist"  80  ""
run_test "b04_code"       "Write a Python function to compute fibonacci(n):"    80  ""

# Reasoning
run_test "b05_logic"   "If all bloops are razzles and all razzles are lazzles, are all bloops lazzles? Think step by step:" 60 ""
run_test "b06_math_chain" "A train leaves at 9am going 60mph. Another leaves at 10am going 80mph on the same track. When do they meet? Show work:" 80 ""

# Knowledge depth (where 70B should outshine 8B)
run_test "b07_physics"   "Explain the difference between special and general relativity in 3 sentences:" 80 ""
run_test "b08_history"   "What were the three main causes of World War I? Be specific:" 80 ""
run_test "b09_code_hard" "Implement a red-black tree insertion in C with comments:" 120 ""
run_test "b10_reasoning" "A farmer has chickens and rabbits. There are 20 heads and 56 legs. How many of each? Show work:" 60 ""

# ── Section 2: With FFN compression rank 512 ─────────────────────────────────
echo "" | tee -a "$LOG"
echo "══════════════════════════════════════════════" | tee -a "$LOG"
echo " SECTION 2: FFN COMPRESSION rank=512"           | tee -a "$LOG"
echo "══════════════════════════════════════════════" | tee -a "$LOG"

C512="--axex-ffn-compress --axex-compress-rank 512"

run_test "c512_01_capital"   "The capital of France is"                           30  "$C512"
run_test "c512_02_math"      "What is 17 multiplied by 24?"                       40  "$C512"
run_test "c512_03_story"     "Once upon a time in a galaxy far away, a scientist" 80  "$C512"
run_test "c512_04_logic"     "If all bloops are razzles and all razzles are lazzles, are all bloops lazzles? Think step by step:" 60 "$C512"
run_test "c512_05_math_chain" "A farmer has chickens and rabbits. There are 20 heads and 56 legs. How many of each? Show work:" 60 "$C512"
run_test "c512_06_code"      "Write a Python function to compute fibonacci(n):" 80 "$C512"

# ── Section 3: With FFN compression rank 256 ─────────────────────────────────
echo "" | tee -a "$LOG"
echo "══════════════════════════════════════════════" | tee -a "$LOG"
echo " SECTION 3: FFN COMPRESSION rank=256"           | tee -a "$LOG"
echo "══════════════════════════════════════════════" | tee -a "$LOG"

C256="--axex-ffn-compress --axex-compress-rank 256"

run_test "c256_01_capital"   "The capital of France is"                           30  "$C256"
run_test "c256_02_math"      "What is 17 multiplied by 24?"                       40  "$C256"
run_test "c256_03_story"     "Once upon a time in a galaxy far away, a scientist" 80  "$C256"
run_test "c256_04_logic"     "If all bloops are razzles and all razzles are lazzles, are all bloops lazzles? Think step by step:" 60 "$C256"
run_test "c256_05_math_chain" "A farmer has chickens and rabbits. There are 20 heads and 56 legs. How many of each? Show work:" 60 "$C256"
run_test "c256_06_code"      "Write a Python function to compute fibonacci(n):" 80 "$C256"

# ── Section 4: With FFN compression rank 128 ─────────────────────────────────
echo "" | tee -a "$LOG"
echo "══════════════════════════════════════════════" | tee -a "$LOG"
echo " SECTION 4: FFN COMPRESSION rank=128 (aggressive)" | tee -a "$LOG"
echo "══════════════════════════════════════════════" | tee -a "$LOG"

C128="--axex-ffn-compress --axex-compress-rank 128"

run_test "c128_01_capital"   "The capital of France is"                           30  "$C128"
run_test "c128_02_math"      "What is 17 multiplied by 24?"                       40  "$C128"
run_test "c128_03_logic"     "If all bloops are razzles and all razzles are lazzles, are all bloops lazzles? Think step by step:" 60 "$C128"
run_test "c128_04_math_chain" "A farmer has chickens and rabbits. There are 20 heads and 56 legs. How many of each? Show work:" 60 "$C128"

# ── Summary ───────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG"
echo "══════════════════════════════════════════════" | tee -a "$LOG"
echo " SUMMARY"                                        | tee -a "$LOG"
echo "══════════════════════════════════════════════" | tee -a "$LOG"
echo "Speed results:" | tee -a "$LOG"
grep -E '>>>|Speed:' "$LOG" | paste - - | sed 's/>>> //' | tee -a "$OUT/speed_summary.txt"
echo "" | tee -a "$LOG"
echo "All outputs in $OUT/"
echo "Benchmark complete: $(date)" | tee -a "$LOG"
