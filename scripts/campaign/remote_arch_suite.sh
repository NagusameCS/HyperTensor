#!/usr/bin/env bash
set -euo pipefail

MODEL=""
EXE=""
OUT_DIR=""
COOLDOWN_SEC="20"
LMEVAL_LIMIT="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --exe) EXE="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --cooldown-sec) COOLDOWN_SEC="$2"; shift 2 ;;
    --lm-eval-limit) LMEVAL_LIMIT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="benchmarks/cross_hw_remote_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$OUT_DIR"/logs "$OUT_DIR"/docs_data "$OUT_DIR"/meta

if [[ -z "$EXE" ]]; then
  if [[ -x "./geodessical" ]]; then
    EXE="./geodessical"
  elif [[ -x "/root/HyperTensor/geodessical" ]]; then
    EXE="/root/HyperTensor/geodessical"
  else
    echo "geodessical binary not found; pass --exe" >&2
    exit 1
  fi
fi

if [[ -z "$MODEL" ]]; then
  CANDIDATE=""
  CANDIDATE=$(find /root/models -maxdepth 2 -type f -name '*.gguf' 2>/dev/null | head -n 1 || true)
  if [[ -z "$CANDIDATE" ]]; then
    echo "No model auto-detected under /root/models; pass --model" >&2
    exit 1
  fi
  MODEL="$CANDIDATE"
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  exit 1
fi

if [[ ! -x "$EXE" ]]; then
  echo "Executable not runnable: $EXE" >&2
  exit 1
fi

capture_env() {
  {
    echo "generated_at=$(date -Iseconds)"
    echo "hostname=$(uname -n)"
    echo "kernel=$(uname -srmo)"
    echo "python=$(python --version 2>&1)"
    echo "exe=$EXE"
    echo "model=$MODEL"
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tr '\n' ';')"
    else
      echo "gpu=none"
    fi
    if command -v ncu >/dev/null 2>&1; then
      echo "ncu=$(ncu --version 2>/dev/null | head -n 1)"
    else
      echo "ncu=missing"
    fi
  } > "$OUT_DIR/meta/environment.txt"
}

extract_tps() {
  local file="$1"
  python - "$file" <<'PY'
import re,sys
raw=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
m=re.search(r"Decode-only:\s*prefill\s*[\d.]+\s*ms,\s*([\d.]+)\s*tok/s",raw)
if m:
    print(m.group(1)); raise SystemExit(0)
m=re.search(r"\[GD\]\s*\d+\s+tokens\s+in\s*[\d.]+\s*ms\s*\(([\d.]+)\s*tok/s\)",raw)
if m:
    print(m.group(1)); raise SystemExit(0)
print("NaN")
PY
}

run_ncu_if_present() {
  local log="$OUT_DIR/logs/ncu_l2_profile.log"
  if ! command -v ncu >/dev/null 2>&1; then
    echo "ncu_missing" > "$OUT_DIR/docs_data/ncu_l2_profile_status.txt"
    echo "ncu not installed; skipping" | tee "$log"
    return 0
  fi

  ncu --metrics lts__t_sector_hit_rate.pct,dram__bytes_read.sum --target-processes all --csv --log-file "$OUT_DIR/docs_data/ncu_baseline_raw.csv" -- "$EXE" "$MODEL" -p "The quick brown fox jumps over the lazy dog." -n 10 --temp 0 > "$log" 2>&1
  sleep "$COOLDOWN_SEC"
  ncu --metrics lts__t_sector_hit_rate.pct,dram__bytes_read.sum --target-processes all --csv --log-file "$OUT_DIR/docs_data/ncu_grc1024_raw.csv" -- "$EXE" "$MODEL" -p "The quick brown fox jumps over the lazy dog." -n 10 --temp 0 --axex-compress --axex-skip-o --axex-compress-rank 1024 >> "$log" 2>&1

  python - "$OUT_DIR/docs_data/ncu_baseline_raw.csv" "$OUT_DIR/docs_data/ncu_grc1024_raw.csv" "$OUT_DIR/docs_data/ncu_l2_profile.csv" <<'PY'
import csv,sys
def read(path):
    hit=[];dram=[]
    with open(path,newline='',encoding='utf-8',errors='ignore') as f:
        for r in csv.DictReader(f):
            name=r.get('Metric Name','')
            val=r.get('Metric Value','').replace(',','')
            try:v=float(val)
            except:continue
            if name=='lts__t_sector_hit_rate.pct': hit.append(v)
            if name=='dram__bytes_read.sum': dram.append(v)
    h=sum(hit)/len(hit) if hit else float('nan')
    d=sum(dram) if dram else float('nan')
    return h,d
b=read(sys.argv[1]); g=read(sys.argv[2])
with open(sys.argv[3],'w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['condition','l2_tex_hit_rate','dram_bytes_read'])
    w.writerow(['baseline',b[0],b[1]])
    w.writerow(['grc_k1024',g[0],g[1]])
PY
}

run_context_sweep() {
  local out_csv="$OUT_DIR/docs_data/context_length_sweep.csv"
  : > "$out_csv"
  echo "context_tokens,baseline_mean,grc_mean,grc_over_baseline" >> "$out_csv"
  local contexts=(128 512 1024 2048 4096)
  local words="the quick brown fox jumps over the lazy dog and then continues running through the forest looking for adventure under a clear blue summer sky"

  for ctx in "${contexts[@]}"; do
    local prompt=""
    while [[ ${#prompt} -lt $((ctx*4)) ]]; do prompt+=" $words"; done
    local bsum=0 gsum=0 reps=5
    for _ in $(seq 1 $reps); do
      local bout="$OUT_DIR/logs/context_${ctx}_baseline.txt"
      "$EXE" "$MODEL" -p "$prompt" -n 64 --temp 0 > "$bout" 2>&1
      local btps; btps=$(extract_tps "$bout")
      bsum=$(python - <<PY
print($bsum + float('$btps'))
PY
)
      sleep "$COOLDOWN_SEC"

      local gout="$OUT_DIR/logs/context_${ctx}_grc1024.txt"
      "$EXE" "$MODEL" -p "$prompt" -n 64 --temp 0 --axex-compress --axex-skip-o --axex-compress-rank 1024 > "$gout" 2>&1
      local gtps; gtps=$(extract_tps "$gout")
      gsum=$(python - <<PY
print($gsum + float('$gtps'))
PY
)
      sleep "$COOLDOWN_SEC"
    done

    local bmean gmean ratio
    bmean=$(python - <<PY
print(round($bsum/$reps,6))
PY
)
    gmean=$(python - <<PY
print(round($gsum/$reps,6))
PY
)
    ratio=$(python - <<PY
print(round(($gmean/$bmean) if $bmean > 0 else 0.0,6))
PY
)
    echo "$ctx,$bmean,$gmean,$ratio" >> "$out_csv"
  done
}

run_rank_pareto() {
  local out_csv="$OUT_DIR/docs_data/rank_pareto.csv"
  : > "$out_csv"
  echo "rank,decode_mean,ratio_to_baseline" >> "$out_csv"
  local prompt="The quick brown fox jumps over the lazy dog."
  local reps=7
  local bsum=0
  for _ in $(seq 1 $reps); do
    local bout="$OUT_DIR/logs/rank_baseline.txt"
    "$EXE" "$MODEL" -p "$prompt" -n 64 --temp 0 > "$bout" 2>&1
    local btps; btps=$(extract_tps "$bout")
    bsum=$(python - <<PY
print($bsum + float('$btps'))
PY
)
    sleep "$COOLDOWN_SEC"
  done
  local bmean
  bmean=$(python - <<PY
print(round($bsum/$reps,6))
PY
)
  echo "0,$bmean,1.0" >> "$out_csv"

  local ranks=(512 768 1024 1280 1536)
  for k in "${ranks[@]}"; do
    local s=0
    for _ in $(seq 1 $reps); do
      local out="$OUT_DIR/logs/rank_${k}.txt"
      "$EXE" "$MODEL" -p "$prompt" -n 64 --temp 0 --axex-compress --axex-skip-o --axex-compress-rank "$k" > "$out" 2>&1
      local tps; tps=$(extract_tps "$out")
      s=$(python - <<PY
print($s + float('$tps'))
PY
)
      sleep "$COOLDOWN_SEC"
    done
    local mean ratio
    mean=$(python - <<PY
print(round($s/$reps,6))
PY
)
    ratio=$(python - <<PY
print(round(($mean/$bmean) if $bmean > 0 else 0.0,6))
PY
)
    echo "$k,$mean,$ratio" >> "$out_csv"
  done
}

run_lm_eval_if_available() {
  local log="$OUT_DIR/logs/lm_eval_suite.log"
  local limit_args=()
  if [[ "$LMEVAL_LIMIT" =~ ^[0-9]+$ ]] && [[ "$LMEVAL_LIMIT" -gt 0 ]]; then
    limit_args=(--limit "$LMEVAL_LIMIT")
  fi

  if ! python -m lm_eval --help >/dev/null 2>&1; then
    python -m pip install --upgrade pip >> "$log" 2>&1 || true
    python -m pip install lm-eval >> "$log" 2>&1 || true
  fi
  if ! python -m lm_eval --help >/dev/null 2>&1; then
    echo "lm_eval_missing" > "$OUT_DIR/docs_data/lm_eval_status.txt"
    return 0
  fi

  local port=8081
  "$EXE" "$MODEL" --serve --port "$port" > "$OUT_DIR/logs/geod_server_baseline.log" 2>&1 &
  local pid_base=$!
  sleep 5
  python -m lm_eval --model local-completions --model_args "base_url=http://127.0.0.1:${port}/v1/completions,model=geodessical" --tasks gsm8k,humaneval,mbpp --num_fewshot 0 --output_path "$OUT_DIR/lm_eval_out_baseline" --log_samples "${limit_args[@]}" >> "$log" 2>&1 || true
  kill "$pid_base" >/dev/null 2>&1 || true
  wait "$pid_base" 2>/dev/null || true

  "$EXE" "$MODEL" --serve --port "$port" --axex-compress --axex-skip-o --axex-compress-rank 1536 > "$OUT_DIR/logs/geod_server_grc1536.log" 2>&1 &
  local pid_grc=$!
  sleep 5
  python -m lm_eval --model local-completions --model_args "base_url=http://127.0.0.1:${port}/v1/completions,model=geodessical" --tasks gsm8k,humaneval,mbpp --num_fewshot 0 --output_path "$OUT_DIR/lm_eval_out_grc1536" --log_samples "${limit_args[@]}" >> "$log" 2>&1 || true
  kill "$pid_grc" >/dev/null 2>&1 || true
  wait "$pid_grc" 2>/dev/null || true

  python - "$OUT_DIR/lm_eval_out_baseline" "$OUT_DIR/lm_eval_out_grc1536" "$OUT_DIR/docs_data/lm_eval_results.json" <<'PY'
import json,glob,os,sys
def load(root):
    files=glob.glob(os.path.join(root,'**','results_*.json'),recursive=True)
    if not files: return None
    with open(files[0],encoding='utf-8') as f: return json.load(f)
out={'baseline':load(sys.argv[1]),'grc_k1536':load(sys.argv[2])}
with open(sys.argv[3],'w',encoding='utf-8') as f: json.dump(out,f,indent=2)
PY
}

capture_env
run_ncu_if_present
run_context_sweep
run_rank_pareto
run_lm_eval_if_available

cat > "$OUT_DIR/campaign_manifest.json" <<JSON
{
  "generated_at": "$(date -Iseconds)",
  "hostname": "$(uname -n)",
  "exe": "$EXE",
  "model": "$MODEL",
  "out_dir": "$OUT_DIR"
}
JSON

echo "REMOTE_CAMPAIGN_OUT=$OUT_DIR"
