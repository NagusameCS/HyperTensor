#!/usr/bin/env bash
# scripts/ec2_paperB_ablations/run_paperB_ablations.sh
#
# Paper-B Axiom Gauge (§8) + Online Basis (§10) ablations on a single GPU.
#
# Required env:
#   GPU_NAME       short tag (e.g. "L40S")
#   SRC_DIR        path to source (default /opt/hypertensor)
#   MODEL_PATH     path to Llama-3.1-8B Q4_K_M GGUF
#   OUT_DIR        directory for artifacts (created)
# Optional:
#   N_DECODE       throughput decode tokens (default 32)
#   PROMPT         throughput prompt (default "Hello, the quick brown fox")
#   RANK           compression rank (default 1024)
#   GAUGE_ITER     gauge optimisation iterations (default 0 = auto)
#   PPL_RUNS       PPL repeats per arm (default 1; PPL is deterministic so 1 is enough)
#   THR_RUNS       throughput repeats per arm (default 3)
#   SKIP_NCU       set to 1 to skip optional NCU (default 1; NCU not informative for PPL)
#
# Arms:
#   A. baseline               — no compression
#   B. compress (rank-1024)   — --axex-compress + weight-PCA + attn-only + skip-O
#   C. compress + gauge       — B + --axex-gauge
#   D. compress + online      — B + --axex-online-basis (signal: ONB rejection counts;
#                                speculative-decode is required for the trigger to fire,
#                                so this arm runs PPL with --ott-speculative)
#   E. compress + spec only   — B + --ott-speculative (control for arm D)
#
# Output artifacts in $OUT_DIR:
#   meta.json
#   env.txt nvidia_smi.txt gpu_query.csv
#   <arm>_ppl.log              ([PPL-JSON] line for parsing)
#   <arm>_thr.log              throughput run logs
#   <arm>_thr.txt              extracted decode tok/s lines
#   summary.json               machine-readable summary of all arms
#   summary.txt                human-readable
#   paperB_<GPU>_<ts>.tar.gz   tarball

set -uo pipefail

GPU_NAME="${GPU_NAME:?GPU_NAME required}"
SRC_DIR="${SRC_DIR:-/opt/hypertensor}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH required}"
OUT_DIR="${OUT_DIR:?OUT_DIR required}"
N_DECODE="${N_DECODE:-32}"
PROMPT="${PROMPT:-Hello, the quick brown fox jumps over the lazy dog and runs into the night}"
RANK="${RANK:-1024}"
GAUGE_ITER="${GAUGE_ITER:-0}"
PPL_RUNS="${PPL_RUNS:-1}"
THR_RUNS="${THR_RUNS:-3}"
SKIP_NCU="${SKIP_NCU:-1}"

mkdir -p "$OUT_DIR"
cd "$SRC_DIR"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$SRC_DIR"
export GD_CUDA_KERNELS_PATH="$SRC_DIR/cuda_kernels.so"

EXE="$SRC_DIR/geodessical"
[ -x "$EXE" ] || { echo "ERROR: $EXE not found"; exit 2; }

echo "[paperB] sanity:"
nvidia-smi --query-gpu=name,memory.free --format=csv 2>&1 | head -n 5 || true
echo "[paperB] cuda_kernels: $(ls -la "$GD_CUDA_KERNELS_PATH" 2>&1 | awk '{print $5,$9}')"
ldd "$GD_CUDA_KERNELS_PATH" 2>&1 | grep -E 'libcuda|libcublas' | head -n 5 || true

# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
{
  echo "===== uname -a ====="; uname -a
  echo; echo "===== nvidia-smi ====="; nvidia-smi || true
  echo; echo "===== nvcc --version ====="; (nvcc --version || /usr/local/cuda/bin/nvcc --version) 2>&1 || true
  echo; echo "===== geodessical --help (gauge/onb) ====="
  "$EXE" --help 2>&1 | grep -E 'axex-gauge|axex-online|ppl-eval|ott-spec' || true
} > "$OUT_DIR/env.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv > "$OUT_DIR/gpu_query.csv" 2>&1 || true

cat > "$OUT_DIR/meta.json" <<EOF
{
  "experiment":   "paperB_ablations_gauge_online_basis",
  "gpu_name":     "$GPU_NAME",
  "model_path":   "$MODEL_PATH",
  "rank":         $RANK,
  "gauge_iter":   $GAUGE_ITER,
  "n_decode":     $N_DECODE,
  "ppl_runs":     $PPL_RUNS,
  "thr_runs":     $THR_RUNS,
  "host":         "$(hostname)",
  "started_utc":  "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "git_rev":      "$(git -C "$SRC_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "arms":         ["baseline", "compress", "compress_gauge", "compress_online", "compress_spec"]
}
EOF

# --------------------------------------------------------------------------
# Arm flag tables
# --------------------------------------------------------------------------
COMMON=("$MODEL_PATH" --temp 0)

# baseline: no compression
ARM_BASELINE=()
# compress: GRC attn-only @ rank
ARM_COMPRESS=( --axex-compress --axex-weight-pca --axex-compress-rank "$RANK"
               --axex-attn-only --axex-skip-o --axiom-skip-geodesic )
# +gauge
ARM_GAUGE=( "${ARM_COMPRESS[@]}" --axex-gauge )
[ "$GAUGE_ITER" != "0" ] && ARM_GAUGE+=( --axex-gauge-iter "$GAUGE_ITER" )
# +online-basis (needs spec decode for trigger to fire)
ARM_ONLINE=( "${ARM_COMPRESS[@]}" --axex-online-basis --ott-speculative )
# +spec only (control for online)
ARM_SPEC=( "${ARM_COMPRESS[@]}" --ott-speculative )

declare -A ARMS=(
  [baseline]=""
  [compress]="${ARM_COMPRESS[*]}"
  [compress_gauge]="${ARM_GAUGE[*]}"
  [compress_online]="${ARM_ONLINE[*]}"
  [compress_spec]="${ARM_SPEC[*]}"
)
ARM_ORDER=(baseline compress compress_gauge compress_online compress_spec)

THR_GREP='decode|tok/?s|prefill|TpF|model loaded|backend|axex-gauge|AXEX-GAUGE|ONB\]|axex-compress|axex-pca'

# --------------------------------------------------------------------------
# Run a single arm: PPL eval + N throughput runs.
# --------------------------------------------------------------------------
run_arm() {
  local arm="$1"
  local extra_flags="${ARMS[$arm]}"
  echo
  echo "[paperB] ===== ARM: $arm ====="
  echo "[paperB]  flags: $extra_flags"

  # ------- PPL eval (single run, deterministic) --------
  local ppl_log="$OUT_DIR/${arm}_ppl.log"
  echo "[paperB]  PPL run -> $ppl_log"
  /usr/bin/time -v "$EXE" "${COMMON[@]}" $extra_flags --ppl-eval \
      >"$ppl_log" 2>&1 || echo "[paperB]  ppl exit $?"
  grep -E '\[PPL-JSON\]|\[PPL\]|\[GD\] Model loaded|backend|\[ONB\]|\[AXEX-GAUGE\]' "$ppl_log" | head -n 30 || true

  # ------- Throughput runs --------
  > "$OUT_DIR/${arm}_thr.txt"
  for i in $(seq 1 "$THR_RUNS"); do
    local thr_log="$OUT_DIR/${arm}_thr_run${i}.log"
    echo "[paperB]  thr run $i -> $thr_log"
    /usr/bin/time -v "$EXE" "${COMMON[@]}" -p "$PROMPT" -n "$N_DECODE" $extra_flags \
        >"$thr_log" 2>&1 || echo "[paperB]  thr exit $?"
    {
      echo "===== run $i ====="
      grep -Ei "$THR_GREP" "$thr_log" || true
      echo
    } >> "$OUT_DIR/${arm}_thr.txt"
  done
}

for arm in "${ARM_ORDER[@]}"; do
  run_arm "$arm"
done

# --------------------------------------------------------------------------
# Aggregate JSON summary
# --------------------------------------------------------------------------
python3 - "$OUT_DIR" "${ARM_ORDER[@]}" <<'PY' > "$OUT_DIR/summary.json"
import json, os, re, sys
out_dir = sys.argv[1]
arms    = sys.argv[2:]

def parse_ppl(path):
    if not os.path.exists(path): return None
    txt = open(path, errors="ignore").read()
    m = re.search(r"\[PPL-JSON\]\s*(\{.*?\})", txt)
    if not m: return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None

def parse_thr_run(path):
    if not os.path.exists(path): return None
    txt = open(path, errors="ignore").read()
    # Pattern: "[GD] Decode-only: prefill XX.X ms, YY.Y tok/s"
    m = re.search(r"Decode-only:\s*prefill\s*([\d.]+)\s*ms,\s*([\d.]+)\s*tok/s", txt)
    if m:
        return {"prefill_ms": float(m.group(1)), "decode_tok_s": float(m.group(2))}
    # Fallback to e2e tok/s
    m2 = re.search(r"\[GD\]\s+\d+\s+tokens?\s+in\s+([\d.]+)\s*ms\s*\(([\d.]+)\s*tok/s\)", txt)
    if m2:
        return {"prefill_ms": None, "decode_tok_s": None,
                "e2e_ms": float(m2.group(1)), "e2e_tok_s": float(m2.group(2))}
    return None

def parse_onb(path):
    if not os.path.exists(path): return None
    txt = open(path, errors="ignore").read()
    m = re.search(r"\[ONB\]\s*rejections=(\d+)\s*updates=(\d+)\s*layers_updated=(\d+)", txt)
    if not m: return None
    return {"rejections": int(m.group(1)), "updates": int(m.group(2)),
            "layers_updated": int(m.group(3))}

def parse_gauge(path):
    if not os.path.exists(path): return None
    txt = open(path, errors="ignore").read()
    iters = re.findall(r"\[AXEX-GAUGE\]\s+iter\s+(\d+)/(\d+)\s+\|Newton_grad\|_rms=([\de.+-]+)", txt)
    if not iters: return None
    return {"n_iter": int(iters[-1][0]), "max_iter": int(iters[-1][1]),
            "final_grad_rms": float(iters[-1][2])}

result = {"arms": []}
for arm in arms:
    ppl = parse_ppl(os.path.join(out_dir, f"{arm}_ppl.log"))
    onb = parse_onb(os.path.join(out_dir, f"{arm}_ppl.log"))
    gauge = parse_gauge(os.path.join(out_dir, f"{arm}_ppl.log"))
    thr_runs = []
    n_thr = 0
    for i in range(1, 16):
        p = os.path.join(out_dir, f"{arm}_thr_run{i}.log")
        if not os.path.exists(p): break
        n_thr += 1
        r = parse_thr_run(p)
        if r: thr_runs.append(r)
    decode = [r["decode_tok_s"] for r in thr_runs if r and r.get("decode_tok_s") is not None]
    decode_mean = sum(decode)/len(decode) if decode else None
    decode_min  = min(decode) if decode else None
    decode_max  = max(decode) if decode else None
    arm_row = {
        "arm": arm,
        "ppl": ppl,
        "onb_stats": onb,
        "gauge_optim": gauge,
        "thr_n_runs": n_thr,
        "decode_tok_s_mean": decode_mean,
        "decode_tok_s_min":  decode_min,
        "decode_tok_s_max":  decode_max,
        "thr_runs": thr_runs,
    }
    result["arms"].append(arm_row)
print(json.dumps(result, indent=2))
PY

# --------------------------------------------------------------------------
# Human summary
# --------------------------------------------------------------------------
python3 - "$OUT_DIR/summary.json" <<'PY' > "$OUT_DIR/summary.txt"
import json, sys
data = json.load(open(sys.argv[1]))
print("===== Paper-B ablations summary =====")
print(f"{'arm':<22} {'PPL':>10} {'NLL':>10} {'tok/s':>10} {'ONB rej':>10} {'ONB upd':>10}")
print("-" * 78)
for row in data["arms"]:
    arm = row["arm"]
    ppl = row["ppl"]["ppl"] if row["ppl"] else None
    nll = row["ppl"]["nll"] if row["ppl"] else None
    tps = row["decode_tok_s_mean"]
    onb = row["onb_stats"]
    rej = onb["rejections"] if onb else None
    upd = onb["updates"]    if onb else None
    fmt = lambda v, w, d=2: ("{:>" + str(w) + "." + str(d) + "f}").format(v) if isinstance(v,(int,float)) else f"{'-':>{w}}"
    print(f"{arm:<22} {fmt(ppl,10,4)} {fmt(nll,10,4)} {fmt(tps,10,2)} {fmt(rej,10,0)} {fmt(upd,10,0)}")
print()
# Deltas
ppl_b = next((r["ppl"]["ppl"] for r in data["arms"] if r["arm"]=="baseline" and r["ppl"]), None)
ppl_c = next((r["ppl"]["ppl"] for r in data["arms"] if r["arm"]=="compress" and r["ppl"]), None)
ppl_g = next((r["ppl"]["ppl"] for r in data["arms"] if r["arm"]=="compress_gauge" and r["ppl"]), None)
ppl_o = next((r["ppl"]["ppl"] for r in data["arms"] if r["arm"]=="compress_online" and r["ppl"]), None)
ppl_s = next((r["ppl"]["ppl"] for r in data["arms"] if r["arm"]=="compress_spec" and r["ppl"]), None)
print("===== Deltas =====")
if ppl_b and ppl_c: print(f"  PPL increase from compression:        {(ppl_c-ppl_b)/ppl_b*100:+.2f}%  ({ppl_b:.4f} -> {ppl_c:.4f})")
if ppl_c and ppl_g: print(f"  PPL change Gauge vs compress:         {(ppl_g-ppl_c)/ppl_c*100:+.2f}%  ({ppl_c:.4f} -> {ppl_g:.4f})")
if ppl_s and ppl_o: print(f"  PPL change Online vs compress+spec:   {(ppl_o-ppl_s)/ppl_s*100:+.2f}%  ({ppl_s:.4f} -> {ppl_o:.4f})")
if ppl_b and ppl_g: print(f"  PPL increase compress+gauge vs base:  {(ppl_g-ppl_b)/ppl_b*100:+.2f}%  ({ppl_b:.4f} -> {ppl_g:.4f})")
PY

cat "$OUT_DIR/summary.txt" || true

# --------------------------------------------------------------------------
# Tarball
# --------------------------------------------------------------------------
ts="$(date -u +%Y%m%dT%H%M%SZ)"
TAR_PATH="$OUT_DIR/../paperB_${GPU_NAME}_${ts}.tar.gz"
tar czf "$TAR_PATH" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
echo "[paperB] tarball: $TAR_PATH ($(du -h "$TAR_PATH" | awk '{print $1}'))"
echo "$TAR_PATH" > "$OUT_DIR/tarball_path.txt"

if [ -n "${SHUTDOWN_AT_END:-}" ]; then
  echo "[paperB] requesting shutdown -h +1"
  sudo shutdown -h +1 || true
fi

echo "[paperB] DONE"
