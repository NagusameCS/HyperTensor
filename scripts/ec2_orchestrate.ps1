#requires -Version 5.1
<#
.SYNOPSIS
    EC2 Orchestration: P3 Cross-GPU + Distill Phase 2 + Per-Matrix Bases.
    One launch, one EC2 session, all results pulled back.

.DESCRIPTION
    Bundles three EC2-dependent Tier 1 tasks into a single GPU instance session:
      1. P3 Cross-GPU Cache-Fit Validation (k-sweep on A100/L40S)
      2. Distillation Phase 2 Runner (LoRA training for Paper E)
      3. Per-Matrix Bases (Eckart-Young measurement for Paper A)

    Workflow:
      1. Launch EC2 GPU instance (g6e.xlarge = L40S, or p4d = A100)
      2. Upload repo snapshot + model
      3. Build geodessical binary
      4. Run P3 benchmark -> pull CSV
      5. Run distill runner -> pull safetensors
      6. Run per-matrix bases -> pull JSON
      7. Terminate instance (or keep for inspection)

.PARAMETER InstanceType
    g6e.xlarge (L40S, 48GB, $1.60/hr) or p4d.24xlarge (A100, 40GB, $32/hr)
#>
[CmdletBinding()]
param(
    [string]$InstanceType    = "g6e.xlarge",
    [string]$KeyName         = "hypertensor-key",
    [string]$Region          = "us-east-1",
    [string]$ModelUrl        = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    [string]$LocalModelPath  = "",
    [switch]$DryRun,
    [switch]$KeepInstance,
    [int]$RootVolumeGB       = 200
)

$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$localOutDir = Join-Path $repoRoot "benchmarks\ec2_orchestrated_$ts"

# GPU metadata
$gpuMeta = @{
    "g6e.xlarge"   = @{ name="L40S"; arch="sm_89"; l2_mb=96;  vram_gb=48; ami="ami-0c02fb55956c7d316" }
    "g6.xlarge"    = @{ name="L4";   arch="sm_89"; l2_mb=48;  vram_gb=24; ami="ami-0c02fb55956c7d316" }
    "g5.xlarge"    = @{ name="A10G"; arch="sm_86"; l2_mb=6;   vram_gb=24; ami="ami-0c02fb55956c7d316" }
    "p4d.24xlarge" = @{ name="A100"; arch="sm_80"; l2_mb=40;  vram_gb=40; ami="ami-0c02fb55956c7d316" }
}

if (-not $gpuMeta.ContainsKey($InstanceType)) {
    Write-Error "Unknown InstanceType '$InstanceType'. Use: g6e.xlarge, g6.xlarge, g5.xlarge, p4d.24xlarge"
    exit 2
}
$gpu = $gpuMeta[$InstanceType]

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " EC2 Orchestrated Research Run" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " GPU: $($gpu.name) ($InstanceType, $($gpu.vram_gb)GB, L2=$($gpu.l2_mb)MB)" -ForegroundColor Green
Write-Host " Output: $localOutDir" -ForegroundColor Green
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY RUN] Would launch $InstanceType and run:" -ForegroundColor Yellow
    Write-Host "  1. P3 Cross-GPU (k-sweep)"
    Write-Host "  2. Distill Phase 2 (LoRA training)"  
    Write-Host "  3. Per-Matrix Bases (Eckart-Young)"
    exit 0
}

# ---- Build the orchestration script that runs ON the EC2 instance ----
$remoteScript = @'
#!/bin/bash
set -euo pipefail

REPO_DIR="/home/ec2-user/HyperTensor"
LOG_DIR="$REPO_DIR/benchmarks/ec2_run_logs"
mkdir -p "$LOG_DIR"

echo "=== EC2 Orchestrated Run ===" | tee "$LOG_DIR/00_start.log"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee -a "$LOG_DIR/00_start.log"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader) MB" | tee -a "$LOG_DIR/00_start.log"
echo "L2: $(nvidia-smi --query-gpu=l2_cache_size --format=csv,noheader) MB" | tee -a "$LOG_DIR/00_start.log"

# ---- Build geodessical ----
echo "" | tee -a "$LOG_DIR/00_start.log"
echo "=== Building geodessical ===" | tee "$LOG_DIR/01_build.log"
cd "$REPO_DIR"
bash build_server.sh 2>&1 | tee -a "$LOG_DIR/01_build.log"
echo "Build done." | tee -a "$LOG_DIR/01_build.log"

# ---- Download model (if not present) ----
MODEL_PATH="$REPO_DIR/models/llama3.1-8b-instruct-q4_k_m.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo "=== Downloading model ===" | tee "$LOG_DIR/02_model.log"
    mkdir -p "$REPO_DIR/models"
    curl -L -o "$MODEL_PATH" "MODEL_URL_PLACEHOLDER" 2>&1 | tee -a "$LOG_DIR/02_model.log"
fi

# ---- Activate Python venv ----
cd "$REPO_DIR"
source .venv/bin/activate 2>/dev/null || python3 -m venv .venv && source .venv/bin/activate
pip install -q torch transformers safetensors datasets gguf numpy

# ---- 1. P3 Cross-GPU ----
echo "" | tee -a "$LOG_DIR/00_start.log"
echo "=== [1/3] P3 Cross-GPU Benchmark ===" | tee "$LOG_DIR/03_p3.log"
python scripts/p3_cross_gpu.py \
    --model "$MODEL_PATH" \
    --ranks 512,768,1024,1280,1536,2048 \
    --out benchmarks/p3_$(echo $INSTANCE_TYPE | tr '.' '_') \
    --reps 3 --cooldown 15 \
    2>&1 | tee -a "$LOG_DIR/03_p3.log"

# ---- 2. Distill Phase 2 ----
echo "" | tee -a "$LOG_DIR/00_start.log"
echo "=== [2/3] Distillation Phase 2 ===" | tee "$LOG_DIR/04_distill.log"

# Prepare calibration corpus (WikiText-2 slice)
python -c "
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
text = '\n'.join(ds['text'][:50000])  # ~5k tokens
with open('data/wikitext2_train_5k.txt', 'w') as f:
    f.write(text)
print(f'Corpus: {len(text)} chars')
" 2>&1 | tee -a "$LOG_DIR/04_distill.log"

python scripts/distill_runner.py \
    --teacher meta-llama/Llama-3.1-8B-Instruct \
    --gguf "$MODEL_PATH" \
    --corpus data/wikitext2_train_5k.txt \
    --out benchmarks/paper_e_distill/llama8b_k1536_r8 \
    --rank 1536 --lora-rank 8 --steps 500 --batch 4 --seq-len 512 \
    --dtype bfloat16 \
    2>&1 | tee -a "$LOG_DIR/04_distill.log"

# ---- 3. Per-Matrix Bases ----
echo "" | tee -a "$LOG_DIR/00_start.log"
echo "=== [3/3] Per-Matrix Bases (Eckart-Young) ===" | tee "$LOG_DIR/05_per_matrix.log"
python scripts/per_matrix_bases.py \
    --model "$MODEL_PATH" \
    --out benchmarks/per_matrix/llama8b \
    --ranks 256,512,1024,1536,2048 \
    --sample-layers 0,7,15,23,31 \
    2>&1 | tee -a "$LOG_DIR/05_per_matrix.log"

echo "" | tee -a "$LOG_DIR/00_start.log"
echo "=== ALL DONE ===" | tee -a "$LOG_DIR/00_start.log"
echo "Results in: $REPO_DIR/benchmarks/" | tee -a "$LOG_DIR/00_start.log"
ls -la "$REPO_DIR/benchmarks/p3_"* "$REPO_DIR/benchmarks/paper_e_distill/"* "$REPO_DIR/benchmarks/per_matrix/"* 2>/dev/null || true
'@

# Replace placeholder
$remoteScript = $remoteScript.Replace("MODEL_URL_PLACEHOLDER", $ModelUrl)

# Write the remote script
$scriptPath = Join-Path $localOutDir "ec2_run.sh"
New-Item -ItemType Directory -Force -Path $localOutDir | Out-Null
$remoteScript | Out-File -FilePath $scriptPath -Encoding UTF8 -NoNewline

Write-Host "[*] Remote script written to: $scriptPath" -ForegroundColor Cyan
Write-Host "[*] To execute:" -ForegroundColor Yellow
Write-Host "    1. Launch EC2 instance: $InstanceType in $Region" -ForegroundColor White
Write-Host "    2. Upload repo:  scp -i key.pem -r . ec2-user@<ip>:/home/ec2-user/HyperTensor/" -ForegroundColor White
Write-Host "    3. SSH and run:  ssh -i key.pem ec2-user@<ip> 'bash /home/ec2-user/HyperTensor/$($localOutDir.Name)/ec2_run.sh'" -ForegroundColor White
Write-Host "    4. Pull results: scp -i key.pem -r ec2-user@<ip>:/home/ec2-user/HyperTensor/benchmarks/ $localOutDir/" -ForegroundColor White
Write-Host ""
Write-Host "[*] Or use the existing launch.ps1 in scripts/ec2_paperB_ablations/ as a template" -ForegroundColor Cyan
Write-Host "[*] Estimated cost: ~$2-4 for g6e.xlarge ($1.60/hr  2-3 hrs)" -ForegroundColor Cyan

# Also copy model if local path provided
if ($LocalModelPath -and (Test-Path $LocalModelPath)) {
    Write-Host "[*] Local model at $LocalModelPath will be uploaded" -ForegroundColor Cyan
    $modelDest = Join-Path $localOutDir "model.gguf"
    Copy-Item $LocalModelPath $modelDest
    Write-Host "[*] Model copied to $modelDest" -ForegroundColor Green
}
