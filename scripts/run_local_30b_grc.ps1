# run_local_30b_grc.ps1
# Load Ollama-pulled 30B-class GGUFs (gemma4:31b, qwen3.5:35b) with geodessical
# under GRC/Axex compression so they fit in 8 GB VRAM.
#
# Usage:
#   .\scripts\run_local_30b_grc.ps1 -Model gemma4:31b
#   .\scripts\run_local_30b_grc.ps1 -Model qwen3.5:35b -Prompt "explain attention in 1 sentence"
#
# The fact that these fit at all in an 8 GB consumer GPU IS the data point.
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$Model,
    [string]$Prompt = "Briefly explain Riemannian curvature.",
    [int]$Rank = 128,
    [int]$NPredict = 64,
    [int]$Ctx = 4096,
    [string]$LogDir = "C:\Users\legom\HyperTensor\benchmarks\local_30b_grc",
    [switch]$PplOnly,
    [switch]$NoOffload
)

$ErrorActionPreference = 'Stop'

# ---- Resolve GGUF blob path from Ollama manifest ---------------------------
$modelsRoot = "$env:USERPROFILE\.ollama\models"
$mfPath = Join-Path $modelsRoot ("manifests\registry.ollama.ai\library\" + ($Model -replace ':','\'))
if (-not (Test-Path $mfPath)) {
    throw "Ollama manifest not found: $mfPath  (is the pull complete?)"
}
$mf = Get-Content $mfPath -Raw | ConvertFrom-Json
$ggufLayer = $mf.layers | Where-Object { $_.mediaType -match 'model' } | Select-Object -First 1
if (-not $ggufLayer) { throw "No model layer in manifest" }
$digest = $ggufLayer.digest -replace ':','-'
$blob   = Join-Path $modelsRoot "blobs\$digest"
if (-not (Test-Path $blob)) { throw "Blob missing: $blob" }
$sizeGB = [math]::Round((Get-Item $blob).Length/1GB,2)
Write-Host "[run] model = $Model  blob=$blob  ($sizeGB GB)"

# ---- Build geodessical args ------------------------------------------------
$exe = "C:\Users\legom\HyperTensor\build_host\geodessical.exe"
if (-not (Test-Path $exe)) { throw "geodessical.exe not built: $exe" }

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$tag = ($Model -replace '[:/]','_')
$logOut = Join-Path $LogDir "${tag}_${ts}.log"
$logErr = Join-Path $LogDir "${tag}_${ts}.err"

$args = @(
    $blob,
    '--axex-compress',
    '--axex-ffn-compress',
    '--axex-attn-svd',
    '--axex-compress-rank', $Rank,
    '--ctx-size', $Ctx
)
if (-not $NoOffload) { $args += '--axex-offload' }

if ($PplOnly) {
    $args += '--ppl-eval'
} else {
    $args += @('-p', $Prompt, '-n', $NPredict)
}

Write-Host "[run] cmd: $exe $($args -join ' ')"
Write-Host "[run] log: $logOut"

# ---- VRAM snapshot before/after -------------------------------------------
$vramBefore = (& nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits) -join ' / '
Write-Host "[run] VRAM before (used/free MiB): $vramBefore"

$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $exe @args 1>$logOut 2>$logErr
$rc = $LASTEXITCODE
$sw.Stop()

$vramPeak = (& nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits) -join ' / '
Write-Host "[run] VRAM after  (used/free MiB): $vramPeak"
Write-Host "[run] exit=$rc  elapsed=$([int]$sw.Elapsed.TotalSeconds)s"

if ($rc -ne 0) {
    Write-Host "[run] ERROR — last 20 lines of stderr:"
    Get-Content $logErr -Tail 20
    exit $rc
}

Write-Host "[run] OK — last 10 lines of stdout:"
Get-Content $logOut -Tail 10
