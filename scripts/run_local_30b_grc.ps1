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
    [int]$Ctx = 512,
    [string]$LogDir = "C:\Users\legom\HyperTensor\benchmarks\local_30b_grc",
    [switch]$PplOnly,
    [switch]$NoOffload,
    [float]$MaxErr = 0,
    [switch]$Baseline,
    [switch]$WeightPcaOnly,
    [switch]$LoadOnly
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

$geoArgs = @(
    $blob,
    '--ctx-size', $Ctx
)
if (-not $Baseline) {
    if ($WeightPcaOnly) {
        $geoArgs += @(
            '--axex-weight-pca-only',
            '--axex-compress-rank', $Rank,
            '--axex-compress-max-err', $MaxErr
        )
    } else {
        # Safer path for 30B GQA models: compression with quality gate and raw fallback.
        $geoArgs += @(
            '--axex-compress',
            '--axex-attn-svd',
            '--axex-compress-rank', $Rank,
            '--axex-compress-max-err', $MaxErr
        )
    }
}
if (-not $NoOffload) { $geoArgs += '--axex-offload' }

if ($PplOnly) {
    $geoArgs += '--ppl-eval'
} else {
    $tokens = $NPredict
    if ($LoadOnly) { $tokens = 0 }
    $geoArgs += @('-p', $Prompt, '-n', $tokens)
}

Write-Host "[run] cmd: $exe $($geoArgs -join ' ')"
Write-Host "[run] log: $logOut"

# ---- VRAM snapshot before/after -------------------------------------------
$vramBefore = (& nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits) -join ' / '
Write-Host "[run] VRAM before (used/free MiB): $vramBefore"

$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $exe @geoArgs 1>$logOut 2>$logErr
$rc = $LASTEXITCODE
$sw.Stop()

$vramPeak = (& nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits) -join ' / '
Write-Host "[run] VRAM after  (used/free MiB): $vramPeak"
Write-Host "[run] exit=$rc  elapsed=$([int]$sw.Elapsed.TotalSeconds)s"

$loadedLine = Select-String -Path $logOut -Pattern "\[GD\] Model loaded in" -ErrorAction SilentlyContinue | Select-Object -Last 1
$uploadLine = Select-String -Path $logOut -Pattern "\[GPU\] Uploaded" -ErrorAction SilentlyContinue | Select-Object -Last 1
$offloadLine = Select-String -Path $logOut -Pattern "VRAM tight.*CPU offload|offload" -ErrorAction SilentlyContinue | Select-Object -First 1
$genLine = Select-String -Path $logOut -Pattern "\[GD\] Generating" -ErrorAction SilentlyContinue | Select-Object -Last 1
if ($loadedLine) { Write-Host "[run] $($loadedLine.Line)" }
if ($uploadLine) { Write-Host "[run] $($uploadLine.Line)" }
if ($offloadLine) { Write-Host "[run] offload-note: $($offloadLine.Line)" }
if ($genLine) { Write-Host "[run] $($genLine.Line)" }

# Geodessical v0.6 currently exits 1 after -n 0 even though load and setup complete.
if ($LoadOnly -and $rc -ne 0 -and $genLine -and $genLine.Line -match "Generating 0 tokens") {
    Write-Host "[run] NOTE: treating exit=$rc as success for LoadOnly mode (known -n 0 quirk)."
    $rc = 0
}

if ($rc -ne 0) {
    Write-Host "[run] ERROR — last 20 lines of stderr:"
    Get-Content $logErr -Tail 20
    exit $rc
}

Write-Host "[run] OK — last 10 lines of stdout:"
Get-Content $logOut -Tail 20
# Extract timing stats if present
$timingLines = Select-String -Path $logOut -Pattern "tok/s|ms/tok|tokens/s|timing|elapsed|TIMING|Generated|generated" -ErrorAction SilentlyContinue
if ($timingLines) {
    Write-Host "[run] Timing stats:"
    $timingLines | ForEach-Object { Write-Host "  $($_.Line)" }
}
