# scripts/paperA_proof/expB_thrash_sweep.ps1
# -----------------------------------------------------------------------------
# Experiment B (P2) on RTX 4070 Laptop (Ada).
#
# For each Delta in {0, 8, 16, 22} MB:
#   1. Launch l2_thrash.exe <Delta> in a background process so it streams
#      Delta MB through L2 continuously, evicting persistent lines.
#   2. Run geodessical baseline + GRC k=1024 with NCU instrumentation.
#   3. Capture l2 hit-rate and dram bytes.
#   4. Kill the thrash process and cool down.
#
# Predicted outcome (cache-fit hypothesis):
#   - GRC speed-up flat for Delta = 0
#   - GRC speed-up shrinks as Delta grows
#   - GRC speed-up vanishes once Delta >= 22 MB (32 - 22 = 10 MB < S(1024))
# Predicted outcome (fusion hypothesis): GRC speed-up flat across all Delta.
# -----------------------------------------------------------------------------
[CmdletBinding()]
param(
    [string]$Model      = "C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    [string]$Exe        = "C:\Users\legom\HyperTensor\build_host\geodessical.exe",
    [string]$Thrasher   = "C:\Users\legom\HyperTensor\scripts\paperA_proof\l2_thrash.exe",
    [int[]] $DeltasMB   = @(0, 8, 16, 22),
    [int]   $NReps      = 2,
    [int]   $Tokens     = 16,
    [string]$Prompt     = "The quick brown fox jumps over the lazy dog.",
    [string]$OutDir     = "C:\Users\legom\HyperTensor\docs\figures\paper-a\expB_thrash",
    [int]   $CooldownSec = 20,
    [switch]$NoNcu
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if (-not (Test-Path $Exe))     { throw "Exe missing: $Exe" }
if (-not (Test-Path $Model))   { throw "Model missing: $Model" }
if ((-not (Test-Path $Thrasher)) -and ($DeltasMB | Where-Object { $_ -gt 0 })) {
    throw "Thrasher missing: $Thrasher. Run scripts/paperA_proof/build_thrash.ps1 first."
}

$metrics = @(
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_op_read.sum",
    "dram__bytes_read.sum",
    "gpu__time_duration.sum"
) -join ","

function Start-Thrasher {
    param([int]$DeltaMB)
    if ($DeltaMB -le 0) { return $null }
    Write-Host "[expB] starting thrasher Delta=$DeltaMB MB"
    return Start-Process -FilePath $Thrasher -ArgumentList "$DeltaMB","0" -PassThru -WindowStyle Hidden
}

function Stop-Thrasher {
    param($Proc)
    if ($null -eq $Proc) { return }
    if (-not $Proc.HasExited) {
        Write-Host "[expB] stopping thrasher pid=$($Proc.Id)"
        Stop-Process -Id $Proc.Id -Force
    }
}

function Invoke-GeodCapture {
    param([string]$Label, [string[]]$ExtraArgs, [string]$OutBase)
    $stdoutLog = "$OutBase.stdout.log"
    $ncuCsv = "$OutBase.ncu.csv"
    $args = @($Model, "-p", $Prompt, "-n", $Tokens, "--temp", "0") + $ExtraArgs
    if ($NoNcu) {
        Write-Host "[expB] (no-ncu) $Label $Exe $($args -join ' ')"
        & $Exe @args 2>&1 | Tee-Object -FilePath $stdoutLog
    } else {
        $ncuArgs = @(
            "--metrics", $metrics,
            "--target-processes", "all",
            "--launch-skip", "50",
            "--launch-count", "200",
            "--kernel-name", "regex:gemv",
            "--csv",
            "--log-file", $ncuCsv,
            "--",
            $Exe
        ) + $args
        Write-Host "[expB] $Label ncu $($ncuArgs -join ' ')"
        & ncu @ncuArgs 2>&1 | Tee-Object -FilePath $stdoutLog
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "geod capture $Label exited $LASTEXITCODE"
    }
}

foreach ($delta in $DeltasMB) {
    foreach ($cond in @("baseline", "grc_k1024")) {
        for ($r = 1; $r -le $NReps; $r++) {
            $label = "delta${delta}_${cond}_rep${r}"
            $outBase = Join-Path $OutDir $label
            if (Test-Path "$outBase.ncu.csv") { Write-Host "[skip] $label"; continue }

            $thrasher = Start-Thrasher -DeltaMB $delta
            try {
                Start-Sleep -Seconds 3   # let thrasher warm up
                if ($cond -eq "baseline") {
                    Invoke-GeodCapture $label @() $outBase
                } else {
                    $extra = @("--axex-compress","--axex-attn-only","--axex-weight-pca",
                               "--axiom-skip-geodesic","--axex-skip-o","--axex-compress-rank","1024")
                    Invoke-GeodCapture $label $extra $outBase
                }
            }
            finally {
                Stop-Thrasher $thrasher
            }
            Start-Sleep -Seconds $CooldownSec
        }
    }
}

Write-Host "Exp B complete. Outputs: $OutDir"
Write-Host "Run scripts/paperA_proof/parse_expB.py to aggregate."
