# scripts/paperA_proof/multi_k_ncu_sweep.ps1
# -----------------------------------------------------------------------------
# Experiment A (P1): Multi-k NCU sweep with replicates.
#
# Hypothesis: if the GRC super-baseline is L2-cache-fit driven, then the
# attention-projection kernel L2 hit-rate should be a non-increasing
# function of k in the regime S(k) <= C_L2 = 32 MB, and drop sharply
# once S(k) > C_L2 (predicted around k ~ 1280-1536 on a 4070 32MB L2).
#
# What this script does:
#   1. Runs ncu against geodessical.exe at k in {baseline, 384, 512, 768,
#      1024, 1280, 1536}, n=N_REPS independent invocations each.
#   2. Captures lts__t_sector_hit_rate.pct, lts__t_sectors_op_read.sum,
#      dram__bytes_read.sum, gpu__time_duration.sum, sm__warps_active...
#   3. Writes one CSV per (k, rep) under docs/figures/paper-a/ncu_sweep/.
#   4. Aggregates into a single summary JSON with mean +/- stdev for the
#      attention-projection kernel (gemv_q4_k grid=(512,1,1)).
#
# This script does NOT fabricate measurements. If ncu fails the script
# errors and writes the partial state.
# -----------------------------------------------------------------------------

[CmdletBinding()]
param(
    [string]$Model    = "C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    [string]$Exe      = "C:\Users\legom\HyperTensor\build_host\geodessical.exe",
    [int[]] $Ks       = @(384, 512, 768, 1024, 1280, 1536),
    [int]   $NReps    = 2,
    [int]   $Tokens   = 16,
    [string]$Prompt   = "The quick brown fox jumps over the lazy dog.",
    [string]$OutDir   = "C:\Users\legom\HyperTensor\docs\figures\paper-a\ncu_sweep",
    [int]   $CooldownSec = 20
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command ncu -ErrorAction SilentlyContinue)) {
    throw "ncu not on PATH"
}
if (-not (Test-Path $Exe))   { throw "Exe missing: $Exe" }
if (-not (Test-Path $Model)) { throw "Model missing: $Model" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$metrics = @(
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors_op_read.sum",
    "dram__bytes_read.sum",
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active"
) -join ","

$startedAt = Get-Date
"=== Multi-k NCU sweep started $startedAt ===" | Tee-Object -FilePath (Join-Path $OutDir "sweep.log") -Append

function Invoke-NcuRun {
    param(
        [string]$Label,
        [string[]]$ExtraArgs,
        [string]$OutCsv
    )
    $args = @(
        "--metrics", $metrics,
        "--target-processes", "all",
        "--launch-skip", "50",
        "--launch-count", "200",
        "--kernel-name", "regex:gemv",
        "--csv",
        "--log-file", $OutCsv,
        "--",
        $Exe, $Model, "-p", $Prompt, "-n", $Tokens, "--temp", "0"
    ) + $ExtraArgs
    Write-Host "[ncu] $Label -> $OutCsv"
    & ncu @args 2>&1 | Tee-Object -FilePath (Join-Path $OutDir "$Label.stdout") -Append | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "ncu run $Label exited $LASTEXITCODE (continuing)"
    }
}

# --- baseline (no GRC) ---
for ($r = 1; $r -le $NReps; $r++) {
    $label = "baseline_rep${r}"
    $csv   = Join-Path $OutDir "$label.csv"
    if (Test-Path $csv) { Write-Host "[skip exists] $label"; continue }
    Invoke-NcuRun $label @() $csv
    Start-Sleep -Seconds $CooldownSec
}

# --- multi-k GRC ---
foreach ($k in $Ks) {
    for ($r = 1; $r -le $NReps; $r++) {
        $label = "k${k}_rep${r}"
        $csv   = Join-Path $OutDir "$label.csv"
        if (Test-Path $csv) { Write-Host "[skip exists] $label"; continue }
        $extra = @(
            "--axex-compress",
            "--axex-attn-only",
            "--axex-weight-pca",
            "--axiom-skip-geodesic",
            "--axex-skip-o",
            "--axex-compress-rank", "$k"
        )
        Invoke-NcuRun $label $extra $csv
        Start-Sleep -Seconds $CooldownSec
    }
}

$endedAt = Get-Date
$dur = $endedAt - $startedAt
"=== Multi-k NCU sweep ended $endedAt (duration $($dur.TotalMinutes.ToString('F1')) min) ===" | Tee-Object -FilePath (Join-Path $OutDir "sweep.log") -Append
Write-Host "Sweep complete. CSVs under: $OutDir"
Write-Host "Run scripts/paperA_proof/parse_multi_k.py next to aggregate."
