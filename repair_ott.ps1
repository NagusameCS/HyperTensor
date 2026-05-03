<#
.SYNOPSIS
OTT Speculative Decode — Automated Repair & Calibration Suite
=============================================================
Fixes the two root causes of 0% acceptance:
  1. Degenerate geometry cache (samples_used=0, empty Christoffel symbols)
  2. Poisoned OD table (hidden states from BOS/EOS, not mid-generation)

Run from the repo root:
  .\repair_ott.ps1

Optional parameters let you override defaults without editing this file.
#>

param(
    [string]$ModelPath   = "models\smollm2-135m-instruct-q8_0.gguf",
    [string]$HostExe     = ".\build_host\geodessical.exe",
    [int]   $AxiomSamples = 512,
    [int]   $AxiomProbe   = 2048,
    [int]   $SweepTokens  = 48,
    [int]   $SweepReps    = 3,
    [switch]$SkipSweep    # pass -SkipSweep to skip the C3 calibration sweep
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$LASTEXITCODE = 0  # pre-init so strict-mode doesn't throw before first native exe

function Invoke-Step {
    param([string]$Label, [scriptblock]$Block)
    Write-Host "`n>>> $Label" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne $null) {
        Write-Error "Step failed with exit code $LASTEXITCODE — aborting."
    }
    Write-Host "    OK`n" -ForegroundColor Green
}

#  Preflight checks 
if (-not (Test-Path $HostExe)) {
    Write-Error "geodessical.exe not found at '$HostExe'. Run build_host.ps1 first."
}
if (-not (Test-Path $ModelPath)) {
    Write-Error "Model not found: $ModelPath"
}

Write-Host "============================================================" -ForegroundColor DarkCyan
Write-Host "  OTT Repair & Calibration Suite" -ForegroundColor DarkCyan
Write-Host "  Model  : $ModelPath" -ForegroundColor DarkCyan
Write-Host "  Samples: $AxiomSamples   Probe: $AxiomProbe" -ForegroundColor DarkCyan
Write-Host "============================================================`n" -ForegroundColor DarkCyan

#  STEP 1: Purge degenerate caches 
Invoke-Step "STEP 1: Purging degenerate caches" {
    foreach ($f in @("ott_geometry.bin", "ott_one_decode.bin")) {
        if (Test-Path $f) {
            Remove-Item $f -Force
            Write-Host "  Removed: $f"
        } else {
            Write-Host "  Already absent: $f"
        }
    }
}

#  STEP 2: Cold Riemannian geometry survey 
# Delete the cache file first (already done in Step 1), then run WITHOUT
# --axiom-no-cache so that cfg->reuse_cache=1 and the result gets saved
# to ott_geometry.bin at the end of Phase 5.
# Phases: 1=PCA, 2=Symmetry, 3=Christoffel, 4=Axioms, 5=Geodesic pilot
Invoke-Step "STEP 2: Cold Riemannian survey ($AxiomSamples samples, $AxiomProbe probe)" {
    & $HostExe $ModelPath `
        --ott-full `
        --axiom-samples $AxiomSamples `
        --axiom-probe   $AxiomProbe `
        --axiom-skip-geodesic `
        -p "Hello, how are you today?" `
        -n 4
}

#  Validate geometry 
Write-Host "    Validating geometry..." -ForegroundColor Yellow
$report = Get-Content "axiom_beta_report.json" -Raw | ConvertFrom-Json
$samplesUsed = $report.phase1_manifold.samples_used
$christoffelOk = $report.phase3_curvature.christoffel_computed
$metricPts = $report.phase3_curvature.metric_field_points

if ($samplesUsed -le 0) {
    Write-Error "GEOMETRY VALIDATION FAILED: samples_used=$samplesUsed (still 0). Phase 1 did not run."
}
if ($christoffelOk -ne 1) {
    Write-Warning "WARNING: christoffel_computed=$christoffelOk — GRC subspace may not initialise. Check axiom_beta_report.json."
}
Write-Host "    samples_used=$samplesUsed  christoffel=$christoffelOk  metric_pts=$metricPts" -ForegroundColor Green

#  STEP 3: Bake OD table with ChatML anchor states 
# The prompt must use the ChatML template so hidden states are captured from
# mid-generation positions (after the assistant turn marker), not BOS/EOS.
Invoke-Step "STEP 3: Baking OD table with ChatML anchor states" {
    # Build the prompt string using ChatML tokens. PowerShell backtick-n = newline.
    $chatPrompt = "<|im_start|>user`nExplain Newton's laws.<|im_end|>`n<|im_start|>assistant`n"
    & $HostExe $ModelPath `
        --ott-full `
        -p $chatPrompt `
        -n 4
}

if (-not (Test-Path "ott_one_decode.bin")) {
    Write-Error "OD VALIDATION FAILED: ott_one_decode.bin was not created by Step 3."
}
$odFile = Get-Item "ott_one_decode.bin"
if ($odFile.Length -le 0) {
    Write-Error "OD VALIDATION FAILED: ott_one_decode.bin is empty."
}
Write-Host "    OD cache present: ott_one_decode.bin ($($odFile.Length) bytes)" -ForegroundColor Green

#  Validate OD table 
Write-Host "    Validating OD table..." -ForegroundColor Yellow
$odProbe = & $HostExe $ModelPath `
    --ott-full --ott-speculative --ott-spec-batch 2 --ott-spec-thresh 0.45 `
    -p "Hello" -n 4 2>&1
$odLine = $odProbe | Select-String "\[SPEC-DBG\] od_tok="
if ($odLine) {
    $odStr = $odLine.ToString().Trim()
    Write-Host "    OD probe: $odStr" -ForegroundColor Green
    if ($odStr -match "od_tok=2" -or $odStr -match "piece='<\|im_end\|>'") {
        Write-Warning "OD table still proposes EOS (tok=2). The boundary guard in main.c will block it, but consider re-running Step 3 with a longer -n to bake more anchor states."
    }
} else {
    Write-Host "    (no SPEC-DBG line — OD may not be ready; check full output)" -ForegroundColor Yellow
}

#  STEP 4: C3 Calibration Sweep 
if ($SkipSweep) {
    Write-Host "`n>>> STEP 4: Calibration sweep SKIPPED (-SkipSweep passed)" -ForegroundColor Yellow
} else {
    Invoke-Step "STEP 4: Running C3 calibration sweep (tokens=$SweepTokens reps=$SweepReps)" {
        .\scripts\ott\calibration_sweep.ps1 -MaxTokens $SweepTokens -Reps $SweepReps
    }
}

#  Final smoke test 
Invoke-Step "STEP 5: Final smoke test" {
    $smokeOut = & $HostExe $ModelPath `
        --ott-full --ott-speculative --ott-spec-batch 2 --ott-spec-thresh 0.45 `
        --ott-rejection-log repair_rejection_log.tsv `
        -p "What is the capital of France?" `
        -n 32 2>&1

    $doneLine = $smokeOut | Select-String "\[SPEC\] Done"
    $grcLine  = $smokeOut | Select-String "\[SPEC\] GRC"
    Write-Host "    $doneLine" -ForegroundColor White
    Write-Host "    $grcLine"  -ForegroundColor White

    if (-not $doneLine) {
        Write-Error "Smoke test did not emit [SPEC] Done line."
    }

    if ($doneLine -match "Done:\s+(\d+)\s+tokens") {
        $tokCount = [int]$Matches[1]
        if ($tokCount -le 0) {
            Write-Error "Smoke test generated zero speculative tokens (Done: $tokCount)."
        }
    }

    if ($doneLine -match "acceptance_rate=([\d.]+)%") {
        $acc = [double]$Matches[1]
        if ($acc -ge 20.0) {
            Write-Host "    Acceptance rate: $acc% — PASS (>= 20%)" -ForegroundColor Green
        } else {
            Write-Warning "    Acceptance rate: $acc% — below 20%. GRC may need more correction rounds; try a longer -n run."
        }
    }
}

Write-Host "`n============================================================" -ForegroundColor DarkGreen
Write-Host "  REPAIR COMPLETE" -ForegroundColor DarkGreen
Write-Host "  Next: run D2 taxonomy on repair_rejection_log.tsv" -ForegroundColor DarkGreen
Write-Host "  .\.venv\Scripts\python.exe scripts\ott\rejection_taxonomy.py --log repair_rejection_log.tsv" -ForegroundColor Gray
Write-Host "============================================================`n" -ForegroundColor DarkGreen
