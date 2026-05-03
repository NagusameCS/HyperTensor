#!/usr/bin/env pwsh
# scripts/reproduce_proof_data.ps1
#
# Download and extract a proof data release from GitHub, then re-run
# the analysis pipeline to reproduce Paper A and Paper E measurements.
#
# Usage:
#   .\scripts\reproduce_proof_data.ps1                        # latest proof-data release
#   .\scripts\reproduce_proof_data.ps1 -Tag proof-data-6faf23c
#   .\scripts\reproduce_proof_data.ps1 -ExtractOnly          # skip analysis pipeline
#
# Requirements: gh CLI authenticated, 7-Zip, Python venv at .venv/

param(
    [string] $Tag         = "",
    [switch] $ExtractOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = Split-Path $PSScriptRoot -Parent
$SZ   = "C:\Program Files\7-Zip\7z.exe"
$PY   = "$ROOT\.venv\Scripts\python.exe"

if (-not (Test-Path $SZ))  { Write-Error "7-Zip not found at $SZ" }
if (-not (Test-Path $PY))  { Write-Error "Python venv not found at $PY" }

#  Resolve tag 
if ($Tag -eq "") {
    Write-Host "[reproduce] Fetching latest proof-data release tag..."
    $Tag = gh release list --repo NagusameCS/HyperTensor --limit 20 --json tagName `
        --jq '[.[] | select(.tagName | startswith("proof-data-"))] | .[0].tagName' 2>&1
    $Tag = $Tag.Trim()
    if ($Tag -eq "" -or $Tag -eq "null") {
        Write-Error "No proof-data-* release found. Run scripts\release_proof_data.ps1 first."
    }
    Write-Host "[reproduce] Using latest: $Tag"
}

#  Download archive 
$Archive = "$env:TEMP\hypertensor-$Tag.7z"
if (Test-Path $Archive) {
    Write-Host "[reproduce] Archive already cached: $Archive"
} else {
    Write-Host "[reproduce] Downloading $Tag..."
    gh release download $Tag `
        --repo NagusameCS/HyperTensor `
        --pattern "*.7z" `
        --dir $env:TEMP
    # rename to canonical name if gh saved with a different name
    $downloaded = Get-ChildItem $env:TEMP -Filter "hypertensor-*.7z" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($downloaded.FullName -ne $Archive) {
        Move-Item $downloaded.FullName $Archive -Force
    }
}

#  Extract 
Write-Host "[reproduce] Extracting to $ROOT..."
& $SZ x $Archive "-o$ROOT" -y | Select-String -Pattern "Everything|Error|Warning"

if ($ExtractOnly) {
    Write-Host "[reproduce] ExtractOnly: skipping analysis pipeline."
    exit 0
}

#  Re-run analysis 
Write-Host ""
Write-Host "[reproduce] Running parse_expB.py..."
& $PY "$ROOT\scripts\paperA_proof\parse_expB.py"

Write-Host ""
Write-Host "[reproduce] Running decide_l2_mystery.py..."
& $PY "$ROOT\scripts\paperA_proof\decide_l2_mystery.py"

Write-Host ""
Write-Host "[reproduce] Running grc_distill.py --print-rho (offline, reads from rho_summary.json)..."
Write-Host "[reproduce]   rho_summary.json is pre-computed; skip re-running if model not available."
if (Test-Path "$ROOT\docs\figures\paper-e\rho_sweep\rho_summary.json") {
    & $PY "$ROOT\scripts\paperA_proof\update_paperE_rho.py"
    Write-Host "[reproduce] Paper E tex updated."
} else {
    Write-Host "[reproduce] WARNING: rho_summary.json not found — skipping Paper E update."
}

Write-Host ""
Write-Host "[reproduce] All done. Key outputs:"
Write-Host "  Paper A verdict : docs\figures\paper-a\ncu_sweep\l2_verdict.md"
Write-Host "  Exp B summary   : docs\figures\paper-a\expB_thrash\expB_summary.json"
Write-Host "  Exp B plot      : docs\figures\paper-a\expB_thrash\expB_dram_vs_delta.png"
Write-Host "  rho summary     : docs\figures\paper-e\rho_sweep\rho_summary.json"
