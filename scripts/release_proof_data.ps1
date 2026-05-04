#!/usr/bin/env pwsh
# scripts/release_proof_data.ps1
#
# Bundle all paper proof data (NCU CSVs, sweep summaries, rho data, plots)
# into a 7z LZMA2 archive and upload to a GitHub release.
#
# Usage:
#   .\scripts\release_proof_data.ps1                    # auto-tag from git HEAD
#   .\scripts\release_proof_data.ps1 -Tag v0.3-proof    # explicit tag
#   .\scripts\release_proof_data.ps1 -DryRun            # compress only, no upload
#
# Requirements: 7-Zip at C:\Program Files\7-Zip\7z.exe, gh CLI authenticated.
#
# Compression note: NCU CSVs are repetitive ASCII (~200 rows of identical column headers
# per file). 7z LZMA2 -mx=9 achieves ~98% compression (5.9 MB -> 112 KB in testing).
# Use this script rather than committing raw CSVs to avoid bloating the repo history.

param(
    [string] $Tag     = "",
    [switch] $DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ROOT = Split-Path $PSScriptRoot -Parent
$SZ   = "C:\Program Files\7-Zip\7z.exe"

if (-not (Test-Path $SZ)) {
    Write-Error "7-Zip not found at $SZ. Install with: winget install 7zip.7zip"
}

#  Tag resolution 
if ($Tag -eq "") {
    $commit = (git -C $ROOT rev-parse --short HEAD 2>&1).Trim()
    $Tag    = "proof-data-$commit"
}
Write-Host "[release] Tag: $Tag"

#  Collect source paths 
$DataDirs = @(
    "$ROOT\docs\figures\paper-a\ncu_sweep",
    "$ROOT\docs\figures\paper-a\expB_thrash",
    "$ROOT\docs\figures\paper-e\rho_sweep"
)

$Sources = @()
foreach ($dir in $DataDirs) {
    if (Test-Path $dir) {
        $Sources += $dir
    } else {
        Write-Warning "[release] Directory not found (skipping): $dir"
    }
}

if ($Sources.Count -eq 0) {
    Write-Error "[release] No proof data directories found."
}

#  Build archive 
$Archive = "$env:TEMP\hypertensor-$Tag.7z"
if (Test-Path $Archive) { Remove-Item $Archive -Force }

Write-Host "[release] Compressing $($Sources.Count) directories -> $Archive"
$szArgs = @("a", "-t7z", "-mx=9", "-mmt=on", "-mqs=on", "-bd", $Archive) + $Sources
& $SZ @szArgs | Select-String -Pattern "Everything|Error|Warning"

$origBytes = ($Sources | ForEach-Object { Get-ChildItem $_ -Recurse -File } | Measure-Object -Property Length -Sum).Sum
$compBytes = (Get-Item $Archive).Length
Write-Host ("[release] Original: {0:N0} KB   Compressed: {1:N0} KB   Ratio: {2:N1}%" `
    -f ($origBytes/1KB), ($compBytes/1KB), ($compBytes/$origBytes*100))

if ($DryRun) {
    Write-Host "[release] DryRun: archive at $Archive --- skipping GH upload."
    exit 0
}

#  Create GH release and upload 
$CommitMsg  = (git -C $ROOT log -1 --pretty=%s HEAD 2>&1).Trim()
$Date       = Get-Date -Format "yyyy-MM-dd"
$RelTitle   = "Proof data $Tag ($Date)"
$RelNotes   = @"
## Proof data bundle: $Tag

Losslessly compressed (7z LZMA2 -mx=9) NCU measurement data and sweep
summaries for reproducing Paper A (GRC) and Paper E (GRC-Light) results.

### Contents
- ``docs/figures/paper-a/ncu_sweep/`` --- multi-k NCU sweep (Paper A §sec:cachefit-ncu)
- ``docs/figures/paper-a/expB_thrash/`` --- Exp B L2 thrash sweep (Paper A §sec:falsification P2)
- ``docs/figures/paper-e/rho_sweep/`` --- rho measurement (Paper E §sec:gapbound)

### Key results
- **Exp A (multi-k):** GRC L2 hit-rate +3.9 pp above baseline (7.5% -> 11.5%), flat in k
- **Exp B (thrash):** DRAM speedup proxy = 1.194 across Delta = 0/8/16/22 MB (flat --- structural)
- **L2 verdict:** CONSISTENT-OR-FUSION (kernel fusion, not L2 residency)
- **rho (Paper E):** mean_rho = 0.1340 at k=1024, r=8 (Llama-3.1-8B Q4_K_M, 32 layers)

### Reproduce
\`\`\`powershell
# Download the .7z file from this release, then:
& "C:\Program Files\7-Zip\7z.exe" x hypertensor-$Tag.7z -o"<repo_root>" -y
# Re-run analysis:
python scripts/paperA_proof/parse_expB.py
python scripts/paperA_proof/decide_l2_mystery.py
\`\`\`

Built from commit: $CommitMsg
"@

Write-Host "[release] Creating GH release: $RelTitle"
gh release create $Tag $Archive `
    --title $RelTitle `
    --notes $RelNotes `
    --repo NagusameCS/HyperTensor

Write-Host "[release] Done. Archive: $Archive"
Write-Host "[release] View at: https://github.com/NagusameCS/HyperTensor/releases/tag/$Tag"
