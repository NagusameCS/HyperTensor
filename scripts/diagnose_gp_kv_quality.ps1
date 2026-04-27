param(
  [string]$Model = "C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
  [string]$Exe = ".\build_host\geodessical.exe",
  [int]$Tokens = 128,
  [string]$OutDir = ".\benchmarks\gp_kv_diagnostic_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
)

$ErrorActionPreference = 'Stop'
$Prompt = "Write a Python function that returns prime numbers up to n."

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Run-Test([string]$Label, [string[]]$ExtraArgs) {
    $out = Join-Path $OutDir "$Label.txt"
    $err = Join-Path $OutDir "$Label.err.txt"
    Write-Host "Running: $Label"
    & $Exe $Model @ExtraArgs -p $Prompt -n $Tokens 1> $out 2> $err
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: exit=$($LASTEXITCODE)"
        return $null
    }
    $raw = Get-Content -Raw -Path $out
    $mDec = [regex]::Match($raw, 'Decode-only:\s*prefill\s*([\d.]+)\s*ms,\s*([\d.]+)\s*tok/s')
    $mGd = [regex]::Match($raw, '\[GD\]\s*(\d+)\s+tokens\s+in\s*([\d.]+)\s*ms\s*\(([\d.]+)\s*tok/s\)')
    $decode = if ($mDec.Success) { [double]$mDec.Groups[2].Value } else { $null }
    $prefill = if ($mDec.Success) { [double]$mDec.Groups[1].Value } else { $null }
    $overall = if ($mGd.Success) { [double]$mGd.Groups[3].Value } else { $null }
    Write-Host "  decode=$decode prefill=$prefill overall=$overall"
    return [pscustomobject]@{ label=$Label; decode=$decode; prefill=$prefill; overall=$overall; file=$out }
}

Write-Host "GP K/V Quality Diagnostic"
Write-Host "========================"
Write-Host "Prompt: $Prompt"
Write-Host "Tokens: $Tokens"
Write-Host ""

# 1) Baseline: no compression
Write-Host "PHASE 1: Baseline reference"
$b0 = Run-Test 'baseline' @('--temp', '0')

# 2) Full GP at k=2048 (current suspect regression)
Write-Host ""
Write-Host "PHASE 2: Full GP compression (k=2048)"
$g2 = Run-Test 'gp_k2048_full' @(
    '--axex-compress', '--axex-attn-only', '--axex-skip-o', '--axex-weight-pca',
    '--axex-compress-rank', '2048', '--temp', '0'
)

# 3) Full GP but with Q4 Pt disabled (F16 fallback)
Write-Host ""
Write-Host "PHASE 3: GP k=2048 with Pt F16 (no Q4)"
$env:AXEX_DISABLE_PT_Q4 = '1'
$g2_f16 = Run-Test 'gp_k2048_pt_f16' @(
    '--axex-compress', '--axex-attn-only', '--axex-skip-o', '--axex-weight-pca',
    '--axex-compress-rank', '2048', '--temp', '0'
)
Remove-Item Env:AXEX_DISABLE_PT_Q4 -ErrorAction SilentlyContinue

# 4) Lower rank to narrow regression: k=1536
Write-Host ""
Write-Host "PHASE 4: GP compression (k=1536)"
$g1 = Run-Test 'gp_k1536_full' @(
    '--axex-compress', '--axex-attn-only', '--axex-skip-o', '--axex-weight-pca',
    '--axex-compress-rank', '1536', '--temp', '0'
)

# 5) Even lower: k=1024
Write-Host ""
Write-Host "PHASE 5: GP compression (k=1024)"
$g0 = Run-Test 'gp_k1024_full' @(
    '--axex-compress', '--axex-attn-only', '--axex-skip-o', '--axex-weight-pca',
    '--axex-compress-rank', '1024', '--temp', '0'
)

# Summarize
Write-Host ""
Write-Host "DIAGNOSTIC SUMMARY"
Write-Host "=================="
if ($b0) {
    Write-Host ("Baseline decode: {0:N2} tok/s (reference)" -f $b0.decode)
    $pct1024 = if ($b0.decode) { 100.0 * $g0.decode / $b0.decode } else { $null }
    $pct1536 = if ($b0.decode) { 100.0 * $g1.decode / $b0.decode } else { $null }
    $pct2048 = if ($b0.decode) { 100.0 * $g2.decode / $b0.decode } else { $null }
    $pct2048_f16 = if ($b0.decode) { 100.0 * $g2_f16.decode / $b0.decode } else { $null }
    Write-Host ""
    if ($g0) { Write-Host ("k=1024 decode: {0:N2} tok/s ({1:N1}% of baseline)" -f $g0.decode, $pct1024) }
    if ($g1) { Write-Host ("k=1536 decode: {0:N2} tok/s ({1:N1}% of baseline)" -f $g1.decode, $pct1536) }
    if ($g2) { Write-Host ("k=2048 decode (Q4 Pt): {0:N2} tok/s ({1:N1}% of baseline)" -f $g2.decode, $pct2048) }
    if ($g2_f16) { Write-Host ("k=2048 decode (F16 Pt): {0:N2} tok/s ({1:N1}% of baseline)" -f $g2_f16.decode, $pct2048_f16) }
    Write-Host ""
    if ($pct2048 -and $pct2048_f16) {
        $delta = $pct2048_f16 - $pct2048
        Write-Host ("Q4 Pt impact: {0:+N1}% (F16 - Q4)" -f $delta)
    }
}

Write-Host ""
Write-Host "Output directory: $OutDir"
Write-Host "Investigation files:"
Get-ChildItem -Path $OutDir -Filter "*.txt" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "  $_"
}
