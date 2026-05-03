# scripts/paper_a/multi_k_sweep.ps1
# Paper A Empirical — Multi-rank Pareto sweep
# ============================================
# For each k in K_VALUES runs geodessical.exe on a locked prompt set,
# measures tok/s (throughput) and optionally perplexity, and builds the
# Pareto frontier data for Figure 2 (throughput  PPL vs rank).
#
# Produces:
#   <OutDir>/multi_k_results.csv     — raw per-run data
#   <OutDir>/multi_k_summary.json    — mean ± SD per k
#   <OutDir>/multi_k_report.md       — narrative table for Paper A
#
# Usage:
#   .\scripts\paper_a\multi_k_sweep.ps1
#   .\scripts\paper_a\multi_k_sweep.ps1 -Model <path> -Reps 3 -WithPPL
[CmdletBinding()]
param(
    [string]$Model = "C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    [string]$OutDir    = "C:\Users\legom\HyperTensor\benchmarks\paper_a_multi_k",
    [int]   $NTokens   = 64,
    [int]   $Reps      = 1,
    [switch]$WithPPL           # add --ppl-eval runs (slow; one per k)
)
$ErrorActionPreference = "Stop"
# Use geodessical2.exe — gives proper decode tok/s output without auto-PPL mode
$exe = "C:\Users\legom\HyperTensor\build_host\geodessical2.exe"
if (-not (Test-Path $exe)) { throw "geodessical.exe not found" }
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

#  locked validation prompts (must not change between runs) 
$Prompts = @(
    "Explain the water cycle in three sentences.",
    "What is the capital of France and why is it famous?",
    "Write a Python function that computes Fibonacci numbers.",
    "Describe how a transformer decoder generates tokens step by step.",
    "Summarise the key ideas of Newton's laws of motion.",
    "How does gradient descent work in neural network training?",
    "Compare HTTP/1.1 and HTTP/2 in terms of multiplexing.",
    "What is the cosmic microwave background radiation?"
)

#  rank sweep grid 
$KValues = @(64, 128, 256, 512, 1024)

#  CSV header 
$csvPath = Join-Path $OutDir "multi_k_results.csv"
"k,prompt_idx,rep,tok_s,n_tokens,exit_code" | Out-File $csvPath -Encoding utf8 -Force

# regex to extract tok/s from geodessical stderr
# geodessical2: "[GD] Decode-only: ... 250.5 tok/s"  or  "[TpF] @ 250.5 tok/s:"
# geodessical:  "tps=99.5" (PPL mode)
$re_toks = [regex]'(?:tps=([\d.]+)|([\d.]+)\s+tok(?:ens)?/s)'
$re_gen  = [regex]'Generating\s+(\d+)\s+tokens'

$allResults = [System.Collections.Generic.List[PSObject]]::new()

$totalRuns = ($KValues.Count + 1) * $Prompts.Count * $Reps  # +1 for baseline k=∞
$done = 0

function Run-GeoAndParse {
    param([string[]]$GeoArgs, [string]$Tag)
    # Use Start-Process with an explicitly quoted single command line.
    # This avoids PS native-command stderr warnings being promoted to errors
    # under strict error preference settings.
    $tmpOut = [System.IO.Path]::GetTempFileName()
    $tmpErr = [System.IO.Path]::GetTempFileName()
    try {
        $quoted = @()
        foreach ($a in $GeoArgs) {
            if ($a -match '[\s"]') {
                $quoted += '"' + ($a -replace '"', '\"') + '"'
            } else {
                $quoted += $a
            }
        }
        $argLine = ($quoted -join ' ')
        $proc = Start-Process -FilePath $exe -ArgumentList $argLine `
                    -NoNewWindow -PassThru `
                    -RedirectStandardOutput $tmpOut `
                    -RedirectStandardError  $tmpErr
        $proc.WaitForExit()
        $exitCode = $proc.ExitCode
        $text = (Get-Content $tmpOut -Raw -ErrorAction SilentlyContinue) + "`n" +
                (Get-Content $tmpErr -Raw -ErrorAction SilentlyContinue)
    } finally {
        Remove-Item $tmpOut,$tmpErr -ErrorAction SilentlyContinue
    }
    $tokM  = $re_toks.Match($text)
    # Group 1 = tps=N format, Group 2 = N tok/s format
    $tokS  = if ($tokM.Success) {
        if ($tokM.Groups[1].Value -ne '') { [double]$tokM.Groups[1].Value }
        else { [double]$tokM.Groups[2].Value }
    } else { $null }
    $genM  = $re_gen.Match($text)
    $nToks = if ($genM.Success) { [int]$genM.Groups[1].Value } else { 0 }
    return [PSCustomObject]@{
        TokS    = $tokS
        NTokens = $nToks
        ExitCode= $exitCode
        Text    = $text
    }
}

Write-Host "[multi_k] Starting sweep: $($KValues.Count) rank values  $($Prompts.Count) prompts  $Reps reps" -ForegroundColor Cyan
Write-Host "[multi_k] Model: $Model"
Write-Host "[multi_k] Baseline (no compression) will run with standard args`n"

#  Baseline (no axex flags) 
Write-Host "[multi_k] === BASELINE (k=inf) ===" -ForegroundColor Yellow
$baseResults = @()
foreach ($pi in 0..($Prompts.Count-1)) {
    $p = $Prompts[$pi]
    for ($rep = 0; $rep -lt $Reps; $rep++) {
        $done++
        Write-Progress -Activity "multi_k_sweep" -Status "Baseline prompt=$pi rep=$rep" -PercentComplete ([int](100*$done/$totalRuns))
        $geoArgs = @($Model, '--ctx-size', '512', '-p', $p, '-n', $NTokens, '--temp', '0')
        $r = Run-GeoAndParse -GeoArgs $geoArgs -Tag "baseline"
        $row = [PSCustomObject]@{ k="baseline"; prompt_idx=$pi; rep=$rep; tok_s=$r.TokS; n_tokens=$r.NTokens; exit_code=$r.ExitCode }
        $allResults.Add($row)
        "baseline,$pi,$rep,$($r.TokS),$($r.NTokens),$($r.ExitCode)" | Out-File $csvPath -Append -Encoding utf8
        if ($r.TokS) { Write-Host "  baseline  p=$pi r=$rep  $($r.TokS) tok/s" }
    }
}

#  Per-rank runs 
foreach ($k in $KValues) {
    Write-Host "`n[multi_k] === k=$k ===" -ForegroundColor Yellow
    foreach ($pi in 0..($Prompts.Count-1)) {
        $p = $Prompts[$pi]
        for ($rep = 0; $rep -lt $Reps; $rep++) {
            $done++
            Write-Progress -Activity "multi_k_sweep" -Status "k=$k prompt=$pi rep=$rep" -PercentComplete ([int](100*$done/$totalRuns))
            $gArgs = @(
                $Model,
                '--ctx-size', '512',
                '--axex-compress',
                '--axex-attn-svd',
                '--axex-compress-rank', $k,
                '--axex-compress-max-err', '0',
                '-p', $p, '-n', $NTokens, '--temp', '0'
            )
            $r = Run-GeoAndParse -GeoArgs $gArgs -Tag "k$k"
            $row = [PSCustomObject]@{ k=$k; prompt_idx=$pi; rep=$rep; tok_s=$r.TokS; n_tokens=$r.NTokens; exit_code=$r.ExitCode }
            $allResults.Add($row)
            "$k,$pi,$rep,$($r.TokS),$($r.NTokens),$($r.ExitCode)" | Out-File $csvPath -Append -Encoding utf8
            if ($r.TokS) { Write-Host "  k=$k  p=$pi r=$rep  $($r.TokS) tok/s" }
        }
    }
}

#  Aggregate summary 
Write-Host "`n[multi_k] Computing summary..." -ForegroundColor Cyan

$summary = [System.Collections.Generic.List[PSObject]]::new()
$groupKeys = @("baseline") + ($KValues | ForEach-Object { [string]$_ })

foreach ($kKey in $groupKeys) {
    $rows = $allResults | Where-Object { [string]$_.k -eq $kKey -and $null -ne $_.tok_s }
    if (-not $rows) { continue }
    $vals = @($rows | ForEach-Object { [double]$_.tok_s })
    $mean = ($vals | Measure-Object -Average).Average
    $variance = if ($vals.Count -gt 1) {
        $sq = ($vals | ForEach-Object { ($_ - $mean)*($_ - $mean) } | Measure-Object -Sum).Sum
        [math]::Sqrt($sq / ($vals.Count - 1))
    } else { 0.0 }
    $summary.Add([PSCustomObject]@{
        k        = $kKey
        n_obs    = $vals.Count
        mean_toks= [math]::Round($mean, 2)
        sd_toks  = [math]::Round($variance, 2)
    })
}

$summaryPath = Join-Path $OutDir "multi_k_summary.json"
$summary | ConvertTo-Json -Depth 3 | Set-Content $summaryPath -Encoding UTF8

#  Markdown report 
$baseRow = $summary | Where-Object { $_.k -eq "baseline" }
$baseVal = if ($baseRow) { $baseRow.mean_toks } else { 1.0 }

$md = @("# Paper A — Multi-k Rank Sweep Results", "",
        "**Model**: Llama-3.1-8B-Instruct Q4_K_M  ",
        "**GPU**: RTX 4070 Laptop (8 GB VRAM)  ",
        "**Date**: $(Get-Date -Format 'yyyy-MM-dd')  ",
        "**Reps per cell**: $Reps  $($Prompts.Count) prompts = $Reps$($Prompts.Count) runs per k",
        "",
        "| Rank k | Mean tok/s | ±SD | Speedup vs baseline | N obs |",
        "|--------|-----------|-----|---------------------|-------|"
        )
foreach ($s in ($summary | Sort-Object { if ($_.k -eq 'baseline') { -1 } else { [int]$_.k } })) {
    $speedup = if ($baseVal -gt 0) { [math]::Round($s.mean_toks / $baseVal, 4) } else { "—" }
    $md += "| $($s.k) | $($s.mean_toks) | $($s.sd_toks) | $speedup | $($s.n_obs) |"
}
$md += @("",
         "## Interpretation",
         "",
         "The table above provides the empirical Pareto curve for Paper A Figure 2.",
         "Rank values where speedup > 1.00 confirm L2 cache-fit benefit.",
         "The peak speedup rank and the rank where speedup returns to ~1.00 bound the",
         "working-set size that fits within L2 on this GPU.",
         "")

$reportPath = Join-Path $OutDir "multi_k_report.md"
($md -join "`n") | Set-Content $reportPath -Encoding UTF8

Write-Host "`n[multi_k] Done."
Write-Host "  CSV     -> $csvPath"
Write-Host "  Summary -> $summaryPath"
Write-Host "  Report  -> $reportPath"
