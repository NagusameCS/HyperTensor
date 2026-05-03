<#
.SYNOPSIS
C3 — OTT Speculative Decode Calibration Sweep
==============================================
Grid-searches --ott-spec-thresh x --ott-spec-batch to find the Pareto frontier
of acceptance rate vs verifier throughput for the active model.

.DESCRIPTION
For each (thresh, batch) cell the script:
  1. Runs geodessical.exe on a locked validation prompt set
  2. Parses the [SPEC] Done line for acceptance_rate, geo_accepted, xfmr, and
     the [SPEC] Verifier decode line for tok/s
  3. Records results to calibration_sweep_results.csv
  4. Prints a sorted table and marks the Pareto-optimal cells

Usage:
  .\scripts\ott\calibration_sweep.ps1 [-Model <path>] [-MaxTokens <n>] [-Reps <n>] [-OutDir <dir>]

Parameters:
  -Model      Path to the GGUF model file  (default: models\smollm2-135m-instruct-q8_0.gguf)
  -MaxTokens  Tokens to generate per run   (default: 32)
  -Reps       Repetitions per cell         (default: 1 — increase for stable averages)
  -OutDir     Directory for output files   (default: .)

Outputs:
  <OutDir>\calibration_sweep_results.csv
  <OutDir>\calibration_sweep_report.txt   (human-readable Pareto table)

Requires: build_host\geodessical.exe (built with --ott-full support)
#>

param(
    [string]$Model     = "models\smollm2-135m-instruct-q8_0.gguf",
    [int]   $MaxTokens = 32,
    [int]   $Reps      = 1,
    [string]$OutDir    = "."
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$exe = Join-Path $PSScriptRoot "..\..\build_host\geodessical.exe"
if (-not (Test-Path $exe)) { $exe = ".\build_host\geodessical.exe" }
if (-not (Test-Path $exe)) { Write-Error "geodessical.exe not found; run build_host.ps1 first." }

$modelPath = $Model
if (-not (Test-Path $modelPath)) { Write-Error "Model not found: $modelPath" }

#  Validation prompts (locked set — do NOT change between runs) 
$Prompts = @(
    "Explain the water cycle in three sentences.",
    "What is the capital of France?",
    "Write a Python function that reverses a string.",
    "Summarise the key ideas of Newton's first law of motion.",
    "Translate 'Good morning' into Spanish, French, and German."
)

#  Grid 
$Thresholds = @(0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
$BatchSizes = @(1, 2, 3, 4)

$csvPath    = Join-Path $OutDir "calibration_sweep_results.csv"
$reportPath = Join-Path $OutDir "calibration_sweep_report.txt"

$results = [System.Collections.Generic.List[PSCustomObject]]::new()

# CSV header
"thresh,batch,prompt_idx,rep,acceptance_rate,geo_accepted,xfmr_accepted,total_tokens,verifier_tok_s,exit_code" |
    Out-File -FilePath $csvPath -Encoding utf8 -Force

$totalCells = $Thresholds.Count * $BatchSizes.Count * $Prompts.Count * $Reps
$done = 0

Write-Host "`n[C3] Calibration Sweep — $totalCells runs total`n" -ForegroundColor Cyan

foreach ($thresh in $Thresholds) {
    foreach ($batch in $BatchSizes) {
        foreach ($pi in 0..($Prompts.Count - 1)) {
            $prompt = $Prompts[$pi]
            for ($rep = 0; $rep -lt $Reps; $rep++) {
                $done++
                $pct = [int](100 * $done / $totalCells)
                Write-Progress -Activity "C3 Calibration Sweep" `
                    -Status "thresh=$thresh batch=$batch prompt=$pi rep=$rep" `
                    -PercentComplete $pct

                $tmpOut = [System.IO.Path]::GetTempFileName()
                $tmpErr = [System.IO.Path]::GetTempFileName()

                $argList = @(
                    $modelPath,
                    "-p", $prompt,
                    "-n", $MaxTokens,
                    "--ott-full",
                    "--ott-speculative",
                    "--ott-spec-batch", $batch,
                    "--ott-spec-thresh", $thresh
                )

                $proc = Start-Process -FilePath $exe `
                    -ArgumentList $argList `
                    -RedirectStandardOutput $tmpOut `
                    -RedirectStandardError  $tmpErr `
                    -NoNewWindow -Wait -PassThru

                $exitCode = $proc.ExitCode
                $stderr   = Get-Content $tmpErr -Raw -ErrorAction SilentlyContinue
                Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue

                # Parse [SPEC] Done line
                $accRate    = 0.0
                $geoAcc     = 0
                $xfmrAcc    = 0
                $totalToks  = 0
                $verifTokS  = 0.0

                if ($stderr) {
                    # [SPEC] Done: 24 tokens (geo_accepted=2 xfmr=22 od_drafts=3 swarm_k=0, acceptance_rate=8.3%, final_batch=1)
                    if ($stderr -match '\[SPEC\] Done: (\d+) tokens \(geo_accepted=(\d+) xfmr=(\d+) od_drafts=\d+ swarm_k=\d+, acceptance_rate=([\d.]+)%') {
                        $totalToks = [int]$Matches[1]
                        $geoAcc    = [int]$Matches[2]
                        $xfmrAcc   = [int]$Matches[3]
                        $accRate   = [double]$Matches[4]
                    }
                    # [SPEC] Verifier decode: 45.3 tok/s (22 calls in 486 ms)
                    if ($stderr -match '\[SPEC\] Verifier decode: ([\d.]+) tok/s') {
                        $verifTokS = [double]$Matches[1]
                    }
                }

                $row = [PSCustomObject]@{
                    thresh         = $thresh
                    batch          = $batch
                    prompt_idx     = $pi
                    rep            = $rep
                    acceptance_rate= $accRate
                    geo_accepted   = $geoAcc
                    xfmr_accepted  = $xfmrAcc
                    total_tokens   = $totalToks
                    verifier_tok_s = $verifTokS
                    exit_code      = $exitCode
                }
                $results.Add($row)

                "$thresh,$batch,$pi,$rep,$accRate,$geoAcc,$xfmrAcc,$totalToks,$verifTokS,$exitCode" |
                    Add-Content -Path $csvPath -Encoding utf8

                Write-Host ("  [thresh={0:F2} batch={1}] prompt={2} rep={3}  acc={4:F1}%  verif={5:F1}tok/s" -f `
                    $thresh, $batch, $pi, $rep, $accRate, $verifTokS)
            }
        }
    }
}

Write-Progress -Activity "C3 Calibration Sweep" -Completed

#  Aggregate: average over prompts and reps 
$agg = $results | Group-Object { "$($_.thresh),$($_.batch)" } | ForEach-Object {
    $rows = $_.Group
    [PSCustomObject]@{
        thresh          = $rows[0].thresh
        batch           = $rows[0].batch
        avg_acc_rate    = ($rows | Measure-Object -Property acceptance_rate -Average).Average
        avg_verif_tok_s = ($rows | Measure-Object -Property verifier_tok_s  -Average).Average
        n_runs          = $rows.Count
    }
} | Sort-Object avg_acc_rate -Descending

#  Pareto frontier: cells not dominated in both acc_rate and verif_tok_s 
# A cell dominates another if it has >= acc_rate AND >= verif_tok_s
$pareto = [System.Collections.Generic.List[PSCustomObject]]::new()
foreach ($a in $agg) {
    $dominated = $false
    foreach ($b in $agg) {
        if ($b -eq $a) { continue }
        if ($b.avg_acc_rate -ge $a.avg_acc_rate -and $b.avg_verif_tok_s -ge $a.avg_verif_tok_s) {
            $dominated = $true; break
        }
    }
    if (-not $dominated) { $pareto.Add($a) }
}
$paretoKeys = $pareto | ForEach-Object { "$($_.thresh),$($_.batch)" }

#  Report 
$reportLines = @()
$reportLines += ""
$reportLines += "=" * 72
$reportLines += "  C3 OTT Calibration Sweep Report"
$reportLines += "  Model: $modelPath"
$reportLines += "  Prompts: $($Prompts.Count)  Reps/cell: $Reps  MaxTokens: $MaxTokens"
$reportLines += "=" * 72
$reportLines += ("  {0,-8} {1,-6} {2,-12} {3,-14} {4}" -f "thresh","batch","avg_acc(%)","verif(tok/s)","Pareto")
$reportLines += "-" * 72

foreach ($row in $agg) {
    $pk = "$($row.thresh),$($row.batch)"
    $parLabel = if ($paretoKeys -contains $pk) { " *** PARETO" } else { "" }
    $reportLines += ("  {0,-8:F2} {1,-6} {2,-12:F1} {3,-14:F1}{4}" -f `
        $row.thresh, $row.batch, $row.avg_acc_rate, $row.avg_verif_tok_s, $parLabel)
}
$reportLines += "=" * 72
$reportLines += ""
$reportLines += "  Pareto-optimal cells (non-dominated in acceptance AND throughput):"
foreach ($p in ($pareto | Sort-Object avg_acc_rate -Descending)) {
    $reportLines += ("    thresh={0:F2}  batch={1}  acc={2:F1}%  verif={3:F1}tok/s" -f `
        $p.thresh, $p.batch, $p.avg_acc_rate, $p.avg_verif_tok_s)
}
$reportLines += ""
$reportLines += "  Recommended operating point (highest Pareto acc_rate):"
if ($pareto.Count -gt 0) {
    $best = $pareto | Sort-Object avg_acc_rate -Descending | Select-Object -First 1
    $reportLines += ("    --ott-spec-thresh {0:F2} --ott-spec-batch {1}" -f $best.thresh, $best.batch)
}
$reportLines += ""

$reportLines | Out-File -FilePath $reportPath -Encoding utf8 -Force

foreach ($line in $reportLines) { Write-Host $line }

Write-Host "`n[C3] Results: $csvPath" -ForegroundColor Green
Write-Host "[C3] Report:  $reportPath" -ForegroundColor Green
