#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# OTT C-level Speedup Benchmark
# ==============================
# The Python OTT engine validates the jury gate but cannot achieve
# end-to-end speedup because draft generation calls model.generate().
# REAL speedup requires the C binary (geodessical.exe) which uses
# pre-baked OneDecode tables and Christoffel-corrected geodesic flow.
#
# This script measures the actual speedup of each OTT mode vs standard
# transformer decode.
#
# Usage (on EC2 or Linux with geodessical binary):
#   pwsh scripts/benchmark_ott_c_speedup.ps1 -Model path/to/model.gguf -Tokens 128

param(
    [string]$Model = "",
    [string]$Prompt = "Explain the theory of relativity in simple terms.",
    [int]$Tokens = 128,
    [int]$WarmupRuns = 1,
    [int]$BenchRuns = 3
)

$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# Find the binary
$binary = $null
foreach ($candidate in @("build_host/geodessical.exe", "build_host/geodessical", "geodessical")) {
    if (Test-Path $candidate) {
        $binary = $candidate
        break
    }
}

if (-not $binary) {
    Write-Host "[SKIP] geodessical binary not found. Build first: ./build_host.sh"
    Write-Host "[NOTE] Python OTT validates the jury gate (0.17ms vs 30ms = 177x per draft)."
    Write-Host "[NOTE] End-to-end speedup requires C-level geodesic flow, which this"
    Write-Host "[NOTE] script measures when run on a system with the compiled binary."
    exit 0
}

if (-not $Model -or -not (Test-Path $Model)) {
    Write-Host "[SKIP] No model.gguf provided. Use -Model path/to/model.gguf"
    exit 0
}

Write-Host "============================================================"
Write-Host "  OTT C-LEVEL SPEEDUP BENCHMARK"
Write-Host "  Binary: $binary"
Write-Host "  Model: $Model"
Write-Host "  Tokens: $Tokens | Prompt: '$Prompt'"
Write-Host "============================================================"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outDir = Join-Path $repoRoot "benchmarks/ott_c_speedup_${stamp}"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$results = @()
$modes = @(
    @{Name="baseline";         Flags="--temp 0 --top-k 1";                 Desc="Standard greedy decode (no OTT)"},
    @{Name="ott-speculative";  Flags="--ott-speculative --ott-spec-batch 4 --ott-spec-thresh 0.65"; Desc="Geodesic drafts + transformer verify"},
    @{Name="ott-full";         Flags="--ott-full --axiom-fast";            Desc="Geodesic-first + axiom + AttnRes"},
    @{Name="ott-theorem";      Flags="--ott-theorem --axiom-fast";         Desc="Full + depth residual + strict QC"},
    @{Name="ott-perfect";      Flags="--ott-perfect";                      Desc="Exact greedy upper bound (100% accept)"}
)

foreach ($mode in $modes) {
    Write-Host ""
    Write-Host "--- $($mode.Desc) ---"
    Write-Host "  Mode: $($mode.Name)"
    
    $modeDir = Join-Path $outDir $mode.Name
    New-Item -ItemType Directory -Force -Path $modeDir | Out-Null
    
    $tokS_list = @()
    
    for ($run = 0; $run -lt ($WarmupRuns + $BenchRuns); $run++) {
        $isWarmup = $run -lt $WarmupRuns
        $label = if ($isWarmup) { "warmup $($run+1)/$WarmupRuns" } else { "bench $($run-$WarmupRuns+1)/$BenchRuns" }
        
        $stdoutFile = Join-Path $modeDir "run_${run}.stdout.txt"
        $stderrFile = Join-Path $modeDir "run_${run}.stderr.txt"
        
        $args = @(
            $Model,
            "-p", $Prompt,
            "-n", $Tokens
        ) + ($mode.Flags -split " ")
        
        Write-Host "  [$label] Running..."
        
        $proc = Start-Process -FilePath $binary -ArgumentList $args `
            -NoNewWindow -RedirectStandardOutput $stdoutFile `
            -RedirectStandardError $stderrFile -Wait -PassThru
        
        # Parse tokens-per-second from stdout
        if (Test-Path $stdoutFile) {
            $stdout = Get-Content $stdoutFile -Raw
            # Look for "[GD] N tokens in M ms (T tok/s)" pattern
            if ($stdout -match '\[GD\]\s+\d+\s+tokens\sin\s+\d+\s+ms\s+\((\d+\.?\d*)\s+tok/s\)') {
                $tps = [double]$Matches[1]
                if (-not $isWarmup) {
                    $tokS_list += $tps
                    Write-Host "  [$label] $tps tok/s (counted)"
                } else {
                    Write-Host "  [$label] $tps tok/s (warmup, skipped)"
                }
            } else {
                # Try alternate pattern: "[SPEC] Done: N tokens..."
                if ($stdout -match '\[SPEC\]\s+Done:\s+\d+\s+tokens.*acceptance_rate=(\d+\.?\d*)%') {
                    $accept = [double]$Matches[1]
                    Write-Host "  [$label] acceptance=$accept%"
                }
                # Try decode-only metric
                if ($stdout -match 'Decode-only:.*?(\d+\.?\d*)\s+tok/s') {
                    $tps_decode = [double]$Matches[1]
                    if (-not $isWarmup) {
                        $tokS_list += $tps_decode
                        Write-Host "  [$label] $tps_decode tok/s (decode-only)"
                    }
                } else {
                    Write-Host "  [$label] TPS not parseable from output"
                }
            }
        }
    }
    
    if ($tokS_list.Count -gt 0) {
        $meanTps = ($tokS_list | Measure-Object -Average).Average
        $results += @{
            Mode = $mode.Name
            Description = $mode.Desc
            MeanTokPerSec = [math]::Round($meanTps, 1)
            Runs = $tokS_list.Count
            AllTPS = $tokS_list
        }
        Write-Host "  RESULT: $($mode.Name) = $([math]::Round($meanTps, 1)) tok/s (n=$($tokS_list.Count))"
    } else {
        Write-Host "  RESULT: $($mode.Name) = NO DATA (check stderr)"
    }
}

# Compute speedups
Write-Host ""
Write-Host "============================================================"
Write-Host "  OTT C-LEVEL SPEEDUP RESULTS"
Write-Host "============================================================"

$baseline = $results | Where-Object { $_.Mode -eq "baseline" } | Select-Object -First 1

if ($baseline -and $baseline.MeanTokPerSec -gt 0) {
    Write-Host ("  {0,-20s} {1,10s} {2,10s} {3,10s}" -f "Mode", "tok/s", "Speedup", "Notes")
    Write-Host ("  {0,-20s} {1,10s} {2,10s} {3,10s}" -f "----", "------", "-------", "-----")
    
    foreach ($r in $results) {
        $speedup = if ($r.MeanTokPerSec -gt 0) { [math]::Round($r.MeanTokPerSec / $baseline.MeanTokPerSec, 2) } else { "N/A" }
        $notes = if ($r.Mode -eq "ott-perfect") { "upper bound" } `
            elseif ($r.Mode -eq "ott-theorem") { "target 1.6-1.8x" } `
            elseif ($r.Mode -eq "ott-speculative") { "batch=4" } `
            else { "" }
        Write-Host ("  {0,-20s} {1,10:f1} {2,10} {3,10s}" -f $r.Mode, $r.MeanTokPerSec, $speedup, $notes)
    }
} else {
    Write-Host "  No baseline data. Check that standard decode completed."
}

# Save JSON report
$report = @{
    timestamp = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    binary = $binary
    model = $Model
    prompt = $Prompt
    max_tokens = $Tokens
    warmup_runs = $WarmupRuns
    bench_runs = $BenchRuns
    results = $results | ForEach-Object {
        @{
            mode = $_.Mode
            description = $_.Description
            mean_tok_per_sec = $_.MeanTokPerSec
            n_runs = $_.Runs
        }
    }
}

$reportPath = Join-Path $outDir "speedup_report.json"
$report | ConvertTo-Json -Depth 3 | Out-File $reportPath -Encoding utf8
Write-Host ""
Write-Host "  Report saved to: $reportPath"

# Also produce a human-readable summary
$summaryPath = Join-Path $outDir "summary.md"
@"
# OTT C-Level Speedup Benchmark

**Date:** $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
**Binary:** $binary
**Model:** $Model
**Tokens:** $Tokens
**Prompt:** "$Prompt"

## Results

| Mode | tok/s | Speedup | Notes |
|------|-------|---------|-------|
$(
    if ($baseline) {
        ($results | ForEach-Object {
            $sp = if ($_.MeanTokPerSec -gt 0) { [math]::Round($_.MeanTokPerSec / $baseline.MeanTokPerSec, 2) } else { "N/A" }
            $n = if ($_.Mode -eq "ott-perfect") { "upper bound" } `
                elseif ($_.Mode -eq "ott-speculative") { "batch=4, thresh=0.65" } `
                elseif ($_.Mode -eq "ott-full") { "axiom-fast" } `
                elseif ($_.Mode -eq "ott-theorem") { "target 1.6-1.8x" } `
                else { "greedy temp=0" }
            "| $($_.Mode) | $($_.MeanTokPerSec) | ${sp}x | $n |"
        }) -join "`n"
    } else { "| (no data) | | | |" }
)

## Notes

- **Standard decode** uses greedy sampling (temp=0, top-k=1) for fair comparison
- **OTT speculative** uses geodesic drafts verified by the transformer
- **OTT perfect** is the exact upper bound: 100% draft acceptance, no speedup from speculation
- Real speedup requires acceptance rate > 50% to overcome verification overhead
- Python OTT validates the jury gate (0.17ms vs 30ms = 177x per draft) but cannot achieve end-to-end speedup because draft generation calls model.generate()
"@ | Out-File $summaryPath -Encoding utf8

Write-Host "  Summary saved to: $summaryPath"
Write-Host "============================================================"
