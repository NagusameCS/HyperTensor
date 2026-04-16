# benchmark_three_way.ps1
# Three-way head-to-head: Geodessical GPU vs HyperTensor GPU vs Ollama GPU
# Model: Gemma4 2B Q4_0
# Metrics: Decode t/s, Prefill t/s, TTFT ms, E2E ms, GPU%, VRAM, Power W, CPU%, RAM MB
# Design: N_TRIALS measured trials per condition + 2 warmups discarded
# Conditions: 4 prompts x 3 token counts x 3 runtimes = 36 conditions x 30 trials = 1080 runs
#
# Scientific paper grade: 30 measured trials, 2 warmup discards, full resource monitoring,
# nvidia-smi sampled every 500ms, stddev + 95% CI reported, raw CSV exported.

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

# ── Paths ──────────────────────────────────────────────────────────────────
$GEO_EXE = ".\build_host\geodessical.exe"
$HT_EXE  = ".\build_host\hypertensor.exe"
$MODEL   = "C:\Users\legom\TensorOS\models\google_gemma-4-E2B-it-Q4_0.gguf"
$OLL_URL = "http://localhost:11434/api/generate"
$OLL_MDL = "gemma4-2b"

# ── Benchmark parameters ────────────────────────────────────────────────────
$N_TRIALS = 30      # measured trials per condition (scientific paper grade)
$N_WARMUP = 2       # warmup runs discarded before measurement begins
$TMP_MON  = "$env:TEMP\benchmark_monitor.csv"
$OUT_MD   = "benchmark_results.md"
$OUT_CSV  = "benchmark_results_raw.csv"

# ── Test matrix ─────────────────────────────────────────────────────────────
# Prompts: short (burst decode), medium (typical chat), long (prefill stress), code (structured)
$PROMPTS = @(
    [PSCustomObject]@{
        Name = "short"
        Text = "The quick brown fox jumps"
    }
    [PSCustomObject]@{
        Name = "medium"
        Text = "What is the capital of France? Answer in one word."
    }
    [PSCustomObject]@{
        Name = "long"
        Text = "Explain transformer attention mechanisms in detail, including queries, keys, values, and multi-head attention. Discuss layer normalization and its role in training stability. Compare self-attention to cross-attention."
    }
    [PSCustomObject]@{
        Name = "code"
        Text = "Write a Python function that implements binary search on a sorted list."
    }
)
# Token generation counts: 32 (latency-focused), 128 (balanced), 512 (throughput-focused)
$NS = @(32, 128, 512)

# ── Resource monitor (background job, 500ms cadence) ────────────────────────
function Start-Monitor {
    param([string]$TmpFile, [string[]]$ProcNames)
    "" | Set-Content $TmpFile -Encoding UTF8
    Start-Job -ScriptBlock {
        param($f, $pnames)
        while ($true) {
            try {
                $smi   = nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>$null
                $parts = ($smi | Select-Object -First 1).Trim() -split ',\s*'
                $gpuU  = [double]$parts[0]; $vram = [double]$parts[1]; $watt = [double]$parts[2]
            } catch { $gpuU=0; $vram=0; $watt=0 }
            $ramMB = 0
            foreach ($pn in $pnames) {
                try {
                    $proc = Get-Process -Name $pn -EA SilentlyContinue |
                            Sort-Object WorkingSet64 -Descending | Select-Object -First 1
                    if ($proc) { $ramMB = [double]($proc.WorkingSet64/1MB); break }
                } catch {}
            }
            try {
                $cpuPct = [double]((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average)
            } catch { $cpuPct = 0 }
            "$(Get-Date -Format HH:mm:ss.fff),$gpuU,$vram,$watt,$ramMB,$cpuPct" | Add-Content $f
            Start-Sleep -Milliseconds 500
        }
    } -ArgumentList $TmpFile, $ProcNames
}

function Stop-Monitor {
    param($Job, [string]$TmpFile)
    Stop-Job $Job -EA SilentlyContinue
    Remove-Job $Job -Force -EA SilentlyContinue
    $lines = @(Get-Content $TmpFile -EA SilentlyContinue | Where-Object { $_ -match '^\d{2}:\d{2}' })
    if ($lines.Count -eq 0) { return @{AvgGPU=0;MinGPU=0;MaxGPU=0;PeakVRAM=0;AvgVRAM=0;AvgWatt=0;MaxWatt=0;PeakRAM=0;AvgCPU=0;Samples=0} }
    $gArr=@(); $vArr=@(); $wArr=@(); $rArr=@(); $cArr=@()
    foreach ($l in $lines) {
        $p = $l -split ','
        if ($p.Count -ge 6) {
            $gArr += [double]$p[1]; $vArr += [double]$p[2]
            $wArr += [double]$p[3]; $rArr += [double]$p[4]; $cArr += [double]$p[5]
        }
    }
    return @{
        AvgGPU   = [math]::Round(($gArr | Measure-Object -Average).Average, 1)
        MinGPU   = [math]::Round(($gArr | Measure-Object -Minimum).Minimum, 1)
        MaxGPU   = [math]::Round(($gArr | Measure-Object -Maximum).Maximum, 1)
        PeakVRAM = [math]::Round(($vArr | Measure-Object -Maximum).Maximum, 0)
        AvgVRAM  = [math]::Round(($vArr | Measure-Object -Average).Average, 0)
        AvgWatt  = [math]::Round(($wArr | Measure-Object -Average).Average, 1)
        MaxWatt  = [math]::Round(($wArr | Measure-Object -Maximum).Maximum, 1)
        PeakRAM  = [math]::Round(($rArr | Measure-Object -Maximum).Maximum, 0)
        AvgCPU   = [math]::Round(($cArr | Measure-Object -Average).Average, 1)
        Samples  = $lines.Count
    }
}

# ── Inference runners ────────────────────────────────────────────────────────
function Invoke-Geo {
    param([string]$Text, [int]$N)
    try {
        $mon = Start-Monitor -TmpFile $TMP_MON -ProcNames @("geodessical")
        $sw  = [Diagnostics.Stopwatch]::StartNew()
        $raw = (& $GEO_EXE $MODEL -p $Text -n $N 2>&1) -join "`n"
        $sw.Stop()
        $res = Stop-Monitor -Job $mon -TmpFile $TMP_MON

        $mG = [regex]::Match($raw, '\[GD\] (\d+) tokens in (\d+) ms \(([\d.]+) tok/s\)')
        $mP = [regex]::Match($raw, '\[GD\] Decode-only: prefill ([\d.]+) ms, ([\d.]+) tok/s')
        if (-not $mG.Success) { return @{ Err="no GD output: $($raw[-200..-1] -join '')"; Res=$res } }
        $ng  = [int]$mG.Groups[1].Value
        $gms = [int]$mG.Groups[2].Value
        $dec = [double]$mG.Groups[3].Value
        $pms = if ($mP.Success) { [double]$mP.Groups[1].Value } else { 0 }
        $pts = if ($mP.Success) { [double]$mP.Groups[2].Value } else { 0 }
        return @{ Err=$null; DeT=$dec; PrT=$pts; PrMs=$pms; GnMs=$gms; TtMs=[int]($pms+$gms); NG=$ng; WallMs=$sw.ElapsedMilliseconds; Res=$res }
    } catch { return @{ Err=$_.Exception.Message; Res=@{AvgGPU=0;PeakVRAM=0;AvgWatt=0;PeakRAM=0;AvgCPU=0} } }
}

function Invoke-HT {
    param([string]$Text, [int]$N)
    try {
        $mon = Start-Monitor -TmpFile $TMP_MON -ProcNames @("hypertensor")
        $sw  = [Diagnostics.Stopwatch]::StartNew()
        $raw = (& $HT_EXE $MODEL -p $Text -n $N 2>&1) -join "`n"
        $sw.Stop()
        $res = Stop-Monitor -Job $mon -TmpFile $TMP_MON

        $mG = [regex]::Match($raw, '\[HT\] (\d+) tokens in (\d+) ms \(([\d.]+) tok/s\)')
        $mP = [regex]::Match($raw, '\[HT\] Decode-only: prefill ([\d.]+) ms, ([\d.]+) tok/s')
        if (-not $mG.Success) { return @{ Err="no HT output"; Res=$res } }
        $ng  = [int]$mG.Groups[1].Value
        $gms = [int]$mG.Groups[2].Value
        $dec = [double]$mG.Groups[3].Value
        $pms = if ($mP.Success) { [double]$mP.Groups[1].Value } else { 0 }
        $pts = if ($mP.Success) { [double]$mP.Groups[2].Value } else { 0 }
        return @{ Err=$null; DeT=$dec; PrT=$pts; PrMs=$pms; GnMs=$gms; TtMs=[int]($pms+$gms); NG=$ng; WallMs=$sw.ElapsedMilliseconds; Res=$res }
    } catch { return @{ Err=$_.Exception.Message; Res=@{AvgGPU=0;PeakVRAM=0;AvgWatt=0;PeakRAM=0;AvgCPU=0} } }
}

function Invoke-Oll {
    param([string]$Text, [int]$N)
    try {
        $mon  = Start-Monitor -TmpFile $TMP_MON -ProcNames @("ollama_llama_server","ollama_runner","ollama")
        $sw   = [Diagnostics.Stopwatch]::StartNew()
        $opts = @{ num_predict=$N; num_gpu=999; num_ctx=2048 }
        $body = @{ model=$OLL_MDL; prompt=$Text; stream=$false; options=$opts } | ConvertTo-Json
        $resp = Invoke-WebRequest -Uri $OLL_URL -Method POST -Body $body `
                    -ContentType "application/json" -UseBasicParsing -TimeoutSec 600
        $sw.Stop()
        $res  = Stop-Monitor -Job $mon -TmpFile $TMP_MON

        $j   = $resp.Content | ConvertFrom-Json
        $dec = if ($j.eval_duration -gt 0)        { [math]::Round($j.eval_count        / ($j.eval_duration        / 1e9), 1) } else { 0 }
        $prt = if ($j.prompt_eval_duration -gt 0) { [math]::Round($j.prompt_eval_count / ($j.prompt_eval_duration / 1e9), 1) } else { 0 }
        $pms = [math]::Round($j.prompt_eval_duration / 1e6, 1)
        $gms = [math]::Round($j.eval_duration        / 1e6, 0)
        $tms = [math]::Round(($j.eval_duration + $j.prompt_eval_duration) / 1e6, 0)
        return @{ Err=$null; DeT=$dec; PrT=$prt; PrMs=$pms; GnMs=$gms; TtMs=$tms; NG=$j.eval_count; WallMs=$sw.ElapsedMilliseconds; Res=$res }
    } catch { return @{ Err=$_.Exception.Message; Res=@{AvgGPU=0;PeakVRAM=0;AvgWatt=0;PeakRAM=0;AvgCPU=0} } }
}

# ── Ollama health check ───────────────────────────────────────────────────────
function EnsureOllama {
    try { $null = Invoke-WebRequest "http://localhost:11434" -UseBasicParsing -TimeoutSec 3 2>$null; return }
    catch {}
    Write-Host "  Starting ollama serve..." -ForegroundColor Yellow
    $null = Start-Process "ollama" -ArgumentList "serve" -PassThru -WindowStyle Hidden
    Start-Sleep 6
}

# ── Result collection ─────────────────────────────────────────────────────────
$results = [System.Collections.ArrayList]::new()

function Record {
    param([string]$rt, [string]$pname, [int]$n, [int]$tr, [bool]$warmup, $r)
    [void]$results.Add([PSCustomObject]@{
        Runtime=$rt; Prompt=$pname; N=$n; Trial=$tr; Warmup=$warmup
        DeT=$r.DeT; PrT=$r.PrT; PrMs=$r.PrMs; GnMs=$r.GnMs; TtMs=$r.TtMs
        NG=$r.NG; WallMs=$r.WallMs; Err=$r.Err
        AvgGPU=$r.Res.AvgGPU; MinGPU=$r.Res.MinGPU; MaxGPU=$r.Res.MaxGPU
        PeakVRAM=$r.Res.PeakVRAM; AvgVRAM=$r.Res.AvgVRAM
        AvgWatt=$r.Res.AvgWatt; MaxWatt=$r.Res.MaxWatt
        PeakRAM=$r.Res.PeakRAM; AvgCPU=$r.Res.AvgCPU
    })
    if ($r.Err) {
        Write-Host ("  [{0,-12} {1,-6} n={2,3} t={3,2}] ERR: {4}" -f $rt,$pname,$n,$tr,$r.Err) -ForegroundColor Red
    } else {
        $wTag = if ($warmup) { "W" } else { " " }
        Write-Host ("  [{0,-12} {1,-6} n={2,3} t={3,2}]{4} dec={5,6:F1} t/s  ttft={6,6:F0} ms  e2e={7,6:F0} ms  GPU={8,3}%  VRAM={9,5}MB  {10,4:F0}W  CPU={11,4:F1}%" -f `
            $rt,$pname,$n,$tr,$wTag, $r.DeT,$r.PrMs,$r.TtMs,
            $r.Res.AvgGPU,$r.Res.PeakVRAM,$r.Res.AvgWatt,$r.Res.AvgCPU)
    }
}

# ── Stats helpers ─────────────────────────────────────────────────────────────
function Stats {
    param($vals)
    $v = @($vals | Where-Object { $_ -gt 0 })
    if ($v.Count -eq 0) { return @{Mean=0;Std=0;Min=0;Max=0;Median=0;CI95=0;N=0} }
    $mean = ($v | Measure-Object -Average).Average
    $std  = 0
    if ($v.Count -gt 1) {
        $sq = ($v | ForEach-Object { ($_ - $mean)*($_ - $mean) })
        $std = [math]::Sqrt(($sq | Measure-Object -Sum).Sum / ($v.Count - 1))
    }
    $sorted = $v | Sort-Object
    $med = $sorted[[int]($sorted.Count/2)]
    $ci95 = if ($v.Count -gt 0) { 1.96 * $std / [math]::Sqrt($v.Count) } else { 0 }
    return @{
        Mean   = [math]::Round($mean, 2)
        Std    = [math]::Round($std,  2)
        Min    = [math]::Round(($v | Measure-Object -Minimum).Minimum, 2)
        Max    = [math]::Round(($v | Measure-Object -Maximum).Maximum, 2)
        Median = [math]::Round($med,  2)
        CI95   = [math]::Round($ci95, 2)
        N      = $v.Count
    }
}

function F { param($n, $d=1)
    if ($null -eq $n -or ([math]::Abs([double]$n) -lt 0.0001 -and $n -ne 0)) { return "--" }
    [math]::Round([double]$n, $d)
}

# ── Phase 0: Build both local runtimes ────────────────────────────────────────
Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host "  Three-Way Benchmark: Geodessical vs HyperTensor vs Ollama" -ForegroundColor Cyan
Write-Host "  Model: Gemma4 2B Q4_0  |  Trials: $N_TRIALS  |  GPU: CUDA" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan

Write-Host "`n--- Building Geodessical (GPU) ---" -ForegroundColor DarkCyan
$geoBuilt = $false
try {
    & .\build_host.ps1 2>&1 | Where-Object { $_ -match 'SUCCESS|ERROR|CUDA|zig|warning' } | Select-Object -Last 5
    if (Test-Path $GEO_EXE) { $geoBuilt = $true; Write-Host "  Geodessical built OK" -ForegroundColor Green }
    else { Write-Host "  Geodessical EXE not found after build!" -ForegroundColor Red }
} catch { Write-Host "  Build error: $_" -ForegroundColor Red }

# HyperTensor uses existing binary (built separately with batch-attn kernels)
if (Test-Path $HT_EXE) {
    Write-Host "  HyperTensor binary found: $HT_EXE" -ForegroundColor Green
} else {
    Write-Host "  HyperTensor EXE not found at $HT_EXE" -ForegroundColor Red
}

EnsureOllama
Write-Host "  Ollama endpoint: $OLL_URL (model: $OLL_MDL)" -ForegroundColor Green

# ── Phase 1: Warmup (discarded) ───────────────────────────────────────────────
Write-Host "`n--- Warmup Phase ($N_WARMUP runs per condition, discarded) ---" -ForegroundColor DarkCyan
# Use short prompt short N for warmup to be fast
$wpText = $PROMPTS[0].Text
$wpN    = 32

for ($w = 1; $w -le $N_WARMUP; $w++) {
    Write-Host "  Warmup $w/$N_WARMUP..." -NoNewline
    if ($geoBuilt) { $null = Invoke-Geo $wpText $wpN }
    $null = Invoke-HT  $wpText $wpN
    $null = Invoke-Oll $wpText $wpN
    Write-Host " done"
}

# ── Phase 2: Measured trials ──────────────────────────────────────────────────
# Interleave runtimes to reduce thermal/temporal bias
$totalConditions = $PROMPTS.Count * $NS.Count
$totalRuns       = $totalConditions * 3 * $N_TRIALS
Write-Host "`n--- Measured Trials ($N_TRIALS trials x $totalConditions conditions x 3 runtimes = $totalRuns runs) ---" -ForegroundColor Cyan

$runNum = 0
foreach ($p in $PROMPTS) {
    foreach ($n in $NS) {
        Write-Host "`n  [Condition: prompt=$($p.Name) N=$n]" -ForegroundColor Yellow

        for ($t = 1; $t -le $N_TRIALS; $t++) {
            $runNum++
            $pct = [int](100 * $runNum / $totalRuns)
            Write-Progress -Activity "Benchmarking" -Status "Trial $t/$N_TRIALS | $($p.Name) N=$n" -PercentComplete $pct

            # Randomise order per trial to reduce ordering bias
            $order = @("Geo","HT","Oll") | Sort-Object { Get-Random }
            foreach ($rt in $order) {
                switch ($rt) {
                    "Geo" {
                        if ($geoBuilt) {
                            $r = Invoke-Geo $p.Text $n
                            Record "Geodessical" $p.Name $n $t $false $r
                        }
                    }
                    "HT" {
                        $r = Invoke-HT $p.Text $n
                        Record "HyperTensor" $p.Name $n $t $false $r
                    }
                    "Oll" {
                        $r = Invoke-Oll $p.Text $n
                        Record "Ollama" $p.Name $n $t $false $r
                    }
                }
            }
        }
    }
}

Write-Progress -Activity "Benchmarking" -Completed
Write-Host "`n--- Data collection complete. Generating report... ---" -ForegroundColor Cyan

# ── Export raw CSV ─────────────────────────────────────────────────────────────
$measured = $results | Where-Object { -not $_.Warmup }
$measured | Export-Csv -Path "$PSScriptRoot\$OUT_CSV" -NoTypeInformation -Encoding UTF8
Write-Host "  Raw CSV: $OUT_CSV  ($($measured.Count) rows)" -ForegroundColor Green

# ── System info ───────────────────────────────────────────────────────────────
$gpuInfo    = (nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null | Select-Object -First 1)
$cpuInfo    = (Get-WmiObject Win32_Processor | Select-Object -First 1).Name
$sysRAM_GB  = [math]::Round((Get-WmiObject Win32_ComputerSystem).TotalPhysicalMemory/1GB, 0)
$geoVer     = if ($geoBuilt) { (& $GEO_EXE --version 2>&1 | Select-Object -First 1) -replace '.*v(\S+).*','$1' } else { "n/a" }
$htVer      = ((& $HT_EXE --version 2>&1 | Select-Object -First 1) -replace '.*v(\S+).*','$1') 2>$null
$ollVer     = (ollama --version 2>$null | Select-Object -First 1)

# ── Build markdown report ─────────────────────────────────────────────────────
$md = [System.Text.StringBuilder]::new()
$null = $md.AppendLine("# Three-Way Benchmark: Geodessical vs HyperTensor vs Ollama")
$null = $md.AppendLine("")
$null = $md.AppendLine("## System")
$null = $md.AppendLine("")
$null = $md.AppendLine("| Parameter | Value |")
$null = $md.AppendLine("|-----------|-------|")
$null = $md.AppendLine("| Date | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') |")
$null = $md.AppendLine("| CPU | $cpuInfo |")
$null = $md.AppendLine("| GPU | $gpuInfo |")
$null = $md.AppendLine("| System RAM | ${sysRAM_GB} GB |")
$null = $md.AppendLine("| Model | google_gemma-4-E2B-it-Q4_0.gguf (Gemma4 2B Q4_0, 3.2 GB) |")
$null = $md.AppendLine("| Geodessical | v$geoVer (GPU/CUDA, batch prefill) |")
$null = $md.AppendLine("| HyperTensor | v$htVer (GPU/CUDA, batched causal attention) |")
$null = $md.AppendLine("| Ollama | $ollVer (GPU, num_gpu=999) |")
$null = $md.AppendLine("| Trials | **$N_TRIALS** measured per condition + $N_WARMUP warmups discarded |")
$null = $md.AppendLine("| Trial order | Randomised per trial to reduce ordering bias |")
$null = $md.AppendLine("| Monitor cadence | 500 ms (nvidia-smi) |")
$null = $md.AppendLine("")
$null = $md.AppendLine("## Metrics Glossary")
$null = $md.AppendLine("")
$null = $md.AppendLine("| Metric | Description |")
$null = $md.AppendLine("|--------|-------------|")
$null = $md.AppendLine("| Decode t/s | Tokens generated per second (decode phase only) |")
$null = $md.AppendLine("| Prefill t/s | Prompt tokens processed per second |")
$null = $md.AppendLine("| TTFT ms | Time To First Token (prefill latency) |")
$null = $md.AppendLine("| E2E ms | Total wall time: TTFT + decode |")
$null = $md.AppendLine("| ms/tok | Milliseconds per output token |")
$null = $md.AppendLine("| Avg GPU% | Mean GPU SM utilisation during inference |")
$null = $md.AppendLine("| Peak VRAM | Maximum VRAM used (MB) |")
$null = $md.AppendLine("| Avg Watt | Mean GPU power draw (W) |")
$null = $md.AppendLine("| Avg CPU% | Mean system CPU load |")
$null = $md.AppendLine("| Peak RAM | Peak process working set (MB) |")
$null = $md.AppendLine("| t/s/W | Decode tokens per joule (efficiency) |")
$null = $md.AppendLine("| CI 95% | 95% confidence interval half-width (1.96*σ/√n) |")
$null = $md.AppendLine("")

$runtimes = @("Geodessical", "HyperTensor", "Ollama")

# ── Section: Decode throughput summary ────────────────────────────────────────
$null = $md.AppendLine("---")
$null = $md.AppendLine("")
$null = $md.AppendLine("## Decode Throughput (tok/s) — All Conditions")
$null = $md.AppendLine("")
$null = $md.AppendLine("*Higher = better. Mean ± 95%CI over $N_TRIALS trials.*")
$null = $md.AppendLine("")
$null = $md.AppendLine("| Runtime | Prompt | N | Mean (t/s) | ±95%CI | Median | σ | Min | Max | n |")
$null = $md.AppendLine("|---------|--------|--:|:----------:|:------:|:------:|:-:|:---:|:---:|:-:|")
foreach ($rt in $runtimes) {
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS) {
            $rows = $measured | Where-Object { $_.Runtime -eq $rt -and $_.Prompt -eq $p.Name -and $_.N -eq $n -and -not $_.Err }
            if ($rows) {
                $s = Stats ($rows | ForEach-Object { $_.DeT })
                $null = $md.AppendLine("| $rt | $($p.Name) | $n | **$($s.Mean)** | $($s.CI95) | $($s.Median) | $($s.Std) | $($s.Min) | $($s.Max) | $($s.N) |")
            }
        }
    }
}
$null = $md.AppendLine("")

# ── Section: TTFT summary ─────────────────────────────────────────────────────
$null = $md.AppendLine("---")
$null = $md.AppendLine("")
$null = $md.AppendLine("## Time To First Token (ms) — All Conditions")
$null = $md.AppendLine("")
$null = $md.AppendLine("*Lower = better. Mean ± 95%CI over $N_TRIALS trials.*")
$null = $md.AppendLine("")
$null = $md.AppendLine("| Runtime | Prompt | N | Mean (ms) | ±95%CI | Median | σ | Min | Max | n |")
$null = $md.AppendLine("|---------|--------|--:|:---------:|:------:|:------:|:-:|:---:|:---:|:-:|")
foreach ($rt in $runtimes) {
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS) {
            $rows = $measured | Where-Object { $_.Runtime -eq $rt -and $_.Prompt -eq $p.Name -and $_.N -eq $n -and -not $_.Err }
            if ($rows) {
                $s = Stats ($rows | ForEach-Object { $_.PrMs })
                $null = $md.AppendLine("| $rt | $($p.Name) | $n | **$($s.Mean)** | $($s.CI95) | $($s.Median) | $($s.Std) | $($s.Min) | $($s.Max) | $($s.N) |")
            }
        }
    }
}
$null = $md.AppendLine("")

# ── Section: Prefill throughput ───────────────────────────────────────────────
$null = $md.AppendLine("---")
$null = $md.AppendLine("")
$null = $md.AppendLine("## Prefill Throughput (tok/s) — All Conditions")
$null = $md.AppendLine("")
$null = $md.AppendLine("*Higher = better.*")
$null = $md.AppendLine("")
$null = $md.AppendLine("| Runtime | Prompt | N | Mean (t/s) | ±95%CI | Median | σ | Min | Max |")
$null = $md.AppendLine("|---------|--------|--:|:----------:|:------:|:------:|:-:|:---:|:---:|")
foreach ($rt in $runtimes) {
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS) {
            $rows = $measured | Where-Object { $_.Runtime -eq $rt -and $_.Prompt -eq $p.Name -and $_.N -eq $n -and -not $_.Err }
            if ($rows) {
                $vals = @($rows | Where-Object { $_.PrT -gt 0 } | ForEach-Object { $_.PrT })
                if ($vals.Count -gt 0) {
                    $s = Stats $vals
                    $null = $md.AppendLine("| $rt | $($p.Name) | $n | **$($s.Mean)** | $($s.CI95) | $($s.Median) | $($s.Std) | $($s.Min) | $($s.Max) |")
                }
            }
        }
    }
}
$null = $md.AppendLine("")

# ── Section: Head-to-head ratios ──────────────────────────────────────────────
$null = $md.AppendLine("---")
$null = $md.AppendLine("")
$null = $md.AppendLine("## Head-to-Head Ratios vs Ollama")
$null = $md.AppendLine("")
$null = $md.AppendLine("*Ratio = Runtime mean / Ollama mean. >1.0 = faster than Ollama.*")
$null = $md.AppendLine("")
$null = $md.AppendLine("### Decode t/s ratio")
$null = $md.AppendLine("")
$null = $md.AppendLine("| Runtime | Prompt | N | Ratio vs Ollama | Runtime t/s | Ollama t/s |")
$null = $md.AppendLine("|---------|--------|--:|:---------------:|:-----------:|:----------:|")
foreach ($rt in @("Geodessical","HyperTensor")) {
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS) {
            $rtRows = $measured | Where-Object { $_.Runtime -eq $rt -and $_.Prompt -eq $p.Name -and $_.N -eq $n -and -not $_.Err }
            $olRows = $measured | Where-Object { $_.Runtime -eq "Ollama"  -and $_.Prompt -eq $p.Name -and $_.N -eq $n -and -not $_.Err }
            if ($rtRows -and $olRows) {
                $rtDec = (Stats ($rtRows | ForEach-Object { $_.DeT })).Mean
                $olDec = (Stats ($olRows | ForEach-Object { $_.DeT })).Mean
                $ratio = if ($olDec -gt 0) { [math]::Round($rtDec / $olDec, 3) } else { "--" }
                $flag  = if ($ratio -is [double] -and $ratio -ge 1.0) { "✓" } else { "✗" }
                $null = $md.AppendLine("| $rt | $($p.Name) | $n | **${ratio}x** $flag | $rtDec | $olDec |")
            }
        }
    }
}
$null = $md.AppendLine("")

$null = $md.AppendLine("### TTFT ratio (lower ratio = faster)")
$null = $md.AppendLine("")
$null = $md.AppendLine("| Runtime | Prompt | N | Ratio vs Ollama | Runtime ms | Ollama ms |")
$null = $md.AppendLine("|---------|--------|--:|:---------------:|:----------:|:---------:|")
foreach ($rt in @("Geodessical","HyperTensor")) {
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS) {
            $rtRows = $measured | Where-Object { $_.Runtime -eq $rt -and $_.Prompt -eq $p.Name -and $_.N -eq $n -and -not $_.Err }
            $olRows = $measured | Where-Object { $_.Runtime -eq "Ollama"  -and $_.Prompt -eq $p.Name -and $_.N -eq $n -and -not $_.Err }
            if ($rtRows -and $olRows) {
                $rtTTFT = (Stats ($rtRows | ForEach-Object { $_.PrMs })).Mean
                $olTTFT = (Stats ($olRows | ForEach-Object { $_.PrMs })).Mean
                $ratio  = if ($olTTFT -gt 0) { [math]::Round($rtTTFT / $olTTFT, 3) } else { "--" }
                $null = $md.AppendLine("| $rt | $($p.Name) | $n | **${ratio}x** | $rtTTFT | $olTTFT |")
            }
        }
    }
}
$null = $md.AppendLine("")

# ── Section: Resource usage ────────────────────────────────────────────────────
$null = $md.AppendLine("---")
$null = $md.AppendLine("")
$null = $md.AppendLine("## Resource Usage Summary")
$null = $md.AppendLine("")
$null = $md.AppendLine("| Runtime | Prompt | N | Avg GPU% | Peak VRAM MB | Avg Watt | Max Watt | Avg CPU% | Peak RAM MB |")
$null = $md.AppendLine("|---------|--------|--:|:--------:|:------------:|:--------:|:--------:|:--------:|:-----------:|")
foreach ($rt in $runtimes) {
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS) {
            $rows = $measured | Where-Object { $_.Runtime -eq $rt -and $_.Prompt -eq $p.Name -and $_.N -eq $n -and -not $_.Err }
            if ($rows) {
                $aGPU  = (Stats ($rows | ForEach-Object { $_.AvgGPU   })).Mean
                $pVRAM = ($rows | ForEach-Object { $_.PeakVRAM } | Measure-Object -Maximum).Maximum
                $aWatt = (Stats ($rows | ForEach-Object { $_.AvgWatt  })).Mean
                $mWatt = ($rows | ForEach-Object { $_.MaxWatt  } | Measure-Object -Maximum).Maximum
                $aCPU  = (Stats ($rows | ForEach-Object { $_.AvgCPU   })).Mean
                $pRAM  = ($rows | ForEach-Object { $_.PeakRAM  } | Measure-Object -Maximum).Maximum
                $null = $md.AppendLine("| $rt | $($p.Name) | $n | $aGPU | $pVRAM | $aWatt | $mWatt | $aCPU | $pRAM |")
            }
        }
    }
}
$null = $md.AppendLine("")

# ── Section: Efficiency ────────────────────────────────────────────────────────
$null = $md.AppendLine("---")
$null = $md.AppendLine("")
$null = $md.AppendLine("## Efficiency: Decode tok/s per Watt")
$null = $md.AppendLine("")
$null = $md.AppendLine("| Runtime | Prompt | N | Decode t/s | Avg Watt | t/s per W |")
$null = $md.AppendLine("|---------|--------|--:|:----------:|:--------:|:---------:|")
foreach ($rt in $runtimes) {
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS) {
            $rows = $measured | Where-Object { $_.Runtime -eq $rt -and $_.Prompt -eq $p.Name -and $_.N -eq $n -and -not $_.Err }
            if ($rows) {
                $aDec  = (Stats ($rows | ForEach-Object { $_.DeT     })).Mean
                $aWatt = (Stats ($rows | ForEach-Object { $_.AvgWatt })).Mean
                $tpw   = if ($aWatt -gt 0) { [math]::Round($aDec / $aWatt, 4) } else { 0 }
                $null = $md.AppendLine("| $rt | $($p.Name) | $n | $aDec | $aWatt | $tpw |")
            }
        }
    }
}
$null = $md.AppendLine("")

# ── Section: Raw results ───────────────────────────────────────────────────────
$null = $md.AppendLine("---")
$null = $md.AppendLine("")
$null = $md.AppendLine("## Raw Results (all $N_TRIALS trials per condition)")
$null = $md.AppendLine("")
$null = $md.AppendLine("| Runtime | Prompt | N | Trial | N-gen | Decode t/s | Prefill t/s | TTFT ms | E2E ms | Wall ms | GPU% | VRAM MB | Watt | CPU% | RAM MB |")
$null = $md.AppendLine("|---------|--------|--:|------:|------:|:----------:|:-----------:|:-------:|:------:|:-------:|:----:|:-------:|:----:|:----:|:------:|")
foreach ($row in ($measured | Sort-Object Runtime,Prompt,N,Trial)) {
    if ($row.Err) {
        $null = $md.AppendLine("| $($row.Runtime) | $($row.Prompt) | $($row.N) | $($row.Trial) | ERR | $($row.Err) | | | | | | | | | |")
    } else {
        $null = $md.AppendLine("| $($row.Runtime) | $($row.Prompt) | $($row.N) | $($row.Trial) | $($row.NG) | $($row.DeT) | $($row.PrT) | $($row.PrMs) | $($row.TtMs) | $($row.WallMs) | $($row.AvgGPU) | $($row.PeakVRAM) | $($row.AvgWatt) | $($row.AvgCPU) | $($row.PeakRAM) |")
    }
}
$null = $md.AppendLine("")

# ── Write markdown ─────────────────────────────────────────────────────────────
[System.IO.File]::WriteAllText("$PSScriptRoot\$OUT_MD", $md.ToString(), [System.Text.Encoding]::UTF8)
Write-Host "`n== Report written: $OUT_MD ==" -ForegroundColor Green
Write-Host "== CSV written:    $OUT_CSV ==" -ForegroundColor Green
Write-Host "== Total rows:     $($measured.Count) ==" -ForegroundColor Green

# ── Console summary ────────────────────────────────────────────────────────────
Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host "  PERFORMANCE SUMMARY (mean over $N_TRIALS trials)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ("{0,-14} {1,-7} {2,-5} {3,11} {4,9} {5,9} {6,8}" -f "Runtime","Prompt","N","Decode t/s","TTFT ms","E2E ms","ms/tok")
Write-Host ("-" * 70)
foreach ($rt in $runtimes) {
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS) {
            $rows = $measured | Where-Object { $_.Runtime -eq $rt -and $_.Prompt -eq $p.Name -and $_.N -eq $n -and -not $_.Err }
            if ($rows) {
                $sDec  = Stats ($rows | ForEach-Object { $_.DeT  })
                $sTtft = Stats ($rows | ForEach-Object { $_.PrMs })
                $sE2E  = Stats ($rows | ForEach-Object { $_.TtMs })
                $mstok = if ($sDec.Mean -gt 0) { [math]::Round(1000 / $sDec.Mean, 1) } else { "--" }
                Write-Host ("{0,-14} {1,-7} {2,-5} {3,11} {4,9} {5,9} {6,8}" -f `
                    $rt, $p.Name, $n, "$($sDec.Mean)±$($sDec.CI95)", "$($sTtft.Mean)±$($sTtft.CI95)", "$($sE2E.Mean)±$($sE2E.CI95)", $mstok)
            }
        }
    }
    Write-Host ""
}

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host "  RESOURCE SUMMARY" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ("{0,-14} {1,-7} {2,-5} {3,9} {4,11} {5,9} {6,9}" -f "Runtime","Prompt","N","Avg GPU%","Peak VRAM","Avg Watt","Avg CPU%")
Write-Host ("-" * 70)
foreach ($rt in $runtimes) {
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS) {
            $rows = $measured | Where-Object { $_.Runtime -eq $rt -and $_.Prompt -eq $p.Name -and $_.N -eq $n -and -not $_.Err }
            if ($rows) {
                $aGPU  = (Stats ($rows | ForEach-Object { $_.AvgGPU  })).Mean
                $pVRAM = ($rows | ForEach-Object { $_.PeakVRAM } | Measure-Object -Maximum).Maximum
                $aWatt = (Stats ($rows | ForEach-Object { $_.AvgWatt })).Mean
                $aCPU  = (Stats ($rows | ForEach-Object { $_.AvgCPU  })).Mean
                Write-Host ("{0,-14} {1,-7} {2,-5} {3,9} {4,11} {5,9} {6,9}" -f $rt,$p.Name,$n,$aGPU,$pVRAM,$aWatt,$aCPU)
            }
        }
    }
    Write-Host ""
}

Write-Host "`n================================================================" -ForegroundColor Cyan
Write-Host "  RATIO vs OLLAMA (Decode t/s)" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ("{0,-14} {1,-7} {2,-5} {3,12} {4,12} {5,10}" -f "Runtime","Prompt","N","Runtime t/s","Ollama t/s","Ratio")
Write-Host ("-" * 65)
foreach ($rt in @("Geodessical","HyperTensor")) {
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS) {
            $rtRows = $measured | Where-Object { $_.Runtime -eq $rt -and $_.Prompt -eq $p.Name -and $_.N -eq $n -and -not $_.Err }
            $olRows = $measured | Where-Object { $_.Runtime -eq "Ollama" -and $_.Prompt -eq $p.Name -and $_.N -eq $n -and -not $_.Err }
            if ($rtRows -and $olRows) {
                $rtD = (Stats ($rtRows | ForEach-Object { $_.DeT })).Mean
                $olD = (Stats ($olRows | ForEach-Object { $_.DeT })).Mean
                $r   = if ($olD -gt 0) { [math]::Round($rtD / $olD, 3) } else { "--" }
                $col = if ($r -is [double] -and $r -ge 1.0) { "Green" } else { "Red" }
                Write-Host ("{0,-14} {1,-7} {2,-5} {3,12} {4,12} {5,10}" -f $rt,$p.Name,$n,$rtD,$olD,$r) -ForegroundColor $col
            }
        }
    }
    Write-Host ""
}
