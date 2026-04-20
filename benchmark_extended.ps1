# benchmark_extended.ps1
# Extended benchmark: Geodessical (CPU+GPU) vs Ollama (CPU+GPU)
# Models:    smollm2-135m Q8_0 | Phi-3.5 Mini Q4_0 | Gemma-4-E2B Q4_0
# Token counts: 40 / 128 / 512 / 1024
# Prompts:   short / medium / long
# Trials:    1 warmup (discarded) + 3 measured
# Extras:    VRAM MB, Process RAM MB, GPU power W, GPU util% sampled during each run
# Output:    benchmark_extended.csv  +  benchmark_extended.md

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

# ─── Config ───────────────────────────────────────────────────────────────────
$GEO      = ".\build_host\geodessical.exe"
$MODEL_DIR = "C:\Users\legom\TensorOS\models"
$MODELS_GEO = @(
    [PSCustomObject]@{ Name="smollm2-135m";  Gguf="$MODEL_DIR\smollm2-135m-instruct-q8_0.gguf";   Quant="Q8_0";  SizeMB=138  }
    [PSCustomObject]@{ Name="phi35-mini";    Gguf="$MODEL_DIR\Phi-3.5-mini-instruct-Q4_0.gguf";   Quant="Q4_0";  SizeMB=2081 }
    [PSCustomObject]@{ Name="gemma4-2b";     Gguf="$MODEL_DIR\google_gemma-4-E2B-it-Q4_0.gguf";   Quant="Q4_0";  SizeMB=3222 }
)
$MODELS_OLL = @(
    [PSCustomObject]@{ Name="smollm2-135m";  Id="smollm2:135m" }
    [PSCustomObject]@{ Name="phi35-mini";    Id="phi3.5"        }
    [PSCustomObject]@{ Name="gemma4-2b";     Id="gemma4-2b"     }
)
$PROMPTS = @(
    [PSCustomObject]@{ Name="short";  Text="The quick brown fox jumps" }
    [PSCustomObject]@{ Name="medium"; Text="Explain what a transformer neural network is." }
    [PSCustomObject]@{ Name="long";   Text="Explain transformer attention mechanisms in detail, including queries, keys, values, and multi-head attention. Discuss layer normalization and its training benefits." }
)
$NS        = @(40, 128, 512, 1024)
$WARMUPS   = 1
$TRIALS    = 3
$OLL_URL   = "http://localhost:11434/api/generate"
$OUT_CSV   = "benchmark_extended.csv"
$OUT_MD    = "benchmark_extended.md"
$TMP_MON   = "$env:TEMP\geo_ext_monitor.csv"
$TMP_TS_LOG = "$PSScriptRoot\benchmark_timeseries.csv"
$STAGE_FILE = "$env:TEMP\geo_ext_stage.txt"

# ─── Results store ────────────────────────────────────────────────────────────
$results = [System.Collections.ArrayList]::new()

function Record($rt, $be, $mdl, $quant, $pn, $nr, $tr, $r, $hw) {
    [void]$results.Add([PSCustomObject]@{
        Machine="local"; Runtime=$rt; Backend=$be; Model=$mdl; Quant=$quant
        Prompt=$pn; N=$nr; Trial=$tr
        DecodeTS=$r.DeT; PrefillTS=$r.PrT; PrefillMs=$r.PrMs; GenMs=$r.GnMs
        TotalMs=$r.TtMs; NGen=$r.NG; Err=$r.Err
        VramMB=$hw.Vram; RamMB=$hw.Ram; GpuPctAvg=$hw.GpuPct; PowerW=$hw.Power
    })
    if ($r.Err) {
        Write-Host ("  [ERR] {0}/{1} {2} {3} n={4} t={5}: {6}" -f $rt,$be,$mdl,$pn,$nr,$tr,$r.Err) -ForegroundColor Red
    } else {
        Write-Host ("  [{0}/{1}] {2} {3} n={4} t={5}  dec={6} t/s  pre={7} t/s  ttft={8}ms  gen={9}ms  vram={10}MB" -f `
            $rt,$be,$mdl,$pn,$nr,$tr,$r.DeT,$r.PrT,$r.PrMs,$r.GnMs,$hw.Vram)
    }
}

# ─── Hardware sampler ─────────────────────────────────────────────────────────
function Set-Stage($label) {
    $label | Set-Content $STAGE_FILE -Encoding UTF8 -NoNewline
}

function Start-HwMonitor {
    "" | Set-Content $TMP_MON -Encoding UTF8
    $job = Start-Job -ScriptBlock {
        param($f, $tsLog, $sf)
        while ($true) {
            try {
                $smi = nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw `
                    --format=csv,noheader,nounits 2>$null
                $p = ($smi | Select-Object -First 1).Trim() -split ',\s*'
                $line = "$([int]$p[0]),$([int]$p[1]),$([double]$p[2])"
            } catch { $line = "0,0,0" }
            try {
                $geo = Get-Process -Name geodessical -ErrorAction SilentlyContinue |
                    Sort-Object WorkingSet64 -Descending | Select-Object -First 1
                $ram = if ($geo) { [int]($geo.WorkingSet64/1MB) } else { 0 }
            } catch { $ram = 0 }
            "$line,$ram" | Add-Content $f -Encoding UTF8
            try {
                $stage = if (Test-Path $sf) { (Get-Content $sf -Raw -ErrorAction SilentlyContinue).Trim() } else { 'idle' }
                $ts = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
                "$ts,$line,$ram,$stage" | Add-Content $tsLog -Encoding UTF8
            } catch {}
            Start-Sleep -Milliseconds 400
        }
    } -ArgumentList $TMP_MON, $TMP_TS_LOG, $STAGE_FILE
    return $job
}

function Stop-HwMonitor($job) {
    Stop-Job $job -ErrorAction SilentlyContinue
    Remove-Job $job -Force -ErrorAction SilentlyContinue
}

function Read-HwStats {
    try {
        $rows = Get-Content $TMP_MON -ErrorAction SilentlyContinue |
            Where-Object { $_ -match '^\d' } |
            ForEach-Object { $p=$_ -split ','; [PSCustomObject]@{
                Gpu=[int]$p[0]; Vram=[int]$p[1]; Power=[double]$p[2]; Ram=[int]$p[3]
            }}
        if (-not $rows) { return @{GpuPct=0;Vram=0;Power=0;Ram=0} }
        return @{
            GpuPct = [math]::Round(($rows|Measure-Object Gpu -Average).Average,1)
            Vram   = ($rows|Measure-Object Vram -Maximum).Maximum
            Power  = [math]::Round(($rows|Measure-Object Power -Average).Average,1)
            Ram    = ($rows|Measure-Object Ram  -Maximum).Maximum
        }
    } catch { return @{GpuPct=0;Vram=0;Power=0;Ram=0} }
}

# ─── Geodessical runner ───────────────────────────────────────────────────────
function Invoke-Geo($gguf, $text, $n) {
    try {
        $raw = (& $GEO $gguf -p $text -n $n 2>&1) -join "`n"
        $mG  = [regex]::Match($raw, '\[GD\] (\d+) tokens in (\d+) ms \(([\d.]+) tok/s\)')
        $mP  = [regex]::Match($raw, '\[GD\] Decode-only: prefill ([\d.]+) ms, ([\d.]+) tok/s')
        if (-not $mG.Success) {
            # Try alternate output format
            $mG2 = [regex]::Match($raw, 'Generated (\d+) tokens.*?(\d+) ms.*?([\d.]+) tok/s')
            if (-not $mG2.Success) { return @{Err="no output pattern matched";DeT=0;PrT=0;PrMs=0;GnMs=0;TtMs=0;NG=0} }
            $ng=[int]$mG2.Groups[1].Value; $gms=[int]$mG2.Groups[2].Value; $dec=[double]$mG2.Groups[3].Value
            return @{Err=$null;DeT=$dec;PrT=0;PrMs=0;GnMs=$gms;TtMs=$gms;NG=$ng}
        }
        $ng  = [int]$mG.Groups[1].Value
        $gms = [int]$mG.Groups[2].Value
        $dec = [double]$mG.Groups[3].Value
        $pms = if ($mP.Success) { [double]$mP.Groups[1].Value } else { 0 }
        $pts = if ($mP.Success) { [double]$mP.Groups[2].Value } else { 0 }
        return @{Err=$null;DeT=$dec;PrT=$pts;PrMs=$pms;GnMs=$gms;TtMs=[int]($gms+$pms);NG=$ng}
    } catch { return @{Err=$_.Exception.Message;DeT=0;PrT=0;PrMs=0;GnMs=0;TtMs=0;NG=0} }
}

# ─── Ollama runner ────────────────────────────────────────────────────────────
function Invoke-OllAPI($mdl, $text, $n, $gpu) {
    try {
        $opts = @{num_predict=$n}
        if (-not $gpu) { $opts.num_gpu = 0 }
        $body = @{model=$mdl;prompt=$text;stream=$false;options=$opts} | ConvertTo-Json -Depth 4
        $resp = Invoke-WebRequest -Uri $OLL_URL -Method POST -Body $body `
            -ContentType "application/json" -UseBasicParsing -TimeoutSec 900
        $j   = $resp.Content | ConvertFrom-Json
        $dec = if ($j.eval_duration -gt 0)        { [math]::Round($j.eval_count/($j.eval_duration/1e9),1) } else { 0 }
        $prt = if ($j.prompt_eval_duration -gt 0) { [math]::Round($j.prompt_eval_count/($j.prompt_eval_duration/1e9),1) } else { 0 }
        $pms = [math]::Round($j.prompt_eval_duration/1e6,1)
        $gms = [math]::Round($j.eval_duration/1e6,0)
        $tms = [math]::Round(($j.eval_duration+$j.prompt_eval_duration)/1e6,0)
        return @{Err=$null;DeT=$dec;PrT=$prt;PrMs=$pms;GnMs=$gms;TtMs=$tms;NG=$j.eval_count}
    } catch { return @{Err=$_.Exception.Message;DeT=0;PrT=0;PrMs=0;GnMs=0;TtMs=0;NG=0} }
}

function EnsureOllama {
    try { $null = Invoke-WebRequest "http://localhost:11434" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop }
    catch {
        Write-Host "  Starting ollama serve..." -ForegroundColor Yellow
        $null = Start-Process "ollama" -ArgumentList "serve" -PassThru -WindowStyle Hidden
        Start-Sleep 5
    }
}

function Avg($vals) {
    $v = $vals | Where-Object { $_ -gt 0 }
    if (-not $v) { return 0 }
    return [math]::Round(($v | Measure-Object -Average).Average, 1)
}

function F($n) { if ($n -eq 0 -or $n -eq $null) { "--" } else { $n } }

# ─── Check prerequisites ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "  HyperTensor Extended Benchmark" -ForegroundColor Cyan
Write-Host "  Models: $($MODELS_GEO.Count)  Prompts: $($PROMPTS.Count)  N: $($NS -join '/') Trials: $TRIALS (+$WARMUPS warmup)" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $GEO)) {
    Write-Host "[bench] geodessical.exe not found. Building CPU variant first..." -ForegroundColor Yellow
    & .\build_host.ps1 -NoCuda 2>&1 | Select-String 'SUCCESS|ERROR|FAIL'
}

$missingModels = $MODELS_GEO | Where-Object { -not (Test-Path $_.Gguf) }
if ($missingModels) {
    Write-Host "[bench] Missing GGUF files:" -ForegroundColor Yellow
    $missingModels | ForEach-Object { Write-Host "  - $($_.Gguf)" -ForegroundColor Yellow }
}

$cpuInfo = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
$gpuInfo = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA|AMD|Intel' } |
    Select-Object -First 1).Name
$ramGB   = [math]::Round((Get-CimInstance Win32_PhysicalMemory |
    Measure-Object Capacity -Sum).Sum / 1GB, 0)

Write-Host "  CPU: $cpuInfo" -ForegroundColor DarkGray
Write-Host "  GPU: $gpuInfo" -ForegroundColor DarkGray
Write-Host "  RAM: ${ramGB} GB" -ForegroundColor DarkGray
Write-Host ""

# Initialize persistent time-series log
"Timestamp,GpuPct,VramMB,PowerW,RamMB,Stage" | Set-Content $TMP_TS_LOG -Encoding UTF8

# ─── Phase 1: Geodessical CPU ─────────────────────────────────────────────────
Write-Host "=== Phase 1: Geodessical CPU ===" -ForegroundColor Cyan
& .\build_host.ps1 -NoCuda 2>&1 | Where-Object { $_ -match 'SUCCESS|ERROR|FAIL|NoCuda' }
Write-Host ""

foreach ($m in $MODELS_GEO) {
    if (-not (Test-Path $m.Gguf)) { Write-Host "  SKIP $($m.Name) — GGUF not found" -ForegroundColor DarkGray; continue }
    Write-Host "  Model: $($m.Name) ($($m.Quant))" -ForegroundColor White
    foreach ($p in $PROMPTS) { foreach ($n in $NS) {
        # Warmup
        for ($w = 1; $w -le $WARMUPS; $w++) {
            Write-Host ("    warmup {0}/{1} {2} n={3}..." -f $w,$WARMUPS,$p.Name,$n) -ForegroundColor DarkGray
            $null = Invoke-Geo $m.Gguf $p.Text $n
        }
        # Measured trials
        for ($t = 1; $t -le $TRIALS; $t++) {
            Set-Stage "GeoCPU|$($m.Name)|$($p.Name)|n$n|t$t"
            $mon = Start-HwMonitor
            "" | Set-Content $TMP_MON -Encoding UTF8   # reset sampler
            $r   = Invoke-Geo $m.Gguf $p.Text $n
            $hw  = Read-HwStats
            Stop-HwMonitor $mon
            Record "Geodessical" "CPU" $m.Name $m.Quant $p.Name $n $t $r $hw
        }
    } }
}

# ─── Phase 2: Geodessical GPU ─────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Phase 2: Geodessical GPU ===" -ForegroundColor Cyan
& .\build_host.ps1 2>&1 | Where-Object { $_ -match 'SUCCESS|ERROR|FAIL|CUDA' }
Write-Host ""

foreach ($m in $MODELS_GEO) {
    if (-not (Test-Path $m.Gguf)) { Write-Host "  SKIP $($m.Name) — GGUF not found" -ForegroundColor DarkGray; continue }
    Write-Host "  Model: $($m.Name) ($($m.Quant))" -ForegroundColor White
    foreach ($p in $PROMPTS) { foreach ($n in $NS) {
        for ($w = 1; $w -le $WARMUPS; $w++) {
            Write-Host ("    warmup {0}/{1} {2} n={3}..." -f $w,$WARMUPS,$p.Name,$n) -ForegroundColor DarkGray
            $null = Invoke-Geo $m.Gguf $p.Text $n
        }
        for ($t = 1; $t -le $TRIALS; $t++) {
            Set-Stage "GeoGPU|$($m.Name)|$($p.Name)|n$n|t$t"
            $mon = Start-HwMonitor
            "" | Set-Content $TMP_MON -Encoding UTF8
            $r   = Invoke-Geo $m.Gguf $p.Text $n
            $hw  = Read-HwStats
            Stop-HwMonitor $mon
            Record "Geodessical" "GPU" $m.Name $m.Quant $p.Name $n $t $r $hw
        }
    } }
}

# ─── Phase 3: Ollama GPU ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Phase 3: Ollama GPU ===" -ForegroundColor Cyan
EnsureOllama
Write-Host ""

foreach ($m in $MODELS_OLL) {
    Write-Host "  Model: $($m.Name)" -ForegroundColor White
    foreach ($p in $PROMPTS) { foreach ($n in $NS) {
        for ($w = 1; $w -le $WARMUPS; $w++) {
            Write-Host ("    warmup {0}/{1} {2} n={3}..." -f $w,$WARMUPS,$p.Name,$n) -ForegroundColor DarkGray
            $null = Invoke-OllAPI $m.Id $p.Text $n $true
        }
        for ($t = 1; $t -le $TRIALS; $t++) {
            Set-Stage "OllGPU|$($m.Name)|$($p.Name)|n$n|t$t"
            $mon = Start-HwMonitor
            "" | Set-Content $TMP_MON -Encoding UTF8
            $r   = Invoke-OllAPI $m.Id $p.Text $n $true
            $hw  = Read-HwStats
            Stop-HwMonitor $mon
            Record "Ollama" "GPU" $m.Name "ollama" $p.Name $n $t $r $hw
        }
    } }
}

# ─── Phase 4: Ollama CPU ──────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Phase 4: Ollama CPU ===" -ForegroundColor Cyan
Write-Host ""

foreach ($m in $MODELS_OLL) {
    Write-Host "  Model: $($m.Name)" -ForegroundColor White
    foreach ($p in $PROMPTS) { foreach ($n in $NS) {
        for ($w = 1; $w -le $WARMUPS; $w++) {
            Write-Host ("    warmup {0}/{1} {2} n={3}..." -f $w,$WARMUPS,$p.Name,$n) -ForegroundColor DarkGray
            $null = Invoke-OllAPI $m.Id $p.Text $n $false
        }
        for ($t = 1; $t -le $TRIALS; $t++) {
            Set-Stage "OllCPU|$($m.Name)|$($p.Name)|n$n|t$t"
            $mon = Start-HwMonitor
            "" | Set-Content $TMP_MON -Encoding UTF8
            $r   = Invoke-OllAPI $m.Id $p.Text $n $false
            $hw  = Read-HwStats
            Stop-HwMonitor $mon
            Record "Ollama" "CPU" $m.Name "ollama" $p.Name $n $t $r $hw
        }
    } }
}

# ─── Write CSV ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Writing Results ===" -ForegroundColor Cyan

$csvHeader = "Machine,Runtime,Backend,Model,Quant,Prompt,N,Trial,NGen,DecodeTS,PrefillTS,PrefillMs,GenMs,TotalMs,VramMB,RamMB,GpuPctAvg,PowerW,Err"
$csvLines  = $results | ForEach-Object {
    "$($_.Machine),$($_.Runtime),$($_.Backend),$($_.Model),$($_.Quant),$($_.Prompt),$($_.N),$($_.Trial),$($_.NGen),$($_.DecodeTS),$($_.PrefillTS),$($_.PrefillMs),$($_.GenMs),$($_.TotalMs),$($_.VramMB),$($_.RamMB),$($_.GpuPctAvg),$($_.PowerW),$($_.Err)"
}
@($csvHeader) + $csvLines | Set-Content $OUT_CSV -Encoding UTF8
Write-Host "  CSV: $OUT_CSV ($($results.Count) rows)" -ForegroundColor Green

# ─── Write Markdown ───────────────────────────────────────────────────────────
$md = [System.Text.StringBuilder]::new()
$null=$md.AppendLine("# HyperTensor Extended Benchmark")
$null=$md.AppendLine("")
$null=$md.AppendLine("**Date:** $(Get-Date -Format 'yyyy-MM-dd HH:mm')")
$null=$md.AppendLine("")
$null=$md.AppendLine("**Machine:** local Windows")
$null=$md.AppendLine("")
$null=$md.AppendLine("**CPU:** $cpuInfo")
$null=$md.AppendLine("")
$null=$md.AppendLine("**GPU:** $gpuInfo")
$null=$md.AppendLine("")
$null=$md.AppendLine("**RAM:** ${ramGB} GB")
$null=$md.AppendLine("")
$null=$md.AppendLine("**Models tested:**")
$null=$md.AppendLine("")
foreach ($m in $MODELS_GEO) { $null=$md.AppendLine("- $($m.Name) ($($m.Quant), $($m.SizeMB) MB)") }
$null=$md.AppendLine("")
$null=$md.AppendLine("**Protocol:** $WARMUPS warmup trial discarded, $TRIALS measured trials averaged")
$null=$md.AppendLine("")
$null=$md.AppendLine("**Token counts:** $($NS -join ' / ')")
$null=$md.AppendLine("")
$null=$md.AppendLine("**Column definitions:**")
$null=$md.AppendLine("- **Decode t/s** = generation tokens/second (decode phase only)")
$null=$md.AppendLine("- **Prefill t/s** = prompt tokens processed/second")
$null=$md.AppendLine("- **TTFT ms** = Time To First Token (prefill wall-time)")
$null=$md.AppendLine("- **Gen ms** = decode wall-time (excludes prefill)")
$null=$md.AppendLine("- **Peak VRAM MB** = maximum VRAM used during run (nvidia-smi)")
$null=$md.AppendLine("- **Avg GPU%** = mean GPU utilization during run")
$null=$md.AppendLine("- **Avg Power W** = mean GPU power draw during run")
$null=$md.AppendLine("")
$null=$md.AppendLine("---")
$null=$md.AppendLine("")

# Summary table
$null=$md.AppendLine("## Summary — Averaged Across All Conditions")
$null=$md.AppendLine("")
$null=$md.AppendLine("| Runtime | Backend | Model | Decode t/s | Prefill t/s | TTFT ms | Peak VRAM MB | Avg GPU% | Avg Power W |")
$null=$md.AppendLine("|---------|---------|-------|:----------:|:-----------:|:-------:|:------------:|:--------:|:-----------:|")
$combos = $results | Select-Object Runtime,Backend,Model -Unique | Sort-Object Runtime,Backend,Model
foreach ($c in $combos) {
    $rows = $results | Where-Object { $_.Runtime -eq $c.Runtime -and $_.Backend -eq $c.Backend -and $_.Model -eq $c.Model -and -not $_.Err }
    $aD=Avg($rows|%{$_.DecodeTS}); $aP=Avg($rows|%{$_.PrefillTS})
    $aT=Avg($rows|%{$_.PrefillMs}); $aV=($rows|Measure-Object VramMB -Maximum).Maximum
    $aG=Avg($rows|%{$_.GpuPctAvg}); $aW=Avg($rows|%{$_.PowerW})
    $null=$md.AppendLine("| $($c.Runtime) | $($c.Backend) | $($c.Model) | $(F $aD) | $(F $aP) | $(F $aT) | $(F $aV) | $(F $aG) | $(F $aW) |")
}
$null=$md.AppendLine("")
$null=$md.AppendLine("---")
$null=$md.AppendLine("")

# Per-model breakdown
foreach ($model in ($results | Select-Object -ExpandProperty Model -Unique | Sort-Object)) {
    $null=$md.AppendLine("## Model: $model")
    $null=$md.AppendLine("")
    foreach ($pname in ($results | Select-Object -ExpandProperty Prompt -Unique | Sort-Object)) {
        $null=$md.AppendLine("### Prompt: $pname")
        $null=$md.AppendLine("")
        $null=$md.AppendLine("| Runtime | Backend | N | N gen | Decode t/s | Prefill t/s | TTFT ms | Gen ms | Total ms | Peak VRAM MB | Avg Power W |")
        $null=$md.AppendLine("|---------|---------|--:|------:|:----------:|:-----------:|:-------:|:------:|:--------:|:------------:|:-----------:|")
        foreach ($rt in @("Geodessical","Ollama")) {
            foreach ($be in @("GPU","CPU")) {
                foreach ($n in $NS) {
                    $rows = $results | Where-Object {
                        $_.Model -eq $model -and $_.Prompt -eq $pname -and
                        $_.Runtime -eq $rt -and $_.Backend -eq $be -and $_.N -eq $n -and -not $_.Err
                    }
                    if ($rows) {
                        $aD=Avg($rows|%{$_.DecodeTS}); $aP=Avg($rows|%{$_.PrefillTS})
                        $aT=Avg($rows|%{$_.PrefillMs}); $aG=Avg($rows|%{$_.GenMs})
                        $aTt=Avg($rows|%{$_.TotalMs}); $aN=[int](Avg($rows|%{$_.NGen}))
                        $aV=($rows|Measure-Object VramMB -Maximum).Maximum
                        $aW=Avg($rows|%{$_.PowerW})
                        $null=$md.AppendLine("| $rt | $be | $n | $aN | $(F $aD) | $(F $aP) | $(F $aT) | $(F $aG) | $(F $aTt) | $(F $aV) | $(F $aW) |")
                    } else {
                        $eRows = $results | Where-Object {
                            $_.Model -eq $model -and $_.Prompt -eq $pname -and
                            $_.Runtime -eq $rt -and $_.Backend -eq $be -and $_.N -eq $n
                        }
                        if ($eRows) { $null=$md.AppendLine("| $rt | $be | $n | -- | ERR | ERR | ERR | ERR | ERR | -- | -- |") }
                    }
                }
            }
        }
        $null=$md.AppendLine("")
    }

    # GPU/CPU speedup table
    $null=$md.AppendLine("### GPU vs CPU Speedup — $model")
    $null=$md.AppendLine("")
    $null=$md.AppendLine("| Runtime | N | GPU Decode t/s | CPU Decode t/s | GPU/CPU Speedup |")
    $null=$md.AppendLine("|---------|--:|:--------------:|:--------------:|:---------------:|")
    foreach ($rt in @("Geodessical","Ollama")) {
        foreach ($n in $NS) {
            $gR = $results | Where-Object { $_.Model -eq $model -and $_.Runtime -eq $rt -and $_.Backend -eq "GPU" -and $_.N -eq $n -and -not $_.Err }
            $cR = $results | Where-Object { $_.Model -eq $model -and $_.Runtime -eq $rt -and $_.Backend -eq "CPU" -and $_.N -eq $n -and -not $_.Err }
            $gD = Avg($gR|%{$_.DecodeTS}); $cD = Avg($cR|%{$_.DecodeTS})
            if ($cD -gt 0 -and $gD -gt 0) {
                $sp = [math]::Round($gD/$cD,1)
                $null=$md.AppendLine("| $rt | $n | $gD | $cD | ${sp}x |")
            }
        }
    }
    $null=$md.AppendLine("")

    # Geodessical vs Ollama GPU head-to-head
    $null=$md.AppendLine("### Geodessical vs Ollama GPU Head-to-Head — $model")
    $null=$md.AppendLine("")
    $null=$md.AppendLine("| N | Geo-GPU t/s | Oll-GPU t/s | Geo Advantage |")
    $null=$md.AppendLine("|--:|:-----------:|:-----------:|:-------------:|")
    foreach ($n in $NS) {
        $gR = $results | Where-Object { $_.Model -eq $model -and $_.Runtime -eq "Geodessical" -and $_.Backend -eq "GPU" -and $_.N -eq $n -and -not $_.Err }
        $oR = $results | Where-Object { $_.Model -eq $model -and $_.Runtime -eq "Ollama"       -and $_.Backend -eq "GPU" -and $_.N -eq $n -and -not $_.Err }
        $gD = Avg($gR|%{$_.DecodeTS}); $oD = Avg($oR|%{$_.DecodeTS})
        if ($gD -gt 0 -and $oD -gt 0) {
            $adv = [math]::Round(($gD-$oD)/$oD*100,1)
            $sign = if ($adv -ge 0) { "+$adv%" } else { "$adv%" }
            $null=$md.AppendLine("| $n | $gD | $oD | $sign |")
        }
    }
    $null=$md.AppendLine("")
    $null=$md.AppendLine("---")
    $null=$md.AppendLine("")
}

# Raw results
$null=$md.AppendLine("## Raw Results (all individual trials)")
$null=$md.AppendLine("")
$null=$md.AppendLine("| Runtime | Backend | Model | Prompt | N | Trial | N gen | Decode t/s | Prefill t/s | TTFT ms | Gen ms | Total ms | VRAM MB | Power W |")
$null=$md.AppendLine("|---------|---------|-------|--------|--:|------:|------:|:----------:|:-----------:|:-------:|:------:|:--------:|:-------:|:-------:|")
foreach ($row in ($results | Sort-Object Runtime,Backend,Model,Prompt,N,Trial)) {
    if ($row.Err) {
        $null=$md.AppendLine("| $($row.Runtime) | $($row.Backend) | $($row.Model) | $($row.Prompt) | $($row.N) | $($row.Trial) | ERR | ERR | ERR | ERR | ERR | ERR | -- | -- |")
    } else {
        $null=$md.AppendLine("| $($row.Runtime) | $($row.Backend) | $($row.Model) | $($row.Prompt) | $($row.N) | $($row.Trial) | $($row.NGen) | $($row.DecodeTS) | $($row.PrefillTS) | $($row.PrefillMs) | $($row.GenMs) | $($row.TotalMs) | $($row.VramMB) | $($row.PowerW) |")
    }
}
$null=$md.AppendLine("")

$md.ToString() | Set-Content $OUT_MD -Encoding UTF8
Write-Host "  MD:  $OUT_MD" -ForegroundColor Green
Write-Host ""
Write-Host "  Done. $(($results|Where-Object{-not $_.Err}).Count) / $($results.Count) runs succeeded." -ForegroundColor Green
Write-Host ""

# ─── Generate graphs ──────────────────────────────────────────────────────────
$pyCmd = if (Get-Command python  -ErrorAction SilentlyContinue) { "python"  } `
         elseif (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" } else { $null }
if ($pyCmd) {
    Write-Host "  Generating resource graphs..." -ForegroundColor Cyan
    & $pyCmd "$PSScriptRoot\benchmark_graph.py"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Graphs saved to: $PSScriptRoot\benchmark_graphs\" -ForegroundColor Green
    }
} else {
    Write-Host "  (python not found — run benchmark_graph.py manually)" -ForegroundColor DarkGray
}
Write-Host ""
