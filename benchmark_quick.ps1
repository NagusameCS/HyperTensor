# benchmark_quick.ps1
# Quick benchmark: all 4 phases (GeoCPU, GeoGPU, OllGPU, OllCPU)
# Models: smollm2-135m (fast) + one medium (gemma4-2b if available)
# N: 40 / 128 / 512   Warmup: 1   Trials: 3
# Writes: benchmark_extended.csv  benchmark_timeseries.csv  benchmark_graphs/

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

# ─── Config ───────────────────────────────────────────────────────────────────
$GEO       = ".\build_host\geodessical.exe"
$MODEL_DIR = "C:\Users\legom\TensorOS\models"
$MODELS_GEO = @(
    [PSCustomObject]@{ Name="smollm2-135m"; Gguf="$MODEL_DIR\smollm2-135m-instruct-q8_0.gguf";   Quant="Q8_0";  SizeMB=138  }
    [PSCustomObject]@{ Name="gemma4-2b";    Gguf="$MODEL_DIR\google_gemma-4-E2B-it-Q4_0.gguf";   Quant="Q4_0";  SizeMB=3222 }
    [PSCustomObject]@{ Name="phi35-mini";   Gguf="$MODEL_DIR\Phi-3.5-mini-instruct-Q4_0.gguf";   Quant="Q4_0";  SizeMB=2081 }
)
$MODELS_OLL = @(
    [PSCustomObject]@{ Name="smollm2-135m"; Id="smollm2:135m" }
    [PSCustomObject]@{ Name="gemma4-2b";    Id="gemma4-2b"     }
    [PSCustomObject]@{ Name="phi35-mini";   Id="phi35test"     }
)
$PROMPTS = @(
    [PSCustomObject]@{ Name="short";  Text="The quick brown fox jumps over the lazy dog." }
    [PSCustomObject]@{ Name="medium"; Text="Explain what a transformer neural network is and how attention works." }
    [PSCustomObject]@{ Name="long";   Text="Explain transformer attention mechanisms in detail, including queries, keys, values, and multi-head attention. Discuss layer normalization and its training benefits. Describe how positional encoding works." }
)
$NS         = @(40, 128, 512)   # used for GPU phases
$NS_CPU     = @(40, 128)        # CPU phases: skip N=512 (slow)
$WARMUPS    = 1
$TRIALS     = 3

# CPU-phase model lists: only fast models to avoid hour-long waits
$MODELS_GEO_CPU = @($MODELS_GEO | Where-Object { $_.Name -eq "smollm2-135m" })
$MODELS_OLL_CPU = @($MODELS_OLL | Where-Object { $_.Name -eq "smollm2-135m" })
$OLL_URL = "http://localhost:11434/api/generate"

$OUT_CSV   = "$PSScriptRoot\benchmark_extended.csv"
$OUT_MD    = "$PSScriptRoot\benchmark_extended.md"
$TMP_MON   = "$env:TEMP\geo_quick_monitor.csv"
$TMP_TS    = "$PSScriptRoot\benchmark_timeseries.csv"
$STAGE_FILE = "$env:TEMP\geo_quick_stage.txt"

$results = [System.Collections.ArrayList]::new()

# ─── GPU availability ──────────────────────────────────────────────────────────
$HAS_GPU = $false
try {
    $smiTest = nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>$null
    $HAS_GPU = ($smiTest -ne $null -and $smiTest -ne "")
} catch {}
Write-Host "  GPU available: $HAS_GPU" -ForegroundColor DarkGray

# ─── Helpers ──────────────────────────────────────────────────────────────────
function Set-Stage($label) { $label | Set-Content $STAGE_FILE -Encoding UTF8 -NoNewline }

function Start-HwMonitor {
    "" | Set-Content $TMP_MON -Encoding UTF8
    $job = Start-Job -ScriptBlock {
        param($monFile, $tsFile, $stageFile, $hasGpu)
        while ($true) {
            $gpu = 0; $vram = 0; $power = 0.0
            if ($hasGpu) {
                try {
                    $smi = nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw `
                        --format=csv,noheader,nounits 2>$null
                    if ($smi) {
                        $p = ($smi | Select-Object -First 1).Trim() -split ',\s*'
                        $gpu   = [int]$p[0]
                        $vram  = [int]$p[1]
                        $power = [double]$p[2]
                    }
                } catch {}
            }
            $ram = 0
            try {
                $geo = Get-Process -Name geodessical -ErrorAction SilentlyContinue |
                    Sort-Object WorkingSet64 -Descending | Select-Object -First 1
                if ($geo) { $ram = [int]($geo.WorkingSet64 / 1MB) }
            } catch {}
            "$gpu,$vram,$power,$ram" | Add-Content $monFile -Encoding UTF8
            try {
                $stage = if (Test-Path $stageFile) {
                    (Get-Content $stageFile -Raw -ErrorAction SilentlyContinue).Trim()
                } else { 'idle' }
                $ts = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
                "$ts,$gpu,$vram,$power,$ram,$stage" | Add-Content $tsFile -Encoding UTF8
            } catch {}
            Start-Sleep -Milliseconds 400
        }
    } -ArgumentList $TMP_MON, $TMP_TS, $STAGE_FILE, $HAS_GPU
    return $job
}

function Stop-HwMonitor($job) {
    Stop-Job  $job -ErrorAction SilentlyContinue
    Remove-Job $job -Force -ErrorAction SilentlyContinue
}

function Read-HwStats {
    try {
        $rows = Get-Content $TMP_MON -ErrorAction SilentlyContinue |
            Where-Object { $_ -match '^\d' } |
            ForEach-Object {
                $p = $_ -split ','
                [PSCustomObject]@{
                    Gpu=[int]$p[0]; Vram=[int]$p[1]; Power=[double]$p[2]; Ram=[int]$p[3]
                }
            }
        if (-not $rows) { return @{GpuPct=0;Vram=0;Power=0;Ram=0} }
        return @{
            GpuPct = [math]::Round(($rows | Measure-Object Gpu   -Average).Average, 1)
            Vram   = ($rows | Measure-Object Vram  -Maximum).Maximum
            Power  = [math]::Round(($rows | Measure-Object Power -Average).Average, 1)
            Ram    = ($rows | Measure-Object Ram   -Maximum).Maximum
        }
    } catch { return @{GpuPct=0;Vram=0;Power=0;Ram=0} }
}

function Avg($vals) {
    $v = $vals | Where-Object { $_ -gt 0 }
    if (-not $v) { return 0 }
    return [math]::Round(($v | Measure-Object -Average).Average, 1)
}
function F($n) { if ($null -eq $n -or $n -eq 0) { "--" } else { $n } }

function Record($rt, $be, $mdl, $quant, $pn, $nr, $tr, $r, $hw) {
    [void]$results.Add([PSCustomObject]@{
        Machine="local"; Runtime=$rt; Backend=$be; Model=$mdl; Quant=$quant
        Prompt=$pn; N=$nr; Trial=$tr
        DecodeTS=$r.DeT; PrefillTS=$r.PrT; PrefillMs=$r.PrMs; GenMs=$r.GnMs
        TotalMs=$r.TtMs; NGen=$r.NG; Err=$r.Err
        VramMB=$hw.Vram; RamMB=$hw.Ram; GpuPctAvg=$hw.GpuPct; PowerW=$hw.Power
    })
    if ($r.Err) {
        Write-Host ("    [ERR] {0} {1} n={2} t={3}: {4}" -f $be,$pn,$nr,$tr,$r.Err) -ForegroundColor Red
    } else {
        Write-Host ("    [{0}] {1} n={2} t={3}  dec={4}t/s  vram={5}MB  gpu={6}%  pwr={7}W" -f `
            $be,$pn,$nr,$tr,$r.DeT,$hw.Vram,$hw.GpuPct,$hw.Power)
    }
}

# ─── Geodessical runner ────────────────────────────────────────────────────────
function Invoke-Geo($gguf, $text, $n) {
    try {
        $raw = (& $GEO $gguf -p $text -n $n 2>&1) -join "`n"
        $mG  = [regex]::Match($raw, '\[GD\] (\d+) tokens in (\d+) ms \(([\d.]+) tok/s\)')
        $mP  = [regex]::Match($raw, '\[GD\] Decode-only: prefill ([\d.]+) ms, ([\d.]+) tok/s')
        if (-not $mG.Success) {
            $mG2 = [regex]::Match($raw, 'Generated (\d+) tokens.*?(\d+) ms.*?([\d.]+) tok/s')
            if ($mG2.Success) {
                return @{Err=$null;DeT=[double]$mG2.Groups[3].Value;PrT=0;PrMs=0
                         GnMs=[int]$mG2.Groups[2].Value;TtMs=[int]$mG2.Groups[2].Value;NG=[int]$mG2.Groups[1].Value}
            }
            # Try simpler pattern
            $mS = [regex]::Match($raw, '([\d.]+)\s+tokens?/s')
            if ($mS.Success) {
                return @{Err=$null;DeT=[double]$mS.Groups[1].Value;PrT=0;PrMs=0;GnMs=0;TtMs=0;NG=$n}
            }
            return @{Err="no output pattern: $($raw.Substring(0,[math]::Min(120,$raw.Length)))";DeT=0;PrT=0;PrMs=0;GnMs=0;TtMs=0;NG=0}
        }
        $ng=[int]$mG.Groups[1].Value; $gms=[int]$mG.Groups[2].Value; $dec=[double]$mG.Groups[3].Value
        $pms=0; $pts=0
        if ($mP.Success) { $pms=[double]$mP.Groups[1].Value; $pts=[double]$mP.Groups[2].Value }
        return @{Err=$null;DeT=$dec;PrT=$pts;PrMs=$pms;GnMs=$gms;TtMs=[int]($gms+$pms);NG=$ng}
    } catch {
        return @{Err=$_.Exception.Message;DeT=0;PrT=0;PrMs=0;GnMs=0;TtMs=0;NG=0}
    }
}

# ─── Ollama runner ─────────────────────────────────────────────────────────────
function Invoke-OllAPI($mdl, $text, $n, $gpu) {
    try {
        $opts = @{num_predict=$n}
        if (-not $gpu) { $opts.num_gpu = 0 }
        $body = @{model=$mdl;prompt=$text;stream=$false;options=$opts} | ConvertTo-Json -Depth 4
        $resp = Invoke-WebRequest -Uri $OLL_URL -Method POST -Body $body `
            -ContentType "application/json" -UseBasicParsing -TimeoutSec 600
        $j   = $resp.Content | ConvertFrom-Json
        $dec = if ($j.eval_duration -gt 0)        { [math]::Round($j.eval_count/($j.eval_duration/1e9),1) } else { 0 }
        $prt = if ($j.prompt_eval_duration -gt 0) { [math]::Round($j.prompt_eval_count/($j.prompt_eval_duration/1e9),1) } else { 0 }
        $pms = [math]::Round($j.prompt_eval_duration/1e6,1)
        $gms = [math]::Round($j.eval_duration/1e6,0)
        return @{Err=$null;DeT=$dec;PrT=$prt;PrMs=$pms;GnMs=$gms
                 TtMs=[math]::Round(($j.eval_duration+$j.prompt_eval_duration)/1e6,0);NG=$j.eval_count}
    } catch {
        return @{Err=$_.Exception.Message;DeT=0;PrT=0;PrMs=0;GnMs=0;TtMs=0;NG=0}
    }
}

function EnsureOllama {
    try { $null = Invoke-WebRequest "http://localhost:11434" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop; return }
    catch {}
    Write-Host "  Starting ollama serve..." -ForegroundColor Yellow
    $null = Start-Process "ollama" -ArgumentList "serve" -PassThru -WindowStyle Hidden
    Start-Sleep 6
}

# ─── Hardware info ─────────────────────────────────────────────────────────────
$cpuInfo = (Get-CimInstance Win32_Processor | Select-Object -First 1).Name
$gpuInfo = (Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA|AMD|Intel' } |
    Select-Object -First 1).Name
$ramGB   = [math]::Round((Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum / 1GB, 0)

Write-Host ""
Write-Host "  HyperTensor Quick Benchmark" -ForegroundColor Cyan
Write-Host "  CPU: $cpuInfo" -ForegroundColor DarkGray
Write-Host "  GPU: $gpuInfo" -ForegroundColor DarkGray
Write-Host "  RAM: ${ramGB} GB" -ForegroundColor DarkGray
Write-Host "  N: $($NS -join ' / ')   Warmup: $WARMUPS   Trials: $TRIALS" -ForegroundColor DarkGray
Write-Host ""

# Init time-series log (with header for graph script)
"Timestamp,GpuPct,VramMB,PowerW,RamMB,Stage" | Set-Content $TMP_TS -Encoding UTF8

# ─── Phase 1: Geodessical CPU ──────────────────────────────────────────────────
Write-Host "=== Phase 1: Geodessical CPU ===" -ForegroundColor Cyan

if (-not (Test-Path $GEO)) {
    Write-Host "  Building CPU binary..." -ForegroundColor Yellow
    & .\build_host.ps1 -NoCuda 2>&1 | Select-String 'SUCCESS|ERROR|FAIL'
}

foreach ($m in $MODELS_GEO_CPU) {
    if (-not (Test-Path $m.Gguf)) {
        Write-Host "  SKIP $($m.Name) — GGUF not found: $($m.Gguf)" -ForegroundColor DarkGray
        continue
    }
    Write-Host "  $($m.Name):" -ForegroundColor White
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS_CPU) {
            for ($w = 1; $w -le $WARMUPS; $w++) { $null = Invoke-Geo $m.Gguf $p.Text $n }
            for ($t = 1; $t -le $TRIALS; $t++) {
                Set-Stage "GeoCPU|$($m.Name)|$($p.Name)|n$n|t$t"
                $mon = Start-HwMonitor
                "" | Set-Content $TMP_MON -Encoding UTF8
                $r  = Invoke-Geo $m.Gguf $p.Text $n
                $hw = Read-HwStats
                Stop-HwMonitor $mon
                Record "Geodessical" "CPU" $m.Name $m.Quant $p.Name $n $t $r $hw
            }
        }
    }
}

# ─── Phase 2: Geodessical GPU ──────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Phase 2: Geodessical GPU ===" -ForegroundColor Cyan

if ($HAS_GPU) {
    Write-Host "  Building GPU binary..." -ForegroundColor Yellow
    & .\build_host.ps1 2>&1 | Select-String 'SUCCESS|ERROR|FAIL|CUDA'
}

foreach ($m in $MODELS_GEO) {
    if (-not (Test-Path $m.Gguf)) { continue }
    Write-Host "  $($m.Name):" -ForegroundColor White
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS) {
            for ($w = 1; $w -le $WARMUPS; $w++) { $null = Invoke-Geo $m.Gguf $p.Text $n }
            for ($t = 1; $t -le $TRIALS; $t++) {
                Set-Stage "GeoGPU|$($m.Name)|$($p.Name)|n$n|t$t"
                $mon = Start-HwMonitor
                "" | Set-Content $TMP_MON -Encoding UTF8
                $r  = Invoke-Geo $m.Gguf $p.Text $n
                $hw = Read-HwStats
                Stop-HwMonitor $mon
                Record "Geodessical" "GPU" $m.Name $m.Quant $p.Name $n $t $r $hw
            }
        }
    }
}

# ─── Phase 3: Ollama GPU ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Phase 3: Ollama GPU ===" -ForegroundColor Cyan
EnsureOllama

foreach ($m in $MODELS_OLL) {
    Write-Host "  $($m.Name):" -ForegroundColor White
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS) {
            for ($w = 1; $w -le $WARMUPS; $w++) { $null = Invoke-OllAPI $m.Id $p.Text $n $true }
            for ($t = 1; $t -le $TRIALS; $t++) {
                Set-Stage "OllGPU|$($m.Name)|$($p.Name)|n$n|t$t"
                $mon = Start-HwMonitor
                "" | Set-Content $TMP_MON -Encoding UTF8
                $r  = Invoke-OllAPI $m.Id $p.Text $n $true
                $hw = Read-HwStats
                Stop-HwMonitor $mon
                Record "Ollama" "GPU" $m.Name "ollama" $p.Name $n $t $r $hw
            }
        }
    }
}

# ─── Phase 4: Ollama CPU ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Phase 4: Ollama CPU ===" -ForegroundColor Cyan

foreach ($m in $MODELS_OLL_CPU) {
    Write-Host "  $($m.Name):" -ForegroundColor White
    foreach ($p in $PROMPTS) {
        foreach ($n in $NS_CPU) {
            for ($w = 1; $w -le $WARMUPS; $w++) { $null = Invoke-OllAPI $m.Id $p.Text $n $false }
            for ($t = 1; $t -le $TRIALS; $t++) {
                Set-Stage "OllCPU|$($m.Name)|$($p.Name)|n$n|t$t"
                $mon = Start-HwMonitor
                "" | Set-Content $TMP_MON -Encoding UTF8
                $r  = Invoke-OllAPI $m.Id $p.Text $n $false
                $hw = Read-HwStats
                Stop-HwMonitor $mon
                Record "Ollama" "CPU" $m.Name "ollama" $p.Name $n $t $r $hw
            }
        }
    }
}

# ─── Write CSV (same format as benchmark_extended.csv) ────────────────────────
Write-Host ""
Write-Host "=== Writing Results ===" -ForegroundColor Cyan

$csvHeader = "Machine,Runtime,Backend,Model,Quant,Prompt,N,Trial,NGen,DecodeTS,PrefillTS,PrefillMs,GenMs,TotalMs,VramMB,RamMB,GpuPctAvg,PowerW,Err"
$csvLines  = $results | ForEach-Object {
    "$($_.Machine),$($_.Runtime),$($_.Backend),$($_.Model),$($_.Quant),$($_.Prompt),$($_.N),$($_.Trial),$($_.NGen),$($_.DecodeTS),$($_.PrefillTS),$($_.PrefillMs),$($_.GenMs),$($_.TotalMs),$($_.VramMB),$($_.RamMB),$($_.GpuPctAvg),$($_.PowerW),$($_.Err)"
}
@($csvHeader) + $csvLines | Set-Content $OUT_CSV -Encoding UTF8
Write-Host "  CSV written: $OUT_CSV ($($results.Count) rows)" -ForegroundColor Green

# ─── Print inline summary table ───────────────────────────────────────────────
Write-Host ""
Write-Host "  ┌────────────────────────────────────────────────────────────────────────────────────────┐" -ForegroundColor Cyan
Write-Host "  │ Runtime       Backend  Model            Dec t/s  Pre t/s  VRAM MB  GPU%   Power W     │" -ForegroundColor Cyan
Write-Host "  ├────────────────────────────────────────────────────────────────────────────────────────┤" -ForegroundColor Cyan

$combos = $results | Select-Object Runtime,Backend,Model -Unique | Sort-Object Runtime,Backend,Model
foreach ($c in $combos) {
    $rows = $results | Where-Object {
        $_.Runtime -eq $c.Runtime -and $_.Backend -eq $c.Backend -and
        $_.Model -eq $c.Model -and -not $_.Err
    }
    if (-not $rows) { continue }
    $aD  = [math]::Round((($rows | Where-Object {$_.DecodeTS -gt 0} | Measure-Object DecodeTS -Average).Average), 1)
    $aP  = [math]::Round((($rows | Where-Object {$_.PrefillTS -gt 0} | Measure-Object PrefillTS -Average).Average), 1)
    $aV  = ($rows | Measure-Object VramMB   -Maximum).Maximum
    $aG  = [math]::Round((($rows | Measure-Object GpuPctAvg -Average).Average), 1)
    $aW  = [math]::Round((($rows | Where-Object {$_.PowerW -gt 0} | Measure-Object PowerW -Average).Average), 1)
    $line = "  │ {0,-13}  {1,-7}  {2,-15}  {3,7}  {4,7}  {5,7}  {6,5}  {7,9} │" -f `
        $c.Runtime, $c.Backend, $c.Model, (F $aD), (F $aP), (F $aV), (F $aG), (F $aW)
    Write-Host $line
}
Write-Host "  └────────────────────────────────────────────────────────────────────────────────────────┘" -ForegroundColor Cyan

# ─── Generate graphs ──────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Generating Graphs ===" -ForegroundColor Cyan
python benchmark_graph.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Graphs written to: benchmark_graphs\" -ForegroundColor Green
    Get-ChildItem benchmark_graphs\*.png | ForEach-Object { Write-Host "    $($_.Name)" }
} else {
    Write-Host "  Graph generation failed (is matplotlib installed?)" -ForegroundColor Yellow
    Write-Host "  Install with: pip install matplotlib numpy" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  Done. Total rows: $($results.Count)" -ForegroundColor Green
