# benchmark_peak.ps1
# Head-to-head: Geodessical GPU vs Ollama GPU on gemma4-2b at optimal conditions
# Metrics: Decode t/s, Prefill t/s, TTFT, E2E ms + GPU util%, VRAM MB, Power W, CPU%, Process RAM
# Optimal = GPU backend, long+short prompts, N=128 and N=512, 5 trials (1 warmup discarded)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$GEO_EXE  = ".\build_host\geodessical.exe"
$M_GEMMA  = "C:\Users\legom\TensorOS\models\google_gemma-4-E2B-it-Q4_0.gguf"
$OLL_URL  = "http://localhost:11434/api/generate"
$OLL_MDL  = "gemma4-2b"
$TRIALS   = 5          # measured trials per condition (plus 1 warmup each)
$TMP_MON  = "$env:TEMP\geo_peak_monitor.csv"
$OUT_MD   = "benchmark_peak.md"

$PROMPTS = @(
    [PSCustomObject]@{ Name="long";  Text="Explain transformer attention mechanisms in detail, including queries, keys, values, and multi-head attention. Discuss layer normalization and its training benefits." }
    [PSCustomObject]@{ Name="short"; Text="The quick brown fox jumps" }
)
$NS = @(128, 512)

# ============================================================
# Resource monitor (background job)
# Samples nvidia-smi + process RAM + system CPU every 500 ms
# ============================================================
function Start-Monitor {
    param([string]$TmpFile, [string[]]$ProcNames)
    "" | Set-Content $TmpFile -Encoding UTF8
    $job = Start-Job -ScriptBlock {
        param($f, $pnames)
        while ($true) {
            # GPU stats
            try {
                $smi   = nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2>$null
                $parts = ($smi | Select-Object -First 1).Trim() -split ',\s*'
                $gpuU  = [int]$parts[0]; $vram = [int]$parts[1]; $watt = [double]$parts[2]
            } catch { $gpuU=0; $vram=0; $watt=0 }
            # Process RAM (largest matching process)
            $ramMB = 0
            foreach ($pn in $pnames) {
                try {
                    $proc = Get-Process -Name $pn -ErrorAction SilentlyContinue |
                            Sort-Object WorkingSet64 -Descending | Select-Object -First 1
                    if ($proc) { $ramMB = [int]($proc.WorkingSet64/1MB); break }
                } catch {}
            }
            # System CPU %
            try {
                $cpuPct = [int]((Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average)
            } catch { $cpuPct = 0 }
            # timestamp,gpuUtil,vramMB,wattW,ramMB,cpuPct
            "$(Get-Date -Format HH:mm:ss.fff),$gpuU,$vram,$watt,$ramMB,$cpuPct" | Add-Content $f
            Start-Sleep -Milliseconds 500
        }
    } -ArgumentList $TmpFile, $ProcNames
    return $job
}

function Stop-Monitor {
    param($Job, [string]$TmpFile)
    Stop-Job  $Job -ErrorAction SilentlyContinue
    Remove-Job $Job -Force  -ErrorAction SilentlyContinue
    $lines = @(Get-Content $TmpFile -ErrorAction SilentlyContinue | Where-Object { $_ -match '^\d{2}:\d{2}' })
    if ($lines.Count -eq 0) { return @{AvgGPU=0; PeakVRAM=0; AvgWatt=0; PeakRAM=0; AvgCPU=0; Samples=0} }
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
        PeakVRAM = [math]::Round(($vArr | Measure-Object -Maximum).Maximum, 0)
        AvgWatt  = [math]::Round(($wArr | Measure-Object -Average).Average, 1)
        PeakRAM  = [math]::Round(($rArr | Measure-Object -Maximum).Maximum, 0)
        AvgCPU   = [math]::Round(($cArr | Measure-Object -Average).Average, 1)
        Samples  = $lines.Count
    }
}

# ============================================================
# Inference runners
# ============================================================
function Invoke-Geo {
    param([string]$Text, [int]$N)
    try {
        $mon = Start-Monitor -TmpFile $TMP_MON -ProcNames @("geodessical")
        $raw = (& $GEO_EXE $M_GEMMA -p $Text -n $N 2>&1) -join "`n"
        $res = Stop-Monitor -Job $mon -TmpFile $TMP_MON

        $mG = [regex]::Match($raw, '\[GD\] (\d+) tokens in (\d+) ms \(([\d.]+) tok/s\)')
        $mP = [regex]::Match($raw, '\[GD\] Decode-only: prefill ([\d.]+) ms, ([\d.]+) tok/s')
        if (-not $mG.Success) { return @{ Err="no GD output"; Res=$res } }
        $ng   = [int]$mG.Groups[1].Value
        $gms  = [int]$mG.Groups[2].Value
        $dec  = [double]$mG.Groups[3].Value
        $pms  = if ($mP.Success) { [double]$mP.Groups[1].Value } else { 0 }
        $pts  = if ($mP.Success) { [double]$mP.Groups[2].Value } else { 0 }
        return @{ Err=$null; DeT=$dec; PrT=$pts; PrMs=$pms; GnMs=$gms; TtMs=[int]($pms+$gms); NG=$ng; Res=$res }
    } catch { return @{ Err=$_.Exception.Message; Res=@{AvgGPU=0;PeakVRAM=0;AvgWatt=0;PeakRAM=0;AvgCPU=0} } }
}

function Invoke-Oll {
    param([string]$Text, [int]$N)
    try {
        # ollama_llama_server or runner_cuda_v0 depending on version; try several names
        $mon  = Start-Monitor -TmpFile $TMP_MON -ProcNames @("ollama_llama_server","ollama_runner","ollama")
        $opts = @{ num_predict=$N; num_gpu=999 }
        $body = @{ model=$OLL_MDL; prompt=$Text; stream=$false; options=$opts } | ConvertTo-Json
        $resp = Invoke-WebRequest -Uri $OLL_URL -Method POST -Body $body `
                    -ContentType "application/json" -UseBasicParsing -TimeoutSec 600
        $res  = Stop-Monitor -Job $mon -TmpFile $TMP_MON

        $j   = $resp.Content | ConvertFrom-Json
        $dec = if ($j.eval_duration -gt 0)        { [math]::Round($j.eval_count        / ($j.eval_duration        / 1e9), 1) } else { 0 }
        $prt = if ($j.prompt_eval_duration -gt 0) { [math]::Round($j.prompt_eval_count / ($j.prompt_eval_duration / 1e9), 1) } else { 0 }
        $pms = [math]::Round($j.prompt_eval_duration / 1e6, 1)
        $gms = [math]::Round($j.eval_duration        / 1e6, 0)
        $tms = [math]::Round(($j.eval_duration + $j.prompt_eval_duration) / 1e6, 0)
        return @{ Err=$null; DeT=$dec; PrT=$prt; PrMs=$pms; GnMs=$gms; TtMs=$tms; NG=$j.eval_count; Res=$res }
    } catch { return @{ Err=$_.Exception.Message; Res=@{AvgGPU=0;PeakVRAM=0;AvgWatt=0;PeakRAM=0;AvgCPU=0} } }
}

# ============================================================
# Ensure Ollama is running
# ============================================================
function EnsureOllama {
    try { $null = Invoke-WebRequest "http://localhost:11434" -UseBasicParsing -TimeoutSec 3 2>$null }
    catch {
        Write-Host "  Starting ollama serve..." -ForegroundColor Yellow
        $null = Start-Process "ollama" -ArgumentList "serve" -PassThru -WindowStyle Hidden
        Start-Sleep 5
    }
}

# ============================================================
# Collect results
# ============================================================
$results = [System.Collections.ArrayList]::new()
function Rec($rt, $pname, $n, $tr, $r) {
    [void]$results.Add([PSCustomObject]@{
        RT=$rt; Pn=$pname; N=$n; Tr=$tr
        DeT=$r.DeT; PrT=$r.PrT; PrMs=$r.PrMs; GnMs=$r.GnMs; TtMs=$r.TtMs; NG=$r.NG; Err=$r.Err
        AvgGPU=$r.Res.AvgGPU; PeakVRAM=$r.Res.PeakVRAM; AvgWatt=$r.Res.AvgWatt
        PeakRAM=$r.Res.PeakRAM; AvgCPU=$r.Res.AvgCPU
    })
    if ($r.Err) {
        Write-Host "  [$rt $pname n=$n t=$tr] ERR: $($r.Err)" -ForegroundColor Red
    } else {
        Write-Host ("  [$rt $pname n=$n t=$tr]  dec={0,6} t/s  ttft={1,5} ms  e2e={2,6} ms  n={3,4}  " +
                    "GPU={4,3}%  VRAM={5,5}MB  {6,4}W  CPU={7,4}%  RAM={8,5}MB") `
            -f $r.DeT, $r.PrMs, $r.TtMs, $r.NG,
               $r.Res.AvgGPU, $r.Res.PeakVRAM, $r.Res.AvgWatt,
               $r.Res.AvgCPU, $r.Res.PeakRAM
    }
}
function Avg { param($vals) $v = $vals | Where-Object {$_ -gt 0}; if (-not $v) {return 0}
    [math]::Round(($v | Measure-Object -Average).Average, 1) }
function F($n,$d=1) { if ([math]::Abs($n) -lt 0.001) { "--" } else { [math]::Round($n,$d) } }

# ============================================================
# Phase 1: Geodessical GPU (build first)
# ============================================================
Write-Host "`n=== Building Geodessical (GPU) ===" -ForegroundColor Cyan
& .\build_host.ps1 2>&1 | Where-Object { $_ -match 'SUCCESS|ERROR|CUDA|warning' }
Write-Host ""

Write-Host "=== Geodessical GPU - Warmup ===" -ForegroundColor DarkCyan
foreach ($p in $PROMPTS) { foreach ($n in $NS) {
    Write-Host "  [Warmup Geo $($p.Name) n=$n]" -NoNewline
    $null = Invoke-Oll $p.Text $n   # just keep GPU alive; don't record
    $wr = Invoke-Geo $p.Text $n; Write-Host "  done (dec=$($wr.DeT) t/s)"
}}

Write-Host "`n=== Geodessical GPU - Measured Trials ===" -ForegroundColor Cyan
foreach ($p in $PROMPTS) { foreach ($n in $NS) { for ($t=1; $t -le $TRIALS; $t++) {
    $r = Invoke-Geo $p.Text $n; Rec "Geo-GPU" $p.Name $n $t $r
}}}

# ============================================================
# Phase 2: Ollama GPU
# ============================================================
EnsureOllama

Write-Host "`n=== Ollama GPU - Warmup (load model + prime KV cache) ===" -ForegroundColor DarkCyan
foreach ($p in $PROMPTS) { foreach ($n in $NS) {
    Write-Host "  [Warmup Oll $($p.Name) n=$n]" -NoNewline
    $wr = Invoke-Oll $p.Text $n; Write-Host "  done (dec=$($wr.DeT) t/s  ttft=$($wr.PrMs) ms)"
}}

Write-Host "`n=== Ollama GPU - Measured Trials ===" -ForegroundColor Cyan
foreach ($p in $PROMPTS) { foreach ($n in $NS) { for ($t=1; $t -le $TRIALS; $t++) {
    $r = Invoke-Oll $p.Text $n; Rec "Oll-GPU" $p.Name $n $t $r
}}}

# ============================================================
# Generate report
# ============================================================
Write-Host "`n=== Generating Report ===" -ForegroundColor Cyan

$gpuInfo = (Get-WmiObject Win32_VideoController | Select-Object -First 1).Name
$cpuInfo = (Get-WmiObject Win32_Processor       | Select-Object -First 1).Name
$vramTotal = (nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1).Trim()

$md = [System.Text.StringBuilder]::new()
$null = $md.AppendLine("# Peak Benchmark: Geodessical GPU vs Ollama GPU")
$null = $md.AppendLine("")
$null = $md.AppendLine("**Date:** $(Get-Date -Format 'yyyy-MM-dd HH:mm')")
$null = $md.AppendLine("**CPU:** $cpuInfo")
$null = $md.AppendLine("**GPU:** $gpuInfo  |  VRAM total: ${vramTotal} MB")
$null = $md.AppendLine("**Model:** gemma4-2b (google_gemma-4-E2B-it Q4_0, 3.2 GB)")
$null = $md.AppendLine("**Trials:** $TRIALS measured per condition + 1 warmup discarded")
$null = $md.AppendLine("**Conditions:** GPU backend only -- optimal settings for each runtime")
$null = $md.AppendLine("")
$null = $md.AppendLine("**Metrics:**")
$null = $md.AppendLine("- Decode t/s = decode-phase tokens per second")
$null = $md.AppendLine("- Prefill t/s = prompt processing throughput")
$null = $md.AppendLine("- TTFT ms = Time To First Token")
$null = $md.AppendLine("- E2E ms = total wall time (TTFT + decode)")
$null = $md.AppendLine("- Avg GPU% = mean GPU utilization during inference")
$null = $md.AppendLine("- Peak VRAM = max VRAM used during inference (MB)")
$null = $md.AppendLine("- Avg Watt = mean GPU power draw (W)")
$null = $md.AppendLine("- Avg CPU% = mean system CPU load during inference")
$null = $md.AppendLine("- Peak RAM = peak process working set (MB)")
$null = $md.AppendLine("")
$null = $md.AppendLine("---")
$null = $md.AppendLine("")

# ---- Performance table ----
$null = $md.AppendLine("## Performance: Head-to-Head")
$null = $md.AppendLine("")
foreach ($pname in @("long","short")) {
    $null = $md.AppendLine("### Prompt: $pname")
    $null = $md.AppendLine("")
    $null = $md.AppendLine("| Runtime | N | N-gen | Decode t/s | Prefill t/s | TTFT ms | E2E ms | ms/tok |")
    $null = $md.AppendLine("|---------|--:|------:|:----------:|:-----------:|:-------:|:------:|:------:|")
    foreach ($rt in @("Geo-GPU","Oll-GPU")) {
        foreach ($n in $NS) {
            $rows = $results | Where-Object { $_.RT -eq $rt -and $_.Pn -eq $pname -and $_.N -eq $n -and -not $_.Err }
            if ($rows) {
                $aDec  = Avg ($rows | ForEach-Object { $_.DeT  })
                $aPre  = Avg ($rows | ForEach-Object { $_.PrT  })
                $aTtft = Avg ($rows | ForEach-Object { $_.PrMs })
                $aE2E  = Avg ($rows | ForEach-Object { $_.TtMs })
                $aNGen = [math]::Round(($rows | ForEach-Object { $_.NG } | Measure-Object -Average).Average, 0)
                $msTok = if ($aNGen -gt 0) { [math]::Round($aE2E / $aNGen, 1) } else { 0 }
                $null = $md.AppendLine("| $rt | $n | $aNGen | $(F $aDec) | $(F $aPre) | $(F $aTtft) | $(F $aE2E) | $(F $msTok) |")
            }
        }
    }
    # ratio row
    foreach ($n in $NS) {
        $gRows = $results | Where-Object { $_.RT -eq "Geo-GPU" -and $_.Pn -eq $pname -and $_.N -eq $n -and -not $_.Err }
        $oRows = $results | Where-Object { $_.RT -eq "Oll-GPU" -and $_.Pn -eq $pname -and $_.N -eq $n -and -not $_.Err }
        if ($gRows -and $oRows) {
            $gDec = Avg ($gRows | ForEach-Object { $_.DeT })
            $oDec = Avg ($oRows | ForEach-Object { $_.DeT })
            if ($oDec -gt 0) {
                $r = [math]::Round($gDec / $oDec, 2)
                $null = $md.AppendLine("| **Geo/Oll ratio** | **$n** | | **${r}x** | | | | |")
            }
        }
    }
    $null = $md.AppendLine("")
}

# ---- Resource table ----
$null = $md.AppendLine("---")
$null = $md.AppendLine("")
$null = $md.AppendLine("## Resource Usage: Head-to-Head")
$null = $md.AppendLine("")
foreach ($pname in @("long","short")) {
    $null = $md.AppendLine("### Prompt: $pname")
    $null = $md.AppendLine("")
    $null = $md.AppendLine("| Runtime | N | Avg GPU% | Peak VRAM MB | Avg Watt | Avg CPU% | Peak RAM MB |")
    $null = $md.AppendLine("|---------|--:|:--------:|:------------:|:--------:|:--------:|:-----------:|")
    foreach ($rt in @("Geo-GPU","Oll-GPU")) {
        foreach ($n in $NS) {
            $rows = $results | Where-Object { $_.RT -eq $rt -and $_.Pn -eq $pname -and $_.N -eq $n -and -not $_.Err }
            if ($rows) {
                $aGPU  = Avg ($rows | ForEach-Object { $_.AvgGPU   })
                $pVRAM = [math]::Round(($rows | ForEach-Object { $_.PeakVRAM } | Measure-Object -Maximum).Maximum, 0)
                $aWatt = Avg ($rows | ForEach-Object { $_.AvgWatt  })
                $aCPU  = Avg ($rows | ForEach-Object { $_.AvgCPU   })
                $pRAM  = [math]::Round(($rows | ForEach-Object { $_.PeakRAM  } | Measure-Object -Maximum).Maximum, 0)
                $null = $md.AppendLine("| $rt | $n | $(F $aGPU) | $(F $pVRAM 0) | $(F $aWatt) | $(F $aCPU) | $(F $pRAM 0) |")
            }
        }
    }
    $null = $md.AppendLine("")
}

# ---- Efficiency: decode t/s per watt ----
$null = $md.AppendLine("---")
$null = $md.AppendLine("")
$null = $md.AppendLine("## Efficiency: Decode t/s per Watt")
$null = $md.AppendLine("")
$null = $md.AppendLine("| Runtime | Prompt | N | Decode t/s | Avg Watt | t/s per W |")
$null = $md.AppendLine("|---------|--------|--:|:----------:|:--------:|:---------:|")
foreach ($rt in @("Geo-GPU","Oll-GPU")) {
    foreach ($pname in @("long","short")) {
        foreach ($n in $NS) {
            $rows = $results | Where-Object { $_.RT -eq $rt -and $_.Pn -eq $pname -and $_.N -eq $n -and -not $_.Err }
            if ($rows) {
                $aDec  = Avg ($rows | ForEach-Object { $_.DeT     })
                $aWatt = Avg ($rows | ForEach-Object { $_.AvgWatt })
                $tpw   = if ($aWatt -gt 0) { [math]::Round($aDec / $aWatt, 3) } else { 0 }
                $null = $md.AppendLine("| $rt | $pname | $n | $(F $aDec) | $(F $aWatt) | $(F $tpw 3) |")
            }
        }
    }
}
$null = $md.AppendLine("")

# ---- Raw results ----
$null = $md.AppendLine("---")
$null = $md.AppendLine("")
$null = $md.AppendLine("## Raw Results (all trials)")
$null = $md.AppendLine("")
$null = $md.AppendLine("| Runtime | Prompt | N | Trial | N-gen | Decode t/s | Prefill t/s | TTFT ms | E2E ms | GPU% | VRAM MB | Watt | CPU% | RAM MB |")
$null = $md.AppendLine("|---------|--------|--:|------:|------:|:----------:|:-----------:|:-------:|:------:|:----:|:-------:|:----:|:----:|:------:|")
foreach ($row in ($results | Sort-Object RT,Pn,N,Tr)) {
    if ($row.Err) {
        $null = $md.AppendLine("| $($row.RT) | $($row.Pn) | $($row.N) | $($row.Tr) | ERR | $($row.Err) | | | | | | | | |")
    } else {
        $null = $md.AppendLine("| $($row.RT) | $($row.Pn) | $($row.N) | $($row.Tr) | $($row.NG) | $($row.DeT) | $($row.PrT) | $($row.PrMs) | $($row.TtMs) | $($row.AvgGPU) | $($row.PeakVRAM) | $($row.AvgWatt) | $($row.AvgCPU) | $($row.PeakRAM) |")
    }
}
$null = $md.AppendLine("")

[System.IO.File]::WriteAllText("$PSScriptRoot\$OUT_MD", $md.ToString(), [System.Text.Encoding]::UTF8)
Write-Host "`n== Report written: $OUT_MD ==" -ForegroundColor Green
Write-Host "Rows: $($results.Count)"

# ---- Console summary ----
Write-Host "`n--- Performance Summary ---" -ForegroundColor Cyan
Write-Host ("{0,-12} {1,-6} {2,-5} {3,10} {4,8} {5,8}" -f "Runtime","Prompt","N","Decode t/s","TTFT ms","E2E ms")
Write-Host ("-" * 60)
foreach ($rt in @("Geo-GPU","Oll-GPU")) { foreach ($pname in @("long","short")) { foreach ($n in $NS) {
    $rows = $results | Where-Object { $_.RT -eq $rt -and $_.Pn -eq $pname -and $_.N -eq $n -and -not $_.Err }
    if ($rows) {
        Write-Host ("{0,-12} {1,-6} {2,-5} {3,10} {4,8} {5,8}" -f `
            $rt, $pname, $n, (Avg ($rows|ForEach-Object{$_.DeT})), (Avg ($rows|ForEach-Object{$_.PrMs})), (Avg ($rows|ForEach-Object{$_.TtMs})))
    }
}}}

Write-Host "`n--- Resource Summary ---" -ForegroundColor Cyan
Write-Host ("{0,-12} {1,-6} {2,-5} {3,9} {4,10} {5,9} {6,8} {7,9}" -f "Runtime","Prompt","N","Avg GPU%","Peak VRAM","Avg Watt","Avg CPU%","Peak RAM")
Write-Host ("-" * 75)
foreach ($rt in @("Geo-GPU","Oll-GPU")) { foreach ($pname in @("long","short")) { foreach ($n in $NS) {
    $rows = $results | Where-Object { $_.RT -eq $rt -and $_.Pn -eq $pname -and $_.N -eq $n -and -not $_.Err }
    if ($rows) {
        $pVRAM = [math]::Round(($rows|ForEach-Object{$_.PeakVRAM}|Measure-Object -Maximum).Maximum,0)
        $pRAM  = [math]::Round(($rows|ForEach-Object{$_.PeakRAM }|Measure-Object -Maximum).Maximum,0)
        Write-Host ("{0,-12} {1,-6} {2,-5} {3,9} {4,10} {5,9} {6,8} {7,9}" -f `
            $rt, $pname, $n,
            (Avg ($rows|ForEach-Object{$_.AvgGPU})), $pVRAM,
            (Avg ($rows|ForEach-Object{$_.AvgWatt})),
            (Avg ($rows|ForEach-Object{$_.AvgCPU})), $pRAM)
    }
}}}
