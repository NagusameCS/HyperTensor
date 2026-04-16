$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

$GEO       = ".\build_host\geodessical.exe"
$M_SMOL    = "C:\Users\legom\TensorOS\models\smollm2-135m-instruct-q8_0.gguf"
$M_GEMMA   = "C:\Users\legom\TensorOS\models\google_gemma-4-E2B-it-Q4_0.gguf"
$API       = "http://localhost:11434/api/generate"
$O_SMOL    = "smollm2:135m"
$O_GEMMA   = "gemma4-2b"
$OUT_MD    = "benchmark_results.md"
$TRIALS    = 2

$prompts = @(
    [PSCustomObject]@{ Name="short"; Text="The quick brown fox jumps" }
    [PSCustomObject]@{ Name="long";  Text="Explain transformer attention mechanisms in detail, including queries, keys, values, and multi-head attention. Discuss layer normalization and its training benefits." }
)
$ns = @(40, 128, 512)
$results = [System.Collections.ArrayList]::new()

function Rec($rt, $be, $mdl, $pn, $nr, $tr, $r) {
    [void]$results.Add([PSCustomObject]@{
        RT=$rt; BE=$be; Mdl=$mdl; Pn=$pn; NR=$nr; Tr=$tr
        DeT=$r.DeT; PrT=$r.PrT; PrMs=$r.PrMs; GnMs=$r.GnMs; TtMs=$r.TtMs; NG=$r.NG; Err=$r.Err
    })
    $s = if ($r.Err) { "(ERR: $(($r.Err).Substring(0,[math]::Min(60,$r.Err.Length))))" } else { "dec=$($r.DeT) t/s  pre=$($r.PrT) t/s  ttft=$($r.PrMs)ms  gen=$($r.GnMs)ms  n=$($r.NG)" }
    Write-Host "  [$rt/$be $mdl $pn n=$nr t=$tr]  $s"
}

function Invoke-Geo($gguf, $text, $n) {
    try {
        $raw = (& $GEO $gguf -p $text -n $n 2>&1) -join "`n"
        $mG  = [regex]::Match($raw, '\[GD\] (\d+) tokens in (\d+) ms \(([\d.]+) tok/s\)')
        $mP  = [regex]::Match($raw, '\[GD\] Decode-only: prefill ([\d.]+) ms, ([\d.]+) tok/s')
        if (-not $mG.Success) { return @{Err="no GD output";DeT=0;PrT=0;PrMs=0;GnMs=0;TtMs=0;NG=0} }
        $ng = [int]$mG.Groups[1].Value; $gms=[int]$mG.Groups[2].Value; $dec=[double]$mG.Groups[3].Value
        $pms = if ($mP.Success) {[double]$mP.Groups[1].Value} else {0}
        $pts = if ($mP.Success) {[double]$mP.Groups[2].Value} else {0}
        return @{Err=$null;DeT=$dec;PrT=$pts;PrMs=$pms;GnMs=$gms;TtMs=[int]($gms+$pms);NG=$ng}
    } catch { return @{Err=$_.Exception.Message;DeT=0;PrT=0;PrMs=0;GnMs=0;TtMs=0;NG=0} }
}

function Invoke-OllAPI($mdl, $text, $n, $gpu) {
    try {
        $opts = @{num_predict=$n}; if (-not $gpu) { $opts.num_gpu=0 }
        $body = @{model=$mdl;prompt=$text;stream=$false;options=$opts} | ConvertTo-Json
        $resp = Invoke-WebRequest -Uri $API -Method POST -Body $body -ContentType "application/json" -UseBasicParsing -TimeoutSec 600
        $j = $resp.Content | ConvertFrom-Json
        $dec  = if ($j.eval_duration -gt 0)        {[math]::Round($j.eval_count/($j.eval_duration/1e9),1)} else {0}
        $prt  = if ($j.prompt_eval_duration -gt 0) {[math]::Round($j.prompt_eval_count/($j.prompt_eval_duration/1e9),1)} else {0}
        $pms  = [math]::Round($j.prompt_eval_duration/1e6,1)
        $gms  = [math]::Round($j.eval_duration/1e6,0)
        $tms  = [math]::Round(($j.eval_duration+$j.prompt_eval_duration)/1e6,0)
        return @{Err=$null;DeT=$dec;PrT=$prt;PrMs=$pms;GnMs=$gms;TtMs=$tms;NG=$j.eval_count}
    } catch { return @{Err=$_.Exception.Message;DeT=0;PrT=0;PrMs=0;GnMs=0;TtMs=0;NG=0} }
}

function EnsureOllama {
    try { $null = Invoke-WebRequest "http://localhost:11434" -UseBasicParsing -TimeoutSec 3 2>$null }
    catch { Write-Host "  Starting ollama..."; $null=Start-Process "ollama" -ArgumentList "serve" -PassThru -WindowStyle Hidden; Start-Sleep 4 }
}

function Avg($vals) {
    $v = $vals | Where-Object {$_ -gt 0}
    if (-not $v) { return 0 }
    return [math]::Round(($v|Measure-Object -Average).Average,1)
}
function F($n) { if ($n -eq 0) {"--"} else {$n} }

$geoMdls = @(
    [PSCustomObject]@{Name="smollm2-135m";Gguf=$M_SMOL}
    [PSCustomObject]@{Name="gemma4-2b";   Gguf=$M_GEMMA}
)
$ollMdls = @(
    [PSCustomObject]@{Name="smollm2-135m";Id=$O_SMOL}
    [PSCustomObject]@{Name="gemma4-2b";   Id=$O_GEMMA}
)

Write-Host "`n=== Phase 1: Geodessical CPU ===" -ForegroundColor Cyan
& .\build_host.ps1 -NoCuda 2>&1 | Where-Object {$_ -match 'SUCCESS|ERROR|NoCuda'}
Write-Host ""
foreach ($m in $geoMdls) { foreach ($p in $prompts) { foreach ($n in $ns) { for ($t=1;$t -le $TRIALS;$t++) {
    $r = Invoke-Geo $m.Gguf $p.Text $n; Rec "Geodessical" "CPU" $m.Name $p.Name $n $t $r
} } } }

Write-Host "`n=== Phase 2: Geodessical GPU ===" -ForegroundColor Cyan
& .\build_host.ps1 2>&1 | Where-Object {$_ -match 'SUCCESS|ERROR|CUDA'}
Write-Host ""
foreach ($m in $geoMdls) { foreach ($p in $prompts) { foreach ($n in $ns) { for ($t=1;$t -le $TRIALS;$t++) {
    $r = Invoke-Geo $m.Gguf $p.Text $n; Rec "Geodessical" "GPU" $m.Name $p.Name $n $t $r
} } } }

Write-Host "`n=== Phase 3: Ollama GPU ===" -ForegroundColor Cyan
EnsureOllama
foreach ($m in $ollMdls) { foreach ($p in $prompts) { foreach ($n in $ns) { for ($t=1;$t -le $TRIALS;$t++) {
    $r = Invoke-OllAPI $m.Id $p.Text $n $true; Rec "Ollama" "GPU" $m.Name $p.Name $n $t $r
} } } }

Write-Host "`n=== Phase 4: Ollama CPU ===" -ForegroundColor Cyan
foreach ($m in $ollMdls) { foreach ($p in $prompts) { foreach ($n in $ns) { for ($t=1;$t -le $TRIALS;$t++) {
    $r = Invoke-OllAPI $m.Id $p.Text $n $false; Rec "Ollama" "CPU" $m.Name $p.Name $n $t $r
} } } }

Write-Host "`n=== Generating Report ===" -ForegroundColor Cyan

$md = [System.Text.StringBuilder]::new()

$gpuInfo = (Get-WmiObject Win32_VideoController|Select-Object -First 1).Name
$cpuInfo = (Get-WmiObject Win32_Processor    |Select-Object -First 1).Name

$null=$md.AppendLine("# TensorOS Inference Benchmark Report")
$null=$md.AppendLine("")
$null=$md.AppendLine("**Date:** $(Get-Date -Format 'yyyy-MM-dd HH:mm')")
$null=$md.AppendLine("")
$null=$md.AppendLine("**CPU:** $cpuInfo")
$null=$md.AppendLine("")
$null=$md.AppendLine("**GPU:** $gpuInfo")
$null=$md.AppendLine("")
$null=$md.AppendLine("**Models:** smollm2-135m-instruct Q8_0 (138 MB) + Gemma-4-E2B Q4_0 (3.2 GB)")
$null=$md.AppendLine("")
$null=$md.AppendLine("**Trials:** $TRIALS per condition | **Token counts:** 40 / 128 / 512")
$null=$md.AppendLine("")
$null=$md.AppendLine("**Column definitions:**")
$null=$md.AppendLine("")
$null=$md.AppendLine("- **Decode t/s** = generation tokens per second (decode phase only)")
$null=$md.AppendLine("- **Prefill t/s** = prompt tokens processed per second")
$null=$md.AppendLine("- **TTFT ms** = Time To First Token (prefill latency)")
$null=$md.AppendLine("- **Gen ms** = decode wall-time (excludes prefill)")
$null=$md.AppendLine("- **Total ms** = gen ms + TTFT ms")
$null=$md.AppendLine("")
$null=$md.AppendLine("---")
$null=$md.AppendLine("")

$null=$md.AppendLine("## Summary -- Average Across All Conditions")
$null=$md.AppendLine("")
$null=$md.AppendLine("| Runtime | Backend | Model | Decode t/s | Prefill t/s | TTFT ms | Gen ms |")
$null=$md.AppendLine("|---------|---------|-------|:----------:|:-----------:|:-------:|:------:|")
$combos = $results | Select-Object RT,BE,Mdl -Unique | Sort-Object RT,BE,Mdl
foreach ($c in $combos) {
    $rows = $results | Where-Object {$_.RT -eq $c.RT -and $_.BE -eq $c.BE -and $_.Mdl -eq $c.Mdl -and -not $_.Err}
    $aD=Avg($rows|ForEach-Object{$_.DeT}); $aP=Avg($rows|ForEach-Object{$_.PrT})
    $aT=Avg($rows|ForEach-Object{$_.PrMs}); $aG=Avg($rows|ForEach-Object{$_.GnMs})
    $null=$md.AppendLine("| $($c.RT) | $($c.BE) | $($c.Mdl) | $(F $aD) | $(F $aP) | $(F $aT) | $(F $aG) |")
}
$null=$md.AppendLine("")
$null=$md.AppendLine("---")
$null=$md.AppendLine("")

foreach ($model in ($results|Select-Object -ExpandProperty Mdl -Unique|Sort-Object)) {
    $null=$md.AppendLine("## Model: $model")
    $null=$md.AppendLine("")
    foreach ($pname in ($results|Select-Object -ExpandProperty Pn -Unique|Sort-Object)) {
        $null=$md.AppendLine("### Prompt: $pname")
        $null=$md.AppendLine("")
        $null=$md.AppendLine("| Runtime | Backend | N | N gen | Decode t/s | Prefill t/s | TTFT ms | Gen ms | Total ms |")
        $null=$md.AppendLine("|---------|---------|--:|------:|:----------:|:-----------:|:-------:|:------:|:--------:|")
        foreach ($rt in @("Geodessical","Ollama")) {
            foreach ($be in @("GPU","CPU")) {
                foreach ($n in $ns) {
                    $rows = $results|Where-Object{$_.Mdl -eq $model -and $_.Pn -eq $pname -and $_.RT -eq $rt -and $_.BE -eq $be -and $_.NR -eq $n -and -not $_.Err}
                    if ($rows) {
                        $aD=Avg($rows|ForEach-Object{$_.DeT}); $aP=Avg($rows|ForEach-Object{$_.PrT})
                        $aT=Avg($rows|ForEach-Object{$_.PrMs}); $aG=Avg($rows|ForEach-Object{$_.GnMs})
                        $aTt=Avg($rows|ForEach-Object{$_.TtMs}); $aN=[int](Avg($rows|ForEach-Object{$_.NG}))
                        $null=$md.AppendLine("| $rt | $be | $n | $aN | $(F $aD) | $(F $aP) | $(F $aT) | $(F $aG) | $(F $aTt) |")
                    } else {
                        $eRows = $results|Where-Object{$_.Mdl -eq $model -and $_.Pn -eq $pname -and $_.RT -eq $rt -and $_.BE -eq $be -and $_.NR -eq $n}
                        if ($eRows) { $null=$md.AppendLine("| $rt | $be | $n | -- | ERR | ERR | ERR | ERR | ERR |") }
                    }
                }
            }
        }
        $null=$md.AppendLine("")
    }
    $null=$md.AppendLine("### GPU vs CPU Speedup -- $model")
    $null=$md.AppendLine("")
    $null=$md.AppendLine("| Runtime | N | GPU Decode t/s | CPU Decode t/s | Speedup |")
    $null=$md.AppendLine("|---------|--:|:--------------:|:--------------:|:-------:|")
    foreach ($rt in @("Geodessical","Ollama")) {
        foreach ($n in $ns) {
            $gR=$results|Where-Object{$_.Mdl -eq $model -and $_.RT -eq $rt -and $_.BE -eq "GPU" -and $_.NR -eq $n -and -not $_.Err}
            $cR=$results|Where-Object{$_.Mdl -eq $model -and $_.RT -eq $rt -and $_.BE -eq "CPU" -and $_.NR -eq $n -and -not $_.Err}
            $gD=Avg($gR|ForEach-Object{$_.DeT}); $cD=Avg($cR|ForEach-Object{$_.DeT})
            if ($cD -gt 0 -and $gD -gt 0) {
                $sp=[math]::Round($gD/$cD,1)
                $null=$md.AppendLine("| $rt | $n | $gD | $cD | ${sp}x |")
            }
        }
    }
    $null=$md.AppendLine("")
    $null=$md.AppendLine("---")
    $null=$md.AppendLine("")
}

$null=$md.AppendLine("## Raw Results (all individual trials)")
$null=$md.AppendLine("")
$null=$md.AppendLine("| Runtime | Backend | Model | Prompt | N | Trial | N gen | Decode t/s | Prefill t/s | TTFT ms | Gen ms | Total ms |")
$null=$md.AppendLine("|---------|---------|-------|--------|--:|------:|------:|:----------:|:-----------:|:-------:|:------:|:--------:|")
foreach ($row in ($results|Sort-Object RT,BE,Mdl,Pn,NR,Tr)) {
    if ($row.Err) {
        $null=$md.AppendLine("| $($row.RT) | $($row.BE) | $($row.Mdl) | $($row.Pn) | $($row.NR) | $($row.Tr) | ERR | ERR | ERR | ERR | ERR | ERR |")
    } else {
        $null=$md.AppendLine("| $($row.RT) | $($row.BE) | $($row.Mdl) | $($row.Pn) | $($row.NR) | $($row.Tr) | $($row.NG) | $(F $row.DeT) | $(F $row.PrT) | $(F $row.PrMs) | $(F $row.GnMs) | $(F $row.TtMs) |")
    }
}
$null=$md.AppendLine("")

# --- End-to-End Latency section ---
function E2ECell($rows, $rt, $be, $field) {
    $r = @($rows | Where-Object { $_.RT -eq $rt -and $_.BE -eq $be -and -not $_.Err })
    if (-not $r) { return "--" }
    $a = Avg($r | ForEach-Object { $_.$field })
    if ($a -eq 0) { return "--" } else { return $a }
}
function MsTokCell($rows, $rt, $be) {
    $r = @($rows | Where-Object { $_.RT -eq $rt -and $_.BE -eq $be -and -not $_.Err -and $_.NG -gt 0 })
    if (-not $r) { return "--" }
    $a = Avg($r | ForEach-Object { [math]::Round($_.TtMs / $_.NG, 1) })
    if ($a -eq 0) { return "--" } else { return $a }
}
$null=$md.AppendLine("---")
$null=$md.AppendLine("")
$null=$md.AppendLine("## End-to-End Latency")
$null=$md.AppendLine("")
$null=$md.AppendLine("**E2E = TTFT + Gen ms** (wall time from prompt submission to last token).")
$null=$md.AppendLine("**ms/tok** = E2E / tokens_generated (user-perceived cost per output token).")
$null=$md.AppendLine("")
foreach ($model in ($results|Select-Object -ExpandProperty Mdl -Unique|Sort-Object)) {
    $null=$md.AppendLine("### $model -- E2E Total ms")
    $null=$md.AppendLine("")
    $null=$md.AppendLine("| Prompt | N | Geo-CPU ms | Geo-GPU ms | Oll-CPU ms | Oll-GPU ms | Fastest |")
    $null=$md.AppendLine("|--------|--:|-----------:|-----------:|-----------:|-----------:|---------|")
    foreach ($pname in @("short","long")) {
        foreach ($n in $ns) {
            $sel = @($results|Where-Object{$_.Mdl -eq $model -and $_.Pn -eq $pname -and $_.NR -eq $n})
            if (-not $sel) { continue }
            $gc=E2ECell $sel "Geodessical" "CPU" "TtMs"; $gg=E2ECell $sel "Geodessical" "GPU" "TtMs"
            $oc=E2ECell $sel "Ollama" "CPU" "TtMs";      $og=E2ECell $sel "Ollama" "GPU" "TtMs"
            $nums=@($gc,$gg,$oc,$og)|Where-Object{$_ -ne "--"}|ForEach-Object{[double]$_}
            $best="--"
            if ($nums.Count -gt 0) {
                $minV=($nums|Measure-Object -Minimum).Minimum
                $lbs=@("Geo-CPU","Geo-GPU","Oll-CPU","Oll-GPU"); $vs=@($gc,$gg,$oc,$og)
                for($i=0;$i-lt 4;$i++){if($vs[$i] -ne "--" -and [double]$vs[$i] -eq $minV){$best=$lbs[$i];break}}
            }
            $null=$md.AppendLine("| $pname | $n | $gc | $gg | $oc | $og | $best |")
        }
    }
    $null=$md.AppendLine("")
    $null=$md.AppendLine("### $model -- ms per Output Token (lower = better)")
    $null=$md.AppendLine("")
    $null=$md.AppendLine("| Prompt | N | Geo-CPU | Geo-GPU | Oll-CPU | Oll-GPU |")
    $null=$md.AppendLine("|--------|--:|--------:|--------:|--------:|--------:|")
    foreach ($pname in @("short","long")) {
        foreach ($n in $ns) {
            $sel = @($results|Where-Object{$_.Mdl -eq $model -and $_.Pn -eq $pname -and $_.NR -eq $n})
            if (-not $sel) { continue }
            $gc=MsTokCell $sel "Geodessical" "CPU"; $gg=MsTokCell $sel "Geodessical" "GPU"
            $oc=MsTokCell $sel "Ollama" "CPU";      $og=MsTokCell $sel "Ollama" "GPU"
            $null=$md.AppendLine("| $pname | $n | $gc | $gg | $oc | $og |")
        }
    }
    $null=$md.AppendLine("")
    $null=$md.AppendLine("### $model -- TTFT ms (Time to First Token)")
    $null=$md.AppendLine("")
    $null=$md.AppendLine("| Prompt | N | Geo-CPU | Geo-GPU | Oll-CPU | Oll-GPU |")
    $null=$md.AppendLine("|--------|--:|--------:|--------:|--------:|--------:|")
    foreach ($pname in @("short","long")) {
        foreach ($n in $ns) {
            $sel = @($results|Where-Object{$_.Mdl -eq $model -and $_.Pn -eq $pname -and $_.NR -eq $n})
            if (-not $sel) { continue }
            $gc=E2ECell $sel "Geodessical" "CPU" "PrMs"; $gg=E2ECell $sel "Geodessical" "GPU" "PrMs"
            $oc=E2ECell $sel "Ollama" "CPU" "PrMs";      $og=E2ECell $sel "Ollama" "GPU" "PrMs"
            $null=$md.AppendLine("| $pname | $n | $gc | $gg | $oc | $og |")
        }
    }
    $null=$md.AppendLine("")
}

# --- Write MD and CSV ---
[System.IO.File]::WriteAllText("$PSScriptRoot\$OUT_MD", $md.ToString(), [System.Text.Encoding]::UTF8)
Write-Host "Report written: $OUT_MD  ($($results.Count) rows)" -ForegroundColor Green

$CSV_OUT = "$PSScriptRoot\benchmark_results.csv"
$results | Select-Object RT,BE,Mdl,Pn,NR,Tr,NG,DeT,PrT,PrMs,GnMs,TtMs,Err | Export-Csv -Path $CSV_OUT -NoTypeInformation -Encoding UTF8
Write-Host "CSV written: $CSV_OUT" -ForegroundColor Green