Set-Location $PSScriptRoot
$MD_IN  = "$PSScriptRoot\benchmark_results.md"
$CSV_OUT = "$PSScriptRoot\benchmark_results.csv"

# --- Parse raw results table from MD ---
$rows = [System.Collections.ArrayList]::new()
$inTable = $false
foreach ($line in [System.IO.File]::ReadAllLines($MD_IN)) {
    if ($line -match '^\| Runtime \| Backend \| Model \| Prompt \| N \| Trial') { $inTable = $true; continue }
    if ($inTable -and $line -match '^\|[-| ]+\|') { continue }
    if ($inTable -and $line -match '^\|') {
        $f = $line -split '\|' | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
        if ($f.Count -ge 12) {
            [void]$rows.Add([PSCustomObject]@{
                Runtime=$f[0]; Backend=$f[1]; Model=$f[2]; Prompt=$f[3]
                N=[int]$f[4]; Trial=[int]$f[5]; NGen=[int]$f[6]
                DecodeTps=[double]$f[7]; PrefillTps=[double]$f[8]
                TTFTms=[double]$f[9]; GenMs=[double]$f[10]; TotalMs=[double]$f[11]
                E2EmsPerTok=if([int]$f[6]-gt 0){[math]::Round([double]$f[11]/[int]$f[6],1)}else{0}
            })
        }
    } elseif ($inTable -and $line.Trim() -eq '') { break }
}
Write-Host "Parsed $($rows.Count) rows"

# --- Export CSV ---
$rows | Export-Csv -Path $CSV_OUT -NoTypeInformation -Encoding UTF8
Write-Host "CSV written: $CSV_OUT"

# --- Build e2e section ---
function Avg2($vals) {
    $v = @($vals | Where-Object { $_ -gt 0 })
    if ($v.Count -eq 0) { return 0 }
    return [math]::Round(($v | Measure-Object -Average).Average, 1)
}
function Cell($rows, $rt, $be, $field) {
    $r = @($rows | Where-Object { $_.Runtime -eq $rt -and $_.Backend -eq $be })
    if (-not $r) { return "--" }
    $vals = $r | ForEach-Object { $_.$field }
    $a = Avg2 $vals
    if ($a -eq 0) { return "--" } else { return $a }
}
function MsTok($rows, $rt, $be) {
    $r = @($rows | Where-Object { $_.Runtime -eq $rt -and $_.Backend -eq $be })
    if (-not $r) { return "--" }
    $a = Avg2 ($r | ForEach-Object { $_.E2EmsPerTok })
    if ($a -eq 0) { return "--" } else { return $a }
}

$sb = [System.Text.StringBuilder]::new()
$null = $sb.AppendLine("---")
$null = $sb.AppendLine("")
$null = $sb.AppendLine("## End-to-End Latency")
$null = $sb.AppendLine("")
$null = $sb.AppendLine("**E2E = TTFT + Gen ms** (wall time from prompt submission to last token).")
$null = $sb.AppendLine("**ms/tok** = E2E / tokens_generated (user-perceived cost per output token).")
$null = $sb.AppendLine("")
$ns  = @(40, 128, 512)
foreach ($model in ($rows | Select-Object -ExpandProperty Model -Unique | Sort-Object)) {
    $null = $sb.AppendLine("### $model -- E2E Total ms")
    $null = $sb.AppendLine("")
    $null = $sb.AppendLine("| Prompt | N | Geo-CPU ms | Geo-GPU ms | Oll-CPU ms | Oll-GPU ms | Fastest |")
    $null = $sb.AppendLine("|--------|--:|-----------:|-----------:|-----------:|-----------:|---------|")
    foreach ($p in @("short","long")) {
        foreach ($n in $ns) {
            $sel = @($rows | Where-Object { $_.Model -eq $model -and $_.Prompt -eq $p -and $_.N -eq $n })
            if (-not $sel) { continue }
            $gc = Cell $sel "Geodessical" "CPU" "TotalMs"
            $gg = Cell $sel "Geodessical" "GPU" "TotalMs"
            $oc = Cell $sel "Ollama"      "CPU" "TotalMs"
            $og = Cell $sel "Ollama"      "GPU" "TotalMs"
            $nums = @($gc,$gg,$oc,$og) | Where-Object {$_ -ne "--"} | ForEach-Object {[double]$_}
            $best = "--"
            if ($nums.Count -gt 0) {
                $minVal = ($nums | Measure-Object -Minimum).Minimum
                $labels = @("Geo-CPU","Geo-GPU","Oll-CPU","Oll-GPU")
                $vals   = @($gc,$gg,$oc,$og)
                for ($i=0;$i -lt 4;$i++) {
                    if ($vals[$i] -ne "--" -and [double]$vals[$i] -eq $minVal) { $best=$labels[$i]; break }
                }
            }
            $null = $sb.AppendLine("| $p | $n | $gc | $gg | $oc | $og | $best |")
        }
    }
    $null = $sb.AppendLine("")

    $null = $sb.AppendLine("### $model -- ms per Output Token (lower = better)")
    $null = $sb.AppendLine("")
    $null = $sb.AppendLine("| Prompt | N | Geo-CPU | Geo-GPU | Oll-CPU | Oll-GPU |")
    $null = $sb.AppendLine("|--------|--:|--------:|--------:|--------:|--------:|")
    foreach ($p in @("short","long")) {
        foreach ($n in $ns) {
            $sel = @($rows | Where-Object { $_.Model -eq $model -and $_.Prompt -eq $p -and $_.N -eq $n })
            if (-not $sel) { continue }
            $gc = MsTok $sel "Geodessical" "CPU"
            $gg = MsTok $sel "Geodessical" "GPU"
            $oc = MsTok $sel "Ollama"      "CPU"
            $og = MsTok $sel "Ollama"      "GPU"
            $null = $sb.AppendLine("| $p | $n | $gc | $gg | $oc | $og |")
        }
    }
    $null = $sb.AppendLine("")

    $null = $sb.AppendLine("### $model -- TTFT ms (Time to First Token)")
    $null = $sb.AppendLine("")
    $null = $sb.AppendLine("| Prompt | N | Geo-CPU | Geo-GPU | Oll-CPU | Oll-GPU |")
    $null = $sb.AppendLine("|--------|--:|--------:|--------:|--------:|--------:|")
    foreach ($p in @("short","long")) {
        foreach ($n in $ns) {
            $sel = @($rows | Where-Object { $_.Model -eq $model -and $_.Prompt -eq $p -and $_.N -eq $n })
            if (-not $sel) { continue }
            $gc = Cell $sel "Geodessical" "CPU" "TTFTms"
            $gg = Cell $sel "Geodessical" "GPU" "TTFTms"
            $oc = Cell $sel "Ollama"      "CPU" "TTFTms"
            $og = Cell $sel "Ollama"      "GPU" "TTFTms"
            $null = $sb.AppendLine("| $p | $n | $gc | $gg | $oc | $og |")
        }
    }
    $null = $sb.AppendLine("")
}

# Append to MD
$existing = [System.IO.File]::ReadAllText($MD_IN, [System.Text.Encoding]::UTF8)
$updated  = $existing.TrimEnd() + "`n`n" + $sb.ToString()
[System.IO.File]::WriteAllText($MD_IN, $updated, [System.Text.Encoding]::UTF8)
Write-Host "E2E section appended to $MD_IN"