$ErrorActionPreference = "Stop"
$out = "C:\Users\legom\HyperTensor\benchmarks\whitepaper_pack_20260425_192208"
$csv = Import-Csv (Join-Path $out "outlier_investigation.csv")

function Mean($a) { ($a | Measure-Object -Average).Average }
function Std($a) {
    $n = ($a | Measure-Object).Count
    if ($n -le 1) { return 0 }
    $mu = Mean $a
    $ss = 0
    foreach ($x in $a) { $ss += [math]::Pow(($x - $mu), 2) }
    [math]::Sqrt($ss / ($n - 1))
}
function CI($a) {
    $n = ($a | Measure-Object).Count
    if ($n -le 1) { return 0 }
    1.96 * (Std $a) / [math]::Sqrt($n)
}

$ciRows = @()
foreach ($name in @("coding", "reasoning")) {
    if ($name -eq "coding") {
        $b = $csv | Where-Object { $_.label -eq "outlier_baseline_coding_256" -and [int]$_.rep -le 5 }
        $g = $csv | Where-Object { $_.label -eq "outlier_grc_coding_256_k2048" -and [int]$_.rep -le 5 }
    } else {
        $b = $csv | Where-Object { $_.label -eq "outlier_baseline_reasoning_256" -and [int]$_.rep -le 5 }
        $g = $csv | Where-Object { $_.label -eq "outlier_grc_reasoning_256_k2048" -and [int]$_.rep -le 5 }
    }

    $bDec = $b.decode_tps | ForEach-Object { [double]$_ }
    $gDec = $g.decode_tps | ForEach-Object { [double]$_ }
    $bOv = $b.overall_tps | ForEach-Object { [double]$_ }
    $gOv = $g.overall_tps | ForEach-Object { [double]$_ }

    $bDecM = Mean $bDec
    $gDecM = Mean $gDec
    $bOvM = Mean $bOv
    $gOvM = Mean $gOv

    $ciRows += [pscustomobject]@{
        case_name = "${name}_256"
        baseline_decode_mean = $bDecM
        baseline_decode_ci95 = (CI $bDec)
        grc_decode_mean = $gDecM
        grc_decode_ci95 = (CI $gDec)
        decode_pct_of_baseline_mean = (100 * $gDecM / $bDecM)
        baseline_overall_mean = $bOvM
        baseline_overall_ci95 = (CI $bOv)
        grc_overall_mean = $gOvM
        grc_overall_ci95 = (CI $gOv)
        overall_pct_of_baseline_mean = (100 * $gOvM / $bOvM)
    }
}

$ciSummaryPath = Join-Path $out "ci_pack_summary.csv"
$ciRawPath = Join-Path $out "ci_pack_raw.csv"
$diagPath = Join-Path $out "outlier_diagnosis.json"

$ciRows | Export-Csv -NoTypeInformation -Path $ciSummaryPath
Copy-Item (Join-Path $out "outlier_investigation.csv") $ciRawPath -Force

$codingB = $csv | Where-Object { $_.label -eq "outlier_baseline_coding_256" }
$codingG = $csv | Where-Object { $_.label -eq "outlier_grc_coding_256_k2048" }
$reasonB = $csv | Where-Object { $_.label -eq "outlier_baseline_reasoning_256" }
$reasonG = $csv | Where-Object { $_.label -eq "outlier_grc_reasoning_256_k2048" }

$codingPct = 100 * (Mean ($codingG.decode_tps | ForEach-Object { [double]$_ })) / (Mean ($codingB.decode_tps | ForEach-Object { [double]$_ }))
$reasonPct = 100 * (Mean ($reasonG.decode_tps | ForEach-Object { [double]$_ })) / (Mean ($reasonB.decode_tps | ForEach-Object { [double]$_ }))

$root = if ([math]::Abs($codingPct - $reasonPct) -lt 5) { "run-condition artifact likely" } else { "prompt-path sensitivity likely" }

[pscustomobject]@{
    coding_decode_pct = $codingPct
    reasoning_decode_pct = $reasonPct
    root_cause = $root
} | ConvertTo-Json | Set-Content -Path $diagPath

Write-Host "CI_SUMMARY=$ciSummaryPath"
Write-Host "CI_RAW=$ciRawPath"
Write-Host "DIAG=$diagPath"
