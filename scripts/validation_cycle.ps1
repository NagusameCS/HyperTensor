# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
# ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
# ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
# ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
# ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
# ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
# :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
# :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
# ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
# :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
# ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
# ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
# :::::::::................................:@@@@@@@@@@%:...............................::::::
# ::::::::..................................*@@@@@@@@@-................................::::::::
# ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
# :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
# :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
# :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
# :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
# :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
# :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
# :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
# :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
# ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
# ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
# :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
# ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
# ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
# :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
# ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
# :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
# :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
# ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
# ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
# :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
# ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
# ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
# ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

param(
    [Parameter(Mandatory = $true)]
    [string]$PackDir,
    [string]$RankDir = "",
    [string]$CiDir = ""
)

$ErrorActionPreference = "Stop"

function Mean($arr) {
    if (-not $arr -or $arr.Count -eq 0) { return $null }
    return ($arr | Measure-Object -Average).Average
}

function Std($arr) {
    if (-not $arr) { return 0.0 }
    $n = $arr.Count
    if ($n -le 1) { return 0.0 }
    $mu = Mean $arr
    $ss = 0.0
    foreach ($x in $arr) { $ss += [math]::Pow(($x - $mu), 2) }
    return [math]::Sqrt($ss / ($n - 1))
}

function CI95($arr) {
    if (-not $arr) { return 0.0 }
    $n = $arr.Count
    if ($n -le 1) { return 0.0 }
    return 1.96 * (Std $arr) / [math]::Sqrt($n)
}

$rankBase = if ($RankDir -and $RankDir.Trim().Length -gt 0) { $RankDir } else { $PackDir }
$ciBase = if ($CiDir -and $CiDir.Trim().Length -gt 0) { $CiDir } else { $PackDir }

$rankRel = Join-Path $rankBase "rank_sweep_relative_to_baseline.csv"
$rankAgg = Join-Path $rankBase "rank_sweep_aggregate.csv"
$ciSum = Join-Path $ciBase "ci_pack_summary.csv"
$pplCsv = Join-Path $ciBase "ci_ppl_5run.csv"

$missing = @()
foreach ($f in @($rankRel, $rankAgg, $ciSum, $pplCsv)) {
    if (-not (Test-Path $f)) { $missing += $f }
}

$agg = @()
$ci = @()
$ppl = @()
if (Test-Path $rankAgg) { $agg = Import-Csv $rankAgg }
if (Test-Path $ciSum) { $ci = Import-Csv $ciSum }
if (Test-Path $pplCsv) { $ppl = Import-Csv $pplCsv }

function Get-RankRow([string]$r) {
    return $agg | Where-Object { $_.rank -eq $r } | Select-Object -First 1
}

$k1024 = Get-RankRow "1024"
$k1536 = Get-RankRow "1536"
$k2048 = Get-RankRow "2048"

$gate_k1024_decode = $false
$gate_k1536_decode = $false
$gate_k2048_decode = $false
$gate_k2048_prefill = $false
if ($k1024 -and $k1536 -and $k2048) {
    $gate_k1024_decode = [double]$k1024.mean_decode_pct_of_baseline -ge 95.0
    $gate_k1536_decode = [double]$k1536.mean_decode_pct_of_baseline -ge 75.0
    $gate_k2048_decode = [double]$k2048.mean_decode_pct_of_baseline -ge 75.0
    $gate_k2048_prefill = [double]$k2048.mean_prefill_pct_of_baseline -le 225.0
} else {
    $missing += "rank_sweep_aggregate.csv missing required rank rows 1024/1536/2048"
}

$coding = $ci | Where-Object { $_.case_name -eq "coding_256" } | Select-Object -First 1
$reason = $ci | Where-Object { $_.case_name -eq "reasoning_256" } | Select-Object -First 1
$coding_lower = $null
$reason_lower = $null
$gate_ci_decode = $false
if ($coding -and $reason -and [double]$coding.baseline_decode_mean -gt 0 -and [double]$reason.baseline_decode_mean -gt 0) {
    $coding_lower = [double]$coding.decode_pct_of_baseline_mean - (1.96 * [double]$coding.grc_decode_ci95 / [double]$coding.baseline_decode_mean * 100.0)
    $reason_lower = [double]$reason.decode_pct_of_baseline_mean - (1.96 * [double]$reason.grc_decode_ci95 / [double]$reason.baseline_decode_mean * 100.0)
    $gate_ci_decode = ($coding_lower -ge 67.0) -and ($reason_lower -ge 67.0)
} else {
    $missing += "ci_pack_summary.csv missing usable coding_256/reasoning_256 rows"
}

$bp = @($ppl | Where-Object { $_.mode -eq "baseline" } | ForEach-Object { [double]$_.ppl })
$gp = @($ppl | Where-Object { $_.mode -eq "grc_k2048" } | ForEach-Object { [double]$_.ppl })
$bpMu = Mean $bp
$gpMu = Mean $gp
$pplDeltaPct = $null
$gate_ppl = $false
if ($bpMu -and $gpMu -and $bpMu -gt 0) {
    $pplDeltaPct = 100.0 * (($gpMu - $bpMu) / $bpMu)
    $gate_ppl = $pplDeltaPct -le 15.0
} else {
    $missing += "ci_ppl_5run.csv missing usable baseline/grc_k2048 PPL rows"
}

$allPassed = $gate_k1024_decode -and $gate_k1536_decode -and $gate_k2048_decode -and $gate_k2048_prefill -and $gate_ci_decode -and $gate_ppl

$result = [pscustomobject]@{
    pack_dir = (Resolve-Path $PackDir).Path
    rank_dir = (Resolve-Path $rankBase).Path
    ci_dir = (Resolve-Path $ciBase).Path
    generated_at = (Get-Date).ToString("s")
    metrics = [pscustomobject]@{
        k1024_decode_pct = if ($k1024) { [double]$k1024.mean_decode_pct_of_baseline } else { $null }
        k1536_decode_pct = if ($k1536) { [double]$k1536.mean_decode_pct_of_baseline } else { $null }
        k2048_decode_pct = if ($k2048) { [double]$k2048.mean_decode_pct_of_baseline } else { $null }
        k2048_prefill_pct = if ($k2048) { [double]$k2048.mean_prefill_pct_of_baseline } else { $null }
        coding_decode_lower95_pct = $coding_lower
        reasoning_decode_lower95_pct = $reason_lower
        ppl_delta_pct = $pplDeltaPct
    }
    gates = [pscustomobject]@{
        k1024_decode_ge_95 = $gate_k1024_decode
        k1536_decode_ge_75 = $gate_k1536_decode
        k2048_decode_ge_75 = $gate_k2048_decode
        k2048_prefill_le_225 = $gate_k2048_prefill
        ci_decode_lower95_ge_67 = $gate_ci_decode
        ppl_delta_le_15 = $gate_ppl
    }
    missing_or_invalid_inputs = $missing
    strong_claim_ready = $allPassed
}

$outJson = Join-Path $PackDir "validation_cycle.json"
$result | ConvertTo-Json -Depth 6 | Set-Content -Path $outJson

Write-Host "VALIDATION_JSON=$outJson"
Write-Host ("STRONG_CLAIM_READY={0}" -f $allPassed)
Write-Host ("k1024_decode={0:N2}% k1536_decode={1:N2}% k2048_decode={2:N2}% k2048_prefill={3:N2}%" -f $result.metrics.k1024_decode_pct, $result.metrics.k1536_decode_pct, $result.metrics.k2048_decode_pct, $result.metrics.k2048_prefill_pct)
Write-Host ("coding_lower95={0:N2}% reasoning_lower95={1:N2}% ppl_delta={2:N2}%" -f $result.metrics.coding_decode_lower95_pct, $result.metrics.reasoning_decode_lower95_pct, $result.metrics.ppl_delta_pct)
