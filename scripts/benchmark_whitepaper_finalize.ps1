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
    [string]$Model = "C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    [string]$Exe = ".\\build_host\\geodessical.exe",
    [string]$OutDir = "",
    [switch]$ReuseLatestPack,
    [int]$RunTimeoutSec = 900,
    [int]$PplTimeoutSec = 1200,
    [int]$CooldownSec = 30
)

$ErrorActionPreference = "Stop"

function Mean($arr) {
    if (-not $arr) { return $null }
    return ($arr | Measure-Object -Average).Average
}

function Std($arr) {
    $n = ($arr | Measure-Object).Count
    if ($n -le 1) { return 0.0 }
    $mu = Mean $arr
    $ss = 0.0
    foreach ($x in $arr) { $ss += [math]::Pow(($x - $mu), 2) }
    return [math]::Sqrt($ss / ($n - 1))
}

function CI95($arr) {
    $n = ($arr | Measure-Object).Count
    if ($n -le 1) { return 0.0 }
    return 1.96 * (Std $arr) / [math]::Sqrt($n)
}

function Parse-Run([string]$stdoutPath) {
    $raw = Get-Content -Raw -Path $stdoutPath
    $mDec = [regex]::Match($raw, 'Decode-only:\s*prefill\s*([\d.]+)\s*ms,\s*([\d.]+)\s*tok/s')
    $mGd = [regex]::Match($raw, '\[GD\]\s*(\d+)\s+tokens\s+in\s*([\d.]+)\s*ms\s*\(([\d.]+)\s*tok/s\)')
    $mCompact = [regex]::Match($raw, '\[(\d+)\s+tok,\s*([\d.]+)\s*tok/s,\s*prefill\s*([\d.]+)\s*ms')

    $decode = $null
    $prefill = $null
    $overall = $null
    $genTok = $null

    if ($mDec.Success) {
        $prefill = [double]$mDec.Groups[1].Value
        $decode = [double]$mDec.Groups[2].Value
    }
    if ($mGd.Success) {
        $genTok = [int]$mGd.Groups[1].Value
        $overall = [double]$mGd.Groups[3].Value
    } elseif ($mCompact.Success) {
        $genTok = [int]$mCompact.Groups[1].Value
        if ($null -eq $decode) { $decode = [double]$mCompact.Groups[2].Value }
        if ($null -eq $prefill) { $prefill = [double]$mCompact.Groups[3].Value }
    }

    if ($null -eq $decode -or $null -eq $prefill) {
        throw "Unable to parse decode/prefill metrics from $stdoutPath"
    }

    return [pscustomobject]@{
        decode_tps = $decode
        overall_tps = $overall
        prefill_ms = $prefill
        generated_tokens = $genTok
    }
}

function Get-RunFailureDetail {
    param(
        [string]$StdoutPath,
        [string]$StderrPath
    )

    $stdoutTail = ""
    $stderrTail = ""
    if (Test-Path $StdoutPath) {
        $stdoutTail = (Get-Content -Path $StdoutPath -Tail 30) -join "`n"
    }
    if (Test-Path $StderrPath) {
        $stderrTail = (Get-Content -Path $StderrPath -Tail 30) -join "`n"
    }

    return "stderr tail:`n$stderrTail`nstdout tail:`n$stdoutTail"
}

function Parse-PPL([string]$stdoutPath) {
    $raw = Get-Content -Raw -Path $stdoutPath
    $m1 = [regex]::Match($raw, 'PPL\s*=\s*([\d.]+)')
    if ($m1.Success) { return [double]$m1.Groups[1].Value }
    $m2 = [regex]::Match($raw, '"ppl":([\d.]+)')
    if ($m2.Success) { return [double]$m2.Groups[1].Value }
    return $null
}

function Stop-StaleGeod {
    $procs = Get-Process geodessical -ErrorAction SilentlyContinue
    foreach ($p in $procs) {
        try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {}
    }
}

function Invoke-GeodProcess {
    param(
        [string[]]$Argv,
        [string]$StdoutPath,
        [string]$StderrPath,
        [int]$TimeoutSec
    )

    Stop-StaleGeod

    if (Test-Path $StdoutPath) { Remove-Item -ErrorAction SilentlyContinue $StdoutPath }
    if (Test-Path $StderrPath) { Remove-Item -ErrorAction SilentlyContinue $StderrPath }

    $safeArgv = @($Argv | Where-Object { $_ -ne $null -and $_.ToString().Length -gt 0 })
    if ($safeArgv.Count -eq 0) {
        throw "Invoke-GeodProcess received an empty argument list"
    }

    $exePath = $Exe
    try {
        $exePath = (Resolve-Path $Exe).Path
    } catch {
        throw "Executable not found: $Exe"
    }
    $cwdPath = (Get-Location).Path

    $job = Start-Job -ScriptBlock {
        param(
            [string]$JobExe,
            [string[]]$JobArgv,
            [string]$JobStdout,
            [string]$JobStderr,
            [string]$JobCwd
        )

        Set-Location $JobCwd

        & $JobExe @JobArgv 1> $JobStdout 2> $JobStderr
        return $LASTEXITCODE
    } -ArgumentList $exePath, $safeArgv, $StdoutPath, $StderrPath, $cwdPath

    $finished = $null -ne (Wait-Job -Job $job -Timeout ([Math]::Max(1, $TimeoutSec)))
    if (-not $finished) {
        try { Stop-Job -Job $job -Force -ErrorAction SilentlyContinue } catch {}
        try { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue } catch {}
        Stop-StaleGeod
        try { [System.IO.File]::WriteAllText($StdoutPath, "") } catch {}
        try { [System.IO.File]::WriteAllText($StderrPath, "TIMEOUT") } catch {}
        return [pscustomobject]@{ exit_code = 124; timed_out = $true }
    }

    $exitCode = 1
    try {
        $result = Receive-Job -Job $job -ErrorAction SilentlyContinue
        if ($null -ne $result) {
            $exitCode = [int]$result
        }
    } finally {
        try { Remove-Job -Job $job -Force -ErrorAction SilentlyContinue } catch {}
    }

    return [pscustomobject]@{ exit_code = $exitCode; timed_out = $false }
}

function Invoke-RunCase {
    param(
        [string]$OutDir,
        [string]$Label,
        [string]$Prompt,
        [int]$Tokens,
        [string[]]$ExtraArgs,
        [int]$Rep = 1,
        [int]$Retries = 5
    )

    $safe = ($Label -replace '[^a-zA-Z0-9_\-]', '_')
    $out = Join-Path $OutDir ("${safe}_rep${Rep}.txt")
    $err = Join-Path $OutDir ("${safe}_rep${Rep}_err.txt")

    $needsRun = $true
    if (Test-Path $out) {
        try {
            $null = Parse-Run $out
            $needsRun = $false
        } catch {
            # Cached output exists but is incomplete/corrupt; regenerate it.
            Remove-Item -ErrorAction SilentlyContinue $out
            Remove-Item -ErrorAction SilentlyContinue $err
            $needsRun = $true
        }
    }

    if ($needsRun) {
        # Cooldown: let GPU temperature stabilise before each fresh run to prevent
        # thermal throttling from biasing throughput measurements.
        if ($CooldownSec -gt 0) {
            Start-Sleep -Seconds $CooldownSec
        }
        $ok = $false
        for ($attempt = 1; $attempt -le $Retries; $attempt++) {
            $runArgs = @($Model) + $ExtraArgs + @('-p', $Prompt, '-n', "$Tokens")
            $run = Invoke-GeodProcess -Argv $runArgs -StdoutPath $out -StderrPath $err -TimeoutSec $RunTimeoutSec
            try {
                if (Test-Path $out) {
                    $null = Parse-Run $out
                    $ok = $true
                    break
                }
            } catch {
                # Keep retrying if output is incomplete or unparseable.
            }
            if ($run.exit_code -eq 0) {
                $ok = $true
                break
            }
        }
        if (-not $ok) {
            try {
                $null = Parse-Run $out
                $ok = $true
            } catch {}
        }
        if (-not $ok) {
            $detail = Get-RunFailureDetail -StdoutPath $out -StderrPath $err
            throw "Failed case '$Label' rep $Rep after $Retries attempts`n$detail"
        }
    }

    $m = Parse-Run $out
    return [pscustomobject]@{
        label = $Label
        rep = $Rep
        prompt = $Prompt
        tokens = $Tokens
        decode_tps = $m.decode_tps
        overall_tps = $m.overall_tps
        prefill_ms = $m.prefill_ms
        generated_tokens = $m.generated_tokens
        stdout = $out
        stderr = $err
    }
}

function Invoke-PPLCase {
    param(
        [string]$OutPath,
        [string]$ErrPath,
        [string[]]$ExtraArgs,
        [int]$Retries = 5
    )

    $needsRun = $true
    if (Test-Path $OutPath) {
        $existingPpl = Parse-PPL $OutPath
        if ($null -ne $existingPpl) {
            $needsRun = $false
        } else {
            Remove-Item -ErrorAction SilentlyContinue $OutPath
            Remove-Item -ErrorAction SilentlyContinue $ErrPath
        }
    }

    if ($needsRun) {
        $ok = $false
        for ($attempt = 1; $attempt -le $Retries; $attempt++) {
            $allArgs = @($Model) + $ExtraArgs
            $run = Invoke-GeodProcess -Argv $allArgs -StdoutPath $OutPath -StderrPath $ErrPath -TimeoutSec $PplTimeoutSec
            $parsed = Parse-PPL $OutPath
            if ($null -ne $parsed) {
                $ok = $true
                break
            }
            if ($run.exit_code -eq 0) {
                $ok = $true
                break
            }
        }
        if (-not $ok) {
            $parsed = Parse-PPL $OutPath
            if ($null -ne $parsed) {
                $ok = $true
            }
        }
        if (-not $ok) {
            $detail = Get-RunFailureDetail -StdoutPath $OutPath -StderrPath $ErrPath
            throw "Failed PPL run after $Retries attempts`n$detail"
        }
    }

    $ppl = Parse-PPL $OutPath
    if ($null -eq $ppl) {
        throw "Unable to parse PPL from $OutPath"
    }
    return $ppl
}

$selectedOutDir = $OutDir
if (-not $selectedOutDir) {
    if ($ReuseLatestPack) {
        $existingPack = Get-ChildItem .\benchmarks -Directory |
            Where-Object { $_.Name -like 'whitepaper_pack_*' } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($existingPack) {
            $selectedOutDir = $existingPack.FullName
        }
    }

    if (-not $selectedOutDir) {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $selectedOutDir = Join-Path ".\\benchmarks" ("whitepaper_pack_" + $stamp)
    }
}

New-Item -ItemType Directory -Force -Path $selectedOutDir | Out-Null
$outDir = (Resolve-Path $selectedOutDir).Path

$codingPrompt = "Write a Python function that returns prime numbers up to n."
$reasonPrompt = "Explain why gradient clipping helps stabilize training in deep networks."
$factualPrompt = "Summarize how TCP congestion control works in modern networks."
$creativePrompt = "Write a short sci-fi paragraph about a city powered by ocean tides."

# 1) Rank sweep table normalized to baseline  [FIRST: run before outlier investigation
#    to avoid measuring at throttled GPU clock after many consecutive GRC runs]
$prompts = @(
    @{ name = 'coding'; text = $codingPrompt },
    @{ name = 'reasoning'; text = $reasonPrompt },
    @{ name = 'factual'; text = $factualPrompt },
    @{ name = 'creative'; text = $creativePrompt }
)
$tokensList = @(128, 256)
$ranks = @(1024, 1536, 2048)

$rankRows = @()
foreach ($p in $prompts) {
    foreach ($t in $tokensList) {
        $rankRows += Invoke-RunCase -OutDir $outDir -Label ("baseline_{0}_{1}" -f $p.name, $t) -Prompt $p.text -Tokens $t -ExtraArgs @('--temp', '0') -Rep 1
        foreach ($r in $ranks) {
            $rankRows += Invoke-RunCase -OutDir $outDir -Label ("grc_k{0}_{1}_{2}" -f $r, $p.name, $t) -Prompt $p.text -Tokens $t -ExtraArgs @('--axex-compress', '--axex-attn-only', '--axex-skip-o', '--axex-weight-pca', '--axex-compress-rank', "$r", '--temp', '0') -Rep 1
        }
    }
}
$rankCsv = Join-Path $outDir "rank_sweep_raw.csv"
$rankRows | Export-Csv -NoTypeInformation -Path $rankCsv

$relativeRows = @()
foreach ($p in $prompts) {
    foreach ($t in $tokensList) {
        $b = $rankRows | Where-Object { $_.label -eq ("baseline_{0}_{1}" -f $p.name, $t) } | Select-Object -First 1
        foreach ($r in $ranks) {
            $g = $rankRows | Where-Object { $_.label -eq ("grc_k{0}_{1}_{2}" -f $r, $p.name, $t) } | Select-Object -First 1
            $relativeRows += [pscustomobject]@{
                prompt = $p.name
                tokens = $t
                rank = $r
                baseline_decode_tps = $b.decode_tps
                grc_decode_tps = $g.decode_tps
                decode_pct_of_baseline = if ($b.decode_tps) { 100.0 * $g.decode_tps / $b.decode_tps } else { $null }
                baseline_overall_tps = $b.overall_tps
                grc_overall_tps = $g.overall_tps
                overall_pct_of_baseline = if ($b.overall_tps) { 100.0 * $g.overall_tps / $b.overall_tps } else { $null }
                baseline_prefill_ms = $b.prefill_ms
                grc_prefill_ms = $g.prefill_ms
                prefill_pct_of_baseline = if ($b.prefill_ms) { 100.0 * $g.prefill_ms / $b.prefill_ms } else { $null }
            }
        }
    }
}
$rankRelCsv = Join-Path $outDir "rank_sweep_relative_to_baseline.csv"
$relativeRows | Export-Csv -NoTypeInformation -Path $rankRelCsv

$rankAgg = $relativeRows | Group-Object rank | ForEach-Object {
    [pscustomobject]@{
        rank = [int]$_.Name
        mean_decode_pct_of_baseline = Mean ($_.Group.decode_pct_of_baseline)
        mean_overall_pct_of_baseline = Mean ($_.Group.overall_pct_of_baseline)
        mean_prefill_pct_of_baseline = Mean ($_.Group.prefill_pct_of_baseline)
    }
}
$rankAggPath = Join-Path $outDir "rank_sweep_aggregate.csv"
$rankAgg | Export-Csv -NoTypeInformation -Path $rankAggPath

# 2) Outlier investigation exists (6 reps x 4 cases)  [SECOND: after rank sweep]
$outlierRows = @()
for ($i = 1; $i -le 6; $i++) {
    $outlierRows += Invoke-RunCase -OutDir $outDir -Label "outlier_baseline_coding_256" -Prompt $codingPrompt -Tokens 256 -ExtraArgs @('--temp', '0') -Rep $i
    $outlierRows += Invoke-RunCase -OutDir $outDir -Label "outlier_grc_coding_256_k2048" -Prompt $codingPrompt -Tokens 256 -ExtraArgs @('--axex-compress', '--axex-attn-only', '--axex-skip-o', '--axex-weight-pca', '--axex-compress-rank', '2048', '--temp', '0') -Rep $i
    $outlierRows += Invoke-RunCase -OutDir $outDir -Label "outlier_baseline_reasoning_256" -Prompt $reasonPrompt -Tokens 256 -ExtraArgs @('--temp', '0') -Rep $i
    $outlierRows += Invoke-RunCase -OutDir $outDir -Label "outlier_grc_reasoning_256_k2048" -Prompt $reasonPrompt -Tokens 256 -ExtraArgs @('--axex-compress', '--axex-attn-only', '--axex-skip-o', '--axex-weight-pca', '--axex-compress-rank', '2048', '--temp', '0') -Rep $i
}
$outlierCsv = Join-Path $outDir "outlier_investigation.csv"
$outlierRows | Export-Csv -NoTypeInformation -Path $outlierCsv

$ob = $outlierRows | Where-Object { $_.label -eq 'outlier_baseline_coding_256' }
$og = $outlierRows | Where-Object { $_.label -eq 'outlier_grc_coding_256_k2048' }
$rb = $outlierRows | Where-Object { $_.label -eq 'outlier_baseline_reasoning_256' }
$rg = $outlierRows | Where-Object { $_.label -eq 'outlier_grc_reasoning_256_k2048' }

$diag = [pscustomobject]@{
    coding_baseline_decode_mean = Mean $ob.decode_tps
    coding_grc_decode_mean = Mean $og.decode_tps
    coding_decode_pct = 100.0 * (Mean $og.decode_tps) / (Mean $ob.decode_tps)
    reasoning_baseline_decode_mean = Mean $rb.decode_tps
    reasoning_grc_decode_mean = Mean $rg.decode_tps
    reasoning_decode_pct = 100.0 * (Mean $rg.decode_tps) / (Mean $rb.decode_tps)
    coding_baseline_generated_tokens_mean = Mean $ob.generated_tokens
    coding_grc_generated_tokens_mean = Mean $og.generated_tokens
    reasoning_baseline_generated_tokens_mean = Mean $rb.generated_tokens
    reasoning_grc_generated_tokens_mean = Mean $rg.generated_tokens
}
$diagPath = Join-Path $outDir "outlier_diagnosis.json"
$diag | ConvertTo-Json | Set-Content -Path $diagPath

# 3) 5-run CI pack for top whitepaper claims, sourced from first 5 outlier reps
$ciRows = @()
for ($i = 1; $i -le 5; $i++) {
    $ciRows += ($outlierRows | Where-Object { $_.label -eq 'outlier_baseline_coding_256' -and $_.rep -eq $i })
    $ciRows += ($outlierRows | Where-Object { $_.label -eq 'outlier_grc_coding_256_k2048' -and $_.rep -eq $i })
    $ciRows += ($outlierRows | Where-Object { $_.label -eq 'outlier_baseline_reasoning_256' -and $_.rep -eq $i })
    $ciRows += ($outlierRows | Where-Object { $_.label -eq 'outlier_grc_reasoning_256_k2048' -and $_.rep -eq $i })
}
$ciRawPath = Join-Path $outDir "ci_pack_raw.csv"
$ciRows | Export-Csv -NoTypeInformation -Path $ciRawPath

$ciSummary = @()
foreach ($caseName in @('coding_256', 'reasoning_256')) {
    if ($caseName -eq 'coding_256') {
        $b = $ciRows | Where-Object { $_.label -eq 'outlier_baseline_coding_256' }
        $g = $ciRows | Where-Object { $_.label -eq 'outlier_grc_coding_256_k2048' }
    } else {
        $b = $ciRows | Where-Object { $_.label -eq 'outlier_baseline_reasoning_256' }
        $g = $ciRows | Where-Object { $_.label -eq 'outlier_grc_reasoning_256_k2048' }
    }

    $bDec = $b.decode_tps
    $gDec = $g.decode_tps
    $bOv = $b.overall_tps
    $gOv = $g.overall_tps

    $bDecMu = Mean $bDec
    $gDecMu = Mean $gDec
    $bOvMu = Mean $bOv
    $gOvMu = Mean $gOv

    $ciSummary += [pscustomobject]@{
        case_name = $caseName
        baseline_decode_mean = $bDecMu
        baseline_decode_ci95 = CI95 $bDec
        grc_decode_mean = $gDecMu
        grc_decode_ci95 = CI95 $gDec
        decode_pct_of_baseline_mean = if ($bDecMu) { 100.0 * $gDecMu / $bDecMu } else { $null }
        baseline_overall_mean = $bOvMu
        baseline_overall_ci95 = CI95 $bOv
        grc_overall_mean = $gOvMu
        grc_overall_ci95 = CI95 $gOv
        overall_pct_of_baseline_mean = if ($bOvMu) { 100.0 * $gOvMu / $bOvMu } else { $null }
    }
}
$ciSummaryPath = Join-Path $outDir "ci_pack_summary.csv"
$ciSummary | Export-Csv -NoTypeInformation -Path $ciSummaryPath

# PPL 5-run pack
$pplRows = @()
for ($i = 1; $i -le 5; $i++) {
    $bo = Join-Path $outDir ("ci_ppl_baseline_rep${i}.txt")
    $be = Join-Path $outDir ("ci_ppl_baseline_rep${i}_err.txt")
    $bp = Invoke-PPLCase -OutPath $bo -ErrPath $be -ExtraArgs @('--ppl-eval')
    $pplRows += [pscustomobject]@{ rep = $i; mode = 'baseline'; ppl = $bp; stdout = $bo; stderr = $be }

    $go = Join-Path $outDir ("ci_ppl_grc2048_rep${i}.txt")
    $ge = Join-Path $outDir ("ci_ppl_grc2048_rep${i}_err.txt")
    $gp = Invoke-PPLCase -OutPath $go -ErrPath $ge -ExtraArgs @('--axex-compress', '--axex-attn-only', '--axex-skip-o', '--axex-weight-pca', '--axex-compress-rank', '2048', '--ppl-eval')
    $pplRows += [pscustomobject]@{ rep = $i; mode = 'grc_k2048'; ppl = $gp; stdout = $go; stderr = $ge }
}
$pplCsv = Join-Path $outDir "ci_ppl_5run.csv"
$pplRows | Export-Csv -NoTypeInformation -Path $pplCsv

$bpArr = ($pplRows | Where-Object { $_.mode -eq 'baseline' }).ppl
$gpArr = ($pplRows | Where-Object { $_.mode -eq 'grc_k2048' }).ppl
$bpMu = Mean $bpArr
$gpMu = Mean $gpArr
$pplPct = if ($bpMu) { 100.0 * $gpMu / $bpMu } else { $null }
$pplDeltaPct = if ($bpMu) { 100.0 * ($gpMu - $bpMu) / $bpMu } else { $null }

Write-Host "OUTDIR=$outDir"
Write-Host "FILES=$outlierCsv;$diagPath;$rankCsv;$rankRelCsv;$rankAggPath;$ciRawPath;$ciSummaryPath;$pplCsv"
Write-Host ("OUTLIER coding decode pct={0:N2} reasoning decode pct={1:N2}" -f $diag.coding_decode_pct, $diag.reasoning_decode_pct)
foreach ($r in ($rankAgg | Sort-Object rank)) {
    Write-Host ("RANK {0} mean decode%={1:N2} overall%={2:N2} prefill%={3:N2}" -f $r.rank, $r.mean_decode_pct_of_baseline, $r.mean_overall_pct_of_baseline, $r.mean_prefill_pct_of_baseline)
}
foreach ($c in $ciSummary) {
    Write-Host ("CI {0}: decode baseline {1:N2}+/-{2:N2}, grc {3:N2}+/-{4:N2}, pct={5:N2}" -f $c.case_name, $c.baseline_decode_mean, $c.baseline_decode_ci95, $c.grc_decode_mean, $c.grc_decode_ci95, $c.decode_pct_of_baseline_mean)
    Write-Host ("CI {0}: overall baseline {1:N2}+/-{2:N2}, grc {3:N2}+/-{4:N2}, pct={5:N2}" -f $c.case_name, $c.baseline_overall_mean, $c.baseline_overall_ci95, $c.grc_overall_mean, $c.grc_overall_ci95, $c.overall_pct_of_baseline_mean)
}
Write-Host ("PPL baseline mean={0:N4}, grc mean={1:N4}, ppl pct of baseline={2:N2}, delta pct={3:N2}" -f $bpMu, $gpMu, $pplPct, $pplDeltaPct)
if ([math]::Abs($diag.coding_decode_pct - $diag.reasoning_decode_pct) -lt 5) {
    Write-Host "ROOT_CAUSE_HINT=Prompt-specific collapse not reproduced under deterministic repeated runs; prior outlier likely run-condition artifact."
} else {
    Write-Host "ROOT_CAUSE_HINT=Prompt-path sensitivity remains; inspect token-pattern-dependent runtime kernels and generated-token behavior."
}
