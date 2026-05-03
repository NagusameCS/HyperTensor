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
    [string]$Model = "C:\Users\legom\TensorOS\models\google_gemma-4-E2B-it-Q4_0.gguf",
    [string]$Prompt = "What is gravity?",
    [int]$Tokens = 128,
    [int]$Batch = 16,
    [int]$SampleMs = 500,
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not $OutputDir) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $repoRoot ("benchmarks\ott-perfect_" + $stamp)
}

$null = New-Item -ItemType Directory -Force -Path $OutputDir

$exePath = Join-Path $repoRoot "build_host\geodessical.exe"
$stdoutPath = Join-Path $OutputDir "stdout.txt"
$stderrPath = Join-Path $OutputDir "stderr.txt"
$rawPath = Join-Path $OutputDir "run_output.txt"
$telemetryPath = Join-Path $OutputDir "telemetry.csv"
$summaryPath = Join-Path $OutputDir "summary.json"
$axiomReportPath = Join-Path $OutputDir "axiom_beta_report.json"
$readyReportPath = Join-Path $OutputDir "ott_readiness_report.json"

function Resolve-SystemPowerCounterPath {
    $candidateSets = @("Power Meter", "Energy Meter")
    foreach ($setName in $candidateSets) {
        try {
            $set = Get-Counter -ListSet $setName -ErrorAction Stop
            $paths = @($set.PathsWithInstances)
            if (-not $paths) {
                continue
            }
            $preferred = $paths |
                Where-Object { $_ -match "_Total" -and $_ -notmatch "Budget" } |
                Select-Object -First 1
            if ($preferred) {
                return $preferred
            }
            $fallback = $paths | Where-Object { $_ -notmatch "Budget" } | Select-Object -First 1
            if ($fallback) {
                return $fallback
            }
        } catch {
        }
    }
    return $null
}

function Get-SystemCpuPercent {
    try {
        $value = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average
        if ($null -eq $value) {
            return $null
        }
        return [math]::Round([double]$value, 1)
    } catch {
        return $null
    }
}

function Get-SystemPowerWatts {
    param(
        [string]$CounterPath
    )

    if (-not $CounterPath) {
        return $null
    }

    try {
        $sample = Get-Counter -Counter $CounterPath -ErrorAction Stop
        $value = ($sample.CounterSamples | Select-Object -First 1).CookedValue
        if ($null -eq $value) {
            return $null
        }
        return [math]::Round([double]$value, 1)
    } catch {
        return $null
    }
}

function Get-GpuTelemetry {
    $default = [ordered]@{
        gpu_util_pct = $null
        gpu_mem_util_pct = $null
        gpu_mem_mb = $null
        gpu_power_w = $null
        gpu_temp_c = $null
    }

    try {
        $cmd = Get-Command nvidia-smi -ErrorAction Stop
    } catch {
        return [PSCustomObject]$default
    }

    try {
        $line = & $cmd.Source --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits 2>$null |
            Select-Object -First 1
        if (-not $line) {
            return [PSCustomObject]$default
        }
        $parts = $line.Trim() -split ",\s*"
        if ($parts.Count -lt 5) {
            return [PSCustomObject]$default
        }
        return [PSCustomObject]@{
            gpu_util_pct = [double]$parts[0]
            gpu_mem_util_pct = [double]$parts[1]
            gpu_mem_mb = [double]$parts[2]
            gpu_power_w = [double]$parts[3]
            gpu_temp_c = [double]$parts[4]
        }
    } catch {
        return [PSCustomObject]$default
    }
}

function Get-Stats {
    param(
        [double[]]$Values,
        [int]$Digits = 1
    )

    $usable = @($Values | Where-Object { $null -ne $_ })
    if (-not $usable -or $usable.Count -eq 0) {
        return [ordered]@{
            avg = $null
            peak = $null
        }
    }

    $avg = ($usable | Measure-Object -Average).Average
    $peak = ($usable | Measure-Object -Maximum).Maximum
    return [ordered]@{
        avg = [math]::Round([double]$avg, $Digits)
        peak = [math]::Round([double]$peak, $Digits)
    }
}

if (-not (Test-Path $exePath)) {
    throw "Missing executable: $exePath"
}

$cpuInfo = (Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)
$gpuInfo = (Get-CimInstance Win32_VideoController | Select-Object -First 1 -ExpandProperty Name)
$powerCounterPath = Resolve-SystemPowerCounterPath

$argsList = @(
    $Model,
    "-p", $Prompt,
    "-n", [string]$Tokens,
    "--ott-perfect",
    "--ott-spec-batch", [string]$Batch,
    "--axiom-report", $axiomReportPath,
    "--ott-ready-report", $readyReportPath
)

$argumentString = ($argsList | ForEach-Object {
    if ($_ -match '[\s"]') {
        '"' + ($_ -replace '"', '\"') + '"'
    } else {
        $_
    }
}) -join " "

"timestamp_ms,proc_cpu_pct,sys_cpu_pct,proc_ram_mb,gpu_util_pct,gpu_mem_util_pct,gpu_mem_mb,gpu_power_w,gpu_temp_c,sys_power_w" |
    Set-Content -Path $telemetryPath -Encoding UTF8

$proc = Start-Process -FilePath $exePath `
    -ArgumentList $argumentString `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$proc.Refresh()
$prevCpuSec = if ($null -ne $proc.CPU) { [double]$proc.CPU } else { 0.0 }
$prevWall = Get-Date

while (-not $proc.HasExited) {
    Start-Sleep -Milliseconds $SampleMs
    try {
        $proc.Refresh()
    } catch {
        break
    }

    $now = Get-Date
    $elapsed = ($now - $prevWall).TotalSeconds
    $procCpuPct = $null
    if ($elapsed -gt 0 -and $null -ne $proc.CPU) {
        $cpuDelta = [double]$proc.CPU - $prevCpuSec
        $procCpuPct = [math]::Round((100.0 * $cpuDelta) / ($elapsed * [Environment]::ProcessorCount), 1)
        $prevCpuSec = [double]$proc.CPU
    }
    $prevWall = $now

    $procRamMb = if ($null -ne $proc.WorkingSet64) {
        [math]::Round([double]$proc.WorkingSet64 / 1MB, 1)
    } else {
        $null
    }
    $gpu = Get-GpuTelemetry
    $sysCpuPct = Get-SystemCpuPercent
    $sysPowerW = Get-SystemPowerWatts -CounterPath $powerCounterPath

    "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f `
        [math]::Round($sw.Elapsed.TotalMilliseconds, 0), `
        $procCpuPct, `
        $sysCpuPct, `
        $procRamMb, `
        $gpu.gpu_util_pct, `
        $gpu.gpu_mem_util_pct, `
        $gpu.gpu_mem_mb, `
        $gpu.gpu_power_w, `
        $gpu.gpu_temp_c, `
        $sysPowerW | Add-Content -Path $telemetryPath
}

$proc.WaitForExit()
$sw.Stop()

@(
    if (Test-Path $stdoutPath) { Get-Content $stdoutPath }
    if (Test-Path $stderrPath) { Get-Content $stderrPath }
) | Set-Content -Path $rawPath -Encoding UTF8

$raw = Get-Content -Path $rawPath -Raw
$telemetryRows = @()
if (Test-Path $telemetryPath) {
    $telemetryRows = @(Import-Csv -Path $telemetryPath)
}

$gdMatch = [regex]::Match($raw, "\[GD\] (\d+) tokens in (\d+) ms \(([\d.]+) tok/s\)")
$decodeMatch = [regex]::Match($raw, "\[GD\] Decode-only: prefill ([\d.]+) ms, ([\d.]+) tok/s")
$specMatch = [regex]::Match($raw, "\[SPEC\] Done: (\d+) tokens \(geo_accepted=(\d+) xfmr=(\d+), acceptance_rate=([\d.]+)%.*, final_batch=(\d+)\)")
$tpfModelMatch = [regex]::Match($raw, "\[TpF\] N=([\d.]+)M\s+b_p=([\d.]+) B/param\s+model=([\d.]+)MB")
$tpfComputeMatch = [regex]::Match($raw, "\[TpF\] @ ([\d.]+) tok/s: ([\d.]+) GFLOPS \(([\d.]+)% of ([\d.]+) TFLOPS peak\)")
$tpfMemoryMatch = [regex]::Match($raw, "\[TpF\] @ ([\d.]+) tok/s: ([\d.]+) GB/s HBM \(([\d.]+)% of ([\d.]+) GB/s peak\)\s+eta_tok=([\d.]+)")

$procCpuStats = Get-Stats -Values ($telemetryRows | ForEach-Object { if ($_.proc_cpu_pct) { [double]$_.proc_cpu_pct } })
$sysCpuStats = Get-Stats -Values ($telemetryRows | ForEach-Object { if ($_.sys_cpu_pct) { [double]$_.sys_cpu_pct } })
$procRamStats = Get-Stats -Values ($telemetryRows | ForEach-Object { if ($_.proc_ram_mb) { [double]$_.proc_ram_mb } })
$gpuUtilStats = Get-Stats -Values ($telemetryRows | ForEach-Object { if ($_.gpu_util_pct) { [double]$_.gpu_util_pct } })
$gpuMemUtilStats = Get-Stats -Values ($telemetryRows | ForEach-Object { if ($_.gpu_mem_util_pct) { [double]$_.gpu_mem_util_pct } })
$gpuMemStats = Get-Stats -Values ($telemetryRows | ForEach-Object { if ($_.gpu_mem_mb) { [double]$_.gpu_mem_mb } })
$gpuPowerStats = Get-Stats -Values ($telemetryRows | ForEach-Object { if ($_.gpu_power_w) { [double]$_.gpu_power_w } })
$gpuTempStats = Get-Stats -Values ($telemetryRows | ForEach-Object { if ($_.gpu_temp_c) { [double]$_.gpu_temp_c } })
$sysPowerStats = Get-Stats -Values ($telemetryRows | ForEach-Object { if ($_.sys_power_w) { [double]$_.sys_power_w } })

$tokensGenerated = if ($gdMatch.Success) { [int]$gdMatch.Groups[1].Value } else { 0 }
$totalMs = if ($gdMatch.Success) { [int]$gdMatch.Groups[2].Value } else { [math]::Round($sw.Elapsed.TotalMilliseconds, 0) }
$tokPerSec = if ($gdMatch.Success) { [double]$gdMatch.Groups[3].Value } else { $null }
$prefillMs = if ($decodeMatch.Success) { [double]$decodeMatch.Groups[1].Value } else { $null }
$decodeTokPerSec = if ($decodeMatch.Success) { [double]$decodeMatch.Groups[2].Value } else { $null }
$timePerTokenMs = if ($tokensGenerated -gt 0) { [math]::Round($totalMs / [double]$tokensGenerated, 2) } else { $null }

$summary = [ordered]@{
    command = [ordered]@{
        exe = $exePath
        args = $argsList
        working_directory = $repoRoot
    }
    environment = [ordered]@{
        cpu = $cpuInfo
        gpu = $gpuInfo
        system_power_counter = $powerCounterPath
    }
    config = [ordered]@{
        model = $Model
        prompt = $Prompt
        tokens_requested = $Tokens
        batch = $Batch
        sample_ms = $SampleMs
        mode = "ott-perfect"
    }
    output = [ordered]@{
        tokens_generated = $tokensGenerated
        total_ms = $totalMs
        tok_per_s = $tokPerSec
        decode_tok_per_s = $decodeTokPerSec
        prefill_ms = $prefillMs
        ms_per_token = $timePerTokenMs
    }
    speculative = [ordered]@{
        geo_accepted = if ($specMatch.Success) { [int]$specMatch.Groups[2].Value } else { $null }
        xfmr_tokens = if ($specMatch.Success) { [int]$specMatch.Groups[3].Value } else { $null }
        acceptance_rate_pct = if ($specMatch.Success) { [double]$specMatch.Groups[4].Value } else { $null }
        final_batch = if ($specMatch.Success) { [int]$specMatch.Groups[5].Value } else { $null }
    }
    tpf = [ordered]@{
        model_millions = if ($tpfModelMatch.Success) { [double]$tpfModelMatch.Groups[1].Value } else { $null }
        bytes_per_param = if ($tpfModelMatch.Success) { [double]$tpfModelMatch.Groups[2].Value } else { $null }
        model_mb = if ($tpfModelMatch.Success) { [double]$tpfModelMatch.Groups[3].Value } else { $null }
        actual_gflops = if ($tpfComputeMatch.Success) { [double]$tpfComputeMatch.Groups[2].Value } else { $null }
        compute_peak_util_pct = if ($tpfComputeMatch.Success) { [double]$tpfComputeMatch.Groups[3].Value } else { $null }
        memory_bw_gbps = if ($tpfMemoryMatch.Success) { [double]$tpfMemoryMatch.Groups[2].Value } else { $null }
        memory_peak_util_pct = if ($tpfMemoryMatch.Success) { [double]$tpfMemoryMatch.Groups[3].Value } else { $null }
        eta_tok = if ($tpfMemoryMatch.Success) { [double]$tpfMemoryMatch.Groups[5].Value } else { $null }
    }
    telemetry = [ordered]@{
        samples = $telemetryRows.Count
        process_cpu_pct = $procCpuStats
        system_cpu_pct = $sysCpuStats
        process_ram_mb = $procRamStats
        gpu_util_pct = $gpuUtilStats
        gpu_mem_util_pct = $gpuMemUtilStats
        gpu_mem_mb = $gpuMemStats
        gpu_power_w = $gpuPowerStats
        gpu_temp_c = $gpuTempStats
        system_power_w = $sysPowerStats
    }
    artifacts = [ordered]@{
        output_dir = $OutputDir
        stdout = $stdoutPath
        stderr = $stderrPath
        raw_output = $rawPath
        telemetry_csv = $telemetryPath
        summary_json = $summaryPath
        axiom_report = $axiomReportPath
        readiness_report = $readyReportPath
    }
}

$summary | ConvertTo-Json -Depth 8 | Set-Content -Path $summaryPath -Encoding UTF8

Write-Host "[bench] OTT perfect benchmark complete"
Write-Host ("[bench] tokens={0} total_ms={1} tok/s={2} ms/tok={3}" -f `
    $summary.output.tokens_generated,
    $summary.output.total_ms,
    $summary.output.tok_per_s,
    $summary.output.ms_per_token)
Write-Host ("[bench] acceptance={0}% gpu_avg={1}% gpu_power_avg={2}W cpu_avg={3}%" -f `
    $summary.speculative.acceptance_rate_pct,
    $summary.telemetry.gpu_util_pct.avg,
    $summary.telemetry.gpu_power_w.avg,
    $summary.telemetry.system_cpu_pct.avg)
Write-Host ("[bench] summary: {0}" -f $summaryPath)