param(
    [string]$Model = "C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    [string]$Exe = ".\build_host\geodessical.exe",
    [string]$OutRoot = "",
    [int]$CooldownSec = 30,
    [int]$LmEvalLimit = 0,
    [switch]$SkipNcu,
    [switch]$SkipLmEval,
    [switch]$SkipContextSweep,
    [switch]$SkipRankPareto
)

$ErrorActionPreference = "Stop"

if (-not $OutRoot) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutRoot = Join-Path ".\benchmarks" "cross_hw_local_$ts"
}

$logsDir = Join-Path $OutRoot "logs"
$dataDir = Join-Path $OutRoot "docs_data"
$metaDir = Join-Path $OutRoot "meta"
New-Item -ItemType Directory -Force -Path $logsDir, $dataDir, $metaDir | Out-Null

$venvScripts = Join-Path (Get-Location).Path ".venv\Scripts"
if (Test-Path $venvScripts) {
    $env:PATH = "$venvScripts;$env:PATH"
}

if (-not (Test-Path $Exe)) { throw "Executable not found: $Exe" }
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Body
    )

    $logPath = Join-Path $logsDir ("{0}.log" -f $Name)
    $status = [ordered]@{
        step = $Name
        ok = $false
        started_at = (Get-Date).ToString("s")
        ended_at = $null
        log = $logPath
        error = $null
    }

    try {
        & $Body *>&1 | Tee-Object -FilePath $logPath | Out-Host
        $status.ok = $true
    } catch {
        $status.error = $_.Exception.Message
        "ERROR: $($_.Exception.Message)" | Add-Content -Path $logPath
    } finally {
        $status.ended_at = (Get-Date).ToString("s")
    }

    return [pscustomobject]$status
}

$meta = [ordered]@{
    generated_at = (Get-Date).ToString("s")
    machine = $env:COMPUTERNAME
    user = $env:USERNAME
    cwd = (Get-Location).Path
    model = (Resolve-Path $Model).Path
    exe = (Resolve-Path $Exe).Path
    git_head = $null
    git_status = $null
    gpu = $null
}

if (Get-Command git -ErrorAction SilentlyContinue) {
    $meta.git_head = (git rev-parse HEAD 2>$null)
    $meta.git_status = (git status --porcelain=v1 2>$null)
}
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    $meta.gpu = (nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>$null)
}
$meta | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $metaDir "environment.json")

$steps = @()

if (-not $SkipNcu) {
    $steps += Invoke-Step "ncu_l2_profile" {
        & ".\scripts\benchmark_ncu_l2_profile.ps1" `
            -Model $Model `
            -Exe $Exe `
            -CooldownSec $CooldownSec `
            -OutCsv (Join-Path $dataDir "ncu_l2_profile.csv") `
            -OutTex (Join-Path $dataDir "ncu_l2_profile.tex")
    }
}

if (-not $SkipLmEval) {
    $steps += Invoke-Step "lm_eval_suite" {
        & ".\scripts\run_lm_eval_suite.ps1" `
            -Model $Model `
            -Exe $Exe `
            -Limit $LmEvalLimit `
            -OutJson (Join-Path $dataDir "lm_eval_results.json") `
            -OutTex (Join-Path $dataDir "lm_eval_results.tex")
    }
}

if (-not $SkipContextSweep) {
    $steps += Invoke-Step "context_length_sweep" {
        & ".\scripts\context_length_sweep.ps1" `
            -Model $Model `
            -Exe $Exe `
            -CooldownSec $CooldownSec `
            -OutCsv (Join-Path $dataDir "context_length_sweep.csv") `
            -OutTex (Join-Path $dataDir "context_length_sweep.tex")
    }
}

if (-not $SkipRankPareto) {
    $steps += Invoke-Step "rank_pareto" {
        & ".\scripts\benchmark_rank_pareto.ps1" `
            -Model $Model `
            -Exe $Exe `
            -CooldownSec $CooldownSec `
            -OutCsv (Join-Path $dataDir "rank_pareto.csv") `
            -OutTex (Join-Path $dataDir "rank_pareto.tex")
    }
}

$summary = [ordered]@{
    out_root = (Resolve-Path $OutRoot).Path
    generated_at = (Get-Date).ToString("s")
    steps = $steps
    all_passed = (@($steps | Where-Object { -not $_.ok }).Count -eq 0)
}

$summary | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $OutRoot "campaign_manifest.json")

Write-Host "CAMPAIGN_OUT=$OutRoot"
Write-Host "ALL_PASSED=$($summary.all_passed)"
