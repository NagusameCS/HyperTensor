# scripts/run_lm_eval_suite.ps1
# -----------------------------------------------------------------------------
# Behavioural-quality benchmark suite: GSM8K, HumanEval, MBPP, baseline vs.
# GRC k=1536. Driven through lm-evaluation-harness against the geodessical
# OpenAI-compatible HTTP server (or via the harness's gguf adapter).
#
# Prereqs:
#   - python with `lm-eval` installed (`pip install lm-eval`)
#   - geodessical built and wired with --serve flag (or your equivalent)
#   - GGUF model file
#
# Output:
#   docs/data/lm_eval_results.json
#   docs/data/lm_eval_results.tex   (\input-able table for Paper A / Paper C)
#
# The script does NOT fabricate scores. If lm-eval errors, the JSON is not
# written.
# -----------------------------------------------------------------------------

param(
    [string]$Model    = "C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    [string]$Exe      = ".\build_host\geodessical.exe",
    [string]$ServerHost = "127.0.0.1",
    [int]   $ServerPort = 8081,
    [string[]]$Tasks  = @("gsm8k", "humaneval", "mbpp"),
    [int]   $NumFewshot = 0,
    [int]   $Limit    = 0,                              # 0 = full set; set >0 for smoke runs
    [string]$OutJson  = "docs/data/lm_eval_results.json",
    [string]$OutTex   = "docs/data/lm_eval_results.tex"
)

$ErrorActionPreference = "Stop"

# --- prereq checks ----------------------------------------------------------
$py = (Get-Command python -ErrorAction SilentlyContinue)
if (-not $py) { throw "python not found on PATH" }
$lmeval = (& python -m lm_eval --help 2>$null)
if ($LASTEXITCODE -ne 0) {
    throw "lm-eval not installed in current python env. Run: pip install lm-eval"
}
if (-not (Test-Path $Exe))   { throw "Executable not found: $Exe" }
if (-not (Test-Path $Model)) { throw "Model not found: $Model" }

# --- launch geodessical server in a child job -------------------------------
function Start-GeodServer {
    param([string]$Label, [string[]]$ExtraArgs)
    $logFile = "lm_eval_$Label.log"
    Write-Host "[$Label] starting server: $Exe -m $Model --serve --port $ServerPort $($ExtraArgs -join ' ')"
    $job = Start-Job -ScriptBlock {
        param($exe, $model, $port, $extra, $log)
        & $exe -m $model --serve --port $port @extra *> $log
    } -ArgumentList $Exe, $Model, $ServerPort, $ExtraArgs, $logFile
    # crude wait-for-ready: poll TCP
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $deadline) {
        try {
            $tcp = New-Object System.Net.Sockets.TcpClient
            $tcp.Connect($ServerHost, $ServerPort)
            $tcp.Close()
            return $job
        } catch { Start-Sleep -Seconds 1 }
    }
    Stop-Job $job; Remove-Job $job
    throw "geodessical server failed to come up within 60s ($logFile)"
}

function Stop-GeodServer { param($job) Stop-Job $job; Remove-Job $job -Force }

# --- run lm-eval against running server -------------------------------------
function Invoke-LmEval {
    param([string]$Label)

    $modelArgs = "base_url=http://$ServerHost`:$ServerPort/v1/completions,model=geodessical"
    $tasksArg  = ($Tasks -join ",")
    $outDir    = "lm_eval_out_$Label"
    $args = @(
        "-m", "lm_eval",
        "--model", "local-completions",
        "--model_args", $modelArgs,
        "--tasks", $tasksArg,
        "--num_fewshot", $NumFewshot,
        "--output_path", $outDir,
        "--log_samples"
    )
    if ($Limit -gt 0) { $args += @("--limit", $Limit) }

    & python @args
    if ($LASTEXITCODE -ne 0) { throw "lm-eval failed for $Label (exit $LASTEXITCODE)" }

    # lm-eval writes results_*.json under $outDir/.../results_*.json
    $resultFile = Get-ChildItem -Path $outDir -Filter "results_*.json" -Recurse | Select-Object -First 1
    if (-not $resultFile) { throw "lm-eval result JSON not found under $outDir" }
    return (Get-Content $resultFile.FullName -Raw | ConvertFrom-Json)
}

# --- run both conditions ----------------------------------------------------
$results = @{}
$job = Start-GeodServer "baseline" @()
try   { $results.baseline = Invoke-LmEval "baseline" }
finally { Stop-GeodServer $job }

$job = Start-GeodServer "grc_k1536" @("--grc-rank", "1536")
try   { $results.grc_k1536 = Invoke-LmEval "grc_k1536" }
finally { Stop-GeodServer $job }

# --- emit JSON ---------------------------------------------------------------
$null = New-Item -ItemType Directory -Force -Path (Split-Path $OutJson -Parent)
$results | ConvertTo-Json -Depth 10 | Set-Content -Path $OutJson
Write-Host "Wrote $OutJson"

# --- emit \input-able LaTeX table -------------------------------------------
function Get-Acc { param($obj, $task)
    if ($obj.results.$task.acc)         { return [math]::Round([double]$obj.results.$task.acc, 4) }
    if ($obj.results.$task.'acc,none')  { return [math]::Round([double]$obj.results.$task.'acc,none', 4) }
    if ($obj.results.$task.'pass@1')    { return [math]::Round([double]$obj.results.$task.'pass@1', 4) }
    return "--"
}

$rows = foreach ($t in $Tasks) {
    $b = Get-Acc $results.baseline   $t
    $g = Get-Acc $results.grc_k1536  $t
    "$t & $b & $g \\"
}

$tex = @"
% auto-generated by scripts/run_lm_eval_suite.ps1 -- do not hand-edit.
\begin{tabular}{lrr}
\toprule
Task & Baseline & GRC \$k\!=\!1536\$ \\
\midrule
$($rows -join "`n")
\bottomrule
\end{tabular}
"@
$null = New-Item -ItemType Directory -Force -Path (Split-Path $OutTex -Parent)
Set-Content -Path $OutTex -Value $tex
Write-Host "Wrote $OutTex"
