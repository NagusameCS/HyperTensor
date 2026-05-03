param(
    [string]$Exe = "C:\Users\legom\HyperTensor\build_host\geodessical.exe",
    [string]$Model = "C:\Users\legom\HyperTensor\models\smollm2-135m-instruct-q8_0.gguf",
    [string]$Prompt = "Explain gravity briefly.",
    [int]$NTokens = 32,
    [int]$Batch = 4,
    [double]$Thresh = 0.45,
    [int]$TimeoutSec = 180,
    [string]$OutDir = "benchmarks\ott_smoke"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Exe)) {
    Write-Error "smoke: exe not found: $Exe"
    exit 2
}
if (-not (Test-Path $Model)) {
    Write-Error "smoke: model not found: $Model"
    exit 2
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $OutDir "smoke_${stamp}.log"
$summaryPath = Join-Path $OutDir "smoke_summary.json"

$args = @(
    $Model,
    "--ott-full", "--ott-speculative",
    "--ott-spec-batch", "$Batch",
    "--ott-spec-thresh", "$Thresh",
    "-p", $Prompt,
    "-n", "$NTokens",
    "--temp", "0"
)

Write-Host "[smoke] running OTT smoke test"
Write-Host "[smoke] exe=$Exe"
Write-Host "[smoke] model=$Model"
Write-Host "[smoke] log=$logPath"

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$text = (& $Exe @args 2>&1 | Out-String)
$exitCode = $LASTEXITCODE
$sw.Stop()

$text | Out-File -FilePath $logPath -Encoding utf8
$hasSpecStart = $false
$hasSpecDone = $false
$hasTokRate = $false
$alpha = $null
$toks = $null

if ($text) {
    $hasSpecStart = $text -match "\[SPEC\]\s+Starting"
    $hasSpecDone = $text -match "\[SPEC\]\s+Done:"

    $tokMatch = [regex]::Match($text, "(\d+(?:\.\d+)?)\s+tok(?:ens)?/s")
    if ($tokMatch.Success) {
        $hasTokRate = $true
        $toks = [double]$tokMatch.Groups[1].Value
    }

    $alphaMatch = [regex]::Match($text, "acceptance_rate=(\d+(?:\.\d+)?)%")
    if ($alphaMatch.Success) {
        $alpha = [double]$alphaMatch.Groups[1].Value
    }
}

$passed = ($exitCode -eq 0 -and $hasSpecStart -and $hasSpecDone -and $hasTokRate)

$summary = [PSCustomObject]@{
    utc = (Get-Date).ToUniversalTime().ToString("o")
    passed = $passed
    exit_code = $exitCode
    elapsed_ms = [int]$sw.ElapsedMilliseconds
    has_spec_start = $hasSpecStart
    has_spec_done = $hasSpecDone
    has_tok_rate = $hasTokRate
    acceptance_rate_pct = $alpha
    tok_s = $toks
    log = $logPath
}

$summary | ConvertTo-Json -Depth 4 | Out-File -FilePath $summaryPath -Encoding utf8

if ($passed) {
    Write-Host "[smoke] PASS exit=$exitCode tok/s=$toks alpha=$alpha"
    Write-Host "[smoke] summary=$summaryPath"
    exit 0
}

Write-Host "[smoke] FAIL exit=$exitCode spec_start=$hasSpecStart spec_done=$hasSpecDone tok_rate=$hasTokRate"
Write-Host "[smoke] summary=$summaryPath"
exit 1
