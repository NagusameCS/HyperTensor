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

param([Parameter(ValueFromRemainingArguments=$true)][string[]]$allargs)
$ErrorActionPreference = "Stop"

$ScriptDir = $PSScriptRoot
$ExePaths = @(
    (Join-Path $ScriptDir "build_host\geodessical.exe"),
    (Join-Path $ScriptDir "geodessical.exe"),
    (Join-Path $env:USERPROFILE ".geod\bin\geodessical.exe")
)
$GeoExe = $null
foreach ($p in $ExePaths) { if (Test-Path $p) { $GeoExe = $p; break } }
if (-not $GeoExe) {
    Write-Host "[geod] geodessical.exe not found. Run .\build_host.ps1 first." -ForegroundColor Red
    exit 1
}

$ModelDirs = @(
    (Join-Path $ScriptDir "models"),
    (Join-Path $ScriptDir "..\models"),
    (Join-Path $ScriptDir "..\..\models"),
    (Join-Path $env:USERPROFILE ".geod\models")
)

function Find-Model([string]$Name) {
    if ($Name -match '\.gguf$' -and (Test-Path $Name)) { return $Name }
    foreach ($d in $ModelDirs) {
        if (-not (Test-Path $d)) { continue }
        $items = Get-ChildItem $d -Filter "*.gguf" -ErrorAction SilentlyContinue
        foreach ($f in $items) {
            $needle = ($Name.ToLower() -replace '[_\-\s]', '.*')
            if ($f.BaseName.ToLower() -match $needle) { return $f.FullName }
        }
    }
    return $null
}

function Show-List {
    Write-Host ""
    Write-Host "Available models:" -ForegroundColor Cyan
    $found = $false
    foreach ($d in $ModelDirs) {
        if (-not (Test-Path $d)) { continue }
        $items = Get-ChildItem $d -Filter "*.gguf" -ErrorAction SilentlyContinue
        foreach ($f in $items) {
            $sz = [math]::Round($f.Length / 1MB, 0)
            Write-Host ("  {0,-52} {1,5} MB" -f $f.Name, $sz)
            $found = $true
        }
    }
    if (-not $found) { Write-Host "  (none -- use: geod pull REPO)" -ForegroundColor Yellow }
    Write-Host ""
}

function Show-Help {
    @"

  geod -- Geodessical AI Runtime CLI
  Usage: geod COMMAND [model] [flags...]

  Commands:
    run     MODEL  One-shot inference
    chat    MODEL  Interactive chat (-i)
    ott     MODEL  OTT full mode   (--ott-full --axiom-fast)
    theorem MODEL  OTT theorem mode (--ott-theorem --axiom-fast)
    fast    MODEL  OTT fast mode   (--ott-fast)
    serve   MODEL  HTTP API server (--serve)
    pull    REPO   Download from HuggingFace
    list           List local models
    version        Show version
    help           Show this help

  Common flags:
    -p, --prompt TEXT        Prompt text
    -n, --tokens N           Max tokens (default 128)
    --temp F                 Temperature (default 0.7)
    --top-k N                Top-K sampling
    --ott-ready-report PATH  Write readiness JSON
    --axiom-fast             Fast axiom survey
    -i, --interactive        Interactive chat
    -v                       Verbose logging

  Examples:
    geod run gemma4 -p "What is entropy?" -n 200
    geod chat gemma4
    geod ott gemma4 -n 128 --ott-ready-report report.json
    geod theorem gemma4 --axiom-fast
    geod fast gemma4 -p "hello"
    geod pull google/gemma-2b-it-GGUF
    geod serve gemma4 --port 8080

"@ | Write-Host
}

if ($allargs.Count -eq 0) { Show-Help; exit 0 }

$Cmd  = $allargs[0]
$Rest = if ($allargs.Count -gt 1) { $allargs[1..($allargs.Count - 1)] } else { @() }

$ValueFlags = @("-p","--prompt","-n","--tokens","-t","--threads","--temp","--top-k",
                "--top-p","--port","--ott-ready-report","--axiom-report","--axiom-seed",
                "--axiom-samples","--axiom-probe","--depth-attn-strength","--depth-attn-window",
                "--attnres-strength","--log-level","--quant","--download")

function Split-ModelAndFlags([string[]]$arr) {
    $model = $null; $flags = @()
    $i = 0
    while ($i -lt $arr.Count) {
        $t = $arr[$i]
        if ($t.StartsWith("-")) {
            $flags += $t
            if ($t -in $ValueFlags) { $i++; if ($i -lt $arr.Count) { $flags += $arr[$i] } }
        } elseif ($null -eq $model) { $model = $t }
        else { $flags += $t }
        $i++
    }
    return $model, $flags
}

switch ($Cmd.ToLower()) {

    "version" {
        Write-Host "geod v1.0 -> geodessical v0.6.0 Synapse" -ForegroundColor Cyan
        Write-Host "  exe: $GeoExe"
        exit 0
    }

    "help" { Show-Help; exit 0 }
    "list" { Show-List; exit 0 }

    "pull" {
        if ($Rest.Count -eq 0) { Write-Host "[geod] pull requires a HuggingFace repo." -ForegroundColor Red; exit 1 }
        $Repo = $Rest[0]
        $xf   = if ($Rest.Count -gt 1) { $Rest[1..($Rest.Count-1)] } else { @() }
        & $GeoExe --download $Repo @xf
        exit $LASTEXITCODE
    }

    { $_ -in @("run","chat","ott","theorem","fast","serve") } {
        $modelArg, $passFlags = Split-ModelAndFlags $Rest
        if (-not $modelArg) {
            Write-Host "[geod] No model specified." -ForegroundColor Red; exit 1
        }
        $modelPath = Find-Model $modelArg
        if (-not $modelPath -and $modelArg -match "/") {
            Write-Host "[geod] Downloading $modelArg ..." -ForegroundColor Yellow
            & $GeoExe --download $modelArg
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
            $modelPath = Find-Model ($modelArg -split "/" | Select-Object -Last 1)
        }
        if (-not $modelPath) {
            Write-Host "[geod] Model not found: $modelArg" -ForegroundColor Red
            Write-Host "       Run 'geod list' or 'geod pull REPO'."
            exit 1
        }
        $preset = switch ($Cmd.ToLower()) {
            "chat"    { @("-i") }
            "ott"     { @("--ott-full","--axiom-fast") }
            "theorem" { @("--ott-theorem","--axiom-fast") }
            "fast"    { @("--ott-fast") }
            "serve"   { @("--serve") }
            default   { @() }
        }
        Write-Host "[geod] $($Cmd.ToUpper()) - $([System.IO.Path]::GetFileName($modelPath))" -ForegroundColor Green
        & $GeoExe $modelPath @preset @passFlags
        exit $LASTEXITCODE
    }

    default {
        # Shorthand: geod MODEL [flags]
        $modelPath = Find-Model $Cmd
        if ($modelPath) {
            Write-Host "[geod] RUN - $([System.IO.Path]::GetFileName($modelPath))" -ForegroundColor Green
            & $GeoExe $modelPath @Rest
            exit $LASTEXITCODE
        }
        Write-Host "[geod] Unknown command or model: $Cmd" -ForegroundColor Red
        Show-Help; exit 1
    }
}
