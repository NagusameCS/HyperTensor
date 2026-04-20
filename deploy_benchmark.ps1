# deploy_benchmark.ps1
# Full two-machine benchmark orchestration:
#   1. Run extended local benchmark (benchmark_extended.ps1)
#   2. Push HyperTensor repo + models to opencs server
#   3. SSH: build Geodessical + run remote benchmark
#   4. Pull remote CSV back
#   5. Generate cross-machine comparison report: benchmark_cross_machine.md
#
# Usage:
#   .\deploy_benchmark.ps1                   # full run
#   .\deploy_benchmark.ps1 -LocalOnly        # skip remote steps
#   .\deploy_benchmark.ps1 -RemoteOnly       # skip local benchmark
#   .\deploy_benchmark.ps1 -SkipBuild        # reuse existing binaries on remote
#   .\deploy_benchmark.ps1 -ReportOnly       # just regenerate the cross-machine report
#                                              (requires local + remote CSVs to exist)
#
# Prerequisites:
#   - cloudflared at 'C:\Program Files (x86)\cloudflared\cloudflared.exe'
#   - SSH config entry 'opencs' in ~/.ssh/config
#   - GGUF models at C:\Users\legom\TensorOS\models\
#   - zig in PATH (for local build)

param(
    [switch]$LocalOnly,
    [switch]$RemoteOnly,
    [switch]$SkipBuild,
    [switch]$ReportOnly,
    [switch]$SkipLocalBuild   # use existing local benchmark_extended.csv
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$LOCAL_CSV  = "benchmark_extended.csv"
$REMOTE_CSV = "benchmark_remote_results.csv"
$CROSS_MD   = "benchmark_cross_machine.md"
$MODEL_DIR  = "C:\Users\legom\TensorOS\models"
$REMOTE_HOST= "opencs"
$REMOTE_DIR = "/root/HyperTensor"
$REMOTE_MDL = "/root/models"

$MODELS_TO_PUSH = @(
    "smollm2-135m-instruct-q8_0.gguf"
    "Phi-3.5-mini-instruct-Q4_0.gguf"
    "google_gemma-4-E2B-it-Q4_0.gguf"
)

# ─── Helpers ─────────────────────────────────────────────────────────────────
function Write-Step($msg) {
    Write-Host ""
    Write-Host ">>> $msg" -ForegroundColor Cyan
    Write-Host ""
}

function Run-SSH($cmd) {
    & ssh $REMOTE_HOST $cmd
    if ($LASTEXITCODE -ne 0) { throw "SSH command failed (exit $LASTEXITCODE): $cmd" }
}

function Run-SCP($local, $remote) {
    & scp $local "${REMOTE_HOST}:${remote}"
    if ($LASTEXITCODE -ne 0) { throw "SCP failed: $local -> $remote" }
}

function Pull-SCP($remote, $local) {
    & scp "${REMOTE_HOST}:${remote}" $local
    if ($LASTEXITCODE -ne 0) { throw "SCP pull failed: $remote -> $local" }
}

# ─── Phase 1: Local extended benchmark ───────────────────────────────────────
if (-not $RemoteOnly -and -not $ReportOnly) {
    if ($SkipLocalBuild -and (Test-Path $LOCAL_CSV)) {
        Write-Step "Skipping local benchmark (using existing $LOCAL_CSV)"
    } else {
        Write-Step "Running local extended benchmark..."
        & .\benchmark_extended.ps1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[deploy] Local benchmark exited with non-zero code — continuing anyway." -ForegroundColor Yellow
        }
        if (-not (Test-Path $LOCAL_CSV)) {
            throw "Local benchmark did not produce $LOCAL_CSV"
        }
        Write-Host "[deploy] Local benchmark complete. ($((Get-Item $LOCAL_CSV).Length) bytes)" -ForegroundColor Green
    }
}

# ─── Phase 2: Push repo and models to remote ──────────────────────────────────
if (-not $LocalOnly -and -not $ReportOnly) {
    Write-Step "Pushing HyperTensor repo to $REMOTE_HOST..."

    # Create remote directories
    Run-SSH "mkdir -p $REMOTE_DIR $REMOTE_MDL"

    # Rsync the repo (exclude build artifacts, large model files, etc.)
    # Fall back to scp of key source directories if rsync not available
    $rsync = "rsync"
    if (-not (Get-Command $rsync -ErrorAction SilentlyContinue)) {
        Write-Host "[deploy] rsync not available locally — using targeted scp..." -ForegroundColor Yellow
        # Push key subdirs individually
        $dirsToSync = @("host", "runtime", "kernel", "boot", "scripts")
        foreach ($d in $dirsToSync) {
            if (Test-Path $d) {
                Write-Host "  scp -r $d -> $REMOTE_DIR/$d"
                & scp -r $d "${REMOTE_HOST}:${REMOTE_DIR}/"
            }
        }
        # Push root files
        $filesToPush = @(
            "benchmark_remote.sh",
            "CMakeLists.txt"
        )
        foreach ($f in $filesToPush) {
            if (Test-Path $f) { Run-SCP $f "$REMOTE_DIR/$f" }
        }
    } else {
        & $rsync -avz --delete `
            --exclude='.git' `
            --exclude='build_host/' `
            --exclude='build/' `
            --exclude='build_arm64/' `
            --exclude='*.gguf' `
            --exclude='models/' `
            --exclude='archives/' `
            "./" "${REMOTE_HOST}:${REMOTE_DIR}/"
    }

    Write-Step "Pushing GGUF models to $REMOTE_HOST..."
    foreach ($mf in $MODELS_TO_PUSH) {
        $src = Join-Path $MODEL_DIR $mf
        if (-not (Test-Path $src)) {
            Write-Host "  SKIP $mf — not found locally" -ForegroundColor Yellow
            continue
        }
        $sizeMB = [math]::Round((Get-Item $src).Length / 1MB, 0)
        Write-Host "  Uploading $mf ($sizeMB MB)..."
        & scp $src "${REMOTE_HOST}:${REMOTE_MDL}/${mf}"
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  [warn] Upload of $mf failed — continuing" -ForegroundColor Yellow
        } else {
            Write-Host "  -> OK" -ForegroundColor Green
        }
    }
    # Symlink models into repo models/ dir for the benchmark script
    Run-SSH "mkdir -p $REMOTE_DIR/models; cd $REMOTE_MDL && for f in *.gguf; do ln -sf \$PWD/\$f $REMOTE_DIR/models/\$f 2>/dev/null || true; done"

    # ─── Phase 3: Remote build + benchmark ─────────────────────────────────────
    Write-Step "Running remote benchmark on $REMOTE_HOST..."

    $buildFlag = if ($SkipBuild) { "--skip-build" } else { "" }

    $remoteCmd = @"
set -euo pipefail
cd $REMOTE_DIR
chmod +x benchmark_remote.sh
export MODEL_DIR=$REMOTE_MDL
bash benchmark_remote.sh $buildFlag 2>&1
"@
    Run-SSH $remoteCmd

    # ─── Phase 4: Pull remote results ──────────────────────────────────────────
    Write-Step "Pulling remote results..."
    Pull-SCP "/tmp/benchmark_remote_results.csv" $REMOTE_CSV
    Write-Host "[deploy] Remote CSV: $REMOTE_CSV ($((Get-Item $REMOTE_CSV).Length) bytes)" -ForegroundColor Green
}

# ─── Phase 5: Cross-machine comparison report ─────────────────────────────────
Write-Step "Generating cross-machine comparison report..."

if (-not (Test-Path $LOCAL_CSV)) {
    Write-Host "[deploy] No local CSV found at $LOCAL_CSV — skipping cross-machine report." -ForegroundColor Yellow
    exit 0
}

# ─── Load and parse CSVs ─────────────────────────────────────────────────────
function Load-CSV($path) {
    if (-not (Test-Path $path)) { return @() }
    return Import-Csv $path
}

$localRows  = Load-CSV $LOCAL_CSV
$remoteRows = Load-CSV $REMOTE_CSV

function Where-Valid($rows) { $rows | Where-Object { -not $_.Err -and $_.DecodeTS -gt 0 } }
function Avg-Field($rows, $field) {
    $vals = $rows | ForEach-Object { [double]$_.$field } | Where-Object { $_ -gt 0 }
    if (-not $vals) { return 0 }
    return [math]::Round(($vals | Measure-Object -Average).Average, 1)
}
function Max-Field($rows, $field) {
    $vals = $rows | ForEach-Object { [double]$_.$field }
    if (-not $vals) { return 0 }
    return ($vals | Measure-Object -Maximum).Maximum
}
function F($n) { if ($n -eq 0 -or $n -eq $null) { "--" } else { $n } }

# Machine info from local CSV headers
$localMachine  = "RTX 4070 / Ryzen 9 7940HS 8c (Windows)"
$remoteMachine = "RTX 3050 / 32c Linux (opencs)"

$md = [System.Text.StringBuilder]::new()
$null=$md.AppendLine("# HyperTensor Cross-Machine Benchmark")
$null=$md.AppendLine("")
$null=$md.AppendLine("**Generated:** $(Get-Date -Format 'yyyy-MM-dd HH:mm')")
$null=$md.AppendLine("")
$null=$md.AppendLine("| | Local | Remote (opencs) |")
$null=$md.AppendLine("|---|---|---|")
$null=$md.AppendLine("| CPU | AMD Ryzen 9 7940HS (8c/16t) | 32-core Arch Linux server |")
$null=$md.AppendLine("| GPU | RTX 4070 Laptop 8 GB | RTX 3050 6 GB |")
$null=$md.AppendLine("| OS | Windows | Arch Linux |")
$null=$md.AppendLine("| Runtime | Geodessical + Ollama | Geodessical + Ollama |")
$null=$md.AppendLine("")
$null=$md.AppendLine("---")
$null=$md.AppendLine("")

# ─── Summary: all combos both machines ───────────────────────────────────────
$null=$md.AppendLine("## Summary — Both Machines")
$null=$md.AppendLine("")
$null=$md.AppendLine("| Machine | Runtime | Backend | Model | Decode t/s | Prefill t/s | TTFT ms | Peak VRAM MB | Avg Power W |")
$null=$md.AppendLine("|---------|---------|---------|-------|:----------:|:-----------:|:-------:|:------------:|:-----------:|")

foreach ($dataset in @(@{Label="Local"; Rows=$localRows}, @{Label="Remote"; Rows=$remoteRows})) {
    $valid = Where-Valid $dataset.Rows
    $combos = $valid | Select-Object Runtime,Backend,Model -Unique | Sort-Object Runtime,Backend,Model
    foreach ($c in $combos) {
        $r = $valid | Where-Object { $_.Runtime -eq $c.Runtime -and $_.Backend -eq $c.Backend -and $_.Model -eq $c.Model }
        $aD = Avg-Field $r "DecodeTS"; $aP = Avg-Field $r "PrefillTS"
        $aT = Avg-Field $r "PrefillMs"; $aV = Max-Field $r "VramMB"
        $aW = Avg-Field $r "PowerW"
        $null=$md.AppendLine("| $($dataset.Label) | $($c.Runtime) | $($c.Backend) | $($c.Model) | $(F $aD) | $(F $aP) | $(F $aT) | $(F $aV) | $(F $aW) |")
    }
}
$null=$md.AppendLine("")
$null=$md.AppendLine("---")
$null=$md.AppendLine("")

# ─── GPU head-to-head: models present on both machines ────────────────────────
$null=$md.AppendLine("## GPU Head-to-Head: RTX 4070 vs RTX 3050")
$null=$md.AppendLine("")
$null=$md.AppendLine("Same model, same prompt class, same N. Geodessical GPU decode t/s.")
$null=$md.AppendLine("")

# Find model name mappings between local and remote (gemma4-2b appears on both)
$crossModels = @(
    [PSCustomObject]@{ LocalName="gemma4-2b"; RemoteName="gemma4-2b"; Display="Gemma-4-E2B Q4_0" }
    [PSCustomObject]@{ LocalName="smollm2-135m"; RemoteName="smollm2-135m"; Display="SmolLM2-135M Q8_0" }
    [PSCustomObject]@{ LocalName="phi35-mini"; RemoteName="phi35-mini"; Display="Phi-3.5 Mini Q4_0" }
)

foreach ($xm in $crossModels) {
    $lRows = Where-Valid $localRows  | Where-Object { $_.Model -eq $xm.LocalName  -and $_.Runtime -eq "Geodessical" -and $_.Backend -eq "GPU" }
    $rRows = Where-Valid $remoteRows | Where-Object { $_.Model -eq $xm.RemoteName -and $_.Runtime -eq "Geodessical" -and $_.Backend -eq "GPU" }
    if (-not $lRows -and -not $rRows) { continue }

    $null=$md.AppendLine("### $($xm.Display)")
    $null=$md.AppendLine("")
    $null=$md.AppendLine("| N | Local RTX 4070 t/s | Remote RTX 3050 t/s | RTX 4070 Advantage |")
    $null=$md.AppendLine("|--:|:------------------:|:-------------------:|:------------------:|")
    foreach ($n in @(40, 128, 512, 1024)) {
        $lN = $lRows | Where-Object { $_.N -eq $n }
        $rN = $rRows | Where-Object { $_.N -eq $n }
        $lD = Avg-Field $lN "DecodeTS"; $rD = Avg-Field $rN "DecodeTS"
        if ($lD -eq 0 -and $rD -eq 0) { continue }
        $adv = if ($rD -gt 0 -and $lD -gt 0) { "+$([math]::Round(($lD-$rD)/$rD*100,1))%" } else { "N/A" }
        $null=$md.AppendLine("| $n | $(F $lD) | $(F $rD) | $adv |")
    }
    $null=$md.AppendLine("")
}
$null=$md.AppendLine("---")
$null=$md.AppendLine("")

# ─── CPU head-to-head: 8-core Windows vs 32-core Linux ─────────────────────
$null=$md.AppendLine("## CPU Head-to-Head: 8-core Windows vs 32-core Linux")
$null=$md.AppendLine("")
$null=$md.AppendLine("Geodessical CPU decode t/s.")
$null=$md.AppendLine("")

foreach ($xm in $crossModels) {
    $lRows = Where-Valid $localRows  | Where-Object { $_.Model -eq $xm.LocalName  -and $_.Runtime -eq "Geodessical" -and $_.Backend -eq "CPU" }
    $rRows = Where-Valid $remoteRows | Where-Object { $_.Model -eq $xm.RemoteName -and $_.Runtime -eq "Geodessical" -and $_.Backend -eq "CPU" }
    if (-not $lRows -and -not $rRows) { continue }

    $null=$md.AppendLine("### $($xm.Display)")
    $null=$md.AppendLine("")
    $null=$md.AppendLine("| N | Local 8c t/s | Remote 32c t/s | Remote Advantage |")
    $null=$md.AppendLine("|--:|:------------:|:--------------:|:----------------:|")
    foreach ($n in @(40, 128, 512, 1024)) {
        $lN = $lRows | Where-Object { $_.N -eq $n }
        $rN = $rRows | Where-Object { $_.N -eq $n }
        $lD = Avg-Field $lN "DecodeTS"; $rD = Avg-Field $rN "DecodeTS"
        if ($lD -eq 0 -and $rD -eq 0) { continue }
        $adv = if ($lD -gt 0 -and $rD -gt 0) {
            $pct = [math]::Round(($rD-$lD)/$lD*100,1)
            if ($pct -ge 0) { "+${pct}% (remote)" } else { "${pct}% (local wins)" }
        } else { "N/A" }
        $null=$md.AppendLine("| $n | $(F $lD) | $(F $rD) | $adv |")
    }
    $null=$md.AppendLine("")
}
$null=$md.AppendLine("---")
$null=$md.AppendLine("")

# ─── Ollama cross-machine comparison ─────────────────────────────────────────
$null=$md.AppendLine("## Ollama GPU Cross-Machine")
$null=$md.AppendLine("")

# Ollama model names may differ; use displayable name from model field
$ollCombos = @(
    @{ LocMdl="gemma4-2b"; RemMdl="gemma4-e2b"; Display="Gemma-4-E2B" }
    @{ LocMdl="smollm2-135m"; RemMdl="smollm2-135m"; Display="SmolLM2-135M" }
)
foreach ($xm in $ollCombos) {
    $lRows = Where-Valid $localRows  | Where-Object { $_.Model -eq $xm.LocMdl -and $_.Runtime -eq "Ollama" -and $_.Backend -eq "GPU" }
    $rRows = Where-Valid $remoteRows | Where-Object { $_.Model -like "$($xm.RemMdl)*" -and $_.Runtime -eq "Ollama" -and $_.Backend -eq "GPU" }
    if (-not $lRows -and -not $rRows) { continue }

    $null=$md.AppendLine("### $($xm.Display)")
    $null=$md.AppendLine("")
    $null=$md.AppendLine("| N | Local RTX 4070 t/s | Remote RTX 3050 t/s | RTX 4070 Advantage |")
    $null=$md.AppendLine("|--:|:------------------:|:-------------------:|:------------------:|")
    foreach ($n in @(40, 128, 512, 1024)) {
        $lN = $lRows | Where-Object { $_.N -eq $n }
        $rN = $rRows | Where-Object { $_.N -eq $n }
        $lD = Avg-Field $lN "DecodeTS"; $rD = Avg-Field $rN "DecodeTS"
        if ($lD -eq 0 -and $rD -eq 0) { continue }
        $adv = if ($rD -gt 0 -and $lD -gt 0) {
            $pct = [math]::Round(($lD-$rD)/$rD*100,1)
            if ($pct -ge 0) { "+${pct}%" } else { "${pct}% (remote faster)" }
        } else { "N/A" }
        $null=$md.AppendLine("| $n | $(F $lD) | $(F $rD) | $adv |")
    }
    $null=$md.AppendLine("")
}
$null=$md.AppendLine("---")
$null=$md.AppendLine("")

# ─── VRAM efficiency table ────────────────────────────────────────────────────
$null=$md.AppendLine("## GPU Efficiency: Decode t/s per GB VRAM (Geodessical GPU)")
$null=$md.AppendLine("")
$null=$md.AppendLine("| Machine | GPU | VRAM GB | Model | Decode t/s | t/s per GB VRAM |")
$null=$md.AppendLine("|---------|-----|:-------:|-------|:----------:|:---------------:|")
$machineVram = @{ "local"=8; "remote"=6 }
foreach ($dataset in @(@{Label="local"; Rows=$localRows}, @{Label="remote"; Rows=$remoteRows})) {
    $valid = Where-Valid $dataset.Rows | Where-Object { $_.Runtime -eq "Geodessical" -and $_.Backend -eq "GPU" }
    $modelCombos = $valid | Select-Object Model -Unique | Sort-Object Model
    foreach ($mc in $modelCombos) {
        $r = $valid | Where-Object { $_.Model -eq $mc.Model }
        $dec = Avg-Field $r "DecodeTS"
        $vramGB = $machineVram[$dataset.Label]
        $gpuLabel = if ($dataset.Label -eq "local") { "RTX 4070" } else { "RTX 3050" }
        $eff = if ($dec -gt 0 -and $vramGB -gt 0) { [math]::Round($dec/$vramGB,1) } else { "--" }
        $null=$md.AppendLine("| $($dataset.Label) | $gpuLabel | $vramGB | $($mc.Model) | $(F $dec) | $eff |")
    }
}
$null=$md.AppendLine("")
$null=$md.AppendLine("---")
$null=$md.AppendLine("")

# ─── Geodessical vs Ollama: both machines ─────────────────────────────────────
$null=$md.AppendLine("## Geodessical vs Ollama: GPU — Both Machines")
$null=$md.AppendLine("")
$null=$md.AppendLine("Average decode t/s across all conditions (GPU backend).")
$null=$md.AppendLine("")
$null=$md.AppendLine("| Machine | Model | Geodessical t/s | Ollama t/s | Geo Advantage |")
$null=$md.AppendLine("|---------|-------|:---------------:|:----------:|:-------------:|")
foreach ($dataset in @(@{Label="local"; Rows=$localRows}, @{Label="remote"; Rows=$remoteRows})) {
    $valid = Where-Valid $dataset.Rows | Where-Object { $_.Backend -eq "GPU" }
    $modelCombos = $valid | Select-Object Model -Unique | Sort-Object Model
    foreach ($mc in $modelCombos) {
        $geoR = $valid | Where-Object { $_.Model -eq $mc.Model -and $_.Runtime -eq "Geodessical" }
        $ollR = $valid | Where-Object { $_.Model -eq $mc.Model -and $_.Runtime -eq "Ollama" }
        $geoD = Avg-Field $geoR "DecodeTS"; $ollD = Avg-Field $ollR "DecodeTS"
        if ($geoD -eq 0 -and $ollD -eq 0) { continue }
        $adv = if ($geoD -gt 0 -and $ollD -gt 0) {
            $pct = [math]::Round(($geoD-$ollD)/$ollD*100,1)
            if ($pct -ge 0) { "+${pct}% (Geo)" } else { "${pct}% (Oll)" }
        } else { "N/A" }
        $null=$md.AppendLine("| $($dataset.Label) | $($mc.Model) | $(F $geoD) | $(F $ollD) | $adv |")
    }
}
$null=$md.AppendLine("")

$md.ToString() | Set-Content $CROSS_MD -Encoding UTF8
Write-Host "[deploy] Cross-machine report: $CROSS_MD" -ForegroundColor Green
Write-Host ""
Write-Host "  All done." -ForegroundColor Green
Write-Host ""
Write-Host "  Files generated:" -ForegroundColor White
if (Test-Path $LOCAL_CSV)  { Write-Host "    $LOCAL_CSV" }
if (Test-Path $REMOTE_CSV) { Write-Host "    $REMOTE_CSV" }
if (Test-Path $CROSS_MD)   { Write-Host "    $CROSS_MD" }
Write-Host ""
