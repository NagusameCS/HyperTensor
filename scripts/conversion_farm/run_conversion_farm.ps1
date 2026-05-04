# scripts/conversion_farm/run_conversion_farm.ps1
# Crash-safe, resumable, multi-paper GRC conversion and benchmarking farm.
# Compatible with PowerShell 5.1. NO ConvertFrom-Json -AsHashtable, NO += on
# hashtable array values, NO PSObject.Properties on strings.
#
# Checkpoint format (flat, safe):
#   { run_id, saved_utc, completed: ["id1","id2",...], failed: {"id": N_fails} }
#
# Usage:
#   .\scripts\conversion_farm\run_conversion_farm.ps1              # fresh run
#   .\scripts\conversion_farm\run_conversion_farm.ps1 -Resume      # skip done jobs
#   .\scripts\conversion_farm\run_conversion_farm.ps1 -MaxJobs 3   # limit to 3 jobs this run
#   .\scripts\conversion_farm\run_conversion_farm.ps1 -PackageOnly # just zip existing outputs
[CmdletBinding()]
param(
    [string]$ManifestPath = "C:\Users\legom\HyperTensor\scripts\conversion_farm\farm_manifest.json",
    [string]$OutRoot      = "C:\Users\legom\HyperTensor\benchmarks\conversion_farm",
    [switch]$Resume,
    [int]$MaxJobs  = 0,
    [switch]$PackageOnly
)
$ErrorActionPreference = "Stop"
$REPO = "C:\Users\legom\HyperTensor"

#  Find Python 
function Resolve-PyExe {
    $cands = @("$REPO\.venv\Scripts\python.exe", "python")
    foreach ($c in $cands) {
        if ($c -eq "python") {
            $cmd = Get-Command python -ErrorAction SilentlyContinue
            if ($cmd) { return $cmd.Source }
        } elseif (Test-Path $c) { return $c }
    }
    throw "Python not found"
}

#  Resolve Ollama blob 
function Resolve-Blob {
    param([string]$Tag)
    $root = Join-Path $env:USERPROFILE ".ollama\models"
    # e.g.  gemma4:31b  ->  manifests/registry.ollama.ai/library/gemma4/31b
    $rel  = ($Tag -replace ':','\')
    $mfp  = Join-Path $root "manifests\registry.ollama.ai\library\$rel"
    if (-not (Test-Path $mfp)) { throw "Ollama manifest not found for $Tag at $mfp" }
    $mf   = Get-Content $mfp -Raw | ConvertFrom-Json
    $lay  = $mf.layers | Where-Object { $_.mediaType -match "model" } | Select-Object -First 1
    if (-not $lay) { throw "No model layer in manifest for $Tag" }
    $blob = Join-Path $root ("blobs\" + ($lay.digest -replace ':','-'))
    if (-not (Test-Path $blob)) { throw "Blob missing for $Tag at $blob" }
    return $blob
}

#  Run a process, capture stdout/stderr to files 
function Invoke-Timed {
    param(
        [string]   $Exe,
        [string[]] $ProcArgs,
        [string]   $StdoutFile,
        [string]   $StderrFile,
        [int]      $TimeoutSec = 0
    )
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $StdoutFile) | Out-Null
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $StderrFile) | Out-Null
    $proc = Start-Process -FilePath $Exe -ArgumentList $ProcArgs `
        -WorkingDirectory $REPO -NoNewWindow -PassThru `
        -RedirectStandardOutput $StdoutFile `
        -RedirectStandardError  $StderrFile
    if ($TimeoutSec -gt 0) {
        $finished = $proc.WaitForExit($TimeoutSec * 1000)
        if (-not $finished) {
            try { $proc.Kill() } catch {}
            return 124   # timeout sentinel
        }
    } else {
        $proc.WaitForExit()
    }
    return [int]$proc.ExitCode
}

#  Build [exe, @args] for a job 
function Build-Cmd {
    param($Job, [string]$PyExe, [string]$LogDir, [int]$Attempt)

    $id   = [string]$Job.id
    $mode = [string]$Job.mode

    switch ($mode) {

        "run_local" {
            $logSwitch = Join-Path $LogDir "${id}_a${Attempt}"
            $a = [System.Collections.Generic.List[string]]::new()
            $a.AddRange([string[]]@("-NoProfile","-ExecutionPolicy","Bypass","-File",
                "scripts\run_local_30b_grc.ps1",
                "-Model",[string]$Job.model,
                "-LogDir",$logSwitch))
            if ($null -ne $Job.rank)     { $a.Add("-Rank");     $a.Add([string]$Job.rank) }
            if ($null -ne $Job.npredict) { $a.Add("-NPredict"); $a.Add([string]$Job.npredict) }
            if ($null -ne $Job.ctx)      { $a.Add("-Ctx");      $a.Add([string]$Job.ctx) }
            if ($null -ne $Job.max_err)  { $a.Add("-MaxErr");   $a.Add([string]$Job.max_err) }
            if ($Job.baseline)        { $a.Add("-Baseline") }
            if ($Job.weight_pca_only) { $a.Add("-WeightPcaOnly") }
            if ($Job.no_offload)      { $a.Add("-NoOffload") }
            if ($Job.load_only)       { $a.Add("-LoadOnly") }
            if ($Job.ppl_only)        { $a.Add("-PplOnly") }
            if ($Job.prompt)          { $a.Add("-Prompt"); $a.Add([string]$Job.prompt) }
            return [PSCustomObject]@{ Exe = "powershell"; Args = $a.ToArray() }
        }

        "compute_kint" {
            $blob = Resolve-Blob -Tag ([string]$Job.model)
            $a    = [System.Collections.Generic.List[string]]::new()
            $a.AddRange([string[]]@("scripts\paperA_proof\compute_kint.py",
                "--blob",$blob, "--name",[string]$Job.name))
            if ($Job.layers -and ($Job.layers | Measure-Object).Count -gt 0) {
                $a.Add("--layers")
                foreach ($l in $Job.layers) { $a.Add([string]$l) }
            }
            $outDir = if ($Job.out_dir) { [string]$Job.out_dir } else { "docs/figures/paper-a/kint_30b" }
            $a.AddRange([string[]]@("--out",$outDir))
            return [PSCustomObject]@{ Exe = $PyExe; Args = $a.ToArray() }
        }

        "rho" {
            $blob = Resolve-Blob -Tag ([string]$Job.model)
            $a    = [System.Collections.Generic.List[string]]::new()
            $a.AddRange([string[]]@("scripts\grc_distill.py","--print-rho",
                "--model",$blob,
                "--rank",[string]$Job.rank,
                "--lora-rank",[string]$Job.lora_rank,
                "--out",[string]$Job.out_dir))
            return [PSCustomObject]@{ Exe = $PyExe; Args = $a.ToArray() }
        }

        "script" {
            $a = [System.Collections.Generic.List[string]]::new()
            if ($Job.args) { foreach ($x in $Job.args) { $a.Add([string]$x) } }
            return [PSCustomObject]@{ Exe = [string]$Job.file; Args = $a.ToArray() }
        }

        default { throw "Unknown mode: $mode  (job id=$id)" }
    }
}

#  Checkpoint I/O (only primitives --- no nested hashtable deserialisation) 
function Read-Checkpoint {
    param([string]$Path)
    $done  = @{}
    $fails = @{}
    if (-not (Test-Path $Path)) { return $done, $fails }
    $raw = Get-Content $Path -Raw | ConvertFrom-Json
    # completed is an array of strings
    foreach ($id in $raw.completed) {
        $done[[string]$id] = $true
    }
    # failed is a JSON object  { "id": N, ... }
    # PSObject.Properties is safe here because $raw.failed itself is a PSCustomObject,
    # not a string, so its properties ARE the key/value pairs we stored.
    foreach ($prop in $raw.failed.PSObject.Properties) {
        $fails[[string]$prop.Name] = [int]$prop.Value
    }
    return $done, $fails
}

function Write-Checkpoint {
    param([string]$Path, [hashtable]$Done, [hashtable]$Fails, [string]$RunId)
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null

    # Build a plain PSCustomObject so ConvertTo-Json is trivial
    $failObj = New-Object -TypeName PSObject
    foreach ($k in $Fails.Keys) {
        $failObj | Add-Member -NotePropertyName ([string]$k) -NotePropertyValue ([int]$Fails[$k])
    }

    $ckpt = [PSCustomObject]@{
        run_id    = $RunId
        saved_utc = (Get-Date).ToUniversalTime().ToString("o")
        completed = @($Done.Keys | ForEach-Object { [string]$_ })
        failed    = $failObj
    }
    $ckpt | ConvertTo-Json -Depth 4 | Set-Content -Path $Path -Encoding UTF8
}

#  Summary 
function Build-Summary {
    param([hashtable]$Done, [hashtable]$Fails, $Jobs, [string]$RunId)
    $total = ($Jobs | Measure-Object).Count
    $ok    = $Done.Count
    $hard  = 0
    foreach ($id in $Fails.Keys) { $hard++ }   # anything still in Fails = not done

    $byPaper = $Jobs | Group-Object { [string]$_.paper } | ForEach-Object {
        $pJobs = @($_.Group)
        $pOk   = ($pJobs | Where-Object { $Done[[string]$_.id] } | Measure-Object).Count
        [PSCustomObject]@{
            paper     = [string]$_.Name
            completed = $pOk
            total     = ($pJobs | Measure-Object).Count
        }
    }

    [PSCustomObject]@{
        run_id    = $RunId
        saved_utc = (Get-Date).ToUniversalTime().ToString("o")
        totals    = [PSCustomObject]@{
            jobs      = $total
            completed = $ok
            failed    = $hard
            pending   = ($total - $ok - $hard)
        }
        by_paper  = @($byPaper)
        done_ids  = @($Done.Keys | ForEach-Object { [string]$_ })
    }
}

#  Package outputs into a zip 
function Save-Package {
    param([hashtable]$Done, [hashtable]$Fails, [string]$RunId,
          [string]$RunDir, [string]$PkgDir, [string]$ZipPath)
    New-Item -ItemType Directory -Force -Path $PkgDir | Out-Null
    Copy-Item $ManifestPath (Join-Path $PkgDir "farm_manifest.json") -Force
    $ckptP = Join-Path $RunDir "state\checkpoint.json"
    $summP = Join-Path $RunDir "state\summary.json"
    if (Test-Path $ckptP) { Copy-Item $ckptP (Join-Path $PkgDir "checkpoint.json") -Force }
    if (Test-Path $summP) { Copy-Item $summP (Join-Path $PkgDir "summary.json") -Force }
    $logDir = Join-Path $RunDir "logs"
    if (Test-Path $logDir) {
        Copy-Item $logDir (Join-Path $PkgDir "logs") -Recurse -Force
    }
    foreach ($j in $manifest.jobs) {
        if ($j.artifacts) {
            foreach ($art in $j.artifacts) {
                $src = Join-Path $REPO ([string]$art)
                if (Test-Path $src) {
                    $rel = ([string]$art).TrimStart('/').TrimStart('\')
                    $dst = Join-Path $PkgDir "artifacts\$rel"
                    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
                    Copy-Item $src $dst -Force
                }
            }
        }
    }
    if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
    Compress-Archive -Path "$PkgDir\*" -DestinationPath $ZipPath -CompressionLevel Optimal
    Write-Host "[farm] package -> $ZipPath"
}

# 
# Main
# 
$manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$runId    = if ($manifest.run_id) { [string]$manifest.run_id } else {
                "farm_" + (Get-Date -Format "yyyyMMdd_HHmmss")
            }
$runDir   = Join-Path $OutRoot $runId
$ckptPath = Join-Path $runDir "state\checkpoint.json"
$summPath = Join-Path $runDir "state\summary.json"
$logDir   = Join-Path $runDir "logs"
$pkgDir   = Join-Path $runDir "package"
$zipPath  = Join-Path $runDir ("conversion_farm_" + $runId + ".zip")

if ($PackageOnly) {
    $done, $fails = Read-Checkpoint -Path $ckptPath
    $summary = Build-Summary -Done $done -Fails $fails -Jobs $manifest.jobs -RunId $runId
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $summPath) | Out-Null
    $summary | ConvertTo-Json -Depth 8 | Set-Content $summPath -Encoding UTF8
    Save-Package -Done $done -Fails $fails -RunId $runId `
        -RunDir $runDir -PkgDir $pkgDir -ZipPath $zipPath
    exit 0
}

# Fresh run: wipe any existing checkpoint for this run_id
if (-not $Resume) {
    if (Test-Path $ckptPath) { Remove-Item $ckptPath -Force }
}
$done, $fails = Read-Checkpoint -Path $ckptPath

$pyExe     = Resolve-PyExe
$processed = 0

Write-Host "[farm] run_id=$runId  manifest=$ManifestPath"
Write-Host "[farm] checkpoint=$ckptPath"
if ($Resume) { Write-Host "[farm] RESUME mode: $($done.Count) already done" }

foreach ($job in $manifest.jobs) {
    $id = [string]$job.id

    # Skip disabled
    if ($null -ne $job.enabled -and -not [bool]$job.enabled) {
        Write-Host "[farm] SKIP(disabled) $id"
        continue
    }
    # Skip already completed
    if ($done[$id]) {
        Write-Host "[farm] SKIP(done)     $id"
        continue
    }
    # MaxJobs cap
    if ($MaxJobs -gt 0 -and $processed -ge $MaxJobs) {
        Write-Host "[farm] MaxJobs=$MaxJobs reached, stopping"
        break
    }

    $retries    = if ($null -ne $job.retries)     { [int]$job.retries }     else { 1 }
    $timeout    = if ($null -ne $job.timeout_sec) { [int]$job.timeout_sec } else { 0 }
    $cooldown   = if ($null -ne $job.cooldown_sec){ [int]$job.cooldown_sec } else { 5 }
    $priorFails = if ($fails[$id]) { [int]$fails[$id] } else { 0 }

    Write-Host "`n[farm]  START $id  paper=$($job.paper)  mode=$($job.mode)  retries=$retries"

    $ok = $false
    $attempt = $priorFails
    while ($attempt -lt $retries -and -not $ok) {
        $attempt++
        $ts       = Get-Date -Format "yyyyMMdd_HHmmss"
        $stdoutF  = Join-Path $logDir "${id}_a${attempt}_${ts}.out.log"
        $stderrF  = Join-Path $logDir "${id}_a${attempt}_${ts}.err.log"

        try {
            $cmd     = Build-Cmd -Job $job -PyExe $pyExe -LogDir $logDir -Attempt $attempt
            $exe     = [string]$cmd.Exe
            $cmdArgs = [string[]]$cmd.Args
            Write-Host "[farm]   attempt $attempt/$retries"
            Write-Host "[farm]   exe: $exe"
            Write-Host "[farm]   args: $($cmdArgs -join ' ')"
            $rc = Invoke-Timed -Exe $exe -ProcArgs $cmdArgs -StdoutFile $stdoutF -StderrFile $stderrF -TimeoutSec $timeout
        } catch {
            Write-Host "[farm]   BUILD/LAUNCH ERROR: $($_.Exception.Message)" -ForegroundColor Yellow
            $rc = -99
        }

        if ($rc -eq 0) {
            $done[$id] = $true
            $fails.Remove($id)
            Write-Checkpoint -Path $ckptPath -Done $done -Fails $fails -RunId $runId
            Write-Host "[farm]   OK  $id  (exit 0)" -ForegroundColor Green
            $ok = $true
        } else {
            $fails[$id] = $attempt
            Write-Checkpoint -Path $ckptPath -Done $done -Fails $fails -RunId $runId
            Write-Host "[farm]   FAIL $id  exit=$rc  attempt=$attempt/$retries" -ForegroundColor Yellow
            if ($rc -eq 124) { Write-Host "[farm]   (timeout after ${timeout}s)" -ForegroundColor Yellow }
            if ($attempt -lt $retries -and $cooldown -gt 0) {
                Write-Host "[farm]   cooldown ${cooldown}s ..."
                Start-Sleep -Seconds $cooldown
            }
        }
    }

    if (-not $ok) {
        Write-Host "[farm] EXHAUSTED $id after $retries attempt(s)" -ForegroundColor Red
    }
    $processed++
}

# Final summary and package
$summary = Build-Summary -Done $done -Fails $fails -Jobs $manifest.jobs -RunId $runId
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $summPath) | Out-Null
$summary | ConvertTo-Json -Depth 8 | Set-Content $summPath -Encoding UTF8
Save-Package -Done $done -Fails $fails -RunId $runId -RunDir $runDir -PkgDir $pkgDir -ZipPath $zipPath

Write-Host "`n[farm] DONE  run_id=$runId  processed=$processed  completed=$($done.Count)  failed=$($fails.Count)"
Write-Host "[farm] summary   -> $summPath"
Write-Host "[farm] package   -> $zipPath"

#  Auto-generate benchmark report after every run 
$reportScript = Join-Path $REPO "scripts\gen_benchmark_report.py"
if (Test-Path $reportScript) {
    Write-Host "[farm] generating benchmark report ..."
    $rptOut  = Join-Path $REPO "docs\benchmark_report"
    $rptLog  = Join-Path $runDir "report_gen.log"
    $rptArgs = @($reportScript, "--farm-run-id", $runId, "--out-dir", $rptOut)
    try {
        & $pyExe @rptArgs 2>&1 | Tee-Object -FilePath $rptLog
        Write-Host "[farm] report  -> $rptOut\MASTER_REPORT.md"
    } catch {
        Write-Warning "[farm] report generation failed: $_"
    }
}
