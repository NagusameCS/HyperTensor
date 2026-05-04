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

# benchmark_ec2.ps1
# Full EC2 benchmark: BASELINE  FLAT-SVD  GRC across 4 models, 3 prompts each
# Saves all raw output for manual coherence checking.
# Appends structured results to benchmark_results.md.
#
# Usage:
#   .\benchmark_ec2.ps1
#   .\benchmark_ec2.ps1 -Models SmolLM2,Phi   (subset)
#   .\benchmark_ec2.ps1 -DryRun               (print commands only)

param(
    [string[]]$Models   = @("SmolLM2", "Phi", "Gemma", "Llama70B"),
    [string]  $OutDir   = "benchmarks\ec2_runs",
    [string]  $ResultsMd = "benchmark_results.md",
    [switch]  $DryRun
)

$ErrorActionPreference = "Continue"
$exe = ".\build_host\geodessical.exe"

#  Model definitions 
$modelDefs = [ordered]@{
    "SmolLM2"  = [PSCustomObject]@{
        label     = "SmolLM2-135M"
        path      = "C:\Users\legom\TensorOS\models\smollm2-135m-instruct-q8_0.gguf"
        n_tokens  = 40
        calib     = 64
        skip_grc  = $false
    }
    "Phi"      = [PSCustomObject]@{
        label     = "Phi-3.5-mini"
        path      = "C:\Users\legom\TensorOS\models\Phi-3.5-mini-instruct-Q4_0.gguf"
        n_tokens  = 40
        calib     = 32
        skip_grc  = $false
    }
    "Gemma"    = [PSCustomObject]@{
        label     = "Gemma-4-E2B"
        path      = "C:\Users\legom\TensorOS\models\google_gemma-4-E2B-it-Q4_0.gguf"
        n_tokens  = 40
        calib     = 32
        skip_grc  = $false
    }
    "Llama70B" = [PSCustomObject]@{
        label     = "Llama3.1-70B"
        path      = "C:\Users\legom\TensorOS\models\llama31-70b-iq2xs.gguf"
        n_tokens  = 10
        calib     = 0
        skip_grc  = $true  # calibration infeasible with CPU-offloaded layers
    }
}

#  Method definitions 
# FLAT-SVD-FULL on 70B uses --axex-attn-svd too so the whole model fits on GPU
$methodDefs = [ordered]@{
    "BASELINE"      = @()
    "FLAT-SVD"      = @("--axex-ffn-compress", "--axex-compress-rank", "128")
    "FLAT-SVD-FULL" = @("--axex-ffn-compress", "--axex-attn-svd", "--axex-compress-rank", "64")
    "GRC"           = @("--axex-compress", "--axiom-skip-geodesic")
}

# Which methods to run per model key
# NOTE: GRC is only viable for small models (SmolLM2) --- for larger models the
# GRC is now viable for Phi too --- axpca_compute_topk uses Gram-matrix path (nn) when
# n_samples < ff_dim, so even ff_dim=8192 calibration with n=32 samples runs a 3232
# Jacobi in microseconds.  The prior O(d²) concern predates the Gram-matrix fast path.
$modelMethods = @{
    "SmolLM2"  = @("BASELINE", "FLAT-SVD", "GRC")
    "Phi"      = @("BASELINE", "FLAT-SVD", "GRC")
    "Gemma"    = @("BASELINE", "FLAT-SVD")
    "Llama70B" = @("BASELINE", "FLAT-SVD-FULL")
}

#  Prompts 
$prompts = @(
    "The capital of France is"
    "What is the speed of light?"
    "Once upon a time there was a"
)

#  Output directory 
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

#  Parsing helper 
function Parse-Run {
    param([string[]]$lines)

    $r = @{
        decode_toks  = $null
        prefill_toks = $null
        vram_mb      = $null
        ffn_compr_pct = $null   # mean compression % from [AXEX-COMPRESS] summary
        ffn_lw_pct    = $null   # % from [AXEX-FFN-DOWN] Total
        attn_matrices = $null   # count from [AXEX-MANIFOLD] Uploaded N W_proj
        output_text   = ""
        error         = $null
    }

    $inGen   = $false
    $textBuf = [System.Collections.Generic.List[string]]::new()

    foreach ($line in $lines) {
        # VRAM
        if ($line -match 'GPU-resident forward pass ready \(total VRAM: (\d+) MB\)') {
            $r.vram_mb = [int]$Matches[1]
        }
        # decode tok/s  e.g. "[GD] 40 tokens in 823 ms (48.6 tok/s)"
        if ($line -match '\[GD\]\s+\d+ tokens in .+\((\d+\.?\d*) tok/s\)') {
            $r.decode_toks = [float]$Matches[1]
        }
        # prefill tok/s  e.g. "[GD] Decode-only: prefill 312.0 ms, 166.0 tok/s"
        if ($line -match 'Decode-only: prefill .+,\s+(\d+\.?\d*) tok/s') {
            $r.prefill_toks = [float]$Matches[1]
        }
        # FLAT-SVD compression summary: "[AXEX-COMPRESS] 240 matrices compressed | mean size 12.3% of original"
        if ($line -match '\[AXEX-COMPRESS\]\s+(\d+) matrices compressed \| mean size (\d+\.?\d*)% of original') {
            $r.ffn_compr_pct = [float]$Matches[2]
        }
        # FFN-DOWN layerwise: "[AXEX-FFN-DOWN] Total: 30 layers | 45 MB -> 20 MB (55.6% reduction)"
        if ($line -match '\[AXEX-FFN-DOWN\] Total:.*\((\d+\.?\d*)% reduction\)') {
            $r.ffn_lw_pct = [float]$Matches[1]
        }
        # Attn manifold: "[AXEX-MANIFOLD] Uploaded 0 W_proj matrices to GPU"
        if ($line -match '\[AXEX-MANIFOLD\] Uploaded (\d+) W_proj matrices') {
            $r.attn_matrices = [int]$Matches[1]
        }
        # Errors / OOM
        if ($line -match '(OOM|out of memory|Error:|failed|FATAL)') {
            $r.error = $line.Trim()
        }
        # Generated text: capture between "Generating N tokens..." and "[GD] N tokens in"
        if ($line -match '\[GD\] Generating \d+ tokens') { $inGen = $true; continue }
        if ($inGen) {
            if ($line -match '^\[GD\]\s+\d+ tokens in') { $inGen = $false }
            elseif ($line.Trim() -ne "" -and -not ($line.TrimStart().StartsWith("["))) {
                $textBuf.Add($line.Trim())
            }
        }
    }
    $r.output_text = ($textBuf -join " ").Trim()
    return $r
}

#  Formatting helpers 
function Fmt-Toks($v) { if ($null -eq $v) { return "---" }; return "$v" }
function Fmt-VRAM($v) { if ($null -eq $v) { return "---" }; return "$v" }

function Compress-Label($parsed, $method) {
    # Returns a short compression description for the table
    if ($method -eq "BASELINE") { return "0%" }
    if ($method -like "FLAT-SVD*") {
        if ($null -ne $parsed.ffn_compr_pct) {
            $kept = [math]::Round($parsed.ffn_compr_pct, 1)
            $red  = [math]::Round(100.0 - $kept, 1)
            return "${red}% FFN"
        }
        return "SVD (parsing n/a)"
    }
    if ($method -eq "GRC") {
        $parts = @()
        if ($null -ne $parsed.ffn_lw_pct -and $parsed.ffn_lw_pct -gt 0) {
            $parts += "$($parsed.ffn_lw_pct)% FFN-GP"
        }
        if ($null -ne $parsed.attn_matrices) {
            $parts += "$($parsed.attn_matrices) attn matrices"
        }
        if ($parts.Count -eq 0) { return "0% (guards)" }
        return $parts -join ", "
    }
    return "---"
}

#  Run a single trial 
function Run-Trial {
    param(
        [string]   $modelKey,
        [PSObject] $mDef,
        [string]   $method,
        [string[]] $mFlags,
        [string]   $prompt,
        [int]      $promptIdx,
        [string]   $outFile
    )
    $allArgs = @($mDef.path) + $mFlags + @(
        "--axex-calib-samples", "$($mDef.calib)",
        "-p", $prompt,
        "-n", "$($mDef.n_tokens)",
        "--temp", "0"
    )
    # Don't pass --axex-calib-samples 0 (baseline/flat-svd don't need it)
    if ($mDef.calib -eq 0 -or $method -notlike "GRC*") {
        $allArgs = @($mDef.path) + $mFlags + @(
            "-p", $prompt,
            "-n", "$($mDef.n_tokens)",
            "--temp", "0"
        )
    }

    $cmd = "$exe " + (($allArgs | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join " ")
    Write-Host "  CMD: $cmd" -ForegroundColor DarkGray

    if ($DryRun) { return $null }

    $t0 = Get-Date
    & $exe @allArgs > $outFile 2>&1
    $elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

    $lines  = Get-Content $outFile -ErrorAction SilentlyContinue
    $parsed = Parse-Run $lines
    $parsed.elapsed = $elapsed
    return $parsed
}

#  Main loop 
$runDate  = Get-Date -Format "yyyy-MM-dd HH:mm"
$allRows  = [System.Collections.Generic.List[PSObject]]::new()

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host " EC2 BENCHMARK  ---  $runDate" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

foreach ($modelKey in $Models) {
    if (-not $modelDefs.Contains($modelKey)) {
        Write-Warning "Unknown model key: $modelKey (valid: $($modelDefs.Keys -join ', '))"
        continue
    }
    $mDef    = $modelDefs[$modelKey]
    $methods = $modelMethods[$modelKey]

    Write-Host " $($mDef.label) " -ForegroundColor Yellow

    foreach ($method in $methods) {
        if (-not $methodDefs.Contains($method)) { continue }
        $mFlags = $methodDefs[$method]

        Write-Host "`n  [$method]" -ForegroundColor Magenta

        $trials = [System.Collections.Generic.List[PSObject]]::new()

        for ($pi = 0; $pi -lt $prompts.Count; $pi++) {
            $prompt  = $prompts[$pi]
            $safeTag = "$($modelKey)__${method}__p${pi}" -replace '[^A-Za-z0-9_\-]', '_'
            $outFile = Join-Path $OutDir "$safeTag.txt"

            Write-Host "    Trial $($pi+1)/3: `"$prompt`"" -NoNewline -ForegroundColor Cyan

            $p = Run-Trial -modelKey $modelKey -mDef $mDef -method $method `
                           -mFlags $mFlags -prompt $prompt -promptIdx $pi `
                           -outFile $outFile

            if ($DryRun) { Write-Host " [dry-run]"; continue }

            $decStr = if ($p.decode_toks)  { "$($p.decode_toks) tok/s" } else { "n/a" }
            $preStr = if ($p.prefill_toks) { "$($p.prefill_toks) tok/s prefill" } else { "" }
            $txtStr = if ($p.output_text)  { "`"$($p.output_text.Substring(0, [Math]::Min(60,$p.output_text.Length)))`"" } else { "(no text)" }
            $errStr = if ($p.error)        { " [ERR: $($p.error.Substring(0,[Math]::Min(40,$p.error.Length)))]" } else { "" }

            Write-Host " -> $decStr  $preStr  $($p.elapsed)s$errStr" -ForegroundColor Green
            Write-Host "      output: $txtStr" -ForegroundColor Gray

            $p.prompt      = $prompt
            $p.prompt_idx  = $pi
            $p.method      = $method
            $p.model_label = $mDef.label
            $p.raw_file    = $outFile
            $trials.Add($p)
        }

        if ($DryRun) { continue }
        if ($trials.Count -eq 0) { continue }

        # Average over trials
        $decVals = $trials | Where-Object { $null -ne $_.decode_toks  } | ForEach-Object { $_.decode_toks }
        $preVals = $trials | Where-Object { $null -ne $_.prefill_toks } | ForEach-Object { $_.prefill_toks }
        $avgDec  = if ($decVals) { [math]::Round(($decVals | Measure-Object -Average).Average, 1) } else { $null }
        $avgPre  = if ($preVals) { [math]::Round(($preVals | Measure-Object -Average).Average, 1) } else { $null }
        $vram    = ($trials | Where-Object { $null -ne $_.vram_mb } | Select-Object -First 1).vram_mb
        $comprLabel = Compress-Label $trials[-1] $method

        Write-Host "    AVG: decode=$(Fmt-Toks $avgDec)  prefill=$(Fmt-Toks $avgPre)  VRAM=$(Fmt-VRAM $vram)" -ForegroundColor White

        $allRows.Add([PSCustomObject]@{
            model      = $mDef.label
            method     = $method
            avg_decode = $avgDec
            avg_prefill= $avgPre
            vram       = $vram
            compr      = $comprLabel
            trials     = $trials
        })
    }
    Write-Host ""
}

if ($DryRun) {
    Write-Host "`n[DRY-RUN] No files written." -ForegroundColor Yellow
    exit 0
}

#  Build markdown section 
$md = [System.Collections.Generic.List[string]]::new()
$md.Add("")
$md.Add("---")
$md.Add("")
$md.Add("## EC2 Full Benchmark --- $runDate")
$md.Add("")
$md.Add("geodessical · RTX 3070 8GB (40 TFLOPS) · `--temp 0` · 3 prompts averaged · raw outputs below")
$md.Add("")

# Summary table
$md.Add("### Performance Summary")
$md.Add("")
$md.Add("| Model | Method | Decode avg (tok/s) | Prefill avg (tok/s) | VRAM (MB) | Compression |")
$md.Add("|---|---|---:|---:|---:|---|")
foreach ($r in $allRows) {
    $dec = Fmt-Toks $r.avg_decode
    $pre = Fmt-Toks $r.avg_prefill
    $vr  = Fmt-VRAM $r.vram
    $md.Add("| $($r.model) | $($r.method) | $dec | $pre | $vr | $($r.compr) |")
}
$md.Add("")

# Per-trial decode table (all individual readings)
$md.Add("### Per-Trial Decode Speeds")
$md.Add("")
$md.Add("| Model | Method | T1 (tok/s) | T2 (tok/s) | T3 (tok/s) |")
$md.Add("|---|---|---:|---:|---:|")
foreach ($r in $allRows) {
    $t = $r.trials
    $v = @(0,1,2) | ForEach-Object { if ($_ -lt $t.Count -and $null -ne $t[$_].decode_toks) { $t[$_].decode_toks } else { "---" } }
    $md.Add("| $($r.model) | $($r.method) | $($v[0]) | $($v[1]) | $($v[2]) |")
}
$md.Add("")

# Generated output log
$md.Add("### Generated Output Log")
$md.Add("")
$md.Add("All outputs captured at `--temp 0` for deterministic coherence checking.")
$md.Add("")

foreach ($r in $allRows) {
    $md.Add("#### $($r.model) --- $($r.method)")
    $md.Add("")
    foreach ($t in $r.trials) {
        $output = if ($t.output_text) { $t.output_text } else { "*(no output captured)*" }
        $errNote = if ($t.error) { " WARN: $($t.error)" } else { "" }
        $md.Add("**Prompt**: $($t.prompt)$errNote")
        $md.Add("")
        $md.Add("> $output")
        $md.Add("")
        $dec = if ($t.decode_toks) { "$($t.decode_toks) tok/s decode" } else { "n/a" }
        $pre = if ($t.prefill_toks) { "$($t.prefill_toks) tok/s prefill" } else { "" }
        $md.Add("_${dec} · ${pre} · raw: $(Split-Path $t.raw_file -Leaf)_")
        $md.Add("")
    }
}

# Write to file
Add-Content -Path $ResultsMd -Value ($md -join "`n") -Encoding UTF8

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " DONE  ---  $($allRows.Count) configurations run" -ForegroundColor Cyan
Write-Host " Results appended to: $ResultsMd" -ForegroundColor Cyan
Write-Host " Raw files in:        $OutDir" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan
