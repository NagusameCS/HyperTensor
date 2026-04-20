# benchmark_flags.ps1 — Flag configuration sweep for Geodessical
# Tests: CPU vs GPU, thread counts, OTT modes, speculative decode variants

param(
    [string]$Model = "smollm",   # smollm | phi | gemma
    [int]$N = 100
)

$GEO  = "$PSScriptRoot\build_host\geodessical.exe"
$CPUB = "$PSScriptRoot\build_host\geodessical_cpu.exe"

$GGUF = switch ($Model) {
    "phi"   { "C:\Users\legom\TensorOS\models\Phi-3.5-mini-instruct-Q4_0.gguf" }
    "gemma" { "C:\Users\legom\TensorOS\models\google_gemma-4-E2B-it-Q4_0.gguf" }
    default { "C:\Users\legom\TensorOS\models\smollm2-135m-instruct-q8_0.gguf" }
}

$P = "Explain how neural networks learn using backpropagation and gradient descent."

Write-Host ""
Write-Host "  Geodessical Flag Benchmark" -ForegroundColor Cyan
Write-Host "  Model: $Model ($GGUF)" -ForegroundColor DarkCyan
Write-Host "  N=$N tokens" -ForegroundColor DarkCyan
Write-Host ""

$rows = [System.Collections.Generic.List[PSCustomObject]]::new()

function Run-Config($bin, $label, $extraArgs) {
    Write-Host "  Running: $label ..." -NoNewline
    $allArgs = @($GGUF, "-p", $P, "-n", $N) + $extraArgs
    $t0 = [datetime]::UtcNow
    $out = (& $bin @allArgs 2>&1) -join "`n"
    $elapsed = [math]::Round(([datetime]::UtcNow - $t0).TotalSeconds, 1)

    $dec  = [regex]::Match($out, 'Decode-only.*?([\d.]+) tok/s').Groups[1].Value
    $pre  = [regex]::Match($out, 'prefill ([\d.]+) ms, ([\d.]+) tok/s').Groups[2].Value
    $ttft = [regex]::Match($out, 'prefill ([\d.]+) ms').Groups[1].Value
    $vram = [regex]::Match($out, 'total VRAM: (\d+) MB').Groups[1].Value
    $hbm  = [regex]::Match($out, '([\d.]+)% of 336 GB/s').Groups[1].Value
    $toks = [regex]::Match($out, '\[GD\] (\d+) tokens in').Groups[1].Value
    $backend = if ($out -match 'Backend: cuda') { 'GPU' } elseif ($out -match 'Backend: cpu') { 'CPU' } else { '?' }

    if (-not $dec) { $dec = "—" }
    if (-not $ttft) { $ttft = "—" }
    if (-not $vram) { $vram = "—" }
    if (-not $hbm)  { $hbm  = "—" }

    Write-Host (" dec={0,7} t/s  ttft={1,6}ms  vram={2,6}MB  HBM={3,5}%  [{4}s]" -f $dec,$ttft,$vram,$hbm,$elapsed)

    $rows.Add([PSCustomObject]@{
        Config  = $label
        Backend = $backend
        Dec     = $dec
        PreFill = $pre
        TTFT    = $ttft
        VRAM    = $vram
        HBM     = $hbm
        Elapsed = $elapsed
    })
}

# ── CPU: thread scaling (smollm only — larger models hang on multi-thread CPU) ──
if ($Model -eq "smollm") {
    Write-Host "── CPU: Thread Scaling ──" -ForegroundColor Yellow
    Run-Config $CPUB "CPU / 1 thread"   @("-t", "1")
    Run-Config $CPUB "CPU / 2 threads"  @("-t", "2")
    Write-Host ""
}

# ── GPU: baseline and mode flags ─────────────────────────────────────────────
Write-Host "── GPU: Decode Modes ──" -ForegroundColor Yellow
Run-Config $GEO  "GPU / baseline"       @()
Run-Config $GEO  "GPU / attnres"        @("--attnres")
Run-Config $GEO  "GPU / attnres 0.7"    @("--attnres", "--attnres-strength", "0.7")
Run-Config $GEO  "GPU / ott-fast"       @("--ott-fast")
Run-Config $GEO  "GPU / no-verifier"    @("--no-verifier")
Run-Config $GEO  "GPU / ott-spec"       @("--ott-speculative")
Run-Config $GEO  "GPU / ott-spec b=6"   @("--ott-speculative", "--ott-spec-batch", "6")
Run-Config $GEO  "GPU / ott-od"         @("--ott-od")
Run-Config $GEO  "GPU / one-decode"     @("--one-decode")
Run-Config $GEO  "GPU / ott-full"       @("--ott-full")

Write-Host ""
Write-Host "── Summary ──" -ForegroundColor Cyan
$rows | Format-Table -AutoSize

# Write CSV
$csvPath = "$PSScriptRoot\benchmark_flags_results.csv"
$rows | Export-Csv -Path $csvPath -NoTypeInformation
Write-Host "Results saved: $csvPath"
