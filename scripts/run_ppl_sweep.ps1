# run_ppl_sweep.ps1
# ==================
# Measure WikiText-2 perplexity at GRC ranks {1024, 1536, 2048} on
# Llama-3.1-8B-Instruct-Q4_K_M and the matching baseline.
#
# Usage:
#   .\scripts\run_ppl_sweep.ps1 [-Reps 1] [-CooldownSec 30]
#
# Output: docs/figures/ppl_sweep/<timestamp>_<rank>.{stdout,stderr,json}.txt
#         and a summary CSV at docs/figures/ppl_sweep/summary.csv

[CmdletBinding()]
param(
  [int]$Reps = 1,
  [int]$CooldownSec = 30,
  [string]$Model = 'C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF\snapshots\bf5b95e96dac0462e2a09145ec66cae9a3f12067\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'
)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot '..\build_host\geodessical.exe' | Resolve-Path
$outDir = Join-Path $PSScriptRoot '..\docs\figures\ppl_sweep' | Resolve-Path -ErrorAction SilentlyContinue
if (-not $outDir) {
  New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot '..\docs\figures\ppl_sweep') | Out-Null
  $outDir = Join-Path $PSScriptRoot '..\docs\figures\ppl_sweep' | Resolve-Path
}

if (-not (Test-Path $Model)) { throw "Model not found: $Model" }
$modelMB = [math]::Round((Get-Item $Model).Length / 1MB, 1)
Write-Host "[ppl-sweep] model = $Model ($modelMB MB)"
Write-Host "[ppl-sweep] exe   = $exe"

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$summaryPath = Join-Path $outDir 'summary.csv'
if (-not (Test-Path $summaryPath)) {
  'timestamp,case,rank,rep,wall_s,exit_code,ppl,decode_ratio,stdout_path' | Out-File -FilePath $summaryPath -Encoding UTF8
}

function Invoke-One {
  param(
    [string]$Label,
    [string[]]$ExtraArgs,
    [int]$Rep
  )
  $stem = "${ts}_${Label}_rep$Rep"
  $oo = Join-Path $outDir "$stem.stdout.txt"
  $oe = Join-Path $outDir "$stem.stderr.txt"

  Write-Host ""
  Write-Host "[ppl-sweep] -> $Label  rep=$Rep"
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $proc = Start-Process -FilePath $exe -ArgumentList (@($Model) + $ExtraArgs) `
    -NoNewWindow -PassThru -Wait `
    -RedirectStandardOutput $oo -RedirectStandardError $oe
  $sw.Stop()
  $wall = [math]::Round($sw.Elapsed.TotalSeconds, 1)

  $stdout = Get-Content -Raw $oo -ErrorAction SilentlyContinue
  $ppl = $null
  if ($stdout) {
    $m = [regex]::Match($stdout, 'perplexity\D+(\d+\.\d+)', 'IgnoreCase')
    if ($m.Success) { $ppl = [double]$m.Groups[1].Value }
  }
  Write-Host "[ppl-sweep]    wall=${wall}s  exit=$($proc.ExitCode)  ppl=$ppl"
  $row = "$ts,$Label,$($ExtraArgs -join ' '),$Rep,$wall,$($proc.ExitCode),$ppl,,$oo"
  Add-Content -Path $summaryPath -Value $row

  if ($CooldownSec -gt 0) {
    Write-Host "[ppl-sweep]    cooldown ${CooldownSec}s..."
    Start-Sleep -Seconds $CooldownSec
  }
}

# Cases
$cases = @(
  @{ Label = 'baseline';  Args = @('--ppl-eval') },
  @{ Label = 'grc_k1024'; Args = @('--axex-compress','--axex-attn-only','--axex-skip-o','--axex-weight-pca','--axex-compress-rank','1024','--ppl-eval') },
  @{ Label = 'grc_k1536'; Args = @('--axex-compress','--axex-attn-only','--axex-skip-o','--axex-weight-pca','--axex-compress-rank','1536','--ppl-eval') },
  @{ Label = 'grc_k2048'; Args = @('--axex-compress','--axex-attn-only','--axex-skip-o','--axex-weight-pca','--axex-compress-rank','2048','--ppl-eval') }
)

for ($r = 1; $r -le $Reps; $r++) {
  foreach ($c in $cases) {
    Invoke-One -Label $c.Label -ExtraArgs $c.Args -Rep $r
  }
}

Write-Host ""
Write-Host "[ppl-sweep] DONE. Summary: $summaryPath"
Get-Content $summaryPath | Select-Object -Last ($cases.Count * $Reps + 1)
