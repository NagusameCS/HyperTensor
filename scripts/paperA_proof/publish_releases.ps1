<#
.SYNOPSIS
  Publish HyperTensor pre-computed W_proj caches as GitHub Releases.

.DESCRIPTION
  Reads docs/data/release_manifest.json, computes SHA256 of each asset,
  and creates one GitHub Release per entry via the gh CLI. Idempotent:
  releases that already exist are skipped (use -ForceUpload to re-attach
  an asset to an existing release).

  Caches stay local; only their bytes are uploaded to github.com.

.PARAMETER DryRun
  Print what would be done without contacting GitHub.

.PARAMETER Repo
  owner/repo on GitHub. Default: NagusameCS/HyperTensor.

.PARAMETER Only
  Optional comma-separated list of release tags to publish. All others
  are skipped. Useful for incremental uploads.

.PARAMETER ForceUpload
  If a release with the tag already exists, attempt to upload the asset
  anyway (gh release upload --clobber).

.EXAMPLE
  powershell -File scripts\paperA_proof\publish_releases.ps1 -DryRun

.EXAMPLE
  # Upload just the headline cache first
  powershell -File scripts\paperA_proof\publish_releases.ps1 -Only wproj-cache-2405A3B6
#>
[CmdletBinding()]
param(
  [switch]$DryRun,
  [string]$Repo = 'NagusameCS/HyperTensor',
  [string]$Only = '',
  [switch]$ForceUpload
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$manifestPath = Join-Path $root 'docs\data\release_manifest.json'

if (-not (Test-Path $manifestPath)) { Write-Error "Manifest not found: $manifestPath" }
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  Write-Error 'gh CLI required. Install from https://cli.github.com/ and run `gh auth login`.'
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
$onlyList = if ($Only) { $Only.Split(',') | ForEach-Object { $_.Trim() } } else { @() }

$existingTags = @()
if (-not $DryRun) {
  $existingTags = (gh release list --repo $Repo --limit 200 --json tagName | ConvertFrom-Json).tagName
}

$summary = @()
foreach ($r in $manifest.releases) {
  if ($onlyList.Count -gt 0 -and ($onlyList -notcontains $r.tag)) { continue }

  $asset = Join-Path $root $r.asset_filename
  Write-Host ""
  Write-Host "=== $($r.tag) ===" -ForegroundColor Cyan
  Write-Host "  asset: $($r.asset_filename)"

  if (-not (Test-Path $asset)) {
    Write-Warning "  asset not found locally; skipping"
    $summary += [pscustomobject]@{tag=$r.tag; status='missing-asset'}
    continue
  }

  $sizeMB = [math]::Round((Get-Item $asset).Length / 1MB, 2)
  Write-Host "  size: $sizeMB MB"

  $exists = $existingTags -contains $r.tag
  if ($exists -and -not $ForceUpload) {
    Write-Host "  release exists, skipping (use -ForceUpload to re-attach)" -ForegroundColor Yellow
    $summary += [pscustomobject]@{tag=$r.tag; status='exists-skipped'; size_mb=$sizeMB}
    continue
  }

  Write-Host "  hashing (this takes ~10-30 s for a 1 GB file) ..."
  $sha = (Get-FileHash $asset -Algorithm SHA256).Hash

  $model = $manifest.models.($r.model_id)
  $notes = @"
HyperTensor pre-computed shared-basis projection cache (GRC).

| field | value |
|---|---|
| Base model | $($model.huggingface_repo) |
| Quantisation | $($model.quantisation) |
| n_in (model dim) | $($model.n_in) |
| n_layers | $($model.n_layers) |
| Rank (layer 0) | $($r.rank_layer0) |
| File size | $sizeMB MB |
| SHA256 | ``$sha`` |

$($r.performance_notes)

## How to use

``````powershell
gh release download $($r.tag) --repo $Repo --pattern '*.bin'
Get-FileHash $($r.asset_filename) -Algorithm SHA256   # must match above

# Then run geodessical with the matching rank in the SAME working directory:
.\build_host\geodessical.exe ``
  --model models\$($model.huggingface_repo.Split('/')[-1])\<your.gguf> ``
  --axex-compress --axex-attn-only --axex-weight-pca ``
  --axex-compress-rank $($r.rank_layer0) ``
  --tokens 64 --prompt "Hello"
``````

The runtime auto-detects ``ott_wproj_cache_*.bin`` files in cwd and matches
them by their 8-byte ``cache_key`` (computed over model hash + rank +
slot config). If the key matches, calibration is skipped entirely.

## Format

Binary, little-endian. Magic ``0x3130564A4F525057`` (\"WPROJV01\"). Full
layout in ``runtime/nn/axiom_exploit.c`` (search for ``AXEX_WPROJ_MAGIC``).

## Caveats

- Tied to the exact base-model GGUF commit hash this cache was built
  from. Re-quantisations of the same model produce a different cache_key
  and will not match.
- Per-layer rank may differ from the layer-0 rank shown above; inspect
  the file with ``scripts/paperA_proof/dump_cache_header.ps1`` for the
  full rank distribution.
- Resource estimates assume a single-stream decode at batch 1 on
  $($model.huggingface_repo); larger batch sizes scale VRAM linearly.
"@

  if ($DryRun) {
    Write-Host "  [dry-run] would create release $($r.tag) (sha256=$($sha.Substring(0,16))...)"
    $summary += [pscustomobject]@{tag=$r.tag; status='dry-run'; size_mb=$sizeMB; sha256=$sha}
    continue
  }

  if ($exists -and $ForceUpload) {
    Write-Host "  re-uploading asset to existing release $($r.tag) ..." -ForegroundColor Yellow
    gh release upload $r.tag $asset --repo $Repo --clobber
    $code = $LASTEXITCODE
  } else {
    Write-Host "  creating release $($r.tag) ..." -ForegroundColor Green
    $notesFile = New-TemporaryFile
    $notes | Set-Content -Path $notesFile -Encoding UTF8
    try {
      gh release create $r.tag $asset --repo $Repo --title $r.title --notes-file $notesFile
      $code = $LASTEXITCODE
    } finally {
      Remove-Item $notesFile -ErrorAction SilentlyContinue
    }
  }

  if ($code -eq 0) {
    Write-Host "  OK" -ForegroundColor Green
    $summary += [pscustomobject]@{tag=$r.tag; status='uploaded'; size_mb=$sizeMB; sha256=$sha}
  } else {
    Write-Warning "  gh failed for $($r.tag) (exit $code)"
    $summary += [pscustomobject]@{tag=$r.tag; status="failed-$code"; size_mb=$sizeMB; sha256=$sha}
  }
}

Write-Host ""
Write-Host "=== SUMMARY ===" -ForegroundColor Cyan
$summary | Format-Table -AutoSize
$total = ($summary | Where-Object { $_.size_mb } | Measure-Object size_mb -Sum).Sum
Write-Host ("Total bytes processed: {0:N1} MB" -f $total)
