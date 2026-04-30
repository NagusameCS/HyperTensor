<#
.SYNOPSIS
  Publish HyperTensor pre-computed caches as GitHub Releases.

.DESCRIPTION
  Reads docs/data/release_manifest.json and creates one GitHub Release per
  entry, attaching the matching ott_wproj_cache_*.bin asset. Idempotent:
  skips releases that already exist. Uses the gh CLI; you must have run
  `gh auth login` first.

  All cache files stay local; this script ONLY uploads to github.com.

.PARAMETER DryRun
  Print what would be done without contacting GitHub.

.PARAMETER Repo
  The owner/name of the GitHub repository. Default: NagusameCS/HyperTensor.

.EXAMPLE
  pwsh scripts/paperA_proof/publish_releases.ps1 -DryRun

.EXAMPLE
  pwsh scripts/paperA_proof/publish_releases.ps1
#>
[CmdletBinding()]
param(
  [switch]$DryRun,
  [string]$Repo = 'NagusameCS/HyperTensor'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$manifestPath = Join-Path $root 'docs\data\release_manifest.json'

if (-not (Test-Path $manifestPath)) {
  Write-Error "Manifest not found: $manifestPath"
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
  Write-Error 'gh CLI is required. Install from https://cli.github.com/ then run `gh auth login`.'
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

# Existing releases on the repo
$existingTags = @()
if (-not $DryRun) {
  $existingTags = (gh release list --repo $Repo --limit 200 --json tagName | ConvertFrom-Json).tagName
}

foreach ($r in $manifest.releases) {
  $tag    = $r.tag
  $title  = $r.title
  $asset  = Join-Path $root $r.asset_filename
  $sizeMB = [math]::Round((Get-Item $asset -ErrorAction SilentlyContinue).Length / 1MB, 2)

  Write-Host ""
  Write-Host "=== $tag ===" -ForegroundColor Cyan
  Write-Host "  asset: $($r.asset_filename) ($sizeMB MB)"
  Write-Host "  title: $title"

  if (-not (Test-Path $asset)) {
    Write-Warning "  asset not found locally, skipping"
    continue
  }
  if ($existingTags -contains $tag) {
    Write-Host "  release already exists, skipping" -ForegroundColor Yellow
    continue
  }

  $sha = (Get-FileHash $asset -Algorithm SHA256).Hash
  $notes = @"
HyperTensor pre-computed cache asset.

- **Base model**: $($r.base_model.name) ($($r.base_model.quantisation))
- **Rank k**: $($r.rank_k)
- **File size**: $sizeMB MB
- **SHA256**: ``$sha``
- **VRAM at decode**: $($r.vram_at_decode_mib) MiB ($($r.vram_delta_vs_baseline_mib) MiB over baseline)
- **Host RAM required**: $($r.host_ram_required_mib) MiB
- **Min compute capability**: $($r.min_compute_capability)
- **Calibration time skipped**: $($r.calibration_time_seconds_skipped) seconds

$($r.performance_notes)

## Use

``````powershell
gh release download $tag --repo $Repo --pattern 'ott_wproj_cache_*.bin'
.\build_host\geodessical.exe --model your-model.gguf ``
  --axex-compress --axex-attn-only --axex-weight-pca ``
  --axex-compress-rank $($r.rank_k) --tokens 64 --prompt "..."
``````

The runtime will detect the cache file in the working directory and load it on
launch, skipping calibration. See the
[reproduction guide](https://nagusamecs.github.io/HyperTensor/research/repro/paper-a.html)
for full instructions.
"@

  if ($DryRun) {
    Write-Host "  [dry-run] gh release create $tag $asset --repo $Repo --title `"$title`""
    continue
  }

  gh release create $tag $asset --repo $Repo --title $title --notes $notes
  if ($LASTEXITCODE -ne 0) { Write-Warning "  gh release create failed for $tag" }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
