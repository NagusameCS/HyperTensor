# enable_70b_jobs.ps1
# Run this once `ollama pull llama3.1:70b` completes.
# Enables the four llama3.1:70b jobs in farm_manifest.json and re-runs
# the farm with -Resume so only the new 70B jobs execute.
[CmdletBinding()]
param(
    [string]$ManifestPath = "C:\Users\legom\HyperTensor\scripts\conversion_farm\farm_manifest.json"
)
$ErrorActionPreference = "Stop"

# Verify model is actually present
$modelsRoot = "$env:USERPROFILE\.ollama\models"
$mfPath = Join-Path $modelsRoot "manifests\registry.ollama.ai\library\llama3.1\70b"
if (-not (Test-Path $mfPath)) {
    throw "llama3.1:70b manifest not found at $mfPath — is the pull complete?"
}
Write-Host "[enable_70b] llama3.1:70b manifest confirmed."

# Patch manifest: flip enabled: false -> true for all llama31_70b jobs
$content = Get-Content $ManifestPath -Raw
$ids = @("A_kint_llama31_70b","B_loadonly_llama31_70b","C_decode_llama31_70b_r128","E_rho_llama31_70b")
$patched = $content
foreach ($id in $ids) {
    # Match the block for this job id and flip enabled from false to true
    $patched = $patched -replace (
        '("id":\s*"' + [regex]::Escape($id) + '"[^}]*?"enabled":\s*)false'
    ), '${1}true'
}

if ($patched -eq $content) {
    Write-Warning "No 'enabled: false' entries found for 70B jobs — may already be enabled or pattern mismatch."
} else {
    Set-Content $ManifestPath -Value $patched -Encoding UTF8 -NoNewline
    Write-Host "[enable_70b] Enabled 70B jobs in $ManifestPath"
}

# Remove 70B jobs from failed checkpoint so they get fresh retries
$ckpt = "C:\Users\legom\HyperTensor\benchmarks\conversion_farm\grc_farm_spectrum_v1\state\checkpoint.json"
if (Test-Path $ckpt) {
    $raw = Get-Content $ckpt -Raw | ConvertFrom-Json
    # Remove 70B IDs from failed object
    $failedNew = New-Object -TypeName PSObject
    foreach ($prop in $raw.failed.PSObject.Properties) {
        if ($prop.Name -notmatch "llama31_70b") {
            $failedNew | Add-Member -NotePropertyName $prop.Name -NotePropertyValue $prop.Value
        }
    }
    $raw.failed = $failedNew
    $raw | ConvertTo-Json -Depth 4 | Set-Content $ckpt -Encoding UTF8
    Write-Host "[enable_70b] Cleared 70B entries from checkpoint."
}

Write-Host ""
Write-Host "[enable_70b] Ready. Run the farm with -Resume to execute 70B jobs:"
Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File scripts/conversion_farm/run_conversion_farm.ps1 -Resume"
