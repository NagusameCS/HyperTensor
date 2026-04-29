# wait_and_run_phase3.ps1 — polls BITS transfer, then runs Phase 3 benchmark
param(
    [string]$Model = "C:\Users\legom\models\mistral-7b-v0.1\mistral-7b-v0.1.Q4_K_M.gguf",
    [string]$ModelTag = "mistral7b_v0.1",
    [int]$PollSec = 30
)

Set-Location $PSScriptRoot

Write-Host "[wait_and_run_phase3] Watching BITS transfer 'Mistral7B-Q4KM'..."
while ($true) {
    $job = Get-BitsTransfer | Where-Object { $_.DisplayName -eq "Mistral7B-Q4KM" } | Select-Object -First 1
    if (-not $job) {
        Write-Host "[wait_and_run_phase3] BITS job not found — checking if file exists..."
        break
    }
    $pct = if ($job.BytesTotal -gt 0) { [math]::Round(100 * $job.BytesTransferred / $job.BytesTotal, 1) } else { 0 }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] State=$($job.JobState)  $([math]::Round($job.BytesTransferred/1MB,0)) MB / $([math]::Round($job.BytesTotal/1MB,0)) MB  ($pct%)"

    if ($job.JobState -eq "Transferred") {
        Write-Host "[wait_and_run_phase3] Transfer complete — finalizing..."
        Complete-BitsTransfer -BitsJob $job
        break
    }
    if ($job.JobState -eq "Error" -or $job.JobState -eq "TransientError") {
        Write-Host "[wait_and_run_phase3] BITS job in error state: $($job.JobState). Aborting."
        exit 1
    }
    Start-Sleep -Seconds $PollSec
}

# Verify file exists
if (-not (Test-Path $Model)) {
    Write-Host "[wait_and_run_phase3] ERROR: Model file not found at $Model"
    exit 1
}

$sizeMB = [math]::Round((Get-Item $Model).Length / 1MB, 1)
Write-Host "[wait_and_run_phase3] Model ready: $sizeMB MB — launching Phase 3 benchmark..."

& "$PSScriptRoot\scripts\phase3_transfer.ps1" `
    -Model $Model `
    -ModelTag $ModelTag `
    -CooldownSec 30
