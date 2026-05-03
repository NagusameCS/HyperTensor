# Fetch fresh remote logs, run the analyser, optionally commit.
[CmdletBinding()]
param(
    [string]$Root = "benchmarks\cross_hw_local_fix_20260428_192807",
    [switch]$Commit
)

$ErrorActionPreference = 'Stop'

$dst = Join-Path $Root 'session_logs\remote'
New-Item -ItemType Directory -Force -Path $dst | Out-Null

Write-Host '[1/3] Pulling remote logs...'
scp -o ConnectTimeout=20 `
    "ssh.opencs.dev:/root/HyperTensor/grc_8b_k256.log" `
    "ssh.opencs.dev:/root/HyperTensor/wproj_cache_8b_k1024.log" `
    "ssh.opencs.dev:/tmp/smollm_run.log" `
    $dst 2>&1 | Out-Null

Write-Host '[2/3] Running analyser...'
& "$PSScriptRoot\analyse_cross_device.ps1" -Root $Root

if ($Commit) {
    Write-Host '[3/3] Committing...'
    git add -f "$Root\session_logs\remote\*.log" "$Root\cross_device_results.*" "$Root\cross_device_summary.md"
    git commit -m "Cross-device: refresh 8B k=256 logs and analysis"
    git push origin main
} else {
    Write-Host '[3/3] Skipping commit (use -Commit to push).'
}
