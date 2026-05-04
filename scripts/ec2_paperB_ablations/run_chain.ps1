# Serial cross-HW chain: g6e.xlarge (L40S) -> g5.xlarge (A10G)
# Waits for vCPU quota to free between runs.
# Idempotent: re-running picks up wherever the previous run left off
# (each launch_detached.ps1 invocation is self-contained).
#
# Background-friendly: writes per-run logs to:
#   benchmarks\paperB_chain_<ts>\g6e.xlarge.log
#   benchmarks\paperB_chain_<ts>\g5.xlarge.log
[CmdletBinding()]
param(
    [string[]]$InstanceTypes = @('g6e.xlarge','g5.xlarge'),
    [int]$MaxRuntimeMinutes  = 120,
    [int]$WaitBetweenSeconds = 60
)

$ErrorActionPreference = "Continue"
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repo
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$chainDir = Join-Path $repo "benchmarks\paperB_chain_$ts"
New-Item -ItemType Directory -Force -Path $chainDir | Out-Null
$chainLog = Join-Path $chainDir "chain.log"
function ChainLog($msg) { $line = "[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $msg; $line | Tee-Object -FilePath $chainLog -Append; Write-Host $line }
ChainLog "Chain started for: $($InstanceTypes -join ', ')  -> dir=$chainDir"

function Wait-NoActiveGInstances {
    param([int]$TimeoutSec = 1200)
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        $active = aws ec2 describe-instances --region us-east-1 `
            --filters "Name=instance-state-name,Values=pending,running,stopping,shutting-down" `
            --query 'Reservations[].Instances[?starts_with(InstanceType,`g`)].[InstanceId,InstanceType,State.Name]' `
            --output text 2>$null
        if (-not $active -or $active.Trim() -eq '') {
            ChainLog "  no active G-family instances --- quota free"
            return $true
        }
        ChainLog "  active G-instances: $($active -replace '\s+',' ') --- waiting"
        Start-Sleep -Seconds 30
    }
    return $false
}

foreach ($it in $InstanceTypes) {
    ChainLog "===== $it ====="
    if (-not (Wait-NoActiveGInstances -TimeoutSec 1500)) {
        ChainLog "TIMEOUT waiting for quota --- aborting chain at $it"; break
    }
    Start-Sleep -Seconds $WaitBetweenSeconds  # let AWS catch up
    $runLog = Join-Path $chainDir "$it.log"
    ChainLog "  launching $it (log: $runLog)"
    & "$repo\scripts\ec2_paperB_ablations\launch_detached.ps1" `
        -InstanceType $it -MaxRuntimeMinutes $MaxRuntimeMinutes *>&1 | Tee-Object -FilePath $runLog
    $rc = $LASTEXITCODE
    ChainLog "  $it finished, exit=$rc"
}
ChainLog "Chain complete."
