param(
    [string]$SshHost = "ssh.opencs.dev",
    [string[]]$FallbackHosts = @("root@159.198.42.248"),
    [string]$RemoteRoot = "/root/HyperTensor",
    [string]$RemoteExe = "/root/HyperTensor/geodessical",
    [string]$RemoteModel = "",
    [string]$RemoteOutDir = "",
    [string]$LocalPullDir = "",
    [int]$CooldownSec = 20,
    [int]$LmEvalLimit = 0
)

$ErrorActionPreference = "Stop"

$localScript = ".\scripts\campaign\remote_arch_suite.sh"
if (-not (Test-Path $localScript)) {
    throw "Local script not found: $localScript"
}

if (-not $RemoteOutDir) {
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $RemoteOutDir = "/tmp/cross_hw_remote_$ts"
}
if (-not $LocalPullDir) {
    $LocalPullDir = Join-Path ".\benchmarks" ("cross_hw_remote_pull_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
}

$sshOpts = @(
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2"
)
$ssh = $sshOpts + @($SshHost)
$scpOpts = @(
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=20",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2"
)

function Invoke-WithRetry {
    param(
        [scriptblock]$Body,
        [string]$Label,
        [int]$Retries = 3
    )

    for ($i = 1; $i -le $Retries; $i++) {
        & $Body
        if ($LASTEXITCODE -eq 0) {
            return $true
        }
        Write-Host "[remote] $Label attempt $i/$Retries failed (exit $LASTEXITCODE)" -ForegroundColor Yellow
    }
    return $false
}

$hostCandidates = @($SshHost) + @($FallbackHosts)
$hostCandidates = $hostCandidates | Where-Object { $_ -and $_.Trim().Length -gt 0 } | Select-Object -Unique

$activeHost = $null
foreach ($candidate in $hostCandidates) {
    Write-Host "[remote] probing host $candidate ..."
    $probe = {
        & ssh @($sshOpts + @($candidate)) "echo ok"
    }
    if (Invoke-WithRetry -Body $probe -Label "probe $candidate" -Retries 2) {
        $activeHost = $candidate
        break
    }
}

if (-not $activeHost) {
    throw "No reachable SSH host from candidate list: $($hostCandidates -join ', ')"
}

Write-Host "[remote] using host: $activeHost"
$ssh = $sshOpts + @($activeHost)

Write-Host "[remote] preparing directories on $activeHost ..."
$mk = { & ssh @ssh "mkdir -p '$RemoteRoot/scripts/campaign' '$RemoteOutDir'" }
if (-not (Invoke-WithRetry -Body $mk -Label "prepare directories")) { throw "Failed to create remote directories" }

Write-Host "[remote] copying benchmark harness ..."
$cpScript = { & scp -q @scpOpts $localScript "$($activeHost):$RemoteRoot/scripts/campaign/remote_arch_suite.sh" }
if (-not (Invoke-WithRetry -Body $cpScript -Label "copy harness")) { throw "Failed to copy remote harness script" }

Write-Host "[remote] running benchmark campaign ..."
$modelArg = ""
if ($RemoteModel) {
    $modelArg = " --model '$RemoteModel'"
}
$remoteCmd = @(
    "chmod +x '$RemoteRoot/scripts/campaign/remote_arch_suite.sh'",
    "cd '$RemoteRoot'",
    "bash ./scripts/campaign/remote_arch_suite.sh --exe '$RemoteExe' --out-dir '$RemoteOutDir' --cooldown-sec '$CooldownSec' --lm-eval-limit '$LmEvalLimit'$modelArg"
) -join "; "

& ssh @ssh $remoteCmd
$remoteExitCode = $LASTEXITCODE

Write-Host "[remote] pulling artifacts back to local workspace ..."
New-Item -ItemType Directory -Force -Path $LocalPullDir | Out-Null
$pullWithWildcard = { & scp -q -r @scpOpts "$($activeHost):$RemoteOutDir/*" "$LocalPullDir" }
$pullWholeDir = { & scp -q -r @scpOpts "$($activeHost):$RemoteOutDir" "$LocalPullDir" }

if (-not (Invoke-WithRetry -Body $pullWithWildcard -Label "pull artifacts (wildcard)" -Retries 2)) {
    if (-not (Invoke-WithRetry -Body $pullWholeDir -Label "pull artifacts (dir)" -Retries 2)) {
        throw "Failed to pull remote artifacts"
    }
}

if ($remoteExitCode -ne 0) {
    throw "Remote campaign failed (exit $remoteExitCode). Artifacts were pulled to $LocalPullDir"
}

Write-Host "REMOTE_OUT=$RemoteOutDir"
Write-Host "LOCAL_PULL=$LocalPullDir"
