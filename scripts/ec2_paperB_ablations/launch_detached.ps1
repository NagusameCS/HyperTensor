#requires -Version 5.1
<#
.SYNOPSIS
    Launch an EC2 GPU instance, build geodessical, run the Paper-B
    Axiom Gauge + Online Basis ablation suite, pull artifacts back,
    then terminate the instance.

.DESCRIPTION
    Same orchestrator pattern as ec2_paperA_cachefit\launch.ps1.
    Differences: copies up scripts/ec2_paperB_ablations/run_paperB_ablations.sh
    instead of run_paperA.sh, and writes results into
    benchmarks\paperB_ablations_<GPU>_<ts>\.

.PARAMETER InstanceType
    EC2 instance type. Default g6e.xlarge (L40S).
#>
[CmdletBinding()]
param(
    [string]$InstanceType    = "g6e.xlarge",
    [string]$KeyName         = "hypertensor-key",
    [string]$PemPath         = "",
    [string]$Region          = "us-east-1",
    [int]   $MaxRuntimeMinutes = 240,
    [string]$HFToken         = "",
    [string]$ModelUrl        = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    [string]$LocalOutDir     = "",
    [switch]$DryRun,
    [switch]$KeepInstance,
    [int]   $RootVolumeGB    = 100
)

$ErrorActionPreference = "Continue"

function Log($msg, $color="Cyan") { Write-Host ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $msg) -ForegroundColor $color }
function Warn($msg) { Log $msg Yellow }
function Err($msg)  { Log $msg Red }
function Ok($msg)   { Log $msg Green }

$gpuMeta = @{
    "g6e.xlarge"    = @{ name="L40S";  arch="sm_89"; l2_mb=96; vram_gb=48 }
    "g6e.2xlarge"   = @{ name="L40S";  arch="sm_89"; l2_mb=96; vram_gb=48 }
    "g6.xlarge"     = @{ name="L4";    arch="sm_89"; l2_mb=48; vram_gb=24 }
    "g5.xlarge"     = @{ name="A10G";  arch="sm_86"; l2_mb=6;  vram_gb=24 }
    "g5.2xlarge"    = @{ name="A10G";  arch="sm_86"; l2_mb=6;  vram_gb=24 }
    "p4d.24xlarge"  = @{ name="A100";  arch="sm_80"; l2_mb=40; vram_gb=40 }
    "p5.48xlarge"   = @{ name="H100";  arch="sm_90"; l2_mb=50; vram_gb=80 }
}
if (-not $gpuMeta.ContainsKey($InstanceType)) {
    Err "Unknown InstanceType '$InstanceType'."
    exit 2
}
$gpu = $gpuMeta[$InstanceType]
$gpuName = $gpu.name
Ok "Target: $InstanceType  ($gpuName, $($gpu.arch), L2=$($gpu.l2_mb) MB, VRAM=$($gpu.vram_gb) GB)"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $LocalOutDir) {
    $LocalOutDir = Join-Path $repoRoot ("benchmarks\paperB_ablations_{0}_{1}" -f $gpuName, $ts)
}
$staging = Join-Path $env:TEMP ("hypertensor_ec2B_$ts")
New-Item -ItemType Directory -Path $LocalOutDir -Force | Out-Null
New-Item -ItemType Directory -Path $staging     -Force | Out-Null
Log "Local out dir: $LocalOutDir"
Log "Staging:       $staging"

if (-not $PemPath) { $PemPath = Join-Path $HOME ".ssh\$KeyName.pem" }
if (-not (Test-Path $PemPath)) { Err "PEM not found: $PemPath"; exit 2 }

if (-not $HFToken) { $HFToken = $env:HF_TOKEN }
if (-not $HFToken) { Warn "No HuggingFace token; gated GGUF download may fail." }

Log "Looking up latest Deep Learning Base AMI in $Region..."
$amiQuery = aws ec2 describe-images --region $Region --owners amazon `
    --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*' `
    --query 'reverse(sort_by(Images,&CreationDate))[0].[ImageId,Name]' --output text
if (-not $amiQuery -or $amiQuery -match 'None') { Err "Could not find DLAMI"; exit 3 }
$parts = $amiQuery -split "\s+"
$amiId  = $parts[0]
Ok "AMI: $amiId"

$vpcId = aws ec2 describe-vpcs --region $Region --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text
$subnetRows = aws ec2 describe-subnets --region $Region --filters "Name=vpc-id,Values=$vpcId" --query 'Subnets[].[SubnetId,AvailabilityZone]' --output text
$subnets = @()
foreach ($line in ($subnetRows -split "`n")) {
    $line = $line.Trim(); if (-not $line) { continue }
    $cols = $line -split "\s+"
    $subnets += [PSCustomObject]@{ Id = $cols[0]; Az = $cols[1] }
}
$azPref = 'a','b','c','f','e','d'
$subnets = $subnets | Sort-Object @{ Expression = { $azPref.IndexOf($_.Az.Substring($_.Az.Length-1,1)) } }

try {
    $myIp = (Invoke-RestMethod -Uri "https://checkip.amazonaws.com" -TimeoutSec 10).Trim()
} catch {
    $myIp = (Invoke-RestMethod -Uri "https://api.ipify.org" -TimeoutSec 10).Trim()
}
Log "My public IP: $myIp"

$sgName = "hypertensor-paperA-ssh"  # share SG with paperA campaign
$sgId = aws ec2 describe-security-groups --region $Region --filters "Name=group-name,Values=$sgName" "Name=vpc-id,Values=$vpcId" --query 'SecurityGroups[0].GroupId' --output text 2>$null
if ($sgId -eq 'None' -or -not $sgId) {
    $sgId = aws ec2 create-security-group --region $Region --group-name $sgName --description "HyperTensor SSH" --vpc-id $vpcId --query GroupId --output text
}
& aws ec2 authorize-security-group-ingress --region $Region --group-id $sgId --protocol tcp --port 22 --cidr "$myIp/32" 2>&1 | Out-Null

$srcTar = Join-Path $staging "hypertensor_src.tar.gz"
Log "git archive HEAD -> $srcTar"
git archive --format=tar.gz --output="$srcTar" HEAD
$srcSizeMB = [math]::Round((Get-Item $srcTar).Length / 1MB, 2)
Log "Source tarball: $srcSizeMB MB"

Copy-Item "$PSScriptRoot\build_ubuntu_cuda.sh"      $staging
Copy-Item "$PSScriptRoot\run_paperB_ablations.sh"   $staging

$userData = @"
#!/bin/bash
echo "shutdown -h now" | sudo at now + $MaxRuntimeMinutes minutes 2>/dev/null || \
  (sleep $($MaxRuntimeMinutes * 60) && shutdown -h now) &
mkdir -p /opt/hypertensor
chown ubuntu:ubuntu /opt/hypertensor
touch /var/log/hypertensor_user_data_done
"@
$udB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($userData))

if ($DryRun) { Ok "DryRun, exiting"; exit 0 }

$bdMap = @"
[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":$RootVolumeGB,"VolumeType":"gp3","DeleteOnTermination":true}}]
"@
$bdMapFile = Join-Path $staging "bdmap.json"
$bdMap | Set-Content -Path $bdMapFile -Encoding ASCII

Log "aws ec2 run-instances..."
$tagSpec = "ResourceType=instance,Tags=[{Key=Name,Value=hypertensor-paperB-$gpuName-$ts},{Key=Project,Value=HyperTensor},{Key=Purpose,Value=PaperB-Ablations}]"
$instance = $null
foreach ($sn in $subnets) {
    Log "  trying AZ=$($sn.Az) subnet=$($sn.Id)"
    $instanceJson = aws ec2 run-instances `
        --region $Region --image-id $amiId --instance-type $InstanceType `
        --key-name $KeyName --security-group-ids $sgId --subnet-id $sn.Id `
        --block-device-mappings "file://$bdMapFile" `
        --instance-initiated-shutdown-behavior terminate `
        --user-data $udB64 --tag-specifications $tagSpec --output json 2>&1
    if ($LASTEXITCODE -eq 0) {
        $instance = ($instanceJson | ConvertFrom-Json).Instances[0]
        Ok "  launched in $($sn.Az)"; break
    } else {
        $msgStr = ($instanceJson | Out-String).Trim()
        if ($msgStr -match "InsufficientInstanceCapacity|Unsupported") { Warn "  AZ unavailable, next..."; continue }
        Err "  fatal: $msgStr"; exit 5
    }
}
if (-not $instance) { Err "All AZs exhausted"; exit 5 }
$instanceId = $instance.InstanceId
Ok "Launched $instanceId"

$eipAlloc = $null
try {
    Log "Waiting for instance running..."
    aws ec2 wait instance-running --region $Region --instance-ids $instanceId
    $desc = aws ec2 describe-instances --region $Region --instance-ids $instanceId --output json | ConvertFrom-Json
    $publicIp = $desc.Reservations[0].Instances[0].PublicIpAddress
    if ($publicIp -like "100.*") {
        Log "Public IP in 100.x/8 (often blocked); allocating EIP..."
        $eipJson = aws ec2 allocate-address --region $Region --domain vpc --output json
        $eipAlloc = ($eipJson | ConvertFrom-Json)
        aws ec2 associate-address --region $Region --instance-id $instanceId --allocation-id $eipAlloc.AllocationId --output json | Out-Null
        Start-Sleep -Seconds 5
        $publicIp = $eipAlloc.PublicIp
    }
    Ok "Public IP: $publicIp"

    Log "Waiting for SSH..."
    $sshOpts = @(
        "-i", $PemPath,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=$($env:TEMP)\hypertensor_known_$ts",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=20",
        "-o", "ServerAliveCountMax=6",
        "-o", "TCPKeepAlive=yes"
    )
    $sshReady = $false
    for ($i=0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 10
        & ssh @sshOpts "ubuntu@$publicIp" "echo ssh-ok && [ -f /var/log/hypertensor_user_data_done ] && echo ud-done || echo ud-pending" 2>$null
        if ($LASTEXITCODE -eq 0) { $sshReady = $true; break }
        Write-Host "." -NoNewline
    }
    Write-Host ""
    if (-not $sshReady) { throw "SSH never became ready" }
    Ok "SSH ready"

    Log "scp source + scripts..."
    & scp @sshOpts "$srcTar" "ubuntu@${publicIp}:/tmp/hypertensor_src.tar.gz"
    if ($LASTEXITCODE -ne 0) { throw "scp source failed" }
    & scp @sshOpts "$staging\build_ubuntu_cuda.sh" "$staging\run_paperB_ablations.sh" "ubuntu@${publicIp}:/tmp/"
    if ($LASTEXITCODE -ne 0) { throw "scp scripts failed" }

    $remoteScript = @"
#!/usr/bin/env bash
set -uxo pipefail
exec > >(tee -a /tmp/paperB_remote.log) 2>&1

GPU_NAME='$gpuName'
SM_ARCH='$($gpu.arch)'
MODEL_URL='$ModelUrl'
HF_TOKEN='$HFToken'

echo "[remote] === Paper-B start `$(date -u) GPU=`$GPU_NAME ARCH=`$SM_ARCH ==="

sudo mkdir -p /opt/hypertensor
sudo chown ubuntu:ubuntu /opt/hypertensor
cd /opt/hypertensor
tar xzf /tmp/hypertensor_src.tar.gz
echo "[remote] source extracted"

cp /tmp/build_ubuntu_cuda.sh /tmp/run_paperB_ablations.sh /opt/hypertensor/
chmod +x /opt/hypertensor/build_ubuntu_cuda.sh /opt/hypertensor/run_paperB_ablations.sh
SRC_DIR=/opt/hypertensor SM_ARCH=`$SM_ARCH bash /opt/hypertensor/build_ubuntu_cuda.sh

mkdir -p /opt/hypertensor/models
MODEL_PATH=/opt/hypertensor/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
if [ ! -f "`$MODEL_PATH" ]; then
    if [ -n "`$HF_TOKEN" ]; then
        curl -L --fail -H "Authorization: Bearer `$HF_TOKEN" "`$MODEL_URL" -o "`$MODEL_PATH" || \
          curl -L --fail "`$MODEL_URL" -o "`$MODEL_PATH"
    else
        curl -L --fail "`$MODEL_URL" -o "`$MODEL_PATH"
    fi
fi
ls -la "`$MODEL_PATH"

OUT_DIR=/opt/hypertensor/results_paperB_`$GPU_NAME
mkdir -p "`$OUT_DIR"
GPU_NAME=`$GPU_NAME SRC_DIR=/opt/hypertensor MODEL_PATH=`$MODEL_PATH OUT_DIR=`$OUT_DIR \
  bash /opt/hypertensor/run_paperB_ablations.sh

TAR_PATH=`$(cat `$OUT_DIR/tarball_path.txt 2>/dev/null || true)
echo "[remote] DONE tar=`$TAR_PATH"
ls -la /opt/hypertensor/paperB_*.tar.gz 2>/dev/null || true
echo "[remote] === Paper-B end `$(date -u) ==="
"@
    $remoteDriverPath = Join-Path $staging "remote_driver.sh"
    [IO.File]::WriteAllText($remoteDriverPath, ($remoteScript -replace "`r`n","`n"), [Text.UTF8Encoding]::new($false))
    & scp @sshOpts $remoteDriverPath "ubuntu@${publicIp}:/tmp/remote_driver.sh"
    if ($LASTEXITCODE -ne 0) { throw "scp remote_driver failed" }

    $sessionLog = Join-Path $LocalOutDir "remote_session.log"
    Log "Starting remote driver DETACHED (nohup) ..."
    & ssh @sshOpts "ubuntu@$publicIp" "nohup bash /tmp/remote_driver.sh > /tmp/paperB_nohup.log 2>&1 < /dev/null & disown; sleep 1; echo started_pid=`$!"
    if ($LASTEXITCODE -ne 0) { throw "Failed to launch detached driver" }
    Ok "Detached. Polling for completion (poll every 60s, max ${MaxRuntimeMinutes} min)."

    $deadline   = (Get-Date).AddMinutes($MaxRuntimeMinutes)
    $tarballRemote = $null
    $remoteExit = $null
    $iter = 0
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 60
        $iter++
        # Tail the remote log into our session log (append)
        $tail = & ssh @sshOpts "ubuntu@$publicIp" "tail -n 60 /tmp/paperB_remote.log 2>/dev/null; echo ---ENDLOG---; ls -la /opt/hypertensor/paperB_*.tar.gz 2>/dev/null; echo ---ENDTAR---" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Warn "  poll ${iter}: ssh failed (transient?) — retry next cycle"
            continue
        }
        Add-Content -Path $sessionLog -Value "===== poll $iter @ $(Get-Date -Format HH:mm:ss) ====="
        Add-Content -Path $sessionLog -Value $tail
        # Detect completion: tarball exists AND driver printed END marker
        $tarLines = ($tail | Select-String -Pattern 'paperB_.*\.tar\.gz' -SimpleMatch:$false).Matches
        $endMarker = $tail | Select-String -Pattern '=== Paper-B end ' -SimpleMatch
        $lastLines = ($tail -join "`n")
        Log "  poll ${iter}: $(if($endMarker){'END marker seen'}else{'still running'}); tarball-lines=$($tarLines.Count)"
        if ($endMarker) {
            # Get the actual tarball path written by the bench script
            $tarPathRaw = & ssh @sshOpts "ubuntu@$publicIp" "ls /opt/hypertensor/paperB_*.tar.gz 2>/dev/null | head -n1" 2>$null
            if ($tarPathRaw) {
                $tarballRemote = $tarPathRaw.Trim()
                Ok "Tarball detected: $tarballRemote"
            }
            $remoteExit = 0
            break
        }
    }
    if (-not $tarballRemote) { Warn "Polling deadline reached without END marker. Pulling whatever exists." }

    Log "Pulling artifacts..."
    function ScpGet($src, $dst, $recurse=$false) {
        $scpArgs = if ($recurse) { @('-r') + $sshOpts } else { $sshOpts }
        $out = & scp @scpArgs $src $dst 2>&1
        if ($LASTEXITCODE -eq 0) { Log "  scp OK: $src" }
        else { Warn "  scp FAILED ($LASTEXITCODE): $src — $($out -join ' ')" }
    }
    ScpGet "ubuntu@${publicIp}:/tmp/paperB_remote.log"            "$LocalOutDir\paperB_remote.log"
    ScpGet "ubuntu@${publicIp}:/tmp/paperB_nohup.log"             "$LocalOutDir\paperB_nohup.log"
    # tarball — get remote path first so we can scp by exact name (avoids glob issues)
    $tarballs = & ssh @sshOpts "ubuntu@$publicIp" "ls /opt/hypertensor/paperB_*.tar.gz 2>/dev/null" 2>$null
    if ($tarballs) {
        foreach ($tb in ($tarballs | Where-Object { $_.Trim() })) {
            $fname = Split-Path $tb.Trim() -Leaf
            ScpGet "ubuntu@${publicIp}:$($tb.Trim())" "$LocalOutDir\$fname"
        }
    } else {
        Warn "  No tarball found on remote — benchmark may not have finished."
    }
    # full results dir (individual arm logs)
    ScpGet "ubuntu@${publicIp}:/opt/hypertensor/results_paperB_$gpuName" "$LocalOutDir\" -recurse $true

    Log "Local artifacts in $LocalOutDir :"
    Get-ChildItem $LocalOutDir -Recurse -File | ForEach-Object {
        "  {0,-60} {1,10} bytes" -f $_.FullName.Replace($LocalOutDir,"").TrimStart("\"), $_.Length
    } | Write-Host

    # Verify tarball
    $localTar = Get-ChildItem "$LocalOutDir\paperB_*.tar.gz" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($localTar) { Ok "Tarball downloaded: $($localTar.Name) ($($localTar.Length) bytes)" }
    else            { Warn "No tarball in $LocalOutDir — check scp errors above." }

    if (-not $tarballRemote) { Warn "Remote did not finish; partial artifacts pulled." }
    else { Ok "Remote pipeline completed successfully." }
}
catch {
    Err "ERROR: $_"
    Err $_.ScriptStackTrace
}
finally {
    if ($KeepInstance) {
        Warn "KeepInstance — instance $instanceId still running."
    } else {
        Log "Terminating $instanceId..."
        aws ec2 terminate-instances --region $Region --instance-ids $instanceId --output text | Out-Null
        Ok "Terminate request sent."
    }
    if ($eipAlloc) {
        Start-Sleep -Seconds 5
        aws ec2 release-address --region $Region --allocation-id $eipAlloc.AllocationId --output text 2>&1 | Out-Null
    }
    Log "Local artifacts: $LocalOutDir"
}
