#requires -Version 5.1
<#
.SYNOPSIS
    Launch an EC2 GPU instance, build geodessical, run the Paper-A
    cache-fit benchmark suite, pull all artifacts back to the local box,
    then terminate the instance.

.DESCRIPTION
    Targets the L2-capacity dimension of the cache-fit hypothesis using
    AWS-available GPUs. NO benchmark data is left on EC2 --- the local
    machine is the data store.

    Default: g6e.xlarge (L40S, 96 MB L2, 48 GB) --- workstation flagship
    analog of the table in the request.

    Supported -InstanceType values (all single-GPU unless noted):
        g6e.xlarge   L40S      sm_89   96 MB L2   48 GB VRAM   ~$1.86/h
        g6.xlarge    L4        sm_89   48 MB L2   24 GB VRAM   ~$0.80/h
        g5.xlarge    A10G      sm_86    6 MB L2   24 GB VRAM   ~$1.00/h
        p4d.24xlarge A100x8    sm_80   40 MB L2   40 GB VRAM   ~$32/h   (uses GPU 0)
        p5.48xlarge  H100x8    sm_90   50 MB L2   80 GB VRAM   ~$98/h   (uses GPU 0)

.PARAMETER InstanceType
    EC2 instance type. Default g6e.xlarge.

.PARAMETER KeyName
    EC2 key pair name. Default hypertensor-key.

.PARAMETER PemPath
    Local path to PEM file. Default $HOME\.ssh\<KeyName>.pem.

.PARAMETER Region
    AWS region. Default us-east-1.

.PARAMETER MaxRuntimeMinutes
    Hard auto-terminate cap (defence-in-depth via shutdown -h +N).
    Default 120.

.PARAMETER HFToken
    HuggingFace token for gated model download. If empty, reads $env:HF_TOKEN.

.PARAMETER ModelUrl
    GGUF download URL.
    Default bartowski/Meta-Llama-3.1-8B-Instruct-GGUF Q4_K_M (~4.9 GB).

.PARAMETER LocalOutDir
    Local directory to write pulled artifacts.
    Default benchmarks\paperA_cachefit_<ts>\.

.PARAMETER DryRun
    Print plan + AWS resource preview, do not launch.

.PARAMETER KeepInstance
    Skip terminate at end (manual cleanup required, costs money).

.EXAMPLE
    .\scripts\ec2_paperA_cachefit\launch.ps1                       # default L40S
    .\scripts\ec2_paperA_cachefit\launch.ps1 -InstanceType g5.xlarge
    .\scripts\ec2_paperA_cachefit\launch.ps1 -DryRun
#>

[CmdletBinding()]
param(
    [string]$InstanceType    = "g6e.xlarge",
    [string]$KeyName         = "hypertensor-key",
    [string]$PemPath         = "",
    [string]$Region          = "us-east-1",
    [int]   $MaxRuntimeMinutes = 120,
    [string]$HFToken         = "",
    [string]$ModelUrl        = "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    [string]$LocalOutDir     = "",
    [switch]$DryRun,
    [switch]$KeepInstance,
    [int]   $RootVolumeGB    = 100
)

$ErrorActionPreference = "Stop"

# PowerShell 5.1 turns *any* stderr write from a native exe into a terminating
# error when EAP=Stop. AWS CLI commonly writes warnings/info to stderr even on
# success. We rely on explicit $LASTEXITCODE checks below, so use Continue.
$ErrorActionPreference = "Continue"

# ---------------------------------------------------------------------------
# Pretty logging
# ---------------------------------------------------------------------------
function Log($msg, $color="Cyan") { Write-Host ("[{0}] {1}" -f (Get-Date -Format HH:mm:ss), $msg) -ForegroundColor $color }
function Warn($msg) { Log $msg Yellow }
function Err($msg)  { Log $msg Red }
function Ok($msg)   { Log $msg Green }

# ---------------------------------------------------------------------------
# Instance type -> GPU metadata table
# ---------------------------------------------------------------------------
$gpuMeta = @{
    "g6e.xlarge"    = @{ name="L40S";  arch="sm_89"; l2_mb=96; vram_gb=48 }
    "g6e.2xlarge"   = @{ name="L40S";  arch="sm_89"; l2_mb=96; vram_gb=48 }
    "g6.xlarge"     = @{ name="L4";    arch="sm_89"; l2_mb=48; vram_gb=24 }
    "g5.xlarge"     = @{ name="A10G";  arch="sm_86"; l2_mb=6;  vram_gb=24 }
    "g5.2xlarge"    = @{ name="A10G";  arch="sm_86"; l2_mb=6;  vram_gb=24 }
    "p4d.24xlarge"  = @{ name="A100";  arch="sm_80"; l2_mb=40; vram_gb=40 }
    "p4de.24xlarge" = @{ name="A100";  arch="sm_80"; l2_mb=40; vram_gb=80 }
    "p5.48xlarge"   = @{ name="H100";  arch="sm_90"; l2_mb=50; vram_gb=80 }
}
if (-not $gpuMeta.ContainsKey($InstanceType)) {
    Err "Unknown InstanceType '$InstanceType'. Add it to gpuMeta or pick from: $($gpuMeta.Keys -join ', ')"
    exit 2
}
$gpu = $gpuMeta[$InstanceType]
$gpuName = $gpu.name
Ok "Target: $InstanceType  ($gpuName, $($gpu.arch), L2=$($gpu.l2_mb) MB, VRAM=$($gpu.vram_gb) GB)"

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $LocalOutDir) {
    $LocalOutDir = Join-Path $repoRoot ("benchmarks\paperA_cachefit_{0}_{1}" -f $gpuName, $ts)
}
$staging = Join-Path $env:TEMP ("hypertensor_ec2_$ts")
New-Item -ItemType Directory -Path $LocalOutDir -Force | Out-Null
New-Item -ItemType Directory -Path $staging     -Force | Out-Null
Log "Local out dir: $LocalOutDir"
Log "Staging:       $staging"

if (-not $PemPath) { $PemPath = Join-Path $HOME ".ssh\$KeyName.pem" }
if (-not (Test-Path $PemPath)) { Err "PEM not found: $PemPath"; exit 2 }
Log "PEM:           $PemPath"

if (-not $HFToken) { $HFToken = $env:HF_TOKEN }
if (-not $HFToken) {
    Warn "No HuggingFace token set. Llama-3.1-8B GGUF download may fail (gated)."
    Warn "Set `$env:HF_TOKEN or pass -HFToken before re-running, or change -ModelUrl to a non-gated GGUF."
}

# ---------------------------------------------------------------------------
# AMI lookup (Deep Learning Base OSS Nvidia, Ubuntu 22.04, latest)
# ---------------------------------------------------------------------------
Log "Looking up latest Deep Learning Base AMI in $Region..."
$amiQuery = aws ec2 describe-images --region $Region --owners amazon `
    --filters 'Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*' `
    --query 'reverse(sort_by(Images,&CreationDate))[0].[ImageId,Name]' --output text
if (-not $amiQuery -or $amiQuery -match 'None') { Err "Could not find DLAMI"; exit 3 }
$parts = $amiQuery -split "\s+"
$amiId  = $parts[0]
$amiNm  = ($parts[1..($parts.Length-1)] -join ' ')
Ok "AMI: $amiId  ($amiNm)"

# ---------------------------------------------------------------------------
# Default VPC + subnet
# ---------------------------------------------------------------------------
$vpcId = aws ec2 describe-vpcs --region $Region --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text
if ($vpcId -eq 'None' -or -not $vpcId) { Err "No default VPC in $Region"; exit 3 }
# Collect all default-VPC subnets, prefer (a,b,c,f,e,d) --- common GPU capacity preference.
$subnetRows = aws ec2 describe-subnets --region $Region --filters "Name=vpc-id,Values=$vpcId" --query 'Subnets[].[SubnetId,AvailabilityZone]' --output text
$subnets = @()
foreach ($line in ($subnetRows -split "`n")) {
    $line = $line.Trim()
    if (-not $line) { continue }
    $cols = $line -split "\s+"
    $subnets += [PSCustomObject]@{ Id = $cols[0]; Az = $cols[1] }
}
$azPref = 'a','b','c','f','e','d'
$subnets = $subnets | Sort-Object @{ Expression = { $azPref.IndexOf($_.Az.Substring($_.Az.Length-1,1)) } }
if (-not $subnets) { Err "No subnets in default VPC"; exit 3 }
Log ("Subnet candidates (AZ pref order): " + (($subnets | ForEach-Object { "$($_.Az)=$($_.Id)" }) -join ', '))

# ---------------------------------------------------------------------------
# Security group: ssh from this machine's public IP
# ---------------------------------------------------------------------------
try {
    $myIp = (Invoke-RestMethod -Uri "https://checkip.amazonaws.com" -TimeoutSec 10).Trim()
} catch {
    try { $myIp = (Invoke-RestMethod -Uri "https://api.ipify.org" -TimeoutSec 10).Trim() }
    catch { Err "Could not determine public IP"; exit 3 }
}
Log "My public IP: $myIp"

$sgName = "hypertensor-paperA-ssh"
$sgId = aws ec2 describe-security-groups --region $Region --filters "Name=group-name,Values=$sgName" "Name=vpc-id,Values=$vpcId" --query 'SecurityGroups[0].GroupId' --output text 2>$null
if ($sgId -eq 'None' -or -not $sgId) {
    Log "Creating SG $sgName..."
    $sgId = aws ec2 create-security-group --region $Region --group-name $sgName --description "HyperTensor Paper-A SSH" --vpc-id $vpcId --query GroupId --output text
}
Log "SG: $sgId"
# Idempotent ingress: ignore "InvalidPermission.Duplicate".
$ingressOut = & aws ec2 authorize-security-group-ingress --region $Region --group-id $sgId --protocol tcp --port 22 --cidr "$myIp/32" 2>&1
if ($LASTEXITCODE -ne 0) { Log "  (ingress likely already exists: continuing)" "Yellow" }

# ---------------------------------------------------------------------------
# Tarball source via git archive HEAD (clean snapshot, no junk)
# ---------------------------------------------------------------------------
$srcTar = Join-Path $staging "hypertensor_src.tar.gz"
Log "git archive HEAD -> $srcTar"
git archive --format=tar.gz --output="$srcTar" HEAD
if (-not (Test-Path $srcTar)) { Err "git archive failed"; exit 4 }
$srcSizeMB = [math]::Round((Get-Item $srcTar).Length / 1MB, 2)
Log "Source tarball: $srcSizeMB MB"

# Copy bench scripts alongside (in case HEAD doesn't include them yet)
Copy-Item "$PSScriptRoot\build_ubuntu_cuda.sh" $staging
Copy-Item "$PSScriptRoot\run_paperA.sh"        $staging

# ---------------------------------------------------------------------------
# user-data: defence-in-depth auto-shutdown
# ---------------------------------------------------------------------------
$userData = @"
#!/bin/bash
# Hard auto-shutdown after $MaxRuntimeMinutes minutes regardless of orchestrator state.
echo "shutdown -h now" | sudo at now + $MaxRuntimeMinutes minutes 2>/dev/null || \
  (sleep $($MaxRuntimeMinutes * 60) && shutdown -h now) &
mkdir -p /opt/hypertensor
chown ubuntu:ubuntu /opt/hypertensor
touch /var/log/hypertensor_user_data_done
"@
$udB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($userData))

# ---------------------------------------------------------------------------
# Plan summary
# ---------------------------------------------------------------------------
$planText = @"
=== Paper-A Cache-Fit EC2 Launch Plan ===
Region:           $Region
Instance type:    $InstanceType  (cores=$($gpu.name), L2=$($gpu.l2_mb) MB, VRAM=$($gpu.vram_gb) GB)
AMI:              $amiId
VPC / Subnet:     $vpcId / (per-AZ retry across $($subnets.Count) subnets)
Security group:   $sgId  (ssh from $myIp/32)
Key pair:         $KeyName  (PEM=$PemPath)
Root volume:      $RootVolumeGB GB  (terminate on shutdown)
Max runtime:      $MaxRuntimeMinutes min  (hard cap via shutdown -h)
Local out dir:    $LocalOutDir
Source tarball:   $srcTar  ($srcSizeMB MB)
Model:            $ModelUrl
HF token set:     $([bool]$HFToken)
Keep instance:    $($KeepInstance.IsPresent)
"@
Write-Host $planText -ForegroundColor White

if ($DryRun) { Ok "DryRun=`$true, exiting"; exit 0 }

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
$bdMap = @"
[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":$RootVolumeGB,"VolumeType":"gp3","DeleteOnTermination":true}}]
"@
$bdMapFile = Join-Path $staging "bdmap.json"
$bdMap | Set-Content -Path $bdMapFile -Encoding ASCII

Log "Running aws ec2 run-instances..."
$tagSpec = "ResourceType=instance,Tags=[{Key=Name,Value=hypertensor-paperA-$gpuName-$ts},{Key=Project,Value=HyperTensor},{Key=Purpose,Value=PaperA-CacheFit}]"
$instance = $null
foreach ($sn in $subnets) {
    Log "  trying AZ=$($sn.Az) subnet=$($sn.Id)"
    $instanceJson = aws ec2 run-instances `
        --region $Region `
        --image-id $amiId `
        --instance-type $InstanceType `
        --key-name $KeyName `
        --security-group-ids $sgId `
        --subnet-id $sn.Id `
        --block-device-mappings "file://$bdMapFile" `
        --instance-initiated-shutdown-behavior terminate `
        --user-data $udB64 `
        --tag-specifications $tagSpec `
        --output json 2>&1
    if ($LASTEXITCODE -eq 0) {
        $instance = ($instanceJson | ConvertFrom-Json).Instances[0]
        Ok "  launched in $($sn.Az)"
        break
    } else {
        $msgStr = ($instanceJson | Out-String).Trim()
        if ($msgStr -match "InsufficientInstanceCapacity|Unsupported") {
            Warn "  AZ $($sn.Az) unavailable, trying next..."
            continue
        }
        Err "  fatal: $msgStr"
        exit 5
    }
}
if (-not $instance) { Err "All AZs exhausted with InsufficientInstanceCapacity. Try a different region or instance type."; exit 5 }
$instanceId = $instance.InstanceId
Ok "Launched $instanceId"

# Hook a try/finally for guaranteed terminate
$terminated = $false
$eipAlloc = $null
try {
    Log "Waiting for instance running..."
    aws ec2 wait instance-running --region $Region --instance-ids $instanceId
    $desc = aws ec2 describe-instances --region $Region --instance-ids $instanceId --output json | ConvertFrom-Json
    $publicIp = $desc.Reservations[0].Instances[0].PublicIpAddress
    if (-not $publicIp) { throw "No public IP assigned" }
    Log "Initial public IP: $publicIp"

    # Some ISPs (Telmex/T-Mobile etc.) block AWS' 100.x/8 carrier-grade NAT
    # range entirely. Allocate an Elastic IP (always 52.x/54.x) to dodge.
    if ($publicIp -like "100.*") {
        Log "Public IP is in 100.x/8 range (often ISP-blocked); allocating EIP..."
        $eipJson = aws ec2 allocate-address --region $Region --domain vpc --output json
        $eipAlloc = ($eipJson | ConvertFrom-Json)
        Log "  EIP $($eipAlloc.PublicIp) (alloc=$($eipAlloc.AllocationId))"
        $assoc = aws ec2 associate-address --region $Region --instance-id $instanceId --allocation-id $eipAlloc.AllocationId --output json | ConvertFrom-Json
        Start-Sleep -Seconds 5
        $publicIp = $eipAlloc.PublicIp
    }
    Ok "Public IP: $publicIp"

    # Wait for SSH
    Log "Waiting for SSH (up to 5 min)..."
    $sshReady = $false
    $sshOpts = @(
        "-i", $PemPath,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=$($env:TEMP)\hypertensor_known_$ts",
        "-o", "ConnectTimeout=8",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=240",
        "-o", "TCPKeepAlive=yes"
    )
    for ($i=0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 10
        & ssh @sshOpts "ubuntu@$publicIp" "echo ssh-ok && [ -f /var/log/hypertensor_user_data_done ] && echo ud-done || echo ud-pending" 2>$null
        if ($LASTEXITCODE -eq 0) { $sshReady = $true; break }
        Write-Host "." -NoNewline
    }
    Write-Host ""
    if (-not $sshReady) { throw "SSH never became ready" }
    Ok "SSH ready"

    # ---------------------------------------------------------------------------
    # Upload source + scripts
    # ---------------------------------------------------------------------------
    Log "scp source tarball + scripts..."
    & scp @sshOpts "$srcTar" "ubuntu@${publicIp}:/tmp/hypertensor_src.tar.gz"
    if ($LASTEXITCODE -ne 0) { throw "scp source failed" }
    & scp @sshOpts "$staging\build_ubuntu_cuda.sh" "$staging\run_paperA.sh" "ubuntu@${publicIp}:/tmp/"
    if ($LASTEXITCODE -ne 0) { throw "scp scripts failed" }

    # ---------------------------------------------------------------------------
    # Compose the remote driver script. We write it locally (here-string)
    # then scp it to avoid quoting hell.
    # ---------------------------------------------------------------------------
    $remoteScript = @"
#!/usr/bin/env bash
set -uxo pipefail
exec > >(tee -a /tmp/paperA_remote.log) 2>&1

GPU_NAME='$gpuName'
SM_ARCH='$($gpu.arch)'
MODEL_URL='$ModelUrl'
HF_TOKEN='$HFToken'

echo "[remote] === Paper-A start `$(date -u) GPU=`$GPU_NAME ARCH=`$SM_ARCH ==="

# 1. Stage source
sudo mkdir -p /opt/hypertensor
sudo chown ubuntu:ubuntu /opt/hypertensor
cd /opt/hypertensor
tar xzf /tmp/hypertensor_src.tar.gz
echo "[remote] source extracted: `$(ls | head -n10)"

# 2. Build
cp /tmp/build_ubuntu_cuda.sh /tmp/run_paperA.sh /opt/hypertensor/
chmod +x /opt/hypertensor/build_ubuntu_cuda.sh /opt/hypertensor/run_paperA.sh
SRC_DIR=/opt/hypertensor SM_ARCH=`$SM_ARCH bash /opt/hypertensor/build_ubuntu_cuda.sh

# 3. Download model
mkdir -p /opt/hypertensor/models
MODEL_PATH=/opt/hypertensor/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
if [ ! -f "`$MODEL_PATH" ]; then
    echo "[remote] downloading model..."
    if [ -n "`$HF_TOKEN" ]; then
        curl -L --fail -H "Authorization: Bearer `$HF_TOKEN" "`$MODEL_URL" -o "`$MODEL_PATH" || \
          { echo "[remote] HF token download failed, trying anonymous..."; curl -L --fail "`$MODEL_URL" -o "`$MODEL_PATH"; }
    else
        curl -L --fail "`$MODEL_URL" -o "`$MODEL_PATH"
    fi
fi
ls -la "`$MODEL_PATH"

# 4. Run benchmarks
OUT_DIR=/opt/hypertensor/results_`$GPU_NAME
mkdir -p "`$OUT_DIR"
GPU_NAME=`$GPU_NAME SRC_DIR=/opt/hypertensor MODEL_PATH=`$MODEL_PATH OUT_DIR=`$OUT_DIR \
  bash /opt/hypertensor/run_paperA.sh

# 5. Stage tarball for pull
TAR_PATH=`$(cat `$OUT_DIR/tarball_path.txt 2>/dev/null || true)
echo "[remote] DONE tar=`$TAR_PATH"
ls -la /opt/hypertensor/paperA_*.tar.gz 2>/dev/null || true
echo "[remote] === Paper-A end `$(date -u) ==="
"@
    $remoteDriverPath = Join-Path $staging "remote_driver.sh"
    # write LF endings
    [IO.File]::WriteAllText($remoteDriverPath, ($remoteScript -replace "`r`n","`n"), [Text.UTF8Encoding]::new($false))
    & scp @sshOpts $remoteDriverPath "ubuntu@${publicIp}:/tmp/remote_driver.sh"
    if ($LASTEXITCODE -ne 0) { throw "scp remote_driver failed" }

    # ---------------------------------------------------------------------------
    # Run and stream
    # ---------------------------------------------------------------------------
    $sessionLog = Join-Path $LocalOutDir "remote_session.log"
    Log "Running remote driver (live tail to $sessionLog)..."
    & ssh @sshOpts "ubuntu@$publicIp" "bash /tmp/remote_driver.sh" 2>&1 | Tee-Object -FilePath $sessionLog
    $remoteExit = $LASTEXITCODE
    Log "Remote driver exit=$remoteExit"

    # ---------------------------------------------------------------------------
    # Pull artifacts (always, even on partial failure)
    # ---------------------------------------------------------------------------
    Log "Pulling /tmp/paperA_remote.log..."
    & scp @sshOpts "ubuntu@${publicIp}:/tmp/paperA_remote.log" "$LocalOutDir\paperA_remote.log" 2>&1 | Out-Null

    Log "Pulling tarballs and full results dir..."
    & scp @sshOpts "ubuntu@${publicIp}:/opt/hypertensor/paperA_*.tar.gz" "$LocalOutDir\" 2>&1 | Out-Null
    & scp -r @sshOpts "ubuntu@${publicIp}:/opt/hypertensor/results_$gpuName" "$LocalOutDir\" 2>&1 | Out-Null

    Get-ChildItem $LocalOutDir -Recurse | Select-Object FullName,Length | Format-Table -AutoSize | Out-String | Write-Host

    if ($remoteExit -ne 0) { Warn "Remote returned non-zero, but artifacts pulled where possible." }
    else { Ok "Remote pipeline completed successfully." }
}
catch {
    Err "ERROR: $_"
    Err $_.ScriptStackTrace
}
finally {
    if ($KeepInstance) {
        Warn "KeepInstance set --- instance $instanceId still running. Cost is on you."
        Warn "Terminate: aws ec2 terminate-instances --region $Region --instance-ids $instanceId"
    } else {
        Log "Terminating $instanceId..."
        try {
            aws ec2 terminate-instances --region $Region --instance-ids $instanceId --output text | Out-Null
            $terminated = $true
            Ok "Terminate request sent."
        } catch {
            Err "Terminate failed: $_"
            Err "MANUAL ACTION: aws ec2 terminate-instances --region $Region --instance-ids $instanceId"
        }
    }
    if ($eipAlloc) {
        Log "Releasing EIP $($eipAlloc.PublicIp)..."
        # EIP can only be released after disassociation (auto on terminate); add small wait
        Start-Sleep -Seconds 5
        aws ec2 release-address --region $Region --allocation-id $eipAlloc.AllocationId --output text 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Warn "  EIP release failed; manual: aws ec2 release-address --region $Region --allocation-id $($eipAlloc.AllocationId)"
        } else {
            Ok "  EIP released."
        }
    }
    Log "Local artifacts: $LocalOutDir"
}
