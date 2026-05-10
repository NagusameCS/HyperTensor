#!/usr/bin/env pwsh
# scripts/launch_five_principles_ec2.ps1
# May 7, 2026
#
# Launches EC2 L40S instance, runs five principles at 1.5B scale,
# saves results to local benchmarks/, terminates instance.
#
# Prerequisites: AWS CLI configured, key pair 'hypertensor-key' exists,
# security group 'hypertensor-sg' allows SSH.

param(
    [int] $TimeoutMinutes = 60,
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path $PSScriptRoot -Parent

#  Instance config 
$INSTANCE_TYPE = "g6.12xlarge"   # 4× L4, 96GB, ~$2/hr — OR use g5.12xlarge for L40S
$AMI_ID        = "ami-0c55b159cbfafe1f0"  # Deep Learning AMI GPU PyTorch 2.x (us-east-1)
$KEY_NAME      = "hypertensor-key"
$SG_ID         = "sg-hypertensor"
$SUBNET_ID     = "subnet-hypertensor"

Write-Host "================================================================"
Write-Host "HyperTensor — Five Principles EC2 Scale Test"
Write-Host "================================================================"
Write-Host "Instance: $INSTANCE_TYPE"
Write-Host "Timeout:  $TimeoutMinutes min"
Write-Host ""

#  Launch instance 
Write-Host "[1/5] Launching EC2 instance..."
$launch = aws ec2 run-instances `
    --image-id $AMI_ID `
    --instance-type $INSTANCE_TYPE `
    --key-name $KEY_NAME `
    --security-group-ids $SG_ID `
    --subnet-id $SUBNET_ID `
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":200,"VolumeType":"gp3"}}]' `
    --instance-initiated-shutdown-behavior terminate `
    --output json `
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hypertensor-five-principles}]'

if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to launch EC2 instance"
    exit 1
}

$instanceId = ($launch | ConvertFrom-Json).Instances[0].InstanceId
Write-Host "  Instance ID: $instanceId"
Write-Host "  Waiting for running state..."

aws ec2 wait instance-running --instance-ids $instanceId
if ($LASTEXITCODE -ne 0) {
    Write-Error "Instance failed to start"
    exit 1
}

# Get public IP
$desc = aws ec2 describe-instances --instance-ids $instanceId --output json | ConvertFrom-Json
$publicIp = $desc.Reservations[0].Instances[0].PublicIpAddress
Write-Host "  Public IP: $publicIp"
Write-Host ""

#  Wait for SSH 
Write-Host "[2/5] Waiting for SSH ($publicIp)..."
$sshReady = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 10
    $test = ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$HOME/.ssh/$KEY_NAME.pem" "ec2-user@$publicIp" "echo OK" 2>$null
    if ($test -eq "OK") { $sshReady = $true; break }
}
if (-not $sshReady) {
    Write-Error "SSH not ready after 5 minutes"
    aws ec2 terminate-instances --instance-ids $instanceId
    exit 1
}
Write-Host "  SSH ready"
Write-Host ""

#  Copy scripts 
Write-Host "[3/5] Copying test scripts..."
scp -o StrictHostKeyChecking=no -i "$HOME/.ssh/$KEY_NAME.pem" "$ROOT/scripts/five_principles_ec2.py" "ec2-user@$publicIp:~/five_principles_ec2.py"
Write-Host "  Scripts copied"
Write-Host ""

#  Run tests 
Write-Host "[4/5] Running five principles at 1.5B scale..."
Write-Host "  (This will download Qwen2.5-1.5B and run all 5 tests)"
Write-Host "  Estimated time: 15-30 minutes"
Write-Host ""

$remoteCmd = @"
set -e
echo "=== Installing dependencies ==="
pip install --quiet torch transformers 2>&1 | tail -1
echo "=== Running five principles ==="
python3 five_principles_ec2.py 2>&1
echo "=== DONE ==="
"@

# Write command to script, execute via SSH
$remoteCmd | ssh -o StrictHostKeyChecking=no -i "$HOME/.ssh/$KEY_NAME.pem" "ec2-user@$publicIp" "cat > run_test.sh && bash run_test.sh"

if ($LASTEXITCODE -ne 0) {
    Write-Warning "Test script exited with code $LASTEXITCODE"
}

Write-Host ""

#  Download results 
Write-Host "[5/5] Downloading results..."
$localOut = "$ROOT/benchmarks/five_principles_ec2"
New-Item -Force -ItemType Directory $localOut | Out-Null
scp -o StrictHostKeyChecking=no -i "$HOME/.ssh/$KEY_NAME.pem" -r "ec2-user@$publicIp:~/benchmarks/five_principles_ec2/*" "$localOut/"
Write-Host "  Results downloaded to $localOut"
Write-Host ""

#  Terminate instance 
Write-Host "Terminating instance $instanceId..."
aws ec2 terminate-instances --instance-ids $instanceId
Write-Host "Instance terminating. All done."
Write-Host "================================================================"
