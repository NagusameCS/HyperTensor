#!/usr/bin/env pwsh
# =============================================================================
# TensorOS — Push OTA Update to Raspberry Pi over Bluetooth Serial
#
# Usage:
#   .\push_rpi.ps1 -ComPort COM5                    # Chain-load (RAM, fast dev)
#   .\push_rpi.ps1 -ComPort COM5 -Flash             # Persistent SD write
#   .\push_rpi.ps1 -ComPort COM5 -Kernel .\custom.img  # Custom binary
#   .\push_rpi.ps1 -ComPort COM5 -Build             # Build first, then push
#
# The script:
#   1. (Optionally) builds the ARM64 kernel with build_rpi.ps1
#   2. Opens the BT COM port
#   3. Sends the "ota" or "flash" shell command
#   4. Waits for "RDY" response
#   5. Sends: "OTA!" + uint32(size) + raw_bytes + uint32(crc32)
#   6. Waits for "OK!" then "BOOT"
# =============================================================================

param(
    [Parameter(Mandatory = $true)]
    [string]$ComPort,       # e.g. COM5

    [string]$Kernel = "",   # Path to kernel8.img (default: auto-detect)

    [switch]$Flash,         # Use persistent SD flash instead of RAM chain-load
    [switch]$Build,         # Run build_rpi.ps1 before pushing

    [int]$BaudRate = 115200,
    [int]$TimeoutSec = 60
)

$ErrorActionPreference = "Stop"

# ---- CRC-32 (IEEE) ----
function Get-CRC32([byte[]]$Data) {
    [uint32]$crc = 0xFFFFFFFF
    foreach ($b in $Data) {
        $crc = $crc -bxor $b
        for ($j = 0; $j -lt 8; $j++) {
            if ($crc -band 1) {
                $crc = ($crc -shr 1) -bxor 0xEDB88320
            } else {
                $crc = $crc -shr 1
            }
        }
    }
    return $crc -bxor 0xFFFFFFFF
}

# ---- Step 1: Optionally build ----
if ($Build) {
    Write-Host "[BUILD] Running build_rpi.ps1..." -ForegroundColor Cyan
    & "$PSScriptRoot\build_rpi.ps1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[BUILD] Build failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "[BUILD] Build succeeded." -ForegroundColor Green
}

# ---- Step 2: Find kernel binary ----
if ($Kernel -eq "") {
    # Auto-detect: look for build output
    $candidates = @(
        "$PSScriptRoot\build\arm64\kernel8.img",
        "$PSScriptRoot\kernel8.img"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) {
            $Kernel = $c
            break
        }
    }
    if ($Kernel -eq "") {
        Write-Host "[ERROR] No kernel8.img found. Specify -Kernel or use -Build." -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path $Kernel)) {
    Write-Host "[ERROR] Kernel not found: $Kernel" -ForegroundColor Red
    exit 1
}

$kernelBytes = [System.IO.File]::ReadAllBytes($Kernel)
$kernelSize = $kernelBytes.Length
$crc = Get-CRC32 $kernelBytes

Write-Host "[INFO] Kernel: $Kernel ($kernelSize bytes, CRC32=0x$($crc.ToString('X8')))" -ForegroundColor Cyan

# ---- Step 3: Open COM port ----
Write-Host "[COMM] Opening $ComPort at $BaudRate baud..." -ForegroundColor Yellow

$port = New-Object System.IO.Ports.SerialPort $ComPort, $BaudRate, "None", 8, "One"
$port.ReadTimeout = $TimeoutSec * 1000
$port.WriteTimeout = 10000
$port.DtrEnable = $true
$port.RtsEnable = $true
$port.Open()

Write-Host "[COMM] Port opened." -ForegroundColor Green

# Helper: read a line (until \n)
function Read-Line {
    $buf = ""
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if ($port.BytesToRead -gt 0) {
            $ch = [char]$port.ReadByte()
            if ($ch -eq "`n") { return $buf.Trim() }
            $buf += $ch
        } else {
            Start-Sleep -Milliseconds 10
        }
    }
    return $null  # timeout
}

# Helper: drain any pending input
function Drain-Input {
    Start-Sleep -Milliseconds 200
    while ($port.BytesToRead -gt 0) {
        [void]$port.ReadByte()
    }
}

try {
    # ---- Step 4: Send shell command ----
    Drain-Input

    $cmd = if ($Flash) { "flash" } else { "ota" }
    Write-Host "[COMM] Sending '$cmd' command to TensorOS shell..." -ForegroundColor Yellow
    $port.Write("$cmd`r")
    Start-Sleep -Milliseconds 500  # Let shell process + echo

    # ---- Step 5: Wait for RDY ----
    Write-Host "[COMM] Waiting for RDY..." -ForegroundColor Yellow
    $ready = $false
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if ($port.BytesToRead -gt 0) {
            $line = Read-Line
            Write-Host "  < $line" -ForegroundColor DarkGray
            if ($line -match "RDY") {
                $ready = $true
                break
            }
            if ($line -match "ERR") {
                Write-Host "[ERROR] Device reported error: $line" -ForegroundColor Red
                exit 1
            }
        }
        Start-Sleep -Milliseconds 50
    }

    if (-not $ready) {
        Write-Host "[ERROR] Timeout waiting for RDY." -ForegroundColor Red
        exit 1
    }

    Write-Host "[COMM] Device ready. Sending kernel..." -ForegroundColor Green

    # ---- Step 6: Send OTA protocol packet ----
    # Magic: "OTA!"
    $magic = [System.Text.Encoding]::ASCII.GetBytes("OTA!")
    $port.Write($magic, 0, 4)

    # Size: uint32 LE
    $sizeBytes = [System.BitConverter]::GetBytes([uint32]$kernelSize)
    $port.Write($sizeBytes, 0, 4)

    # Data: raw kernel binary (in chunks for flow control)
    $chunkSize = 1024
    $sent = 0
    while ($sent -lt $kernelSize) {
        $remaining = $kernelSize - $sent
        $chunk = [Math]::Min($chunkSize, $remaining)
        $port.Write($kernelBytes, $sent, $chunk)
        $sent += $chunk

        # Progress
        $pct = [int]($sent * 100 / $kernelSize)
        Write-Host "`r[XFER] $sent / $kernelSize bytes ($pct%)" -NoNewline -ForegroundColor Cyan

        # Small delay for flow control (BT SPP can buffer but let's be safe)
        if ($sent % 8192 -eq 0) {
            Start-Sleep -Milliseconds 10
        }
    }
    Write-Host ""  # newline after progress

    # CRC32: uint32 LE
    $crcBytes = [System.BitConverter]::GetBytes([uint32]$crc)
    $port.Write($crcBytes, 0, 4)

    Write-Host "[XFER] Transfer complete. Waiting for verification..." -ForegroundColor Yellow

    # ---- Step 7: Wait for OK! ----
    $verified = $false
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        $line = Read-Line
        if ($line -ne $null) {
            Write-Host "  < $line" -ForegroundColor DarkGray
            if ($line -match "OK!") {
                $verified = $true
                Write-Host "[VERIFY] CRC verified!" -ForegroundColor Green
            }
            if ($line -match "BOOT") {
                Write-Host "[BOOT] Device is booting new kernel!" -ForegroundColor Green
                break
            }
            if ($line -match "ERR") {
                Write-Host "[ERROR] $line" -ForegroundColor Red
                exit 1
            }
            if ($line -match "WARN") {
                Write-Host "[WARN] $line" -ForegroundColor Yellow
            }
        }
    }

    if (-not $verified) {
        Write-Host "[ERROR] Did not receive verification." -ForegroundColor Red
        exit 1
    }

    Write-Host ""
    if ($Flash) {
        Write-Host "[DONE] Kernel flashed to SD card. Device is rebooting." -ForegroundColor Green
    } else {
        Write-Host "[DONE] Kernel chain-loaded into RAM. (Not persisted to SD)" -ForegroundColor Green
        Write-Host "       Use -Flash to write permanently." -ForegroundColor DarkGray
    }
}
finally {
    if ($port.IsOpen) { $port.Close() }
}
