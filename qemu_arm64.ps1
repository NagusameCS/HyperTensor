<#
.SYNOPSIS
    Run TensorOS ARM64 kernel in QEMU (Raspberry Pi 4B emulation)

.DESCRIPTION
    Emulates a Raspberry Pi 4B (BCM2711, Cortex-A72) with 2GB RAM.
    Serial output goes to console (mini UART on serial1).
    
    QEMU 10.x with raspi4b machine support required.
    Install: winget install SoftwareFreedomConservancy.QEMU

.PARAMETER Debug
    Enable QEMU debug logging (exceptions, interrupts) to qemu_debug.log

.PARAMETER Gdb
    Start with GDB server on port 1234 and wait for connection.
    Connect with: gdb-multiarch -ex "target remote :1234" build_arm64/kernel_arm64.elf

.PARAMETER Build
    Build before running (calls build_rpi.ps1)

.PARAMETER Timeout
    Auto-kill QEMU after N seconds (default: 0 = run until Ctrl+C)

.EXAMPLE
    .\qemu_arm64.ps1              # Run interactively
    .\qemu_arm64.ps1 -Build       # Build first, then run
    .\qemu_arm64.ps1 -Debug       # Run with exception logging
    .\qemu_arm64.ps1 -Gdb         # Run with GDB server (paused at start)
    .\qemu_arm64.ps1 -Timeout 10  # Run for 10 seconds then stop
#>
param(
    [switch]$Debug,
    [switch]$Gdb,
    [switch]$Build,
    [int]$Timeout = 0
)

$ErrorActionPreference = "Stop"
$qemu = "C:\Program Files\qemu\qemu-system-aarch64.exe"
$kernel = "build_arm64\kernel8.img"

if (-not (Test-Path $qemu)) {
    Write-Error "QEMU not found at $qemu. Install: winget install SoftwareFreedomConservancy.QEMU"
    return
}

if ($Build) {
    Write-Host "=== Building ===" -ForegroundColor Cyan
    & .\build_rpi.ps1
    if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; return }
}

if (-not (Test-Path $kernel)) {
    Write-Error "Kernel not found: $kernel. Run .\build_rpi.ps1 first."
    return
}

# Base arguments
$args_list = @(
    "-M", "raspi4b"
    "-m", "2G"
    "-kernel", $kernel
    "-serial", "null"           # serial0 = PL011 (unused, suppress)
    "-serial", "mon:stdio"      # serial1 = mini UART → console
    "-display", "none"
    "-no-reboot"
)

if ($Debug) {
    $args_list += @("-d", "int,cpu_reset", "-D", "qemu_debug.log")
    Write-Host "Debug logging -> qemu_debug.log" -ForegroundColor Yellow
}

if ($Gdb) {
    $args_list += @("-s", "-S")
    Write-Host "GDB server on :1234 - waiting for connection..." -ForegroundColor Yellow
    Write-Host "  Connect: gdb-multiarch -ex 'target remote :1234' build_arm64/kernel_arm64.elf" -ForegroundColor DarkGray
}

Write-Host "=== QEMU raspi4b - TensorOS ARM64 ===" -ForegroundColor Green
Write-Host "    Kernel: $kernel ($(( Get-Item $kernel).Length) bytes)" -ForegroundColor DarkGray
Write-Host "    Press Ctrl+A then X to quit QEMU" -ForegroundColor DarkGray
Write-Host ""

if ($Timeout -gt 0) {
    # Background mode with timeout
    Remove-Item qemu_serial1.log -ErrorAction SilentlyContinue
    $args_timeout = $args_list -replace "mon:stdio", "file:qemu_serial1.log"
    $proc = Start-Process -FilePath $qemu -ArgumentList ($args_timeout -join " ") -WorkingDirectory $PWD -PassThru -NoNewWindow
    Write-Host "Running for $Timeout seconds (PID $($proc.Id))..." -ForegroundColor DarkGray
    Start-Sleep $Timeout
    if (-not $proc.HasExited) { Stop-Process $proc -Force; Start-Sleep 1 }
    Write-Host ""
    Write-Host "=== Serial Output ===" -ForegroundColor Cyan
    if (Test-Path qemu_serial1.log) {
        [System.IO.File]::ReadAllText((Resolve-Path qemu_serial1.log))
    } else {
        "No output captured"
    }
    if ($Debug -and (Test-Path qemu_debug.log)) {
        Write-Host "`n=== Debug Log ===" -ForegroundColor Yellow
        Get-Content qemu_debug.log
    }
} else {
    # Interactive mode - serial on stdio
    & $qemu @args_list
}
