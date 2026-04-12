# HyperTensor - Host-Mode Build Script
# Builds the HyperTensor inference runtime for Windows x86_64
#
# Requirements: zig (0.15+)
# Usage: .\build_host.ps1 [-Run] [-Clean]

param(
    [switch]$Run,
    [switch]$Clean,
    [switch]$Cuda,
    [string]$Model
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$BUILD = "build_host"
$OUT = "$BUILD\hypertensor.exe"

$CFLAGS = @(
    "-target", "x86_64-windows-gnu",
    "-O2",
    "-msse2", "-mavx2", "-mfma",
    "-DHYPERTENSOR_HOSTED=1",
    "-Ihost/shims",
    "-I.",
    "-Ihost",
    "-Wno-unused-function", "-Wno-unused-variable", "-Wno-format",
    "-Wno-incompatible-pointer-types", "-Wno-int-conversion",
    "-Wno-sign-compare", "-Wno-missing-field-initializers",
    "-Wno-unused-parameter"
)

$SOURCES = @(
    "host/hal.c",
    "host/main.c",
    "host/api_server.c",
    "runtime/nn/llm.c",
    "runtime/nn/gguf.c",
    "runtime/nn/backend.c",
    "runtime/nn/model_meta.c",
    "runtime/nn/tensor_bridge.c",
    "runtime/jit/x86_jit.c",
    "runtime/jit/llm_jit.c"
)

$LDFLAGS = @(
    "-ladvapi32",
    "-lws2_32"
)

# Optional CUDA backend
if ($Cuda) {
    $CFLAGS += "-DENABLE_CUDA"
    $SOURCES += "runtime/nn/backend_cuda.c"
    Write-Host '  CUDA backend enabled' -ForegroundColor Yellow
}

# Clean
if ($Clean) {
    Write-Host 'Cleaning build_host...' -ForegroundColor Yellow
    if (Test-Path $BUILD) { Remove-Item -Recurse -Force $BUILD }
}

# Build
if (!(Test-Path $BUILD)) { New-Item -ItemType Directory -Path $BUILD | Out-Null }

Write-Host ''
Write-Host '  HyperTensor Build System' -ForegroundColor Cyan
Write-Host '  Host-mode inference runtime (x86_64)' -ForegroundColor Cyan
Write-Host ''

$sw = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host ('Compiling {0} sources...' -f $SOURCES.Length) -ForegroundColor Green

$args_list = @('cc') + $CFLAGS + $SOURCES + @('-o', $OUT) + $LDFLAGS

Write-Host ('zig ' + ($args_list -join ' ')) -ForegroundColor DarkGray

& zig @args_list 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host ('BUILD FAILED (exit code {0})' -f $LASTEXITCODE) -ForegroundColor Red
    exit 1
}

$sw.Stop()
$size = (Get-Item $OUT).Length
$sizeKB = [math]::Round($size / 1024, 1)

Write-Host ''
Write-Host 'BUILD SUCCESS' -ForegroundColor Green
Write-Host ('Output: {0} ({1} KB)' -f $OUT, $sizeKB) -ForegroundColor Green
Write-Host ('Time: {0:F1}s' -f $sw.Elapsed.TotalSeconds) -ForegroundColor Green
Write-Host ''

# Run
if ($Run) {
    if (!$Model) {
        Write-Host 'No model specified. Usage: .\build_host.ps1 -Run -Model path\to\model.gguf' -ForegroundColor Yellow
        exit 0
    }
    Write-Host ('Running: {0} {1}' -f $OUT, $Model) -ForegroundColor Cyan
    & $OUT $Model
}
