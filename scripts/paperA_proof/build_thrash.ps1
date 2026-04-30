# scripts/paperA_proof/build_thrash.ps1
# Compile the L2 thrash co-tenant kernel for the local GPU.
[CmdletBinding()]
param(
    [string]$Out = "C:\Users\legom\HyperTensor\scripts\paperA_proof\l2_thrash.exe",
    [string]$Arch = "sm_89"   # Ada / RTX 4070 Laptop
)
$ErrorActionPreference = "Stop"
$src = "C:\Users\legom\HyperTensor\scripts\paperA_proof\l2_thrash.cu"
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc not on PATH. Source CUDA toolkit env first."
}
Write-Host "[build_thrash] nvcc -O3 -arch=$Arch -o $Out $src"
& nvcc -O3 -arch=$Arch -o $Out $src
if ($LASTEXITCODE -ne 0) { throw "nvcc failed (exit $LASTEXITCODE)" }
Write-Host "[build_thrash] OK -> $Out"
