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

# nvcc on Windows needs cl.exe (MSVC) on PATH. If it's not there, try to
# locate vcvars64.bat under the standard VS BuildTools install root and
# spawn the build inside a cmd shell that has sourced it.
$useVcvars = $false
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vcvars = Get-ChildItem 'C:\Program Files (x86)\Microsoft Visual Studio','C:\Program Files\Microsoft Visual Studio' `
              -Recurse -Filter vcvars64.bat -ErrorAction SilentlyContinue |
              Select-Object -First 1 -ExpandProperty FullName
    if (-not $vcvars) { throw "cl.exe not on PATH and vcvars64.bat not found. Install VS BuildTools or run from a Developer prompt." }
    $useVcvars = $true
    Write-Host "[build_thrash] using $vcvars"
}

Write-Host "[build_thrash] nvcc -O3 -arch=$Arch -o $Out $src"
if ($useVcvars) {
    $cmdLine = "`"$vcvars`" >nul 2>&1 && nvcc -O3 -arch=$Arch -o `"$Out`" `"$src`""
    & cmd /c $cmdLine
} else {
    & nvcc -O3 "-arch=$Arch" -o $Out $src
}
if ($LASTEXITCODE -ne 0) { throw "nvcc failed (exit $LASTEXITCODE)" }
Write-Host "[build_thrash] OK -> $Out"
