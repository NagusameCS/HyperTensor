# Fetch libopenblas.dll for the hypertensor_runtime native binary.
# OpenBLAS 0.3.28 — matches the .lib in host/shims/openblas.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root 'hypertensor_runtime\bin\win-amd64\libopenblas.dll'
if (Test-Path $dest) {
    $sz = (Get-Item $dest).Length
    if ($sz -gt 40MB) { Write-Host "OpenBLAS DLL already present ($sz bytes)"; exit 0 }
}
$url = 'https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.28/OpenBLAS-0.3.28-x64.zip'
$tmp = Join-Path $env:TEMP 'OpenBLAS-0.3.28-x64.zip'
$ext = Join-Path $env:TEMP 'OpenBLAS-0.3.28-extract'
Write-Host "Downloading $url ..."
Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $tmp
if (Test-Path $ext) { Remove-Item -Recurse -Force $ext }
Expand-Archive -Path $tmp -DestinationPath $ext -Force
Copy-Item (Join-Path $ext 'bin\libopenblas.dll') $dest -Force
Remove-Item $tmp -Force
Remove-Item -Recurse -Force $ext
Write-Host "Installed $dest ($((Get-Item $dest).Length) bytes)"
