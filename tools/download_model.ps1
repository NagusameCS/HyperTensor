# TensorOS Model Download Script
# Downloads quantized GGUF models from HuggingFace for LLM inference
#
# Usage:
#   .\tools\download_model.ps1 -Model qwen2.5-0.5b
#   .\tools\download_model.ps1 -Model smollm2-135m
#   .\tools\download_model.ps1 -Model tinyllama
#   .\tools\download_model.ps1 -Model gemma-2-2b
#   .\tools\download_model.ps1 -List
#   .\tools\download_model.ps1 -Url "https://..." -Output models/custom.gguf

param(
    [string]$Model,
    [string]$Url,
    [string]$Output,
    [switch]$List
)

$ModelsDir = Join-Path $PSScriptRoot "..\models"
if (-not (Test-Path $ModelsDir)) {
    New-Item -ItemType Directory $ModelsDir | Out-Null
}

# Model registry: name -> (URL, filename, description, size)
$Registry = @{
    "qwen2.5-0.5b" = @{
        Url  = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf"
        File = "qwen2.5-0.5b-instruct-q4_0.gguf"
        Desc = "Qwen2.5-0.5B-Instruct Q4_0 — Best math for size, 494M params"
        Size = "352 MB"
    }
    "qwen2.5-0.5b-q8" = @{
        Url  = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf"
        File = "qwen2.5-0.5b-instruct-q8_0.gguf"
        Desc = "Qwen2.5-0.5B-Instruct Q8_0 — Higher quality, slower"
        Size = "531 MB"
    }
    "smollm2-135m" = @{
        Url  = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf"
        File = "smollm2-135m-instruct-q8_0.gguf"
        Desc = "SmolLM2-135M-Instruct Q8_0 — Fastest, tiny model"
        Size = "145 MB"
    }
    "smollm2-360m" = @{
        Url  = "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf"
        File = "smollm2-360m-instruct-q4km.gguf"
        Desc = "SmolLM2-360M-Instruct Q4_K_M — Good balance"
        Size = "230 MB"
    }
    "tinyllama" = @{
        Url  = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
        File = "tinyllama-1.1b-chat-q4_0.gguf"
        Desc = "TinyLlama-1.1B-Chat Q4_0 — Classic small LLM"
        Size = "600 MB"
    }
    "gemma-2-2b" = @{
        Url  = "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_0.gguf"
        File = "gemma-2-2b-it-q4_0.gguf"
        Desc = "Gemma-2-2B-IT Q4_0 — Google's model, strong math"
        Size = "1.4 GB"
    }
}

if ($List) {
    Write-Host "`nAvailable Models for TensorOS:" -ForegroundColor Cyan
    Write-Host "=" * 65
    foreach ($key in $Registry.Keys | Sort-Object) {
        $m = $Registry[$key]
        Write-Host "  $key" -ForegroundColor Yellow -NoNewline
        Write-Host "  ($($m.Size))" -ForegroundColor DarkGray
        Write-Host "    $($m.Desc)"
    }
    Write-Host "`nUsage: .\tools\download_model.ps1 -Model <name>" -ForegroundColor Green
    Write-Host "Models are saved to: models\*.gguf" -ForegroundColor DarkGray
    Write-Host "The build script auto-detects them when using -Run`n" -ForegroundColor DarkGray
    exit 0
}

if ($Url) {
    if (-not $Output) {
        $Output = Join-Path $ModelsDir ($Url -split '/')[-1]
    }
    Write-Host "Downloading: $Url" -ForegroundColor Cyan
    Write-Host "Target: $Output" -ForegroundColor DarkGray
    Invoke-WebRequest -Uri $Url -OutFile $Output -UseBasicParsing
    $sz = (Get-Item $Output).Length
    Write-Host "Done: $([math]::Round($sz / 1MB)) MB" -ForegroundColor Green
    exit 0
}

if (-not $Model) {
    Write-Host "Usage: .\tools\download_model.ps1 -Model <name>" -ForegroundColor Yellow
    Write-Host "       .\tools\download_model.ps1 -List" -ForegroundColor Yellow
    exit 1
}

$key = $Model.ToLower()
if (-not $Registry.ContainsKey($key)) {
    Write-Host "Unknown model: $Model" -ForegroundColor Red
    Write-Host "Use -List to see available models" -ForegroundColor Yellow
    exit 1
}

$info = $Registry[$key]
$target = Join-Path $ModelsDir $info.File

if (Test-Path $target) {
    $sz = (Get-Item $target).Length
    Write-Host "Model already downloaded: $($info.File) ($([math]::Round($sz / 1MB)) MB)" -ForegroundColor Green
    exit 0
}

Write-Host "`n=== Downloading $($info.Desc) ===" -ForegroundColor Cyan
Write-Host "  URL: $($info.Url)" -ForegroundColor DarkGray
Write-Host "  Target: $target" -ForegroundColor DarkGray
Write-Host "  Expected size: $($info.Size)" -ForegroundColor DarkGray
Write-Host ""

try {
    $ProgressPreference = 'SilentlyContinue'  # Speed up download
    Invoke-WebRequest -Uri $info.Url -OutFile $target -UseBasicParsing
    $sz = (Get-Item $target).Length
    Write-Host "`n=== Download complete: $([math]::Round($sz / 1MB)) MB ===" -ForegroundColor Green
    Write-Host "  Run: .\build.ps1 -Run" -ForegroundColor Yellow
    Write-Host "  The model will be auto-detected and loaded at boot.`n" -ForegroundColor DarkGray
} catch {
    Write-Host "`nDownload failed: $_" -ForegroundColor Red
    Write-Host "You can manually download from:" -ForegroundColor Yellow
    Write-Host "  $($info.Url)" -ForegroundColor DarkGray
    Write-Host "Save to: models\$($info.File)`n" -ForegroundColor DarkGray
    exit 1
}
