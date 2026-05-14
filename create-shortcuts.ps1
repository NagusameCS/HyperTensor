# Create a desktop shortcut for ht-repro UI
$shortcutPath = [Environment]::GetFolderPath("Desktop") + "\ht-repro UI.lnk"
$targetPath = Join-Path (Get-Location) "ht-repro-ui.bat"
$iconPath = Join-Path (Get-Location) ".venv\Scripts\python.exe"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $targetPath
$Shortcut.WorkingDirectory = (Get-Location).Path
$Shortcut.WindowStyle = 7  # Minimized
$Shortcut.Description = "HyperTensor Reproduction Dashboard"
$Shortcut.IconLocation = "$iconPath,0"

# Also create in Start Menu
$startMenu = [Environment]::GetFolderPath("Programs") + "\HyperTensor"
if (!(Test-Path $startMenu)) { New-Item -ItemType Directory -Path $startMenu -Force }
$startShortcut = Join-Path $startMenu "ht-repro UI.lnk"
$Shortcut2 = $WshShell.CreateShortcut($startShortcut)
$Shortcut2.TargetPath = $targetPath
$Shortcut2.WorkingDirectory = (Get-Location).Path
$Shortcut2.WindowStyle = 7
$Shortcut2.Description = "HyperTensor Reproduction Dashboard"
$Shortcut2.IconLocation = "$iconPath,0"
$Shortcut.Save()
$Shortcut2.Save()

Write-Host "Desktop shortcut created: $shortcutPath" -ForegroundColor Green
Write-Host "Start Menu shortcut created: $startShortcut" -ForegroundColor Green
Write-Host "Double-click ht-repro UI on your desktop to start." -ForegroundColor White
