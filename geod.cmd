@echo off
:: geod.cmd — Windows cmd shim for the geod PowerShell launcher
:: Usage: geod <command> [model] [flags]
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0geod.ps1" %*
