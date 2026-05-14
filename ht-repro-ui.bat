@echo off
title ht-repro
echo.
echo   ht-repro — http://localhost:8765
echo   Press Ctrl+C to stop
echo.
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -m ht_repro serve
) else (
    python -m ht_repro serve
)
pause
