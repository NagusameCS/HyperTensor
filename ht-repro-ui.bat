@echo off
title ht-repro Dashboard
echo Starting ht-repro web UI...
echo.

REM Activate venv if present, otherwise use system Python
if exist "%~dp0.venv\Scripts\python.exe" (
    set PYTHON=%~dp0.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

REM Start the server
%PYTHON% -m ht_repro serve
pause
