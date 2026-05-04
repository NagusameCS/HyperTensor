@echo off
REM ISAGI Local with Streaming --- Qwen2.5-7B 4-bit on RTX 4070 Laptop
cd /d "%~dp0"
echo.
echo   ISAGI v1.0 --- The Adaptive Living Model
echo   Streaming mode ^| Qwen2.5-7B-Instruct 4-bit ^| RTX 4070
echo   Stack: GTC + OTT + GRC + UGT + Safe OGD + Snipe + COG
echo.
.venv\Scripts\python scripts\isagi_chat.py --model Qwen/Qwen2.5-7B-Instruct --4bit --stream --max-tokens 300 %*
