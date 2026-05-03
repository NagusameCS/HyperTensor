@echo off
REM ── HyperChat Local Launcher ──
REM Launches the HyperTensor living model on local RTX 4070 (8GB VRAM)
REM Uses 4-bit NF4 quantization to fit the 7B model in 8GB

cd /d %~dp0..
echo ============================================
echo   HyperChat — Local 4-bit Launcher
echo   Model: Qwen2.5-7B-Instruct (NF4 4-bit)
echo   Stack: UGT + Safe OGD + Snipe + COG+TEH
echo ============================================

.venv\Scripts\python scripts\hyper_chat.py --4bit %*
