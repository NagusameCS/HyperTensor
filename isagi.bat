@echo off
REM ── ISAGI Local Launcher (7B, fits RTX 4070 8GB) ──
cd /d %~dp0..
echo ============================================
echo   ISAGI v1.0 — The Adaptive Living Model
echo   Model: Qwen2.5-7B-Instruct (4-bit NF4)
echo   Stack: GTC + OTT + GRC + UGT + Safe OGD + Snipe + COG+TEH
echo ============================================
.venv\Scripts\python scripts\isagi_chat.py --model Qwen/Qwen2.5-7B-Instruct --4bit %*
