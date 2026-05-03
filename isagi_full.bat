@echo off
REM ── ISAGI Full Launcher (32B, for EC2 L40S 46GB) ──
REM Deploy to EC2: scp scripts/isagi_chat.py ubuntu@100.30.187.224:/tmp/
REM Then SSH: ssh ubuntu@100.30.187.224 "~/venv/bin/python /tmp/isagi_chat.py"
cd /d %~dp0..
echo ============================================
echo   ISAGI v1.0 — FULL (32B)
echo   Model: Qwen2.5-32B-Instruct (4-bit NF4)
echo   Requires: 20GB+ VRAM (EC2 L40S)
echo ============================================
.venv\Scripts\python scripts\isagi_chat.py --model Qwen/Qwen2.5-32B-Instruct --4bit %*
