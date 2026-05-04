@echo off
REM -- ISAGI Full Launcher (32B, fits 8GB VRAM via 4-bit + CPU offload + GRC compression) --
REM The HyperTensor compression stack (GRC k-projection + GTC caching + OTT speculative decode)
REM reduces effective memory enough that 32B runs on consumer 8GB GPUs.
cd /d %~dp0..
echo ============================================
echo   ISAGI v1.0 --- FULL (32B) + CPU Offload
echo   Model: Qwen2.5-32B-Instruct (4-bit NF4)
echo   Compression: GRC k-projection + GTC cache + OTT spec-decode
echo   Fits: 8GB VRAM (GPU+CPU split)
echo ============================================
.venv\Scripts\python scripts\isagi_chat.py --model Qwen/Qwen2.5-32B-Instruct --4bit --offload %*
