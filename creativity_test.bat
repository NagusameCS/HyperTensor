@echo off
REM -- MIKU Creativity Benchmark Launcher --
REM Runs the 5-dimension creativity test on the local model

cd /d %~dp0..
echo ============================================
echo   MIKU Creativity Benchmark (MCB v1)
echo   5 Dimensions · 10 samples × 5 items
echo ============================================

.venv\Scripts\python scripts\creativity_benchmark.py --4bit %*
