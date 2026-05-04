@echo off
REM ISAGI DUEL — Quick local test (10 turns, math proof)
cd /d "%~dp0"
echo.
echo   ISAGI DUEL v1.0 — Dual Agent Collaboration
echo   Task: math_proof ^| Turns: 10 ^| 4-bit
echo.
.venv\Scripts\python scripts\isagi_duel.py --task math_proof --turns 10 --4bit --save duel_outputs\duel_math_quick.json %*
