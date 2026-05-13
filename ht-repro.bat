@echo off
REM ht-repro — HyperTensor Reproduction CLI (Windows launcher)
REM Run from anywhere: ht-repro smoke, ht-repro all-t1, ht-repro list
python "%~dp0scripts\ht_repro.py" %*
