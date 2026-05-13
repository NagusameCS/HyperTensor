@echo off
REM ht-repro — HyperTensor Reproduction CLI (Windows launcher)
REM Run: ht-repro smoke | ht-repro all-t1 | ht-repro list
REM Install: pip install ht-repro   OR   pip install -e ht_repro/
python -m ht_repro %*
