#!/bin/bash
while ! grep -q 'DONE' ~/train_qwen_math.log 2>/dev/null; do sleep 30; done
echo 'MATH DONE, starting LANGUAGE'
/home/ubuntu/venv/bin/python3 -u ~/train_qwen_pure.py --skill language --steps 4000 > ~/train_qwen_lang.log 2>&1
echo 'LANG DONE, starting CECI'
/home/ubuntu/venv/bin/python3 -u ~/ceci_qwen.py > ~/ceci_qwen.log 2>&1
echo 'ALL DONE'
