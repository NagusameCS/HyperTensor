#!/bin/bash
pkill -f geodessical 2>/dev/null
sleep 1
nohup /root/geodessical /root/models/llama31-70b-iq2xs.gguf -p 'The capital of France is' -n 5 --temp 0 > /tmp/test70b_embd.txt 2>&1 &
disown $!
echo "Launched PID=$!"
