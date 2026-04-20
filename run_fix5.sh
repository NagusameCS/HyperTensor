#!/bin/bash
pkill -f 'llama31-70b' 2>/dev/null
sleep 1
nohup timeout 600 /root/geodessical /root/models/llama31-70b-iq2xs.gguf -p 'The capital of France is' -n 5 --temp 0 </dev/null >/tmp/test70b_fix5.txt 2>&1 &
disown
echo "LAUNCHED_PID=$!"
