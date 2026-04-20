#!/bin/bash
pkill -f gd70b 2>/dev/null
sleep 1
setsid /root/gd70b /root/models/llama31-70b-iq2xs.gguf -p 'The capital of France is' -n 5 --temp 0 > /tmp/test70b_v2.txt 2>&1 &
echo "LAUNCHED_PID=$!"
