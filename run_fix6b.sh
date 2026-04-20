#!/bin/bash
# Kill old runs
pkill -9 -f 'llama31-70b' 2>/dev/null
sleep 2
# Launch test fully detached
setsid timeout 700 /root/geodessical /root/models/llama31-70b-iq2xs.gguf \
    -p 'The capital of France is' -n 5 --temp 0 \
    </dev/null >/tmp/test70b_fix6.txt 2>&1
echo "TEST_EXIT=$?" >> /tmp/test70b_fix6.txt
