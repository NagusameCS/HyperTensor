#!/bin/bash
# Kill all old processes
pkill -9 -f geodessical 2>/dev/null
pkill -9 -f gd_daemon 2>/dev/null
sleep 5

# Check VRAM
echo "VRAM after kill:"
nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "no apps"

# Wait for VRAM to free (max 60s)
for i in $(seq 1 12); do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    echo "VRAM_USED=$USED"
    if [ "${USED:-9999}" -lt 1000 ]; then
        echo "VRAM_FREE_OK"
        break
    fi
    sleep 5
done

# Run test
nohup timeout 600 /root/geodessical /root/models/llama31-70b-iq2xs.gguf \
    -p 'The capital of France is' -n 5 --temp 0 \
    </dev/null >/tmp/test70b_fix6.txt 2>&1
echo "TEST_EXIT=$?"
