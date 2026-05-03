cd /root/HyperTensor
# smollm baseline (non-greedy)
nohup bash -c 'timeout 120 ./geodessical /root/models/smollm2-135m-q8_0.gguf -p "the quick brown fox jumps over" -n 16 --temp 0.7 > smollm_3050_baseline.log 2>&1' </dev/null >/dev/null 2>&1 &
disown
echo "smollm baseline PID started"
sleep 1
ls -la /root/models/ | head -5