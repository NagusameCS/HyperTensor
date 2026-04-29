cd /root/HyperTensor
nohup bash -c '
echo === BASELINE ===
timeout 60 ./geodessical /root/models/smollm2-135m-instruct-q8_0.gguf -p "the quick brown fox jumps over" -n 16 --temp 0.7 2>&1
echo === GRC k=512 ===
timeout 120 ./geodessical /root/models/smollm2-135m-instruct-q8_0.gguf -p "the quick brown fox jumps over" -n 16 --temp 0.7 --axex-compress --axex-compress-rank 512 2>&1
' > smollm_3050_combined.log 2>&1 </dev/null >/dev/null 2>&1 &
disown
echo "started pid=$!"