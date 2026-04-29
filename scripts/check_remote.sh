ls -la /root/models/*.gguf 2>/dev/null
echo --- HyperTensor logs ---
ls -la /root/HyperTensor/*.log | tail -10
echo --- last lines ---
for f in /root/HyperTensor/smollm_3050_baseline.log; do
  if [ -f "$f" ]; then
    echo "=== $f ==="
    tail -8 "$f"
  fi
done