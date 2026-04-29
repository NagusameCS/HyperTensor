# Poll the remote 8B GRC k=256 run.  Idempotent.
$ErrorActionPreference = 'SilentlyContinue'
$bash = @'
LOG=/root/HyperTensor/grc_8b_k256.log
echo "=== $(date -u) ==="
ps -eo pid,etime,pcpu,rss,stat,cmd | grep -E "[g]eodessical /root/models/llama31-8b" | head -3
echo "--- log lines: $(wc -l < $LOG) ---"
tail -3 $LOG
LAYERS=$(grep -c "running weight-PCA eigvec" $LOG)
echo "Layers started: $LAYERS / 32"
'@
ssh -T -o ConnectTimeout=20 -o ServerAliveInterval=8 ssh.opencs.dev $bash
