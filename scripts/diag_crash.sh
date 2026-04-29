tail -20 /root/HyperTensor/grc_8b_k256.log
echo ---DMESG---
dmesg | tail -20
echo ---COREDUMP---
coredumpctl list --no-pager 2>/dev/null | tail -5