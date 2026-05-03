#!/usr/bin/env bash
set -x
ps -ef | grep -E 'sleep|shutdown' | grep -v grep
sudo pkill -9 -f 'sleep 1800' || true
sudo pkill -9 -f 'sleep [0-9]*' || true
sleep 1
ps -ef | grep -E 'sleep|shutdown' | grep -v grep || echo all-clear
sudo bash -c 'nohup bash -c "sleep 5400; shutdown -h now" >/dev/null 2>&1 &'
echo "new-90min-cap-set"
ps -ef | grep -E 'sleep|shutdown' | grep -v grep
