#!/usr/bin/env bash
set -x
ps -ef | grep -E 'sleep [0-9]' | grep -v grep
sudo pkill -9 -f 'sleep 5400' || true
sudo pkill -9 -f 'sleep 9000' || true
sleep 1
sudo bash -c 'nohup bash -c "sleep 9000 ; shutdown -h now" >/dev/null 2>&1 &'
echo cap-extended-150min
sleep 1
ps -ef | grep -E 'sleep [0-9]' | grep -v grep
