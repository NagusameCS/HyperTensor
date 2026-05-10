# scripts/poll_remote.cmd
@echo off
type %~dp0poll_remote.sh | ssh -T -o ConnectTimeout=20 -o ServerAliveInterval=8 ssh.opencs.dev "bash -s"
