#!/usr/bin/env bash
# start_rerun.sh — launch compress arm rerun detached
export GPU_NAME=A10G
export MODEL_PATH=/opt/hypertensor/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
export OUT_DIR=/opt/hypertensor/results_paperB_A10G
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/opt/hypertensor
export GD_CUDA_KERNELS_PATH=/opt/hypertensor/cuda_kernels.so
nohup bash /tmp/rerun_compress_arms.sh > /tmp/paperB_rerun.log 2>&1 &
echo "launched pid=$!"
