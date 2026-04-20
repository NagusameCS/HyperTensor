#!/bin/bash
GD_CUDA_KERNELS_PATH=/tmp/cuda_kernels_batched.so
OPENBLAS_NUM_THREADS=8
export GD_CUDA_KERNELS_PATH OPENBLAS_NUM_THREADS
/tmp/geodessical.batched /root/models/llama31-8b-q8_0.gguf --axiom-skip-geodesic --axex-ffn-compress --axex-compress-rank 256 -p "The capital of France is" -n 20 > /tmp/test_r256.txt 2>&1
echo "DONE_RC=$?" >> /tmp/test_r256.txt
