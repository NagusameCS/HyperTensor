#!/bin/sh
/tmp/llm_test_runner_70b /root/models/llama31-70b-iq2xs.gguf -p 'The capital of France is' -n 5 --temp 0 > /tmp/test70b_v3.txt 2>&1
echo "EXIT=$?" >> /tmp/test70b_v3.txt
