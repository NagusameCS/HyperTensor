#!/usr/bin/env python3
"""ISAGI EC2 Setup: Download 32B model and verify."""
import os, torch
# Set your HF_TOKEN in environment before running:
#   export HF_TOKEN=hf_your_token_here
#   python isagi_ec2_setup.py

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print('[ISAGI-EC2] Downloading Qwen2.5-32B-Instruct...')
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-32B-Instruct',
    quantization_config=bnb,
    device_map='auto',
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B-Instruct')
vram = torch.cuda.memory_allocated() / 1e9
print(f'ISAGI-EC2-READY: VRAM={vram:.1f}GB, d={model.config.hidden_size}, layers={model.config.num_hidden_layers}')

# Quick personality test
msg = [{'role': 'user', 'content': 'ISAGI, can you solve the Riemann Hypothesis?'}]
prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
enc = tok(prompt, return_tensors='pt').to('cuda')
out = model.generate(**enc, max_new_tokens=100, do_sample=True, temperature=0.7)
resp = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
print(f'Test response: {resp[:300]}')
