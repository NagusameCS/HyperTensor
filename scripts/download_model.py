"""Download models for ISAGI with HF_TOKEN."""
import os, torch
# HF_TOKEN already set in environment
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys

model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
use_4bit = "--4bit" in sys.argv or "32B" in model_id

print(f"Downloading {model_id}...")
print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN', ''))}")
print(f"HF_HOME: {os.environ.get('HF_HOME', 'default')}")

if use_4bit:
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )

tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
vram = torch.cuda.memory_allocated() / 1e9
print(f"READY: VRAM={vram:.1f}GB, d={model.config.hidden_size}, layers={model.config.num_hidden_layers}")
