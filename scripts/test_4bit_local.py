"""Quick test: load Qwen2.5-7B-Instruct with 4-bit quantization locally."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("[1/3] Loading Qwen2.5-7B-Instruct with 4-bit NF4...")
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

vram = torch.cuda.memory_allocated() / 1e9
print(f"[2/3] Model loaded: {vram:.1f}GB VRAM, d={model.config.hidden_size}, layers={model.config.num_hidden_layers}")

# Quick generation test
msg = [{"role": "user", "content": "Say hello in exactly 10 words."}]
prompt = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
enc = tok(prompt, return_tensors="pt").to("cuda")
out = model.generate(**enc, max_new_tokens=20, do_sample=True, temperature=0.7)
resp = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
print(f"[3/3] Test response: {resp.strip()}")
print(f"VRAM peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
print("SUCCESS - 7B model runs locally!")
