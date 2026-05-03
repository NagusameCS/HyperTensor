import os
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B", dtype="bfloat16", device_map="auto")
path = os.path.abspath("outputs/pure_models/qwen2.5-1.5b-math-pure/final")
print(f"Path: {path}")
print(f"adapter_config: {os.path.exists(os.path.join(path, 'adapter_config.json'))}")
print(f"adapter_model: {os.path.exists(os.path.join(path, 'adapter_model.safetensors'))}")
m = PeftModel.from_pretrained(base, path)
m = m.merge_and_unload()
print("OK")
