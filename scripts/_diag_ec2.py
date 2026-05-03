import torch, subprocess, os
print("torch:", torch.__version__)
free, total = torch.cuda.mem_get_info()
print(f"VRAM free: {free/1e9:.1f}GB / {total/1e9:.1f}GB")
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
# Check disk
result = subprocess.run(["df", "-h", "/home/ubuntu"], capture_output=True, text=True)
print(result.stdout)
# Check for models
for d in ["/home/ubuntu/.cache/huggingface/hub", "/home/ubuntu/models"]:
    if os.path.exists(d):
        models = [m for m in os.listdir(d) if "smol" in m.lower() or "llama" in m.lower() or "qwen" in m.lower() or "gemma" in m.lower()]
        print(f"Models in {d}: {models}")
