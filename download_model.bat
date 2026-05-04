@echo off
REM -- Download Qwen2.5-7B-Instruct for local use --
REM Set HF_TOKEN environment variable for faster downloads
REM huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir .cache\huggingface\hub\models--Qwen--Qwen2.5-7B-Instruct

echo ============================================
echo   Downloading Qwen2.5-7B-Instruct (~15GB)
echo   For faster downloads, set HF_TOKEN first:
echo     set HF_TOKEN=hf_your_token_here
echo ============================================

cd /d %~dp0..
.venv\Scripts\python -c "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig; import torch; print('Starting download...'); bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4'); m=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', quantization_config=bnb, device_map='auto', trust_remote_code=True); t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct'); print(f'SUCCESS: {torch.cuda.memory_allocated()/1e9:.1f}GB VRAM'); print('Model cached locally. Run chat.bat to start.')"
