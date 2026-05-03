#!/usr/bin/env python3
"""
OLLAMA EXPORTER — Convert LoRA-trained pure models to Ollama GGUF.

Merges LoRA adapter into base SmolLM2-135M weights, exports as safetensors,
then creates Ollama Modelfile for deployment.

Usage:
  python scripts/export_to_ollama.py --model outputs/pure_models/smollm2-135m-math-pure/final --name math-pure
  python scripts/export_to_ollama.py --model outputs/pure_models/smollm2-135m-language-pure/final --name language-pure
"""

import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
OLLAMA_MODELS = Path(os.environ.get("OLLAMA_MODELS", 
                    os.path.expandvars("%USERPROFILE%\\.ollama\\models")))


def merge_and_export(adapter_path: str, output_dir: str, model_name: str):
    """Merge LoRA adapter into base, export as full model."""
    print("=" * 60)
    print(f"OLLAMA EXPORT: {model_name}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Load base model
    print("\n[1] Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    # Merge LoRA
    print("[2] Merging LoRA adapter...")
    is_lora = (Path(adapter_path) / "adapter_config.json").exists()
    if is_lora:
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
        print("  LoRA merged successfully")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path, dtype=torch.bfloat16, trust_remote_code=True
        )
        print("  Full model loaded (no LoRA merge needed)")
    
    # Save merged model
    print("[3] Saving merged model...")
    model.save_pretrained(str(out), safe_serialization=True)
    tokenizer.save_pretrained(str(out))
    
    # Create GGUF via llama.cpp convert script if available
    gguf_path = out / f"{model_name}.gguf"
    convert_script = Path(".").glob("**/convert-hf-to-gguf.py")
    convert_script = next(convert_script, None)
    
    if convert_script and convert_script.exists():
        print(f"[4] Converting to GGUF via {convert_script}...")
        subprocess.run([
            sys.executable, str(convert_script),
            str(out), "--outfile", str(gguf_path),
            "--outtype", "f16"
        ], check=False)
    else:
        print("[4] GGUF conversion script not found — skipping.")
        print("    Install: pip install gguf && python -m gguf.convert")
        gguf_path = None
    
    return out, gguf_path


def create_modelfile(model_name: str, gguf_path: Path = None, 
                     model_dir: Path = None, skill: str = "math"):
    """Create Ollama Modelfile for the exported model."""
    
    if skill == "math":
        system_prompt = (
            "You are a pure mathematical reasoning model. "
            "Answer questions with rigorous, step-by-step mathematical proofs. "
            "Use formal notation where appropriate. Do not use conversational language."
        )
    else:
        system_prompt = (
            "You are a pure language and prose model. "
            "Write fluently with proper grammar, rich vocabulary, and natural flow. "
            "Do not attempt mathematical reasoning or code generation."
        )
    
    modelfile = f"""# Ollama Modelfile for {model_name}
# HyperTensor Paper X — Pure Single-Skill Model

FROM {gguf_path or model_dir}

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# Template for SmolLM2/Llama-3 chat format
TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
\"\"\"
"""
    
    modelfile_path = Path(f"modelfiles/{model_name}.modelfile")
    modelfile_path.parent.mkdir(exist_ok=True)
    modelfile_path.write_text(modelfile)
    
    print(f"\n[5] Modelfile created: {modelfile_path}")
    return modelfile_path


def register_with_ollama(model_name: str, modelfile_path: Path):
    """Register the model with Ollama."""
    print(f"\n[6] Registering with Ollama...")
    
    # Check if ollama is available
    ollama_path = shutil.which("ollama")
    if not ollama_path:
        print("   Ollama not found in PATH. Install from https://ollama.com")
        print(f"  Manual registration: ollama create {model_name} -f {modelfile_path}")
        return False
    
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", str(modelfile_path)],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print(f"   Model '{model_name}' registered with Ollama!")
        print(f"  Run: ollama run {model_name}")
        return True
    else:
        print(f"   Registration failed: {result.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export LoRA model to Ollama")
    parser.add_argument("--model", required=True, help="Path to trained model (adapter or full)")
    parser.add_argument("--name", required=True, help="Ollama model name")
    parser.add_argument("--skill", default="math", choices=["math", "language"])
    parser.add_argument("--output", default=None, help="Output directory for merged model")
    args = parser.parse_args()
    
    output_dir = args.output or f"outputs/ollama/{args.name}"
    
    # Merge and export
    model_dir, gguf_path = merge_and_export(args.model, output_dir, args.name)
    
    # Create Modelfile
    modelfile = create_modelfile(args.name, gguf_path, model_dir, args.skill)
    
    # Register with Ollama
    register_with_ollama(args.name, modelfile)
    
    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETE: {args.name}")
    print(f"  Merged model: {model_dir}")
    if gguf_path:
        print(f"  GGUF: {gguf_path}")
    print(f"  Modelfile: {modelfile}")
    print(f"  Ollama: ollama run {args.name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
