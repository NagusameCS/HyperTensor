#!/usr/bin/env python3
"""
Phase 3: Twin Training --- continued pretraining of SmolLM2-135M on pure data.

CRITICAL DESIGN FOR CECI:
1. Both models start from the EXACT SAME base weights (SmolLM2-135M)
   -> This is STRONGER than "same seed" --- the weights are literally identical
2. SHARED tokenizer (SmolLM2/Llama-3) for splice compatibility
3. CONTINUED PRETRAINING (not from scratch) on pure skill-specific data
   -> Each model diverges from the shared base, specializing in its domain
   -> Needs ~100M tokens per model (vs 3.3B from scratch)
4. Saves in safetensors format for CECI protocol loading
5. Optimized for RTX 4070 8GB VRAM (SmolLM2-135M ~270MB in bf16)

Usage:
  python scripts/train_pure_model.py --skill math --max-steps 30000
  python scripts/train_pure_model.py --skill language --max-steps 30000

Both commands load the same SmolLM2-135M base, guaranteeing shared init.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset


# ===========================================================================
# Configuration
# ===========================================================================

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
# Alternative if above fails: "HuggingFaceTB/SmolLM2-135M-Instruct"
OUTPUT_ROOT = Path("outputs/pure_models")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Training hyperparams optimized for continued pretraining on RTX 4070 8GB
TRAINING_ARGS = {
    "per_device_train_batch_size": 8,        # Bumped --- 39% VRAM used at batch=4
    "gradient_accumulation_steps": 2,        # Effective batch = 16 (enough for 135M)
    "learning_rate": 1e-5,                  # Lower LR for continued pretraining
    "warmup_steps": 200,
    "save_steps": 1000,
    "eval_steps": 500,
    "logging_steps": 25,                     # More frequent logging
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "optim": "adamw_8bit",
    "bf16": True,
    "gradient_checkpointing": True,
    "max_grad_norm": 1.0,
    "save_total_limit": 3,
    "dataloader_num_workers": 4,            # More workers for data loading
    "dataloader_prefetch_factor": 2,         # Prefetch batches
    "report_to": "none",
}


# ===========================================================================
# Pure Text Dataset
# ===========================================================================

class PureTextDataset(Dataset):
    """Simple dataset wrapping tokenized texts from JSONL files."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load text data
        self.texts = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        text = item.get("text", "")
                        if text and len(text) > 50:
                            self.texts.append(text)
                    except json.JSONDecodeError:
                        continue
        
        print(f"Loaded {len(self.texts)} texts from {data_path}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize with truncation
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,  # Return lists, not tensors
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }


@dataclass
class PureDataCollator:
    """Collator that pads to max length in batch."""
    tokenizer: any
    max_length: int = 2048
    
    def __call__(self, batch: list[dict]) -> dict:
        # Pad sequences to same length
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        
        # Dynamic padding: pad to longest in batch (up to max_length)
        max_len = min(max(len(ids) for ids in input_ids), self.max_length)
        
        padded_ids = []
        padded_masks = []
        labels = []
        
        for ids, mask in zip(input_ids, attention_masks):
            # Truncate if needed
            ids = ids[:max_len]
            mask = mask[:max_len]
            
            # Pad
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.tokenizer.pad_token_id or 0] * pad_len)
            padded_masks.append(mask + [0] * pad_len)
            # Labels: same as input_ids, with padding ignored (-100)
            labels.append(ids + [-100] * pad_len)
        
        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ===========================================================================
# Training function
# ===========================================================================

def train_pure_model(
    skill: str,
    seed: int = 42,
    data_path: Optional[str] = None,
    max_steps: int = 50000,
    resume_from: Optional[str] = None,
):
    """
    Train a pure single-skill model from scratch.
    
    Args:
        skill: "math" or "language"
        seed: Random seed for initialization (use same for both models!)
        data_path: Path to JSONL training data
        max_steps: Maximum training steps
        resume_from: Checkpoint to resume from (optional)
    """
    
    assert skill in ("math", "language"), f"Unknown skill: {skill}"
    
    # Set seed BEFORE model creation for identical initialization
    set_seed(seed)
    
    output_name = f"smollm2-135m-{skill}-pure"
    output_dir = OUTPUT_ROOT / output_name
    
    print("=" * 70)
    print(f"PURE MODEL TRAINING: {skill.upper()}")
    print(f"  Seed: {seed} (for identical init with sibling model)")
    print(f"  Output: {output_dir}")
    print(f"  Max steps: {max_steps}")
    print("=" * 70)
    
    # Determine data path
    if data_path is None:
        data_path = f"data/pure_{skill}/{skill}_pure_train.jsonl"
    
    if not os.path.exists(data_path):
        print(f"\n[!] Data file not found: {data_path}")
        print("[!] Run scripts/prepare_pure_{skill}.py first to build the corpus.")
        print("[!] Falling back to direct dataset loading...")
        return _train_from_hf_datasets(skill, seed, output_dir, max_steps, resume_from)
    
    # === Step 1: Load tokenizer ===
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, 
        trust_remote_code=True,
        use_fast=True,
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    
    # === Step 2: Load pretrained model (continued pretraining) ===
    print("\n[2/4] Loading pretrained SmolLM2-135M base model...")
    
    # Load the PRETRAINED model --- both Model M and Model L start from
    # the exact same weights, guaranteeing identical initialization
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16 if TRAINING_ARGS["bf16"] else torch.float32,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Enable gradient checkpointing
    if TRAINING_ARGS["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Seed: {seed} (identical init enabled)")
    
    # === Step 3: Load dataset ===
    print(f"\n[3/4] Loading dataset from {data_path}...")
    dataset = PureTextDataset(data_path, tokenizer, max_length=2048)
    print(f"  Samples: {len(dataset)}")
    
    # Split 95/5 for train/eval
    split_idx = int(0.95 * len(dataset))
    train_dataset = torch.utils.data.Subset(dataset, range(split_idx))
    eval_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # === Step 4: Train ===
    print("\n[4/4] Starting training...")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        **{k: v for k, v in TRAINING_ARGS.items() if k != "gradient_checkpointing"},
        max_steps=max_steps,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        dataloader_pin_memory=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=PureDataCollator(tokenizer, max_length=2048),
    )
    
    # Resume if requested
    if resume_from:
        print(f"  Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()
    
    # === Save final model ===
    print("\nSaving final model...")
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE: {skill.upper()}")
    print(f"  Model: {final_path}")
    print(f"  Ready for CECI splice protocol")
    print(f"{'='*70}")
    
    return str(final_path)


def _train_from_hf_datasets(skill, seed, output_dir, max_steps, resume_from):
    """Fallback: train using direct HuggingFace dataset loading."""
    print("  Using HuggingFace datasets as fallback...")
    
    set_seed(seed)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    
    # Load PRETRAINED model (continued pretraining from base)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16 if TRAINING_ARGS["bf16"] else torch.float32,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    if TRAINING_ARGS["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Load HF datasets directly
    if skill == "math":
        ds_names = [
            ("gsm8k", "main", "train"),
        ]
    else:
        ds_names = [
            ("wikitext", "wikitext-103-raw-v1", "train"),
        ]
    
    all_texts = []
    for ds_path, ds_config, ds_split in ds_names:
        try:
            ds = load_dataset(ds_path, ds_config, split=ds_split)
            for row in ds:
                text = row.get("text", "") or row.get("question", "") or ""
                if text and len(text) > 50:
                    all_texts.append(text)
                if len(all_texts) >= 10000:
                    break
        except Exception as e:
            print(f"  {ds_path}: {e}")
    
    print(f"  Loaded {len(all_texts)} texts from HF")
    
    class SimpleDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len=2048):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            tokens = self.tokenizer(self.texts[idx], truncation=True, 
                                     max_length=self.max_len, padding=False)
            return {"input_ids": tokens["input_ids"], 
                    "attention_mask": tokens["attention_mask"]}
    
    dataset = SimpleDataset(all_texts, tokenizer)
    split_idx = int(0.95 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(split_idx))
    eval_ds = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=min(max_steps, 5000),  # Shorter for fallback
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        warmup_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        save_steps=1000,
        logging_steps=50,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        save_total_limit=2,
        report_to="none",
        dataloader_pin_memory=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=PureDataCollator(tokenizer),
    )
    
    trainer.train(resume_from_checkpoint=resume_from if resume_from else False)
    
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    print(f"\nTraining complete: {final_path}")
    return str(final_path)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train a pure single-skill SmolLM2-135M model for CECI splicing"
    )
    parser.add_argument(
        "--skill", type=str, required=True, choices=["math", "language"],
        help="Which skill to train (math or language)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (use SAME seed for both models for identical init!)"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to JSONL training data"
    )
    parser.add_argument(
        "--max-steps", type=int, default=50000,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from checkpoint"
    )
    
    args = parser.parse_args()
    
    # Verify CUDA
    if not torch.cuda.is_available():
        print("\n[WARNING] CUDA not available --- training on CPU will be SLOW.")
        print("  SmolLM2-135M on CPU: ~2-5 steps/sec (expect ~7 hours for 50K steps)")
        print("  Consider installing CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu126")
        response = input("  Continue on CPU? [y/N]: ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name} ({vram:.1f} GB VRAM)")
    
    result = train_pure_model(
        skill=args.skill,
        seed=args.seed,
        data_path=args.data,
        max_steps=args.max_steps,
        resume_from=args.resume,
    )
    
    print(f"\nModel saved to: {result}")
    print("Run the geometric pre-checks next (scripts/verify_pure_model.py)")


if __name__ == "__main__":
    main()
