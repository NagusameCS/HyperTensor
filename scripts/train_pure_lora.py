#!/usr/bin/env python3
"""
FAST TWIN TRAINING — LoRA continued pretraining of SmolLM2-135M on pure data.

WHY LORA:
1. 5-10x faster per step (only adapter weights have gradients)
2. Base weights stay IDENTICAL — both M and L share exact same base
3. LoRA adapters capture the skill specialization (math vs language)
4. For CECI splice: compute subspaces from base weights + merge adapters
5. RTX 4070: expect 8-12 steps/sec (vs 0.2 steps/sec with full training)

Usage:
  python scripts/train_pure_lora.py --skill math --steps 10000
  python scripts/train_pure_lora.py --skill language --steps 10000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"
OUTPUT_ROOT = Path("outputs/pure_models")

# LoRA config — targets all linear layers for full adaptation
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                        # Rank — small, fast, effective
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
)


class FastTextDataset(torch.utils.data.Dataset):
    """Pre-tokenized dataset for fast iteration."""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.max_length = max_length
        
        # Load texts
        texts = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    t = item.get("text", "")
                    if t and len(t) > 50:
                        texts.append(t)
                except json.JSONDecodeError:
                    continue
        
        # Pre-tokenize ALL texts upfront (no lazy loading)
        print(f"  Pre-tokenizing {len(texts)} texts...")
        self.input_ids = []
        self.attention_masks = []
        
        for i, text in enumerate(texts):
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            self.input_ids.append(tokens["input_ids"])
            self.attention_masks.append(tokens["attention_mask"])
            if (i + 1) % 10000 == 0:
                print(f"  ... {i+1}/{len(texts)} tokenized")
        
        print(f"  Tokenized {len(self.input_ids)} samples")
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
        }


def collate_batch(batch, tokenizer, max_length=2048):
    """Fast collation with dynamic padding."""
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    
    max_len = min(max(len(ids) for ids in input_ids), max_length)
    
    padded_ids = []
    padded_masks = []
    labels = []
    
    for ids, mask in zip(input_ids, attention_masks):
        ids = ids[:max_len]
        mask = mask[:max_len]
        pad_len = max_len - len(ids)
        
        pad_id = tokenizer.pad_token_id or 0
        padded_ids.append(ids + [pad_id] * pad_len)
        padded_masks.append(mask + [0] * pad_len)
        labels.append(ids + [-100] * pad_len)
    
    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", required=True, choices=["math", "language"])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    
    set_seed(42)
    
    name = f"smollm2-135m-{args.skill}-pure"
    out_dir = OUTPUT_ROOT / name
    data_path = f"data/pure_{args.skill}/{args.skill}_pure_train.jsonl"
    
    print("=" * 70)
    print(f"FAST LORA TRAINING: {args.skill.upper()}")
    print(f"  Steps: {args.steps}, Batch: {args.batch_size}, LR: {args.lr}")
    print(f"  Output: {out_dir}")
    print("=" * 70)
    
    # Load tokenizer
    print("\n[1] Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    print(f"  Vocab: {tokenizer.vocab_size}, Pad: {tokenizer.pad_token}")
    
    # Load base model
    print("[2] Base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        base_model = base_model.cuda()
    base_model.gradient_checkpointing_enable()
    
    # Apply LoRA
    print("[3] LoRA adapters...")
    model = get_peft_model(base_model, LORA_CONFIG)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    # Load & pre-tokenize dataset
    print(f"[4] Dataset: {data_path}")
    dataset = FastTextDataset(data_path, tokenizer)
    
    split = int(0.95 * len(dataset))
    train_ds = torch.utils.data.Subset(dataset, range(split))
    eval_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))
    print(f"  Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    
    # Train
    print(f"[5] Training ({args.steps} steps)...")
    
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        warmup_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        save_steps=1000,
        eval_steps=500,
        logging_steps=25,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        max_grad_norm=1.0,
        save_total_limit=2,
        dataloader_num_workers=0,      # Windows-safe
        report_to="none",
        save_strategy="steps",
        eval_strategy="steps",
        logging_strategy="steps",
    )
    
    from functools import partial
    collator = partial(collate_batch, tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )
    
    trainer.train(resume_from_checkpoint=True)
    
    # Save
    final_path = out_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    print(f"\nDONE: {final_path}")
    print(f"  Adapter: {final_path}/adapter_model.safetensors")
    print(f"  Base weights: unchanged from {BASE_MODEL}")
    print(f"  Ready for CECI splice at k>=128")


if __name__ == "__main__":
    main()
