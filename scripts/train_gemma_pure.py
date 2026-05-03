#!/usr/bin/env python3
"""
EC2 Gemma-4-2B Pure LoRA Training.
Self-contained script — upload to EC2 and run directly.
Trains a pure single-skill model on either math or language data.

Usage on EC2:
  python3 train_gemma_pure.py --skill math --steps 5000
  python3 train_gemma_pure.py --skill language --steps 5000
  
Requirements: torch, transformers, peft, datasets, accelerate
EC2: g6e.xlarge (L40S 46GB) — Gemma-4-2B fits in BF16 with LoRA (~8GB)
"""

import argparse, json, os, time
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset

BASE_MODEL = "Qwen/Qwen2.5-1.5B"  # d=1536, 28 layers, freely available, no auth
OUTPUT_BASE = Path("outputs/pure_models")

# ===========================================================================
# Pure Math Dataset (curated for non-overlap with language)
# ===========================================================================
MATH_TEXTS = [
    # GSM8K-style grade school math
    "Question: Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the rest for $2 each. How much does she make daily?\nAnswer: 16-3-4=9 eggs sold. 9×$2=$18. Janet makes $18 per day.",
    "Question: A store sells pencils in packs of 12 for $3. How many pencils can you buy with $15?\nAnswer: $15/$3=5 packs. 5×12=60 pencils.",
    "Question: If a car travels at 60 mph for 2.5 hours, how far does it go?\nAnswer: 60×2.5=150 miles.",
    "Question: Solve for x: 3x + 7 = 22\nAnswer: 3x=15, x=5.",
    "Question: What is 15% of 200?\nAnswer: 0.15×200=30.",
    "Question: A rectangle has length 8m and width 5m. Find its area and perimeter.\nAnswer: Area=8×5=40m². Perimeter=2(8+5)=26m.",
    "Question: Factor: x² - 9\nAnswer: (x+3)(x-3) — difference of squares.",
    "Question: What is the square root of 196?\nAnswer: 14, since 14²=196.",
    "Question: If f(x)=2x²+3x-5, find f(2).\nAnswer: f(2)=2(4)+6-5=8+6-5=9.",
    "Question: A train leaves at 8:00 AM traveling 80 mph. Another train leaves the same station at 9:30 AM traveling 100 mph. When does the second train catch up?\nAnswer: Head start: 1.5×80=120 miles. Relative speed: 20 mph. 120/20=6 hours after 9:30 AM = 3:30 PM.",
    "Question: Solve the system: 2x + y = 7, x - y = 2\nAnswer: From second: x=y+2. Substitute: 2(y+2)+y=7 → 2y+4+y=7 → 3y=3 → y=1, x=3.",
    "Question: What is the probability of rolling a sum of 7 with two dice?\nAnswer: 6 favorable outcomes (1+6,2+5,3+4,4+3,5+2,6+1) out of 36. P=6/36=1/6.",
    "Question: Find the derivative of f(x)=3x⁴-2x²+5x-7.\nAnswer: f'(x)=12x³-4x+5.",
    "Question: A circle has radius 5. Find its circumference and area.\nAnswer: C=2π×5=10π≈31.4. A=π×5²=25π≈78.5.",
    "Question: Convert 0.625 to a fraction in simplest form.\nAnswer: 625/1000 = 5/8.",
    "Question: What is the median of {3,7,2,9,5,11,6}?\nAnswer: Sorted: {2,3,5,6,7,9,11}. Median=6.",
    "Question: If log₁₀(1000)=x, what is x?\nAnswer: x=3 since 10³=1000.",
    "Question: A triangle has sides 3, 4, 5. Is it a right triangle?\nAnswer: 3²+4²=9+16=25=5². Yes, it's a 3-4-5 right triangle.",
    "Question: Find the sum of the first 10 positive integers.\nAnswer: n(n+1)/2 = 10×11/2 = 55.",
    "Question: Simplify: (a²b³)⁴\nAnswer: a⁸b¹².",
] * 250  # 5000 texts

# ===========================================================================
# Pure Language Dataset (curated for non-overlap with math)
# ===========================================================================
LANGUAGE_TEXTS = [
    "The morning sun cast golden rays across the meadow, illuminating droplets of dew that clung to every blade of grass like tiny diamonds scattered by a generous hand.",
    "She walked through the ancient library, her fingers tracing the spines of leather-bound books that had not been opened in centuries, each one holding secrets of forgotten civilizations.",
    "The old fisherman sat at the end of the pier, his weathered hands mending nets with the same patience he had learned from his grandfather sixty years ago, when the sea still teemed with life.",
    "In the heart of the city, beneath the neon glow of countless advertisements, a small coffee shop served as a sanctuary for writers, artists, and dreamers who found solace in the bitter aroma of freshly ground beans.",
    "The symphony began with a whisper of violins, building slowly like a storm gathering on the horizon, until the full orchestra erupted in a cascade of sound that left the audience breathless.",
    "He remembered the taste of his grandmother's apple pie — cinnamon and nutmeg dancing on his tongue, the flaky crust crumbling perfectly with each bite, a recipe passed down through four generations.",
    "The desert stretched endlessly before them, a vast ocean of sand dunes sculpted by millennia of wind, where the only sound was the soft whisper of grains shifting in the afternoon breeze.",
    "Language is not merely a tool for communication but the very fabric of thought itself, shaping our perception of reality in ways we are only beginning to understand through cognitive science.",
    "The detective examined the crime scene with meticulous care, noting every detail — the position of the overturned chair, the half-empty glass of wine, the faint footprint in the carpet.",
    "As autumn painted the forest in shades of amber and crimson, the old oak stood sentinel, its branches reaching toward the pale October sky like the arms of a patient guardian.",
    "The chef moved through the kitchen with practiced grace, his knife dancing across the cutting board in a rhythm perfected over decades, transforming simple ingredients into culinary art.",
    "History teaches us that the greatest discoveries often come not from methodical research but from serendipitous accidents — penicillin, X-rays, and microwave ovens were all born from unexpected observations.",
    "The garden bloomed in defiance of the urban landscape surrounding it, a riot of color and fragrance that attracted butterflies and hummingbirds who seemed unaware they were in the middle of a metropolis.",
    "She opened the letter with trembling hands, the yellowed paper crackling softly as she unfolded decades of silence, tears forming as she recognized her mother's elegant handwriting.",
    "The philosopher argued that consciousness arises not from any single region of the brain but from the complex interactions between neural networks that create our sense of self and agency.",
    "Beneath the star-filled sky, the campfire crackled and popped, sending sparks spiraling upward to join the distant constellations that had guided travelers across these same mountains for thousands of years.",
    "The architect designed buildings that seemed to defy gravity, glass and steel structures that captured light in unexpected ways, transforming the city skyline into a work of modern art.",
    "Poetry is the art of saying the unsayable — capturing in a few carefully chosen words the depth of human emotion that prose requires pages to express inadequately.",
    "The old train station stood abandoned, its Victorian ironwork rusted but still beautiful, a monument to an era when railways connected the nation and travel was an adventure rather than an inconvenience.",
    "She discovered that true happiness lay not in achieving her goals but in the journey itself — the small victories, the lessons from failures, the people who walked beside her along the way.",
] * 250  # 5000 texts

# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skill', required=True, choices=['math', 'language'])
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lora-r', type=int, default=8)
    args = parser.parse_args()
    
    output_dir = OUTPUT_BASE / f"qwen2.5-1.5b-{args.skill}-pure"
    output_final = output_dir / "final"
    output_final.mkdir(parents=True, exist_ok=True)
    
    effective_batch = args.batch_size * args.grad_accum
    
    print("=" * 60)
    print(f"QWEN2.5-1.5B PURE LORA: {args.skill.upper()}")
    print(f"  Steps: {args.steps}, Batch: {args.batch_size}×{args.grad_accum}={effective_batch}")
    print(f"  LR: {args.lr}, LoRA r={args.lora_r}")
    print(f"  Output: {output_final}")
    print("=" * 60)
    
    # Load base model
    print("\n[1] Loading Qwen2.5-1.5B base...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_time = time.perf_counter() - t0
    print(f"  Loaded in {load_time:.0f}s")
    print(f"  Model params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    
    # Apply LoRA
    print("\n[2] Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Prepare data
    print(f"\n[3] Preparing {args.skill} data...")
    texts = MATH_TEXTS if args.skill == 'math' else LANGUAGE_TEXTS
    texts = texts[:args.steps * effective_batch]  # Scale to steps
    
    def tokenize_fn(examples):
        result = tokenizer(examples["text"], truncation=True, 
                         max_length=512, padding="max_length")
        result["labels"] = result["input_ids"].copy()
        return result
    
    dataset = HFDataset.from_dict({"text": texts})
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    # Manual split
    n_total = len(tokenized)
    n_train = int(n_total * 0.95)
    indices = list(range(n_total))
    import random; random.seed(42); random.shuffle(indices)
    train_ds = tokenized.select(indices[:n_train])
    eval_ds = tokenized.select(indices[n_train:])
    
    print(f"  Train: {len(train_ds)} samples, Eval: {len(eval_ds)} samples")
    
    # Train
    print(f"\n[4] Training ({args.steps} steps)...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=100,  # effectively unlimited, max_steps controls
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=100,
        save_steps=1000,
        eval_steps=500,
        eval_strategy="steps",
        save_total_limit=2,
        bf16=True,
        dataloader_num_workers=2,
        report_to="none",
        max_steps=args.steps,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False),
    )
    
    trainer.train()
    
    # Save
    print(f"\n[5] Saving to {output_final}...")
    model.save_pretrained(str(output_final))
    tokenizer.save_pretrained(str(output_final))
    
    # Metadata
    meta = {
        "skill": args.skill,
        "base_model": BASE_MODEL,
        "steps": args.steps,
        "lora_r": args.lora_r,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": effective_batch,
        "lr": args.lr,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with open(output_final / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nDONE: {args.skill.upper()} model saved to {output_final}")
    print(f"  Trainable: {meta['trainable_params']/1e6:.1f}M params")

if __name__ == "__main__":
    main()
