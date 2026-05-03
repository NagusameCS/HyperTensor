#!/usr/bin/env python3
"""
UGT PHASE 5: Functional Multi-Head Training
==========================================
Redesigned from Phase 1-4 failures. Key changes:

1. TOPLoss v2: targets HEALTHY overlap (5%), not zero
   - Zero overlap = broken zones (Phase 1-4 failure mode)
   - 5% overlap = functional connectivity between zones
   
2. Multi-head output: each zone has its own lm_head
   - Zone 1 (syntax): predicts tokens from grammar subspace
   - Zone 2 (algorithmic): predicts tokens from reasoning subspace  
   - Zone 3 (factual): predicts tokens from knowledge subspace
   
3. Zone competition: soft winner-take-all per token
   - Each token position: which zone predicts it best?
   - Winning zone gets more gradient → specializes
   - Entropy bonus prevents one zone from dominating
   
4. Mixed training data:
   - Syntax examples (grammar exercises)
   - Factual examples (trivia, world knowledge)
   - General text (FineWeb-Edu)

Usage (local test):
  python scripts/ugt_phase5.py --steps 5000 --k 96 --top-lambda 0.01 --lr 1e-4

Usage (EC2 full):
  python -u scripts/ugt_phase5.py --steps 100000 --k 256 --top-lambda 0.01
"""

import json, os, sys, time, gc, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

OUT = Path("benchmarks/ugt_phase5")
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ugt_infrastructure import TOPLoss, UGTAdapter, TOPMonitor

# ══════════════════════════════════════════════════════════════════════
# Task-specific training data
# ══════════════════════════════════════════════════════════════════════

SYNTAX_TEMPLATES = [
    "The {adj} {noun} {verb} the {adj} {noun}.",
    "After {verb}ing the {noun}, she {verb}ed to the {noun}.",
    "Although it was {adj}, the {noun} still {verb}ed.",
    "The {noun} that {verb}ed the {noun} was very {adj}.",
    "If you {verb} the {noun}, then the {noun} will {verb}.",
    "Neither the {noun} nor the {noun} {verb}ed the {noun}.",
    "The {noun}, which had been {verb}ing, suddenly {verb}ed.",
    "By {verb}ing the {noun}, we can {verb} the {noun}.",
]

FACTUAL_QA = [
    ("The capital of France is", " Paris"),
    ("Water freezes at", " 0 degrees Celsius"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("The chemical symbol for gold is", " Au"),
    ("The speed of light is approximately", " 300,000 kilometers per second"),
    ("The human body has", " 206 bones"),
    ("The longest river in the world is", " the Nile"),
    ("The Earth orbits around", " the Sun"),
    ("Photosynthesis produces", " oxygen"),
    ("DNA stands for", " deoxyribonucleic acid"),
    ("The smallest prime number is", " 2"),
    ("Mount Everest is located in", " the Himalayas"),
    ("The process of cell division is called", " mitosis"),
    ("Einstein's famous equation is", " E=mc²"),
    ("The first element on the periodic table is", " hydrogen"),
]

REASONING_EXAMPLES = [
    "If all A are B, and all B are C, then all A are",
    "2 + 2 =",
    "The square root of 144 is",
    "If it takes 5 machines 5 minutes to make 5 widgets, then 100 machines would take",
    "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. The ball costs",
    "In a race, if you overtake the second person, you are now in",
    "If yesterday was Monday, then tomorrow will be",
    "Complete the sequence: 1, 1, 2, 3, 5, 8,",
]


def build_task_dataset(tokenizer, n_samples=2000):
    """Build a mixed dataset with syntax, factual, and reasoning examples."""
    import random
    random.seed(42)
    
    adjectives = ["quick", "brown", "lazy", "smart", "tall", "red", "happy", "cold", "bright", "dark"]
    nouns = ["fox", "dog", "cat", "bird", "tree", "car", "house", "book", "river", "mountain"]
    verbs = ["jumps", "runs", "flies", "swims", "reads", "writes", "builds", "breaks", "finds", "loses"]
    
    samples = []
    
    # Syntax examples (40%)
    n_syntax = int(n_samples * 0.4)
    for _ in range(n_syntax):
        template = random.choice(SYNTAX_TEMPLATES)
        text = template.format(
            adj=random.choice(adjectives),
            noun=random.choice(nouns),
            verb=random.choice(verbs)
        )
        samples.append({"text": text, "task": "syntax"})
    
    # Factual examples (30%)
    n_fact = int(n_samples * 0.3)
    for _ in range(n_fact):
        prompt, completion = random.choice(FACTUAL_QA)
        samples.append({"text": prompt + completion, "task": "factual"})
    
    # Reasoning examples (20%)
    n_reason = int(n_samples * 0.2)
    for _ in range(n_reason):
        example = random.choice(REASONING_EXAMPLES)
        samples.append({"text": example, "task": "reasoning"})
    
    # General text (10%) — from built-in knowledge
    n_general = n_samples - len(samples)
    general_texts = [
        "The history of civilization spans thousands of years of human development.",
        "Machine learning is a subset of artificial intelligence that focuses on data-driven algorithms.",
        "The ocean covers approximately 71 percent of the Earth's surface.",
        "Music theory encompasses the study of rhythm, harmony, and melody.",
        "The Renaissance was a period of great cultural and artistic achievement in Europe.",
    ]
    for _ in range(n_general):
        samples.append({"text": random.choice(general_texts), "task": "general"})
    
    random.shuffle(samples)
    return samples


def main():
    import argparse
    p = argparse.ArgumentParser(description="UGT Phase 5: Functional Multi-Head Training")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--top-lambda", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--overlap-target", type=float, default=0.05, help="Target healthy overlap (0.02-0.15)")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--test", action="store_true", help="Quick test with k=96, 500 steps")
    p.add_argument("--basis-only-steps", type=int, default=200,
                   help="Phase A: train only basis+heads (model frozen) to establish healthy overlap")
    args = p.parse_args()
    
    if args.test:
        args.k = 96
        args.steps = 500
        args.save_every = 500
        args.eval_every = 50
    
    zones = [args.k//3, args.k*2//3, args.k]
    
    print("=" * 60)
    print(f"UGT PHASE 5: Functional Multi-Head Training")
    print(f"  k={args.k}, steps={args.steps}, overlap_target={args.overlap_target}")
    print(f"  TOP lambda={args.top_lambda}, lr={args.lr}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ──── 1. Model ────
    if args.resume:
        print(f"\n[1] Resuming from {args.resume}...")
        model = AutoModelForCausalLM.from_pretrained(args.resume, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(args.resume)
        basis_path = Path(args.resume) / "taxonomic_basis.pt"
    else:
        print("\n[1] Building SmolLM2-135M from pretrained...")
        model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceTB/SmolLM2-135M-Instruct",
            torch_dtype=torch.float32,
        )
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
        basis_path = None
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {n_params/1e6:.1f}M")
    print(f"    Vocab size: {model.config.vocab_size}")
    
    # ──── 2. UGT Adapter ────
    print(f"\n[2] Wrapping with UGTAdapter v2 (k={args.k}, multi-head)...")
    ugt = UGTAdapter(
        model, k=args.k, zones=zones,
        top_lambda=args.top_lambda,
        overlap_target=args.overlap_target,
    )
    ugt = ugt.to(device)
    
    if basis_path and basis_path.exists():
        ugt.taxonomic_basis.data = torch.load(basis_path, map_location=device, weights_only=True)
        print("    Loaded basis from checkpoint")
    else:
        with torch.no_grad():
            # Initialize with slight overlap (~5%)
            raw = torch.randn(ugt.d, args.k, device=device) * 0.1
            shared = torch.randn(ugt.d, 1, device=device) * 0.05
            for z_start, z_end in zip([0] + zones[:-1], zones):
                raw[:, z_start:z_end] = raw[:, z_start:z_end] * 0.95 + shared * 0.05
            ugt.taxonomic_basis.data = raw / (torch.norm(raw, dim=0, keepdim=True) + 1e-10)
    
    top_fn = TOPLoss(k=args.k, zones=zones, overlap_target=args.overlap_target)
    init_purity = top_fn.purity_score(ugt.taxonomic_basis.data)
    _, init_ov, _ = top_fn(ugt.taxonomic_basis.data)
    print(f"    Initial purity: {init_purity:.4f}")
    for k, v in init_ov.items():
        print(f"      {k}: {v:.6f}")
    
    # ──── 3. Training data ────
    print("\n[3] Building task-specific training data...")
    task_samples = build_task_dataset(tokenizer, n_samples=2000 if args.test else 5000)
    task_counts = {}
    for s in task_samples:
        task_counts[s["task"]] = task_counts.get(s["task"], 0) + 1
    print(f"    Samples: {len(task_samples)}")
    for task, count in sorted(task_counts.items()):
        print(f"      {task}: {count}")
    
    # Also try to load FineWeb-Edu for general text
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        fw_texts = []
        for i, row in enumerate(ds):
            if len(row["text"].strip()) > 100:
                fw_texts.append(row["text"])
            if len(fw_texts) >= 1000:
                break
        print(f"    FineWeb-Edu supplement: {len(fw_texts)} texts")
    except:
        fw_texts = []
        print(f"    FineWeb-Edu unavailable, using task data only")
    
    # ──── 3.5 Phase A: Basis-only pre-training ────
    if args.basis_only_steps > 0:
        print(f"\n[3.5] Phase A: Basis-only pre-training ({args.basis_only_steps} steps)...")
        print(f"    Freezing model weights, training only basis + zone heads")
        
        # Freeze model
        for param in ugt.model.parameters():
            param.requires_grad = False
        
        basis_opt = torch.optim.AdamW(
            list(ugt.zone_lm_heads.parameters()) + [ugt.taxonomic_basis],
            lr=args.lr * 2.0  # Higher LR for basis-only phase
        )
        
        top_fn_strong = TOPLoss(k=args.k, zones=zones, overlap_target=args.overlap_target)
        # Use strong penalty during basis-only phase
        top_fn_strong.too_low_scale = 5000.0
        
        ugt.train()
        for bs in range(args.basis_only_steps):
            sample = random.choice(task_samples)
            enc = tokenizer(sample["text"], return_tensors="pt", truncation=True, max_length=128)
            enc = {k: v.to(device) for k, v in enc.items()}
            if enc["input_ids"].shape[1] < 4:
                continue
            
            basis_opt.zero_grad()
            
            # Forward with zone competition
            with torch.no_grad():
                outputs = ugt.model(**enc, output_hidden_states=True)
                hidden = outputs.hidden_states[-1].float()
            
            # Zone projections
            zone_hiddens = ugt._get_zone_hidden(hidden)
            zone_logits = []
            for i, hz in enumerate(zone_hiddens):
                zone_logits.append(ugt.zone_lm_heads[i](hz))
            all_logits = torch.stack(zone_logits, dim=0)
            
            shift_logits = all_logits[..., :-1, :].contiguous()
            shift_labels = enc["input_ids"][..., 1:].contiguous()
            
            zone_losses = []
            for i in range(len(zones)):
                z_loss = F.cross_entropy(
                    shift_logits[i].view(-1, ugt.vocab_size),
                    shift_labels.view(-1), reduction='none'
                ).view(shift_labels.shape)
                zone_losses.append(z_loss)
            zone_losses = torch.stack(zone_losses, dim=0)
            
            zone_weights = F.softmax(-zone_losses / ugt.zone_temp, dim=0)
            weighted_loss = (zone_losses * zone_weights.detach()).sum(dim=0).mean()
            
            zone_usage = zone_weights.mean(dim=(1, 2))
            target_usage = 1.0 / len(zones)
            entropy_bonus = ((zone_usage - target_usage) ** 2).sum() * 1.0
            
            # TOP penalty (strong)
            top_pen, overlaps, healthy = top_fn_strong(ugt.taxonomic_basis)
            
            loss = weighted_loss + entropy_bonus + top_pen * 0.5  # Heavy TOP weight
            loss.backward()
            torch.nn.utils.clip_grad_norm_([ugt.taxonomic_basis] + list(ugt.zone_lm_heads.parameters()), 1.0)
            basis_opt.step()
            
            with torch.no_grad():
                ugt.taxonomic_basis.data = ugt.taxonomic_basis.data / (
                    torch.norm(ugt.taxonomic_basis.data, dim=0, keepdim=True) + 1e-10)
            
            if (bs + 1) % max(1, args.basis_only_steps // 5) == 0 or bs == 0:
                purity = top_fn_strong.purity_score(ugt.taxonomic_basis.data)
                ov_str = " ".join(f"{v:.4f}" for v in overlaps.values())
                zu_str = " ".join(f"{v:.3f}" for v in zone_usage.detach().tolist())
                healthy_mark = "✓ HEALTHY" if healthy else "✗"
                print(f"    Basis step {bs+1:4d}: purity={purity:.4f}, ov=[{ov_str}] {healthy_mark}, zu=[{zu_str}]")
        
        # Unfreeze model for full training
        for param in ugt.model.parameters():
            param.requires_grad = True
        
        print(f"    Phase A complete. Overlap established.")
        print(f"    Final basis purity: {top_fn_strong.purity_score(ugt.taxonomic_basis.data):.4f}")
    
    # ──── 4. Training loop ────
    optimizer = torch.optim.AdamW(ugt.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    monitor = TOPMonitor()
    monitor.log(0, init_purity, init_ov)
    
    print(f"\n[4] Training ({args.steps} steps)...")
    ugt.train()
    t0 = time.time()
    total_loss = 0.0
    grad_accum = 4
    
    for step in range(args.steps):
        # Mix: 70% task data, 30% general text
        if random.random() < 0.7 or not fw_texts:
            sample = random.choice(task_samples)
            text = sample["text"]
        else:
            text = random.choice(fw_texts)
        
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        if enc["input_ids"].shape[1] < 4:
            continue
        
        outputs = ugt(**enc, labels=enc["input_ids"])
        loss = outputs.loss / grad_accum
        loss.backward()
        total_loss += float(loss)
        
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(ugt.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # Normalize basis
            with torch.no_grad():
                ugt.taxonomic_basis.data = ugt.taxonomic_basis.data / (
                    torch.norm(ugt.taxonomic_basis.data, dim=0, keepdim=True) + 1e-10)
        
        # ──── Logging ────
        if (step + 1) % args.eval_every == 0:
            purity = top_fn.purity_score(ugt.taxonomic_basis.data)
            _, overlaps, healthy = top_fn(ugt.taxonomic_basis.data)
            avg_loss = total_loss / max(args.eval_every, 1)
            elapsed = time.time() - t0
            
            zone_usage = getattr(ugt, '_last_zone_usage', {})
            monitor.log(step + 1, purity, overlaps, avg_loss,
                       lm_loss=getattr(ugt, '_last_lm_loss', 0),
                       top_pen=getattr(ugt, '_last_top_penalty', 0),
                       zone_usage=zone_usage)
            
            ov_str = " ".join(f"{v:.4f}" for v in overlaps.values())
            zu_str = " ".join(f"{v:.3f}" for v in zone_usage.values()) if zone_usage else "N/A"
            healthy_mark = "✓" if healthy else "✗"
            print(f"  Step {step+1:5d}/{args.steps}: loss={avg_loss:.4f}, "
                  f"purity={purity:.4f}, ov=[{ov_str}] {healthy_mark}")
            if zone_usage:
                print(f"    Zone usage: [{zu_str}]")
            print(f"    {elapsed:.0f}s, lr={scheduler.get_last_lr()[0]:.2e}")
            total_loss = 0.0
        
        # ──── Checkpoint ────
        if (step + 1) % args.save_every == 0:
            ckpt_dir = OUT / f"checkpoint-{step+1}"
            ckpt_dir.mkdir(exist_ok=True)
            ugt.model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            torch.save(ugt.taxonomic_basis.data.cpu(), ckpt_dir / "taxonomic_basis.pt")
            # Save zone LM heads
            torch.save({
                f"zone_lm_head_{i}": head.state_dict()
                for i, head in enumerate(ugt.zone_lm_heads)
            }, ckpt_dir / "zone_heads.pt")
            print(f"  Saved checkpoint at step {step+1}")
    
    # ──── 5. Final save ────
    elapsed = time.time() - t0
    final_purity = top_fn.purity_score(ugt.taxonomic_basis.data)
    _, final_ov, final_healthy = top_fn(ugt.taxonomic_basis.data)
    zone_usage = getattr(ugt, '_last_zone_usage', {})
    
    monitor.log(args.steps, final_purity, final_ov, zone_usage=zone_usage)
    monitor.save(OUT / "training_monitor.json")
    
    final_dir = OUT / "final"
    final_dir.mkdir(exist_ok=True)
    ugt.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    torch.save(ugt.taxonomic_basis.data.cpu(), final_dir / "taxonomic_basis.pt")
    torch.save({
        f"zone_lm_head_{i}": head.state_dict()
        for i, head in enumerate(ugt.zone_lm_heads)
    }, final_dir / "zone_heads.pt")
    
    print(f"\n[5] Training complete: {elapsed:.0f}s total")
    print(f"    Final purity: {final_purity:.4f}")
    print(f"    Final overlaps: {final_ov}")
    print(f"    Healthy: {final_healthy}")
    print(f"    Zone usage: {zone_usage}")
    print(f"    Monitor: {OUT / 'training_monitor.json'}")
    print(f"    Model: {final_dir}")
    
    # ──── 6. Quick zone ablation test ────
    print(f"\n[6] Zone ablation test...")
    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "The quick brown fox",
    ]
    
    ugt.eval()
    with torch.no_grad():
        for prompt in test_prompts[:3]:
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Full model generation
            out_full = ugt.model.generate(
                **enc, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
            full_text = tokenizer.decode(out_full[0], skip_special_tokens=True)
            
            print(f"  Prompt: \"{prompt}\"")
            print(f"    Full: \"{full_text}\"")
            
            # Per-zone ablation (zero each zone)
            for z in range(3):
                basis_copy = ugt.taxonomic_basis.data.clone()
                z_start = [0] + zones[:-1]
                basis_copy[:, z_start[z]:zones[z]] = 0.0
                ugt.taxonomic_basis.data = basis_copy
                
                out_abl = ugt.model.generate(
                    **enc, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id
                )
                abl_text = tokenizer.decode(out_abl[0], skip_special_tokens=True)
                print(f"    Zone{z+1} ablated: \"{abl_text}\"")
            
            # Restore
            ugt.taxonomic_basis.data = basis_copy  # no, need to reload
            # Quick restore: just reload from checkpoint
            ugt.taxonomic_basis.data = torch.load(final_dir / "taxonomic_basis.pt", map_location=device, weights_only=True)
    
    return monitor

if __name__ == "__main__":
    main()
