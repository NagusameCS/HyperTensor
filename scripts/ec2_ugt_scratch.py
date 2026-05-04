#!/usr/bin/env python3
"""
UGT FROM SCRATCH --- Train SmolLM2-135M with TOP loss active from initialization.
Runs on EC2 L40S (24GB VRAM). Expected runtime: 4-8 hours for 20K steps.

The key difference from post-hoc UGT: the model learns to organize its
representations along taxonomic axes DURING language acquisition, rather
than having zones imposed after the fact.

Usage: python ec2_ugt_scratch.py --steps 20000 --top-lambda 0.01 --k 256
"""

import json, os, sys, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

OUT = Path("benchmarks/ugt_scratch")
OUT.mkdir(parents=True, exist_ok=True)

# UGT imports --- SCP the infrastructure file too
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ugt_infrastructure import TOPLoss, UGTAdapter, TOPMonitor


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--top-lambda", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--eval-every", type=int, default=500)
    args = p.parse_args()
    
    zones = [args.k//3, args.k*2//3, args.k]
    
    print("=" * 60)
    print(f"UGT FROM SCRATCH: SmolLM2-135M, k={args.k}, steps={args.steps}")
    print(f"  TOP lambda={args.top_lambda}, lr={args.lr}, zones={zones}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    print("=" * 60)
    
    # Build model from scratch
    print("\n[1] Building SmolLM2-135M from scratch...")
    config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {n_params/1e6:.1f}M")
    
    # Wrap with UGT
    print(f"[2] Wrapping with UGTAdapter (k={args.k})...")
    ugt = UGTAdapter(model, k=args.k, zones=zones, top_lambda=args.top_lambda)
    ugt = ugt.to(device)
    
    # Initialize taxonomic basis randomly (not orthogonal --- let TOP push it)
    with torch.no_grad():
        random_basis = torch.randn(ugt.d, args.k, device=device) * 0.1
        ugt.taxonomic_basis.data = random_basis / (torch.norm(random_basis, dim=0, keepdim=True) + 1e-10)
    
    top_fn = TOPLoss(k=args.k, zones=zones)
    init_purity = top_fn.purity_score(ugt.taxonomic_basis.data)
    print(f"    Initial purity: {init_purity:.4f}")
    
    # Load training data
    print("[3] Loading training data...")
    wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    texts = [t for t in wiki['text'] if len(t.strip()) > 50]
    print(f"    Training texts: {len(texts)}")
    
    # Load eval data
    wiki_test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    eval_texts = [t for t in wiki_test['text'] if len(t.strip()) > 50][:50]
    
    # Optimizer: train all model params + taxonomic basis + zone heads
    optimizer = torch.optim.AdamW(ugt.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    
    monitor = TOPMonitor()
    monitor.log(0, init_purity, {})
    
    # Training loop
    print(f"\n[4] Training ({args.steps} steps)...")
    ugt.train()
    t0 = time.time()
    total_loss = 0.0
    
    for step in range(args.steps):
        text = texts[step % len(texts)]
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        if enc["input_ids"].shape[1] < 4: continue
        
        outputs = ugt(**enc, labels=enc["input_ids"])
        loss = outputs.loss / args.grad_accum
        loss.backward()
        total_loss += float(loss)
        
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(ugt.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Re-normalize basis
            with torch.no_grad():
                ugt.taxonomic_basis.data = ugt.taxonomic_basis.data / (
                    torch.norm(ugt.taxonomic_basis.data, dim=0, keepdim=True) + 1e-10)
        
        if (step + 1) % args.eval_every == 0:
            purity = top_fn.purity_score(ugt.taxonomic_basis.data)
            _, overlaps = top_fn(ugt.taxonomic_basis.data)
            avg_loss = total_loss / max(args.eval_every, 1)
            elapsed = time.time() - t0
            
            monitor.log(step + 1, purity, overlaps, avg_loss)
            
            ov_str = " ".join(f"{v:.3f}" for v in overlaps.values())
            print(f"  Step {step+1:5d}/{args.steps}: loss={avg_loss:.4f}, "
                  f"purity={purity:.4f}, ov=[{ov_str}], {elapsed:.0f}s, "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
            total_loss = 0.0
        
        if (step + 1) % args.save_every == 0:
            ckpt_dir = OUT / f"checkpoint-{step+1}"
            ckpt_dir.mkdir(exist_ok=True)
            ugt.model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            # Save basis separately
            torch.save(ugt.taxonomic_basis.data.cpu(), ckpt_dir / "taxonomic_basis.pt")
            purity = top_fn.purity_score(ugt.taxonomic_basis.data)
            print(f"  Saved checkpoint at step {step+1}, purity={purity:.4f}")
    
    elapsed = time.time() - t0
    print(f"\n[5] Training complete: {elapsed:.0f}s ({elapsed/args.steps:.2f}s/step)")
    
    # Final metrics
    final_purity = top_fn.purity_score(ugt.taxonomic_basis.data)
    _, final_ov = top_fn(ugt.taxonomic_basis.data)
    
    monitor.log(args.steps, final_purity, final_ov)
    monitor.save(OUT / "training_monitor.json")
    
    print(f"  Final purity: {final_purity:.4f}")
    for k, v in final_ov.items():
        print(f"  {k}: {v:.4f}")
    
    # Save final model
    final_dir = OUT / "final"
    final_dir.mkdir(exist_ok=True)
    ugt.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    torch.save(ugt.taxonomic_basis.data.cpu(), final_dir / "taxonomic_basis.pt")
    
    result = {
        "model": "SmolLM2-135M (from scratch)",
        "k": args.k, "zones": zones,
        "steps": args.steps,
        "top_lambda": args.top_lambda,
        "lr": args.lr,
        "init_purity": round(init_purity, 4),
        "final_purity": round(final_purity, 4),
        "final_overlaps": {k: round(v, 4) for k, v in final_ov.items()},
        "training_time_s": round(elapsed, 1),
        "training_time_h": round(elapsed / 3600, 1),
        "verdict": "TAXONOMY_ENFORCED_FROM_SCRATCH" if final_purity > 0.90 else "TRAINING_INCOMPLETE",
    }
    
    with open(OUT / "scratch_results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n  Saved: {OUT / 'scratch_results.json'}")
    print(f"  Verdict: {result['verdict']}")


if __name__ == "__main__":
    main()
