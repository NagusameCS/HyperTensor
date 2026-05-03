#!/usr/bin/env python3
"""
UGT PHASE 4: 50K-step training on FineWeb-Edu (richer corpus).
Reuses the Phase 3 infrastructure, extended training time.

Usage (on EC2):
  python ~/ugt_phase4.py --steps 50000 --k 256 --top-lambda 0.01
"""
import json, os, sys, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

OUT = Path("benchmarks/ugt_phase4")
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ugt_infrastructure import TOPLoss, UGTAdapter, TOPMonitor

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--top-lambda", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint dir")
    args = p.parse_args()
    
    zones = [args.k//3, args.k*2//3, args.k]
    
    print("=" * 60)
    print(f"UGT PHASE 4: SmolLM2-135M, k={args.k}, steps={args.steps}")
    print(f"  TOP lambda={args.top_lambda}, lr={args.lr}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    if args.resume:
        print(f"\n[1] Resuming from {args.resume}...")
        model = AutoModelForCausalLM.from_pretrained(args.resume, dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(args.resume)
        basis_path = Path(args.resume) / "taxonomic_basis.pt"
    else:
        print("\n[1] Building SmolLM2-135M from scratch...")
        config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
        basis_path = None
    
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    Params: {n_params/1e6:.1f}M")
    
    print(f"[2] Wrapping with UGTAdapter (k={args.k})...")
    ugt = UGTAdapter(model, k=args.k, zones=zones, top_lambda=args.top_lambda)
    ugt = ugt.to(device)
    
    if basis_path and basis_path.exists():
        ugt.taxonomic_basis.data = torch.load(basis_path, map_location=device, weights_only=True)
        print("    Loaded basis from checkpoint")
    else:
        with torch.no_grad():
            random_basis = torch.randn(ugt.d, args.k, device=device) * 0.1
            ugt.taxonomic_basis.data = random_basis / (torch.norm(random_basis, dim=0, keepdim=True) + 1e-10)
    
    top_fn = TOPLoss(k=args.k, zones=zones)
    init_purity = top_fn.purity_score(ugt.taxonomic_basis.data)
    print(f"    Initial purity: {init_purity:.4f}")
    
    # Try FineWeb-Edu, fall back to WikiText-2
    print("[3] Loading training data...")
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        texts = []
        for i, row in enumerate(ds):
            if len(row["text"].strip()) > 100:
                texts.append(row["text"])
            if len(texts) >= 5000: break
        print(f"    FineWeb-Edu: {len(texts)} texts")
    except Exception as e:
        print(f"    FineWeb-Edu unavailable ({e}), using WikiText-2...")
        wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        texts = [t for t in wiki['text'] if len(t.strip()) > 50]
        print(f"    WikiText-2: {len(texts)} texts")
    
    optimizer = torch.optim.AdamW(ugt.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    monitor = TOPMonitor()
    monitor.log(0, init_purity, {})
    
    print(f"\n[4] Training ({args.steps} steps)...")
    ugt.train()
    t0 = time.time()
    total_loss = 0.0; grad_accum = 4
    
    for step in range(args.steps):
        text = texts[step % len(texts)]
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        if enc["input_ids"].shape[1] < 4: continue
        
        outputs = ugt(**enc, labels=enc["input_ids"])
        loss = outputs.loss / grad_accum
        loss.backward()
        total_loss += float(loss)
        
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(ugt.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
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
            torch.save(ugt.taxonomic_basis.data.cpu(), ckpt_dir / "taxonomic_basis.pt")
            print(f"  Saved checkpoint at step {step+1}")
    
    elapsed = time.time() - t0
    final_purity = top_fn.purity_score(ugt.taxonomic_basis.data)
    _, final_ov = top_fn(ugt.taxonomic_basis.data)
    monitor.log(args.steps, final_purity, final_ov)
    monitor.save(OUT / "training_monitor.json")
    
    final_dir = OUT / "final"
    final_dir.mkdir(exist_ok=True)
    ugt.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    torch.save(ugt.taxonomic_basis.data.cpu(), final_dir / "taxonomic_basis.pt")
    
    result = {
        "model": "SmolLM2-135M (Phase 4)", "k": args.k, "steps": args.steps,
        "init_purity": round(init_purity, 4), "final_purity": round(final_purity, 4),
        "training_time_h": round(elapsed / 3600, 1),
        "verdict": "TAXONOMY_ENFORCED_FROM_SCRATCH" if final_purity > 0.90 else "TRAINING_INCOMPLETE",
    }
    with open(OUT / "phase4_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {OUT / 'phase4_results.json'}")
    print(f"  Verdict: {result['verdict']}")

if __name__ == "__main__":
    main()
