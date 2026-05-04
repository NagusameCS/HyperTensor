#!/usr/bin/env python3
"""
UGT (Universal Geodesic Taxonomy) --- TRAINING SCRIPT.
Paper XI: Enforces standardized geometric basis across latent space.

Trains SmolLM2-135M with TOP (Taxonomic Orthogonality Penalty) loss,
measuring taxonomy purity and zone separability during training.

Usage: python scripts/ugt_train.py --steps 500 --k 32 --top-lambda 0.01
"""

import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "ugt_training"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

# Pull in UGT classes
import importlib.util
spec = importlib.util.spec_from_file_location("ugt", ROOT / "scripts" / "ugt_infrastructure.py")
ugt_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ugt_mod)
TOPLoss = ugt_mod.TOPLoss
UGTAdapter = ugt_mod.UGTAdapter
TOPMonitor = ugt_mod.TOPMonitor


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--top-lambda", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--start-random", action="store_true", help="Start from random (non-orthogonal) basis to demonstrate convergence")
    p.add_argument("--eval-every", type=int, default=100)
    args = p.parse_args()

    zones = [12, 24, 32]
    
    print("=" * 60)
    print(f"UGT TRAINING: k={args.k}, zones={zones}, steps={args.steps}")
    print(f"  TOP lambda={args.top_lambda}, lr={args.lr}")
    print("=" * 60)

    # Load model
    print("\n[1] Loading SmolLM2-135M...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, device_map="auto", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print(f"    d={model.config.hidden_size}")

    # Wrap with UGT
    print(f"[2] Wrapping with UGTAdapter (k={args.k})...")
    ugt = UGTAdapter(model, k=args.k, zones=zones, top_lambda=args.top_lambda)
    device = next(model.parameters()).device

    # If random start, replace orthogonal init with biased random basis
    if args.start_random:
        print("    Replacing orthogonal basis with RANDOM (non-orthogonal) basis...")
        with torch.no_grad():
            # Biased toward positive => correlated => non-orthogonal
            random_basis = torch.randn(ugt.d, args.k) * 0.5 + 0.5
            random_basis = random_basis / (torch.norm(random_basis, dim=0, keepdim=True) + 1e-10)
            ugt.taxonomic_basis.data = random_basis.to(device)
        print(f"    Random basis created: {ugt.taxonomic_basis.shape}")

    # Load data
    print("[3] Loading WikiText-2...")
    wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [t for t in wiki['text'] if len(t.strip()) > 50][:200]

    # Baseline: measure purity without training
    print("[4] Baseline purity...")
    basis_init = ugt.taxonomic_basis.data.clone()
    top_fn = TOPLoss(k=args.k, zones=zones)
    init_purity = top_fn.purity_score(basis_init)
    init_penalty, init_overlaps = top_fn(basis_init)
    print(f"    Initial purity: {init_purity:.4f}")
    for k, v in init_overlaps.items():
        print(f"    {k}: {v:.4f}")

    # Optimizer
    trainable = [ugt.taxonomic_basis]
    for head in ugt.zone_heads:
        trainable.extend(head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    monitor = TOPMonitor()
    monitor.log(0, init_purity, init_overlaps)

    # Training loop
    print(f"\n[5] Training ({args.steps} steps)...")
    ugt.train()
    t0 = time.time()

    for step in range(args.steps):
        text = texts[step % len(texts)]
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        if enc["input_ids"].shape[1] < 4:
            continue

        outputs = ugt(**enc, labels=enc["input_ids"])
        loss = outputs.loss
        loss = loss / args.grad_accum
        loss.backward()

        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            # Re-normalize basis after each step
            with torch.no_grad():
                ugt.taxonomic_basis.data = ugt.taxonomic_basis.data / (
                    torch.norm(ugt.taxonomic_basis.data, dim=0, keepdim=True) + 1e-10)

        if (step + 1) % args.eval_every == 0:
            purity = top_fn.purity_score(ugt.taxonomic_basis.data)
            _, overlaps = top_fn(ugt.taxonomic_basis.data)
            monitor.log(step + 1, purity, overlaps, float(loss * args.grad_accum))
            elapsed = time.time() - t0
            overlap_str = " ".join(f"{v:.3f}" for v in overlaps.values())
            print(f"    Step {step+1:4d}/{args.steps}: purity={purity:.4f}, "
                  f"loss={float(loss*args.grad_accum):.4f}, overlaps=[{overlap_str}], "
                  f"{elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"    Training complete: {elapsed:.0f}s ({elapsed/args.steps:.2f}s/step)")

    # Final measurement
    print("\n[6] Results...")
    final_purity = top_fn.purity_score(ugt.taxonomic_basis.data)
    final_penalty, final_overlaps = top_fn(ugt.taxonomic_basis.data)
    monitor.log(args.steps, final_purity, final_overlaps)

    delta_purity = final_purity - init_purity
    print(f"    Initial purity:  {init_purity:.4f}")
    print(f"    Final purity:    {final_purity:.4f}")
    print(f"    Delta:           {delta_purity:+.4f}")
    print(f"    Zone overlaps:")
    for k, v in final_overlaps.items():
        init_v = init_overlaps.get(k, 1.0)
        print(f"      {k}: {v:.4f} (was {init_v:.4f}, delta {v-init_v:+.4f})")

    # Verify taxonomy separability with adversarial test
    print("\n[7] Taxonomy separability test...")
    basis = ugt.taxonomic_basis.data
    # Create test hidden states (random projections)
    test_hidden = torch.randn(8, ugt.d, device=device)
    projections = ugt.get_zone_projections(test_hidden)
    
    sep_scores = {}
    for z_name, proj in projections.items():
        # Norm of projection onto this zone vs total norm
        zone_norm = torch.norm(proj, dim=-1).mean().item()
        total_norm = torch.norm(test_hidden, dim=-1).mean().item()
        sep_scores[z_name] = round(zone_norm / max(total_norm, 1e-10), 4)
        print(f"    {z_name}: {sep_scores[z_name]:.4f} fraction")
    
    # Cross-zone leakage: project zone_1 hidden onto zone_2 basis
    proj_1_on_2 = torch.norm(
        projections["zone_1"] @ basis[:, zones[0]:zones[1]]
    ).item()
    leakage = proj_1_on_2 / max(torch.norm(projections["zone_1"]).item(), 1e-10)
    print(f"    Cross-zone leakage (zone1->zone2): {leakage:.6f}")

    # Save
    result = {
        "model": MODEL_ID,
        "k": args.k,
        "zones": zones,
        "top_lambda": args.top_lambda,
        "steps": args.steps,
        "lr": args.lr,
        "init_purity": round(init_purity, 4),
        "final_purity": round(final_purity, 4),
        "delta_purity": round(delta_purity, 4),
        "init_overlaps": {k: round(v, 4) for k, v in init_overlaps.items()},
        "final_overlaps": {k: round(v, 4) for k, v in final_overlaps.items()},
        "separability": sep_scores,
        "cross_zone_leakage": round(leakage, 6),
        "training_time_s": round(elapsed, 1),
        "verdict": "TAXONOMY_ENFORCED" if final_purity > 0.90 else "WEAK_SEPARATION" if delta_purity > 0.1 else "NO_EFFECT",
    }
    
    out_file = OUT / f"training_k{args.k}_steps{args.steps}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    
    # Also save monitor history
    monitor.save(OUT / f"monitor_k{args.k}_steps{args.steps}.json")
    
    print(f"\n    Saved: {out_file}")
    print(f"    Monitor: {OUT / f'monitor_k{args.k}_steps{args.steps}.json'}")
    print(f"    Verdict: {result['verdict']}")


if __name__ == "__main__":
    main()
