#!/usr/bin/env python3
"""
UGT Phase 2 --- Larger k (256), PCA initialization from real hidden states.
Tests whether a higher-capacity basis captures enough energy for functional taxonomy.

Usage: python scripts/ugt_phase2.py --k 256 --steps 500 --top-lambda 0.01
"""

import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "ugt_phase2"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

# UGT imports
import importlib.util
spec = importlib.util.spec_from_file_location("ugt", ROOT / "scripts" / "ugt_infrastructure.py")
ugt_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ugt_mod)
TOPLoss = ugt_mod.TOPLoss
UGTAdapter = ugt_mod.UGTAdapter

# --- PCA init from real hidden states ---
def collect_hidden_states(model, tokenizer, texts, n_samples=50, device=None):
    """Run forward passes and collect final-layer hidden states."""
    model.eval()
    all_hidden = []
    with torch.no_grad():
        for text in texts[:n_samples]:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            if enc["input_ids"].shape[1] < 4: continue
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[-1][0, :, :].cpu()  # (seq, d)
            all_hidden.append(h)
    if not all_hidden: return None
    H = torch.cat(all_hidden, dim=0).float()  # (total_tokens, d)
    return H.numpy()


def pca_basis(H, k):
    """Compute top-k PCA basis from hidden states."""
    H_centered = H - H.mean(axis=0, keepdims=True)
    C = H_centered.T @ H_centered / (H.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order[:k]], eigvals[order]


# --- Ablation test ---
TEST_PROBES = [
    ("syntax", "Write a grammatically correct sentence about a cat.", ["the","a","cat","is","was","sat","on","mat"]),
    ("syntax", "Complete this sentence: The weather today is", ["warm","cold","nice","sunny","rainy","beautiful","hot"]),
    ("syntax", "Rewrite in proper English: \"he go store yesterday\"", ["went","the","to","store","yesterday"]),
    ("algorithmic", "What is 12 * 7? Answer with just the number.", ["84"]),
    ("algorithmic", "What is 15 + 27? Answer with just the number.", ["42"]),
    ("algorithmic", "If a train travels 60 miles in 2 hours, what is its speed? Answer with just the number.", ["30"]),
    ("algorithmic", "What is the square root of 144? Answer with just the number.", ["12"]),
    ("factual", "What is the capital of France? Answer in one word.", ["Paris"]),
    ("factual", "What is the chemical symbol for water?", ["H2O","h2o","H₂O"]),
    ("factual", "Who wrote Romeo and Juliet? Answer with just the last name.", ["Shakespeare"]),
    ("factual", "What planet is closest to the sun?", ["Mercury"]),
]


def run_ablation(model, tokenizer, basis, zones):
    """Test zone specificity via ablation on the final hidden state."""
    device = next(model.parameters()).device
    d, k = basis.shape
    P = basis.to(device)
    
    zone_masks = []
    prev = 0
    for z_end in zones:
        mask = torch.zeros(k, device=device)
        mask[prev:z_end] = 1.0
        zone_masks.append(mask)
        prev = z_end
    
    zone_names = ["syntax", "algorithmic", "factual"]
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.model.lm_head
    
    results = []
    correct_by_zone = {z: 0 for z in zone_names}
    
    for cat, prompt, checks in TEST_PROBES:
        msgs = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[-1][:, -1, :]  # (1, d) --- last token
            
            # Project and measure energies
            h_k = h @ P  # (1, k)
            h_norm = torch.norm(h)**2
            
            zone_energies = {}
            prev = 0
            for i, z_end in enumerate(zones):
                h_z = h @ P[:, prev:z_end]
                zone_energies[zone_names[i]] = float(torch.norm(h_z)**2 / max(h_norm, 1e-10))
                prev = z_end
            
            # Baseline probability
            logits_base = lm_head(h.unsqueeze(0))[:, -1, :]
            
            # Get correct token IDs
            correct_ids = []
            for chk in checks:
                tid = tokenizer.encode(chk, add_special_tokens=False)
                if tid: correct_ids.extend(tid)
            
            if not correct_ids: continue
            correct_ids = list(set(correct_ids))
            
            base_prob = float(torch.softmax(logits_base, -1)[0, correct_ids].sum())
            
            # Ablate each zone
            zone_deltas = {}
            for ablate_idx in range(len(zones)):
                mask = zone_masks[ablate_idx].unsqueeze(0).unsqueeze(0)
                h_ablated_k = h_k * (1 - mask.squeeze(0))
                h_full_proj = h @ P @ P.T
                h_residual = h - h_full_proj
                h_final = (h_ablated_k @ P.T) + h_residual
                
                logits_ab = lm_head(h_final.unsqueeze(0))[:, -1, :]
                ab_prob = float(torch.softmax(logits_ab, -1)[0, correct_ids].sum())
                zone_deltas[zone_names[ablate_idx]] = round(base_prob - ab_prob, 6)
            
            top_token = tokenizer.decode([torch.argmax(logits_base, -1).item()]).strip()[:30]
            correct = any(chk.lower() in top_token.lower() for chk in checks)
            
            results.append({
                "category": cat, "prompt": prompt[:60], "top_token": top_token,
                "baseline_correct": correct, "energies": zone_energies,
                "deltas": zone_deltas,
            })
    
    # Score
    hit = 0
    for r in results:
        cat = r["category"]
        deltas = r["deltas"]
        max_z = max(deltas, key=deltas.get) if deltas else None
        if max_z == cat and deltas.get(cat, 0) > 0.001: hit += 1
    
    return results, hit


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=256)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--top-lambda", type=float, default=0.01)
    args = p.parse_args()
    
    # Zone split: roughly equal thirds
    z1 = args.k // 3
    z2 = args.k * 2 // 3
    zones = [z1, z2, args.k]
    
    print("=" * 60)
    print(f"UGT PHASE 2: k={args.k}, zones={zones}, steps={args.steps}")
    print("=" * 60)
    
    # Load model
    print("\n[1] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print(f"    d={model.config.hidden_size}, device={device}")
    
    # Collect hidden states for PCA
    print("[2] Collecting hidden states for PCA...")
    wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [t for t in wiki['text'] if len(t.strip()) > 50][:100]
    
    H = collect_hidden_states(model, tokenizer, texts, n_samples=30, device=device)
    if H is None:
        print("    ERROR: No hidden states collected"); return
    
    pca_b, pca_vals = pca_basis(H, args.k)
    var_retained = np.sum(pca_vals[:args.k]) / np.sum(pca_vals)
    print(f"    PCA top-{args.k}: {var_retained*100:.1f}% variance retained")
    print(f"    Eigenvalue range: {pca_vals[0]:.2f} to {pca_vals[args.k-1]:.4f}")
    
    # Initialize UGT adapter with PCA basis
    print(f"[3] Initializing UGT with PCA basis...")
    adapter = UGTAdapter(model, k=args.k, zones=zones, top_lambda=args.top_lambda)
    adapter = adapter.to(device)
    
    with torch.no_grad():
        adapter.taxonomic_basis.data = torch.from_numpy(pca_b).float().to(device)
    
    top_fn = TOPLoss(k=args.k, zones=zones)
    init_purity = top_fn.purity_score(adapter.taxonomic_basis.data)
    print(f"    PCA basis initial purity: {init_purity:.4f}")
    
    # Quick baseline ablation BEFORE training
    print("\n[4] Baseline ablation (PCA init, no training)...")
    baseline_results, baseline_hits = run_ablation(model, tokenizer, adapter.taxonomic_basis.data, zones)
    print(f"    Zone prediction accuracy: {baseline_hits}/{len(TEST_PROBES)} ({baseline_hits/len(TEST_PROBES):.0%})")
    
    # Measure zone energy coverage
    energies = {"syntax":[], "algorithmic":[], "factual":[]}
    for r in baseline_results:
        for z_name in energies:
            energies[z_name].append(r["energies"][z_name])
    for z_name in energies:
        avg_e = np.mean(energies[z_name])
        print(f"    {z_name}: avg energy={avg_e:.4f} (was 0.016 at k=32)")
    
    # Train TOP on the basis
    print(f"\n[5] Training TOP loss ({args.steps} steps)...")
    optimizer = torch.optim.AdamW([adapter.taxonomic_basis], lr=1e-4)
    adapter.train()
    
    for step in range(args.steps):
        text = texts[step % len(texts)]
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        if enc["input_ids"].shape[1] < 4: continue
        
        outputs = adapter(**enc, labels=enc["input_ids"])
        optimizer.zero_grad()
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_([adapter.taxonomic_basis], 1.0)
        optimizer.step()
        
        with torch.no_grad():
            adapter.taxonomic_basis.data = adapter.taxonomic_basis.data / (
                torch.norm(adapter.taxonomic_basis.data, dim=0, keepdim=True) + 1e-10)
        
        if (step + 1) % 200 == 0:
            purity = top_fn.purity_score(adapter.taxonomic_basis.data)
            print(f"    Step {step+1}: purity={purity:.4f}")
    
    final_purity = top_fn.purity_score(adapter.taxonomic_basis.data)
    print(f"    Final purity: {final_purity:.4f}")
    
    # Ablation AFTER training
    print(f"\n[6] Post-training ablation...")
    trained_results, trained_hits = run_ablation(model, tokenizer, adapter.taxonomic_basis.data, zones)
    print(f"    Zone prediction accuracy: {trained_hits}/{len(TEST_PROBES)} ({trained_hits/len(TEST_PROBES):.0%})")
    
    # Show per-probe comparison
    print(f"\n    {'Probe':<14s} {'Baseline':>8s} {'Trained':>8s}")
    for i, (br, tr) in enumerate(zip(baseline_results, trained_results)):
        b_ok = "OK" if br["baseline_correct"] else "--"
        t_ok = "OK" if tr["baseline_correct"] else "--"
        print(f"    {br['category']:<14s} {b_ok:>8s} {t_ok:>8s}")
    
    # Energy delta
    print(f"\n    Energy change from training:")
    for z_name in zone_names:
        b_avg = np.mean([r["energies"][z_name] for r in baseline_results])
        t_avg = np.mean([r["energies"][z_name] for r in trained_results])
        print(f"    {z_name}: {b_avg:.4f} -> {t_avg:.4f} (delta {t_avg-b_avg:+.4f})")
    
    total_energy_base = np.mean([sum(r["energies"].values()) for r in baseline_results])
    total_energy_trained = np.mean([sum(r["energies"].values()) for r in trained_results])
    print(f"    Total subspace energy: {total_energy_base:.4f} -> {total_energy_trained:.4f}")
    
    verdict = "TAXONOMY_PROVEN" if trained_hits >= len(TEST_PROBES) * 0.5 else \
              "TAXONOMY_WEAK" if trained_hits > baseline_hits else "NO_IMPROVEMENT"
    
    result = {
        "k": args.k, "zones": zones, "steps": args.steps,
        "pca_variance_retained": round(var_retained, 4),
        "init_purity": round(init_purity, 4),
        "final_purity": round(final_purity, 4),
        "baseline_hits": baseline_hits,
        "trained_hits": trained_hits,
        "total_probes": len(TEST_PROBES),
        "verdict": verdict,
    }
    
    out_file = OUT / f"phase2_k{args.k}_steps{args.steps}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n  Verdict: {verdict}")
    print(f"  Saved: {out_file}")


if __name__ == "__main__":
    main()
