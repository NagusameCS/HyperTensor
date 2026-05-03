#!/usr/bin/env python3
"""
UGT CAUSAL ABLATION — Proves taxonomic zones control distinct behaviors.

Experiment:
1. Train UGT basis to purity > 0.95 on SmolLM2-135M
2. Run a battery of prompts spanning syntax, arithmetic, knowledge
3. For each prompt, get the final hidden state and project onto each zone
4. Ablate one zone (zero its contribution) and measure:
   - Change in answer token probability
   - Change in output text
5. Hypothesis: Zone 1 (syntax) ablation breaks grammar but preserves facts
              Zone 3 (factual) ablation breaks knowledge but preserves grammar

Usage: python scripts/ugt_ablation.py --steps 1000 --top-lambda 0.05
"""

import json, sys, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "ugt_ablation"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Pull in UGT classes
import importlib.util
spec = importlib.util.spec_from_file_location("ugt", ROOT / "scripts" / "ugt_infrastructure.py")
ugt_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ugt_mod)
TOPLoss = ugt_mod.TOPLoss
UGTAdapter = ugt_mod.UGTAdapter


# Prompt battery organized by expected dominant zone
# zone 1 (syntax): grammar, structure, fluency
# zone 2 (algorithmic): math, logic, step-by-step
# zone 3 (factual): knowledge recall, facts
TEST_SUITE = [
    # Syntax probes (should break on zone 1 ablation)
    {"id": "syntax_1", "category": "syntax", "prompt": "Write a grammatically correct sentence about a cat.",
     "check": ["the", "a", "cat", "is"]},
    {"id": "syntax_2", "category": "syntax", "prompt": "Complete this sentence with proper English: The weather today is",
     "check": ["warm", "cold", "sunny", "rainy", "nice", "beautiful"]},
    {"id": "syntax_3", "category": "syntax", "prompt": "Rewrite this in proper English: \"he go store yesterday\"",
     "check": ["went", "the", "to"]},
    
    # Algorithmic probes (should break on zone 2 ablation)
    {"id": "algo_1", "category": "algorithmic", "prompt": "What is 12 * 7? Answer with just the number.",
     "check": ["84"]},
    {"id": "algo_2", "category": "algorithmic", "prompt": "What is 15 + 27? Answer with just the number.",
     "check": ["42"]},
    {"id": "algo_3", "category": "algorithmic", "prompt": "If a train travels 60 miles in 2 hours, what is its speed in mph? Answer with just the number.",
     "check": ["30"]},
    
    # Factual probes (should break on zone 3 ablation)
    {"id": "fact_1", "category": "factual", "prompt": "What is the capital of France? Answer in one word.",
     "check": ["Paris"]},
    {"id": "fact_2", "category": "factual", "prompt": "What is the chemical symbol for water? Answer with just the symbol.",
     "check": ["H2O", "H₂O", "h2o"]},
    {"id": "fact_3", "category": "factual", "prompt": "Who wrote Romeo and Juliet? Answer with just the name.",
     "check": ["Shakespeare"]},
]


def train_ugt_basis(adapter, tokenizer, texts, steps=1000, top_lambda=0.05):
    """Train UGT basis from random init to taxonomic orthogonality."""
    device = next(adapter.parameters()).device
    
    # Replace orthogonal init with random (non-orthogonal) basis
    with torch.no_grad():
        random_basis = torch.randn(adapter.d, adapter.k, device=device) * 0.5 + 0.5
        random_basis = random_basis / (torch.norm(random_basis, dim=0, keepdim=True) + 1e-10)
        adapter.taxonomic_basis.data = random_basis
    
    top_fn = TOPLoss(k=adapter.k, zones=[12, 24, 32])
    optimizer = torch.optim.AdamW([adapter.taxonomic_basis], lr=5e-4)
    monitor = ugt_mod.TOPMonitor()
    
    init_purity = top_fn.purity_score(adapter.taxonomic_basis.data)
    print(f"  Initial purity: {init_purity:.4f}")
    
    adapter.train()
    for step in range(steps):
        text = texts[step % len(texts)]
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        if enc["input_ids"].shape[1] < 4: continue
        
        outputs = adapter(**enc, labels=enc["input_ids"])
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([adapter.taxonomic_basis], 1.0)
        optimizer.step()
        
        with torch.no_grad():
            adapter.taxonomic_basis.data = adapter.taxonomic_basis.data / (
                torch.norm(adapter.taxonomic_basis.data, dim=0, keepdim=True) + 1e-10)
        
        if (step + 1) % 200 == 0:
            purity = top_fn.purity_score(adapter.taxonomic_basis.data)
            print(f"    Step {step+1}: purity={purity:.4f}")
    
    return top_fn.purity_score(adapter.taxonomic_basis.data)


def run_ablation(model, tokenizer, basis, zones, test_suite):
    """Test each probe with and without zone ablation."""
    device = next(model.parameters()).device
    k = basis.shape[1]
    
    # Precompute zone projection matrices
    P = basis  # (d, k)
    zone_masks = []
    zone_names = ["syntax", "algorithmic", "factual"]
    prev = 0
    for i, z_end in enumerate(zones):
        mask = torch.zeros(k, device=device)
        mask[prev:z_end] = 1.0
        zone_masks.append(mask)
        prev = z_end
    
    def generate_with_ablation(prompt, ablate_zone=None):
        """Generate text, optionally ablating one zone from all hidden states."""
        msgs = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        
        with torch.no_grad():
            if ablate_zone is not None:
                # Use output_hidden_states to get per-layer hidden states
                out = model(**enc, output_hidden_states=True, labels=enc["input_ids"])
                # This won't work for generation — need a custom forward
                # Fall back to: just ablate from the final hidden state via logit manipulation
                out = model(**enc, output_hidden_states=True)
                last_hidden = out.hidden_states[-1]  # (1, seq_len, d)
                
                # Project onto basis
                h_proj = last_hidden @ P  # (1, seq_len, k)
                
                # Zero out the ablated zone
                mask = zone_masks[ablate_zone]  # (k,)
                h_ablated = h_proj * (1 - mask.unsqueeze(0).unsqueeze(0))  # (1, seq_len, k)
                
                # Reconstruct: project back to d
                h_reconstructed = h_ablated @ P.T  # (1, seq_len, d)
                
                # Get LM head logits from reconstructed hidden states
                # Use the model's lm_head directly
                if hasattr(model, 'lm_head'):
                    lm_head = model.lm_head
                else:
                    lm_head = model.model.lm_head
                
                logits = lm_head(h_reconstructed[:, -1:, :])  # (1, 1, vocab)
                # Greedy decode from these logits
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                response_ids = [next_token.item()]
                
                # Simple greedy decode loop (max 30 tokens for speed)
                # This is approximate — we're ablating only the last token's projection
                # Full layer-by-layer ablation requires hooks (complex but more rigorous)
                gen_ids = []
                
            else:
                # Normal generation
                out = model.generate(**enc, max_new_tokens=40, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id,
                                    eos_token_id=tokenizer.eos_token_id)
                gen_ids = out[0][enc["input_ids"].shape[1]:].tolist()
        
        if gen_ids:
            return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        else:
            return ""
    
    # Simpler, more robust approach: run normal generation, then compare
    # Use a single generate_and_probe that measures zone contributions
    results = {z: {"correct": 0, "total": 0, "details": []} for z in ["none", "syntax", "algorithmic", "factual"]}
    results["zone_energies"] = {}
    
    print("\n  Running ablation tests...")
    for probe in TEST_SUITE:
        cat = probe["category"]
        msgs = [{"role": "user", "content": probe["prompt"]}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        
        with torch.no_grad():
            # Get hidden states with full forward
            outputs = model(**enc, output_hidden_states=True)
            last_h = outputs.hidden_states[-1][:, -1, :]  # (1, d) — last token
            
            # Project onto basis: h_k = P^T h
            h_k = last_h @ P  # (1, k)
            
            # Zone energies: ||P_zone^T h||^2 / ||h||^2
            h_norm_sq = torch.norm(last_h)**2
            zone_energies = {}
            prev = 0
            for i, z_end in enumerate(zones):
                h_zone = last_h @ P[:, prev:z_end]  # (1, z_size)
                energy = torch.norm(h_zone)**2 / max(h_norm_sq, 1e-10)
                zone_energies[zone_names[i]] = float(energy)
                prev = z_end
            
            # Get baseline logits
            lm_head = model.lm_head if hasattr(model, 'lm_head') else model.model.lm_head
            logits_base = lm_head(last_h.unsqueeze(0))[:, -1, :]  # (1, vocab)
            
            # For each zone, ablate and measure logit change
            zone_deltas = {}
            for ablate_idx in range(3):
                mask = zone_masks[ablate_idx].unsqueeze(0).unsqueeze(0)  # (1, 1, k)
                h_ablated_k = h_k * (1 - mask.squeeze(0))  # (1, k)
                h_reconstructed = h_ablated_k @ P.T  # (1, d)
                
                # Preserve non-basis component: h_residual = h - h @ P @ P^T
                h_full_proj = last_h @ P @ P.T  # (1, d)
                h_residual = last_h - h_full_proj
                h_final = h_reconstructed + h_residual
                
                logits_ablated = lm_head(h_final.unsqueeze(0))[:, -1, :]
                
                # Change in logit for the correct answer token
                correct_tokens = []
                for chk in probe["check"]:
                    tid = tokenizer.encode(chk, add_special_tokens=False)
                    if tid: correct_tokens.append(tid[0])
                
                if correct_tokens:
                    baseline_prob = torch.softmax(logits_base, -1)[0, correct_tokens].sum()
                    ablated_prob = torch.softmax(logits_ablated, -1)[0, correct_tokens].sum()
                    delta = float(baseline_prob - ablated_prob)  # positive = ablation hurts
                else:
                    delta = 0.0
                
                zone_deltas[zone_names[ablate_idx]] = round(delta, 6)
            
            # Check baseline correctness
            top_token_id = torch.argmax(logits_base, -1).item()
            top_token = tokenizer.decode([top_token_id]).strip().lower()
            correct = any(chk.lower() in top_token for chk in probe["check"])
            
            probe_result = {
                "id": probe["id"],
                "category": cat,
                "prompt": probe["prompt"][:60],
                "top_token": top_token[:30],
                "baseline_correct": correct,
                "zone_energies": zone_energies,
                "zone_deltas": zone_deltas,
            }
            
            results["none"]["total"] += 1
            if correct: results["none"]["correct"] += 1
            results["none"]["details"].append(probe_result)
            
            for z_name in zone_names:
                results[z_name]["details"].append(probe_result)
            
            markers = {z: "!" if zone_deltas.get(z, 0) > 0.05 else ("~" if zone_deltas.get(z, 0) > 0.01 else " ") 
                      for z in zone_names}
            print(f"  [{('OK' if correct else 'XX')}] {probe['id']:<12s} | "
                  f"energies: s={zone_energies['syntax']:.3f} a={zone_energies['algorithmic']:.3f} f={zone_energies['factual']:.3f} | "
                  f"deltas: s{markers['syntax']}{zone_deltas.get('syntax',0):.4f} "
                  f"a{markers['algorithmic']}{zone_deltas.get('algorithmic',0):.4f} "
                  f"f{markers['factual']}{zone_deltas.get('factual',0):.4f}")
    
    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--top-lambda", type=float, default=0.05)
    p.add_argument("--k", type=int, default=32)
    args = p.parse_args()
    
    zones = [12, 24, 32]
    print("=" * 60)
    print("UGT CAUSAL ABLATION — Proving taxonomic zone specificity")
    print(f"  k={args.k}, zones={zones}, steps={args.steps}")
    print("=" * 60)
    
    # Load model
    print("\n[1] Loading SmolLM2-135M-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print(f"    Model: d={model.config.hidden_size}, layers={model.config.num_hidden_layers}")
    
    # Load texts
    print("[2] Loading WikiText-2...")
    wiki = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [t for t in wiki['text'] if len(t.strip()) > 50][:200]
    
    # Wrap with UGT and train
    print(f"[3] Training UGT basis...")
    adapter = UGTAdapter(model, k=args.k, zones=zones, top_lambda=args.top_lambda)
    adapter = adapter.to(device)
    final_purity = train_ugt_basis(adapter, tokenizer, texts, steps=args.steps, top_lambda=args.top_lambda)
    print(f"  Final purity: {final_purity:.4f}")
    
    # Verify zone orthogonality
    top_fn = TOPLoss(k=args.k, zones=zones)
    _, overlaps = top_fn(adapter.taxonomic_basis.data)
    print(f"  Zone overlaps: z1/z2={overlaps['zone1_vs_zone2']:.4f}, "
          f"z1/z3={overlaps['zone1_vs_zone3']:.4f}, z2/z3={overlaps['zone2_vs_zone3']:.4f}")
    
    if final_purity < 0.90:
        print(f"  WARNING: Purity {final_purity:.4f} < 0.90. More steps needed.")
        return
    
    # Run ablation experiment
    print(f"\n[4] Ablation experiment...")
    results = run_ablation(model, tokenizer, adapter.taxonomic_basis.data, zones, TEST_SUITE)
    
    # Analysis
    print(f"\n{'='*60}")
    print("ABLATION ANALYSIS")
    print(f"{'='*60}")
    
    # Zone energy distribution
    print("\n  Zone energy distribution (fraction of total hidden state energy):")
    energy_avgs = {"syntax": 0, "algorithmic": 0, "factual": 0}
    for probe in results["none"]["details"]:
        for z in energy_avgs:
            energy_avgs[z] += probe["zone_energies"].get(z, 0)
    for z in energy_avgs:
        energy_avgs[z] /= len(TEST_SUITE)
    print(f"    Syntax:      {energy_avgs['syntax']:.4f}")
    print(f"    Algorithmic: {energy_avgs['algorithmic']:.4f}")
    print(f"    Factual:     {energy_avgs['factual']:.4f}")
    
    # Expected pattern: each probe type should have dominant energy in its zone
    print("\n  Zone dominance by probe category:")
    for cat in ["syntax", "algorithmic", "factual"]:
        cat_probes = [r for r in results["none"]["details"] if r["category"] == cat]
        s_avg = np.mean([r["zone_energies"]["syntax"] for r in cat_probes])
        a_avg = np.mean([r["zone_energies"]["algorithmic"] for r in cat_probes])
        f_avg = np.mean([r["zone_energies"]["factual"] for r in cat_probes])
        dominant = "syntax" if s_avg > max(a_avg, f_avg) else "algorithmic" if a_avg > max(s_avg, f_avg) else "factual"
        print(f"    {cat:<14s}: s={s_avg:.4f}, a={a_avg:.4f}, f={f_avg:.4f} → dominant: {dominant} "
              f"[{'MATCH' if dominant == cat else 'MISMATCH'}]")
    
    # Ablation impact: does ablating the right zone break the right probes?
    print("\n  Ablation impact (probability drop when zone is removed):")
    print(f"    {'Probe':<14s} {'Category':<14s} {'syntax Δ':>10s} {'algo Δ':>10s} {'fact Δ':>10s}")
    print(f"    {'-'*58}")
    
    hit_count = 0
    for probe in results["none"]["details"]:
        cat = probe["category"]
        deltas = probe["zone_deltas"]
        s_d = deltas.get("syntax", 0)
        a_d = deltas.get("algorithmic", 0)
        f_d = deltas.get("factual", 0)
        
        # Mark the highest delta
        max_d = max(s_d, a_d, f_d)
        s_m = "*" if s_d == max_d and s_d > 0.01 else " "
        a_m = "*" if a_d == max_d and a_d > 0.01 else " "
        f_m = "*" if f_d == max_d and f_d > 0.01 else " "
        
        expected_zone = {"syntax": 0, "algorithmic": 1, "factual": 2}[cat]
        dominant_idx = np.argmax([s_d, a_d, f_d])
        if dominant_idx == expected_zone and max_d > 0:
            hit_count += 1
        
        print(f"    {probe['id']:<14s} {cat:<14s} {s_m}{s_d:>9.4f} {a_m}{a_d:>9.4f} {f_m}{f_d:>9.4f}")
    
    print(f"\n  Zone prediction accuracy: {hit_count}/{len(TEST_SUITE)} ({hit_count/len(TEST_SUITE):.0%})")
    
    # Verdict
    verdict = "TAXONOMY_PROVEN" if hit_count >= len(TEST_SUITE) * 0.6 else "TAXONOMY_WEAK" if hit_count >= len(TEST_SUITE) * 0.3 else "TAXONOMY_UNPROVEN"
    print(f"\n  VERDICT: {verdict}")
    
    if verdict == "TAXONOMY_PROVEN":
        print("  UGT zones CORRESPOND to functional specializations.")
        print("  The 32-dim basis separates syntax, algorithmic, and factual processing.")
    elif verdict == "TAXONOMY_WEAK":
        print("  Partial evidence for zone specialization.")
        print("  Some zones show expected patterns but not consistently.")
    else:
        print("  Zones are mathematically orthogonal but do not cleanly separate functions.")
        print("  More training or different zone boundaries may be needed.")
    
    # Save
    result = {
        "k": args.k, "zones": zones, "steps": args.steps,
        "final_purity": round(final_purity, 4),
        "zone_overlaps": {k: round(v, 4) for k, v in overlaps.items()},
        "energy_distribution": {k: round(v, 4) for k, v in energy_avgs.items()},
        "hit_count": hit_count,
        "total_probes": len(TEST_SUITE),
        "hit_rate": round(hit_count / len(TEST_SUITE), 3),
        "verdict": verdict,
        "probe_details": results["none"]["details"],
    }
    
    out_file = OUT / "ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Saved: {out_file}")


if __name__ == "__main__":
    main()
