#!/usr/bin/env python3
"""UGT from-scratch functional ablation. Runs directly on EC2 where the model lives."""
import json, torch, numpy as np, sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "benchmarks/ugt_scratch/final"
BASIS_PATH = "benchmarks/ugt_scratch/final/taxonomic_basis.pt"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ugt_infrastructure import TOPLoss

# Load model and basis
print("[1] Loading from-scratch UGT model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
device = next(model.parameters()).device
basis = torch.load(BASIS_PATH, map_location=device, weights_only=True)
k = basis.shape[1]
zones = [k//3, k*2//3, k]
print(f"    d={basis.shape[0]}, k={k}, zones={zones}")

# Verify orthogonality
top_fn = TOPLoss(k=k, zones=zones)
purity = top_fn.purity_score(basis)
_, overlaps = top_fn(basis)
print(f"    Purity: {purity:.4f}")
for kk, v in overlaps.items():
    print(f"    {kk}: {v:.4f}")

# Functional ablation
PROBES = [
    ("syntax", "Write a grammatically correct sentence about a cat.", ["the","a","cat","is","was","sat","on","mat","The","A","Cat","It"]),
    ("syntax", "Complete: The weather today is", ["warm","cold","nice","sunny","rainy","beautiful","hot","clear","lovely"]),
    ("syntax", "Fix: \"he go store yesterday\" ->", ["went","the","to","store","yesterday","He"]),
    ("algorithmic", "12 * 7 =", ["84"]),
    ("algorithmic", "15 + 27 =", ["42"]),
    ("algorithmic", "sqrt(144) =", ["12"]),
    ("factual", "Capital of France:", ["Paris"]),
    ("factual", "Chemical symbol for water:", ["H2O","h2o","H₂O"]),
    ("factual", "Author of Romeo and Juliet:", ["Shakespeare"]),
    ("factual", "Closest planet to the sun:", ["Mercury"]),
]

zone_names = ["syntax", "algorithmic", "factual"]
zone_masks = []
prev = 0
for z_end in zones:
    mask = torch.zeros(k, device=device)
    mask[prev:z_end] = 1.0
    zone_masks.append(mask)
    prev = z_end

P = basis.float().to(device)  # (d, k) — use float32 for math
lm_head = model.lm_head if hasattr(model, 'lm_head') else model.model.lm_head

print(f"\n[2] Ablation test ({len(PROBES)} probes)...")
results = []
hit_count = 0

for cat, prompt, checks in PROBES:
    # From-scratch model is not instruct-tuned — use plain text
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][:, -1, :].float()  # (1, d) — float32 for math
        
        # Zone energies
        h_norm = torch.norm(h)**2
        energies = {}
        prev = 0
        for i, z_end in enumerate(zones):
            h_z = h @ P[:, prev:z_end]
            energies[zone_names[i]] = float(torch.norm(h_z)**2 / max(h_norm, 1e-10))
            prev = z_end
        
        # Baseline logits
        logits_base = lm_head(h.half())
        correct_ids = []
        for chk in checks:
            tid = tokenizer.encode(chk, add_special_tokens=False)
            if tid: correct_ids.extend(tid)
        correct_ids = list(set(correct_ids))
        
        if not correct_ids:
            continue
        
        base_prob = float(torch.softmax(logits_base[0], -1)[correct_ids].sum())
        top_tok = tokenizer.decode([int(torch.argmax(logits_base[0], -1))]).strip()
        correct = any(chk.lower() in top_tok.lower() for chk in checks)
        
        # Ablate each zone
        deltas = {}
        for ai in range(3):
            mask = zone_masks[ai]
            h_k = h @ P  # (1, k)
            h_ablated_k = h_k * (1 - mask.unsqueeze(0))
            # Reconstruct: component in basis + residual
            h_basis = h @ P @ P.T
            h_residual = h - h_basis
            h_reconstructed = h_ablated_k @ P.T + h_residual
            
            logits_ab = lm_head(h_reconstructed.half())
            ab_prob = float(torch.softmax(logits_ab[0], -1)[correct_ids].sum())
            deltas[zone_names[ai]] = round(base_prob - ab_prob, 6)
        
        # Which zone had the largest impact?
        max_z = max(deltas, key=deltas.get)
        if max_z == cat and deltas[cat] > 0.001:
            hit_count += 1
        
        results.append({
            "cat": cat, "prompt": prompt[:50], "correct": correct,
            "top_token": top_tok[:30], "energies": energies, "deltas": deltas,
            "dominant": max_z, "dominant_match": max_z == cat
        })
        
        markers = {z: "!" if deltas.get(z,0)>0.01 else ("~" if deltas.get(z,0)>0.001 else " ") 
                  for z in zone_names}
        print(f"  [{('OK' if correct else '--')}] {cat:<14s} | "
              f"s={energies['syntax']:.3f} a={energies['algorithmic']:.3f} f={energies['factual']:.3f} | "
              f"d: s{markers['syntax']}{deltas.get('syntax',0):.4f} "
              f"a{markers['algorithmic']}{deltas.get('algorithmic',0):.4f} "
              f"f{markers['factual']}{deltas.get('factual',0):.4f} "
              f"[{'MATCH' if max_z==cat else 'MISS'}]")

acc = hit_count / len(PROBES)
print(f"\n  Zone prediction accuracy: {hit_count}/{len(PROBES)} ({acc:.0%})")

# Energy analysis
for z in zone_names:
    avg = np.mean([r["energies"][z] for r in results])
    print(f"  {z}: avg energy={avg:.4f}")

# Verdict
if acc >= 0.5:
    verdict = "TAXONOMY_FUNCTIONAL"
    msg = "UGT zones CORRESPOND to functional specializations after from-scratch training."
elif acc >= 0.3:
    verdict = "TAXONOMY_PARTIAL"  
    msg = "Partial zone separation achieved."
else:
    verdict = "TAXONOMY_ORTHOGONAL_ONLY"
    msg = "Zones are orthogonal but not functionally separated."

print(f"\n  VERDICT: {verdict}")
print(f"  {msg}")

# Save
result = {
    "verdict": verdict,
    "accuracy": round(acc, 3),
    "hit_count": hit_count, "total": len(PROBES),
    "zone_energies": {z: round(np.mean([r["energies"][z] for r in results]), 4) for z in zone_names},
    "probes": results,
}
with open("benchmarks/ugt_scratch/ablation_results.json", "w") as f:
    json.dump(result, f, indent=2)
print("\n  Saved: benchmarks/ugt_scratch/ablation_results.json")
