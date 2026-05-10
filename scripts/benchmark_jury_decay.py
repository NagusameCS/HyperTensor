#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#!/usr/bin/env python3
"""
BENCHMARK: J-DECAY CURVE — Jury confidence vs distance from manifold
=====================================================================
Measures the exact relationship between cosine distance from the
manifold and jury confidence J on REAL model hidden states.

This is the central empirical validation of the Unified Manifold
Theory of Truth: J -> 1.0 inside the manifold, J -> 0.5 at the
instinct horizon d_h, J -> 0.0 outside.

Output: benchmarks/jury_decay/jury_decay_curve.json
        benchmarks/jury_decay/jury_decay_chart.png

Usage: python scripts/benchmark_jury_decay.py [--model MODEL_ID]
"""
import torch, json, time, os, sys, math
from pathlib import Path
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from ott_engine import JuryDraftGate

torch.set_grad_enabled(False)
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("benchmarks/jury_decay")
OUT.mkdir(parents=True, exist_ok=True)

# 
# CONFIGURATION
# 

N_JURORS = 7
N_DIST_BINS = 50  # resolution of the J(d) curve
N_SAMPLES_PER_BIN = 100  # statistical robustness

# 
# BUILD MANIFOLD FROM REAL HIDDEN STATES
# 

def build_manifold_from_model(model_id="Qwen/Qwen2.5-0.5B-Instruct"):
    """Extract real hidden states from a model to build the manifold."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"  Loading {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    d_model = model.config.hidden_size
    K = min(d_model // 16, 64)
    
    # Build UGT basis from calibration prompts
    cal_prompts = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius.",
        "The Earth orbits the Sun once per year.",
        "Photosynthesis produces oxygen from carbon dioxide.",
        "The speed of light is approximately 300 million meters per second.",
        "DNA carries genetic information in a double helix.",
        "The Pythagorean theorem relates the sides of a right triangle.",
        "Shakespeare wrote Hamlet and Macbeth.",
        "Newton's second law: force equals mass times acceleration.",
        "The human body has 206 bones.",
        "Gravity accelerates objects at 9.8 meters per second squared.",
        "The Amazon rainforest produces about 20 percent of Earth's oxygen.",
        "Beethoven composed nine symphonies.",
        "Diamond is a form of pure carbon.",
        "The first moon landing was in 1969.",
        "H2O is the chemical formula for water.",
        "The electron has a negative electric charge.",
        "Tokyo is the capital of Japan.",
        "The cheetah is the fastest land animal.",
        "Oxygen is the most abundant element in Earth's crust.",
        "The Great Wall of China is over 13,000 miles long.",
        "The piano has 88 keys.",
        "A haiku is a three-line poem with 5-7-5 syllables.",
        "Leonardo da Vinci painted the Mona Lisa.",
        "Platinum is more expensive than gold.",
        "The human brain has approximately 86 billion neurons.",
        "Sound travels at about 343 meters per second.",
        "Mount Everest is the tallest mountain above sea level.",
        "The Great Pyramid of Giza was built around 2560 BCE.",
        "Vincent van Gogh painted Starry Night.",
    ]
    
    hs_list = []
    for prompt in cal_prompts:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs_list.append(out.hidden_states[-1][0, -1, :].float())
    
    hs_tensor = torch.stack(hs_list)
    hs_centered = hs_tensor - hs_tensor.mean(dim=0)
    U, S, _ = torch.linalg.svd(hs_centered.T, full_matrices=False)
    basis = U[:, :K].float().to(DEVICE)
    
    print(f"  Model: {model_id} | d={d_model} | k={K} | trajectories={len(cal_prompts)}")
    
    return model, tok, basis, d_model, K, cal_prompts


def project_and_build_trajectories(model, tok, basis, prompts):
    """Project hidden states to k-space and build trajectory pool."""
    trajectories = []
    for prompt in prompts:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float()
        hk = (h @ basis).cpu()
        trajectories.append({"proj": hk, "label": prompt[:60]})
    return trajectories


# 
# MEASURE J(d) CURVE
# 

def measure_j_decay(jury, traj_stack, K, n_bins=N_DIST_BINS, n_samples=N_SAMPLES_PER_BIN):
    """Measure jury confidence J as a function of cosine distance from manifold.
    
    Strategy: interpolate between a random anchor trajectory and a far-away
    random point in k-space. As the interpolation weight t goes from 0 to 1,
    the cosine distance from the manifold increases continuously, and J decays.
    
    This produces the full J(d) curve from J~1.0 (inside) through J=0.5 (horizon)
    to J~0.0 (outside).
    """
    traj_norm = F.normalize(traj_stack.float(), dim=1)
    
    results = []
    
    # Use interpolation weights to smoothly walk from inside to outside
    for t in torch.linspace(0.0, 1.0, n_bins):
        t_val = t.item()
        J_vals = []
        cos_dists = []
        
        for _ in range(n_samples):
            # Pick a random trajectory as anchor (inside the manifold)
            anchor_idx = torch.randint(0, len(traj_stack), (1,)).item()
            anchor = traj_stack[anchor_idx]
            
            # Generate a truly random far-away point (outside the manifold)
            far = torch.randn(K) * 5.0  # far from origin
            
            # Interpolate: t=0 -> at anchor (inside), t=1 -> at far point (outside)
            query = (1.0 - t_val) * anchor + t_val * far
            
            # Measure J
            q_norm = F.normalize(query.unsqueeze(0).float(), dim=1)
            sims = (traj_norm @ q_norm.T).squeeze(-1)
            top_sims, top_idx = torch.topk(sims, k=N_JURORS)
            
            # Euclidean distance to nearest trajectory IN K-SPACE
            # Use the actual (unnormalized) trajectory positions
            nearest_idx = top_idx[0].item()
            euclidean_dist = torch.norm(query - traj_stack[nearest_idx]).item()
            
            # Jury formula — distances in k-space Euclidean units
            # Convert cosine distances to Euclidean via ||a-b||^2 = ||a||^2 + ||b||^2 - 2||a||||b||cos_sim
            # But simpler: just use the top-sim distances from the jury formula directly
            # The jury in the code uses cosine_distance = 1 - cos_sim internally
            cos_dists_jury = 1.0 - top_sims
            confidences = torch.exp(-cos_dists_jury / jury.R)
            J = (1.0 - torch.prod(1.0 - confidences)).item()
            
            J_vals.append(J)
            cos_dists.append(actual_dist)
        
        if J_vals:
            mean_J = sum(J_vals) / len(J_vals)
            mean_dist = sum(cos_dists) / len(cos_dists)
            std_J = (sum((j - mean_J)**2 for j in J_vals) / len(J_vals)) ** 0.5
            
            if mean_J >= 0.99:
                status = "deeply_familiar"
            elif mean_J >= 0.85:
                status = "inside_grounded"
            elif mean_J >= 0.50:
                status = "inside_weakening"
            elif mean_J >= 0.25:
                status = "near_horizon"
            elif mean_J >= 0.05:
                status = "outside_extrapolating"
            else:
                status = "deeply_unfamiliar"
            
            results.append({
                "interpolation_t": round(t_val, 4),
                "cosine_distance": round(mean_dist, 6),
                "jury_J_mean": round(mean_J, 6),
                "jury_J_std": round(std_J, 6),
                "n_samples": len(J_vals),
                "status": status,
            })
    
    return results, jury.R, jury.R * (-math.log(1.0 - 0.5 ** (1.0 / N_JURORS)))


# 
# MAIN
# 

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic manifold (no model load)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  J-DECAY CURVE BENCHMARK")
    print("  Jury confidence J as a function of cosine distance d")
    print("=" * 70)
    
    if args.synthetic:
        K = 64
        n_traj = 128
        print(f"  Synthetic manifold: {n_traj} trajectories, k={K}")
        traj_stack = torch.randn(n_traj, K) * 0.3
        trajectories = [{"proj": traj_stack[i], "label": f"topic_{i%4}"}
                       for i in range(n_traj)]
    else:
        print("[1/3] Loading model...")
        model, tok, basis, d_model, K, cal_prompts = build_manifold_from_model(args.model)
        print("[2/3] Building trajectory pool...")
        trajectories = project_and_build_trajectories(model, tok, basis, cal_prompts)
        traj_stack = torch.stack([t["proj"] for t in trajectories])
    
    print("[3/3] Measuring J(d) curve...")
    jury = JuryDraftGate(threshold=0.85, n_jurors=N_JURORS)
    jury.calibrate(trajectories)
    
    curve, R, d_h = measure_j_decay(jury, traj_stack, K)
    
    # Print results
    print(f"\n  Manifold: R={R:.4f} | d_h={d_h:.4f} | N={N_JURORS}")
    print(f"  Instinct horizon at cos_dist ≈ {d_h:.4f}")
    print()
    print(f"  {'cos_dist':>10s} | {'Jury J':>8s} | {'±std':>8s} | {'Status':>25s}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*25}")
    
    near_horizon_idx = -1
    for i, r in enumerate(curve):
        marker = ""
        if abs(r["jury_J_mean"] - 0.5) < 0.15 and near_horizon_idx < 0:
            marker = " <-- INSTINCT HORIZON (J=0.5)"
            near_horizon_idx = i
        # Show every 5th point for readability
        if i % max(1, len(curve)//10) == 0 or marker:
            print(f"  t={r.get('interpolation_t',0):.3f} | "
                  f"cos_dist={r['cosine_distance']:7.4f} | "
                  f"J={r['jury_J_mean']:7.4f} | "
                  f"{r['status']:>25s}{marker}")
    # Always show last point
    if curve:
        r = curve[-1]
        print(f"  t={r.get('interpolation_t',0):.3f} | "
              f"cos_dist={r['cosine_distance']:7.4f} | "
              f"J={r['jury_J_mean']:7.4f} | "
              f"{r['status']:>25s}")
    
    # Find the actual horizon point
    horizon_dist = None
    for r in curve:
        if r["jury_J_mean"] >= 0.48 and r["jury_J_mean"] <= 0.52:
            horizon_dist = r["cosine_distance"]
            break
    if horizon_dist is None:
        # Interpolate
        for i in range(len(curve)-1):
            j1, j2 = curve[i]["jury_J_mean"], curve[i+1]["jury_J_mean"]
            if (j1 - 0.5) * (j2 - 0.5) <= 0:
                d1, d2 = curve[i]["cosine_distance"], curve[i+1]["cosine_distance"]
                t = (0.5 - j1) / (j2 - j1) if j2 != j1 else 0
                horizon_dist = d1 + t * (d2 - d1)
                break
    
    # Compute key metrics
    inside_J = [r["jury_J_mean"] for r in curve if r["cosine_distance"] < d_h]
    outside_J = [r["jury_J_mean"] for r in curve if r["cosine_distance"] > d_h * 1.5]
    inside_mean = sum(inside_J)/len(inside_J) if inside_J else 0
    outside_mean = sum(outside_J)/len(outside_J) if outside_J else 0
    
    print(f"\n  KEY METRICS:")
    print(f"  Coverage radius R:          {R:.4f}")
    print(f"  Theoretical d_h (formula):   {d_h:.4f}")
    print(f"  Measured d_h (J=0.5):        {horizon_dist:.4f}" if horizon_dist else "  Measured d_h: not found")
    print(f"  d_h/R ratio (theoretical):   {d_h/R:.2f}")
    if horizon_dist:
        print(f"  d_h/R ratio (measured):      {horizon_dist/R:.2f}")
    print(f"  Mean J inside d_h:           {inside_mean:.4f}")
    print(f"  Mean J outside 1.5xd_h:      {outside_mean:.4f}")
    print(f"  J-separation (inside-outside): {inside_mean - outside_mean:.4f}")
    
    # Save
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model if not args.synthetic else "synthetic",
        "K": K,
        "n_trajectories": len(trajectories),
        "n_jurors": N_JURORS,
        "coverage_radius_R": round(R, 6),
        "theoretical_d_h": round(d_h, 6),
        "measured_d_h": round(horizon_dist, 6) if horizon_dist else None,
        "mean_J_inside": round(inside_mean, 6),
        "mean_J_outside": round(outside_mean, 6),
        "J_separation": round(inside_mean - outside_mean, 6),
        "curve": curve,
    }
    
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = OUT / f"jury_decay_curve_{stamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Saved to: {report_path}")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    main()
