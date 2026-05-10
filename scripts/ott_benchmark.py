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
OTT BENCHMARK — Measure real geodesic speculative decode + jury gate speedup
=============================================================================

Tests the three OTT pipeline stages end-to-end on a real model:
  1. GeodesicDraftGenerator — PCA-projected geodesic step quality
  2. JuryDraftGate — jury confidence calibration + gating accuracy
  3. OTTEngine — end-to-end speedup vs baseline

Usage:
  python scripts/ott_benchmark.py                              # default 0.5B, quick
  python scripts/ott_benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --n 50
  python scripts/ott_benchmark.py --quick                       # synthetic-only (no model load)

Output:
  benchmarks/ott_engine/benchmark_report.json

William "Nagusame" Stewart — HyperTensor 2026
"""
import torch, json, time, os, sys, argparse, math, random
from pathlib import Path
from collections import defaultdict
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from ott_engine import OTTEngine, JuryDraftGate, GeodesicDraftGenerator

torch.set_grad_enabled(False)
torch.manual_seed(42); random.seed(42)

OUT = Path("benchmarks/ott_engine")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------------------------------
# BENCHMARK 1: Geodesic Step Quality
# -------------------------------------------------------

def benchmark_geodesic(model, tok, basis, d_model, n_trials=100):
    """Measure geodesic step cosine similarity vs. actual next-token embedding."""
    print(f"\n  [1/4] Geodesic Draft Quality ({n_trials} trials)...")
    
    geo = GeodesicDraftGenerator(basis, d_model)
    
    # Collect hidden states for calibration
    prompts = [
        "The mitochondria is the powerhouse of the cell.",
        "Newton's second law states that F equals ma.",
        "Quantum mechanics describes wave-particle duality.",
        "The Pythagorean theorem relates triangle sides.",
        "A transformer uses self-attention for sequences.",
        "Gradient descent minimizes the loss function.",
        "The Riemann zeta function encodes prime distribution.",
        "DNA replication is semiconservative in nature.",
    ]
    hs_list = []
    for p in prompts:
        enc = tok(p, return_tensors="pt", truncation=True, max_length=64).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs_list.append(out.hidden_states[-1][0, -1, :].float())
    
    hs_tensor = torch.stack(hs_list)
    geo.calibrate(hs_tensor)
    
    # Test: predict next token embedding from geodesic step
    cos_sims = []
    l2_errors = []
    
    test_words = ["Therefore", "However", "The answer is", "In conclusion",
                  "Let me explain", "First", "Next", "Finally"]
    
    for word in test_words[:n_trials]:
        enc = tok(word, return_tensors="pt", truncation=True, max_length=4).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        
        if enc.input_ids.shape[1] < 2:
            continue
        
        h_curr = out.hidden_states[-1][0, -1, :].float()
        h_prev = out.hidden_states[-1][0, -2, :].float()
        
        # Geodesic prediction
        p_pred = geo.geodesic_step(h_curr, h_prev)
        e_pred = p_pred @ basis.float().T
        
        # Actual next hidden state (use model's actual next prediction)
        out2 = model.generate(**enc, max_new_tokens=1, output_hidden_states=True,
                               return_dict_in_generate=True, do_sample=False)
        if hasattr(out2, 'hidden_states') and out2.hidden_states:
            h_actual = out2.hidden_states[-1][-1][0, -1, :].float()
        else:
            continue
        
        # Cosine similarity
        cos = F.cosine_similarity(e_pred.unsqueeze(0), h_actual.unsqueeze(0)).item()
        cos_sims.append(cos)
        
        # L2 error
        l2 = torch.norm(e_pred - h_actual).item()
        l2_errors.append(l2)
    
    results = {
        "n_trials": len(cos_sims),
        "cosine_similarity": {
            "mean": round(sum(cos_sims) / len(cos_sims), 4) if cos_sims else 0,
            "min": round(min(cos_sims), 4) if cos_sims else 0,
            "max": round(max(cos_sims), 4) if cos_sims else 0,
        },
        "l2_error": {
            "mean": round(sum(l2_errors) / len(l2_errors), 4) if l2_errors else 0,
        },
        "geodesic_calibrated": geo._n_cal > 0,
        "n_calibration_samples": geo._n_cal,
    }
    
    print(f"    cos_sim: mean={results['cosine_similarity']['mean']:.4f}, "
          f"range=[{results['cosine_similarity']['min']:.4f}, {results['cosine_similarity']['max']:.4f}]")
    print(f"    L2 error: mean={results['l2_error']['mean']:.4f}")
    
    return results


# -------------------------------------------------------
# BENCHMARK 2: Jury Gate Calibration & Accuracy
# -------------------------------------------------------

def benchmark_jury_gate(model, tok, basis, d_model, n_trajectories=64):
    """Measure jury gate calibration and acceptance prediction accuracy."""
    print(f"\n  [2/4] Jury Gate Calibration ({n_trajectories} trajectories)...")
    
    jury = JuryDraftGate(threshold=0.85, n_jurors=7)
    
    # Build trajectory pool from diverse prompts
    diverse_prompts = [
        "The capital of France is Paris.", "Photosynthesis produces oxygen.",
        "The derivative of x squared is 2x.", "Shakespeare wrote Hamlet.",
        "Water boils at 100 degrees Celsius.", "The Earth orbits the Sun.",
        "Python is a programming language.", "The human genome has 23 chromosomes.",
        "Gravity accelerates at 9.8 meters per second squared.",
        "The first law of thermodynamics conserves energy.",
        "Machine learning uses gradient descent.", "The moon affects ocean tides.",
        "Quantum entanglement links particles across distance.",
        "The industrial revolution began in Britain.", "Cells divide through mitosis.",
        "The speed of light is approximately 300 million meters per second.",
        "Natural selection drives evolution.", "The Pythagorean theorem is a²+b²=c².",
    ]
    
    trajectories = []
    for prompt in diverse_prompts[:n_trajectories]:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float()
        hk = h.float() @ basis.float()
        trajectories.append({"proj": hk.cpu(), "label": prompt[:40]})
    
    # Calibrate
    jury.calibrate(trajectories)
    
    # Test: query with in-distribution and out-of-distribution prompts
    in_dist = [
        "The capital of Japan is Tokyo.",
        "Oxygen is produced by plants.",
        "The integral of x is x²/2.",
        "Dickens wrote Oliver Twist.",
    ]
    out_dist = [
        "Zxcvb nmqw er tyu iop asd fgh jkl.",
        "Blargle fnord kazzak zibble wobble.",
        "X7q9 m2p r5t w8y a3 l6k.",
        "asdfghjkl qwertyuiop zxcvbnm.",
    ]
    
    results = {"in_distribution": [], "out_of_distribution": []}
    
    # Test in-distribution
    for prompt in in_dist:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float()
        hk = h.float() @ basis.float()
        
        J, sim, label = jury.jury_confidence(hk)
        accept, _, reason = jury.should_accept(hk)
        results["in_distribution"].append({
            "prompt": prompt[:40],
            "jury_confidence": round(J, 4),
            "similarity": round(sim, 4),
            "accepted": accept,
            "dominant_label": label[:40],
        })
    
    # Test out-of-distribution
    for prompt in out_dist:
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float()
        hk = h.float() @ basis.float()
        
        J, sim, label = jury.jury_confidence(hk)
        accept, _, reason = jury.should_accept(hk)
        results["out_of_distribution"].append({
            "prompt": prompt[:40],
            "jury_confidence": round(J, 4),
            "similarity": round(sim, 4),
            "accepted": accept,
        })
    
    # Summary stats
    in_Js = [r["jury_confidence"] for r in results["in_distribution"]]
    out_Js = [r["jury_confidence"] for r in results["out_of_distribution"]]
    in_accept = sum(1 for r in results["in_distribution"] if r["accepted"])
    out_accept = sum(1 for r in results["out_of_distribution"] if r["accepted"])
    
    stats = {
        "coverage_radius": round(jury.R, 4),
        "n_jurors": len(jury._jurors),
        "in_distribution": {
            "mean_J": round(sum(in_Js) / len(in_Js), 4) if in_Js else 0,
            "acceptance_rate": f"{in_accept}/{len(results['in_distribution'])}",
        },
        "out_of_distribution": {
            "mean_J": round(sum(out_Js) / len(out_Js), 4) if out_Js else 0,
            "acceptance_rate": f"{out_accept}/{len(results['out_of_distribution'])}",
        },
        "separation": round((sum(in_Js)/len(in_Js) - sum(out_Js)/len(out_Js)), 4) if in_Js and out_Js else 0,
        "details": results,
    }
    
    print(f"    Coverage radius R = {stats['coverage_radius']:.4f}")
    print(f"    Jurors: {stats['n_jurors']}")
    print(f"    In-dist:  mean J = {stats['in_distribution']['mean_J']:.4f}, "
          f"accept = {stats['in_distribution']['acceptance_rate']}")
    print(f"    Out-dist: mean J = {stats['out_of_distribution']['mean_J']:.4f}, "
          f"accept = {stats['out_of_distribution']['acceptance_rate']}")
    print(f"    Separation ΔJ = {stats['separation']:.4f}")
    
    return stats


# -------------------------------------------------------
# BENCHMARK 3: End-to-End OTT Speedup
# -------------------------------------------------------

def benchmark_ott_speedup(model, tok, basis, d_model, trajectories, n_turns=30):
    """Measure OTT engine speedup vs baseline (no OTT)."""
    print(f"\n  [3/4] End-to-End OTT Speedup ({n_turns} turns)...")
    
    engine = OTTEngine(basis, d_model, jury_threshold=0.85,
                        acceptance_threshold=0.40, n_drafts=3)
    engine.calibrate_from_trajectories(trajectories)
    engine.reset_stats()
    
    def safe_h(h):
        return h
    
    def to_k(h):
        return h.float() @ basis.float()
    
    def get_h(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        return out.hidden_states[-1][0, -1, :].float()
    
    test_queries = [
        "Explain the theory of relativity in simple terms.",
        "What is the capital of Australia?",
        "How does a neural network learn from data?",
        "Write a haiku about machine learning.",
        "What are the three laws of thermodynamics?",
        "Explain the difference between mitosis and meiosis.",
        "How does public key cryptography work?",
        "What causes ocean tides on Earth?",
        "Describe the process of natural selection.",
        "What is the difference between supervised and unsupervised learning?",
    ]
    
    # Baseline: generate without OTT
    print("    Running baseline (no OTT)...")
    baseline_times = []
    for query in test_queries[:n_turns]:
        enc = tok(query, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=32, do_sample=True,
                                 temperature=0.7, top_p=0.9,
                                 pad_token_id=tok.eos_token_id)
        baseline_times.append((time.time() - t0) * 1000)
    
    # OTT: generate with geodesic drafts + jury gate
    print("    Running OTT (geodesic + jury gate)...")
    ott_times = []
    ott_stats = []
    for query in test_queries[:n_turns]:
        h = get_h(query)
        hk = to_k(safe_h(h))
        
        t0 = time.time()
        # Generate drafts
        drafts = engine.generate_drafts(model, tok, query, h_curr=safe_h(h),
                                         h_prev=None,
                                         vocab_size=model.config.vocab_size)
        
        # Generate actual response
        enc = tok(query, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        np_tok = enc.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=32, do_sample=True,
                                 temperature=0.7, top_p=0.9,
                                 pad_token_id=tok.eos_token_id)
        response = tok.decode(out[0, np_tok:], skip_special_tokens=True).strip()
        
        # Verify
        if drafts:
            best, score, accepted, stats = engine.verify_and_select(
                [response] + drafts, safe_h, to_k, trajectories,
                get_h_func=get_h)
            ott_stats.append(stats)
        
        ott_times.append((time.time() - t0) * 1000)
    
    # Results
    baseline_mean = sum(baseline_times) / len(baseline_times)
    ott_mean = sum(ott_times) / len(ott_times)
    speedup = baseline_mean / ott_mean if ott_mean > 0 else 1.0
    
    engine_st = engine.stats()
    
    results = {
        "n_turns": len(baseline_times),
        "baseline": {
            "mean_ms": round(baseline_mean, 1),
            "p50_ms": round(sorted(baseline_times)[len(baseline_times)//2], 1),
        },
        "ott": {
            "mean_ms": round(ott_mean, 1),
            "p50_ms": round(sorted(ott_times)[len(ott_times)//2], 1),
        },
        "speedup": round(speedup, 2),
        "engine_stats": engine_st,
    }
    
    print(f"    Baseline: {results['baseline']['mean_ms']:.1f}ms mean")
    print(f"    OTT:      {results['ott']['mean_ms']:.1f}ms mean")
    print(f"    Speedup:  {results['speedup']:.2f}×")
    print(f"    Jury bypass: {engine_st['jury_accepted']}/{engine_st['total_drafts']} "
          f"({engine_st['jury_acceptance_rate']}%)")
    print(f"    Est. speedup: {engine_st['estimated_speedup_vs_baseline']}×")
    
    return results


# -------------------------------------------------------
# BENCHMARK 4: Jury vs. No-Jury Ablation
# -------------------------------------------------------

def benchmark_ablation(model, tok, basis, d_model, trajectories, n_drafts=100):
    """Ablation: measure jury gate time vs. full verification time per draft."""
    print(f"\n  [4/4] Jury vs. Verify Ablation ({n_drafts} drafts)...")
    
    engine = OTTEngine(basis, d_model, jury_threshold=0.85, n_drafts=3)
    engine.calibrate_from_trajectories(trajectories)
    
    def safe_h(h):
        return h
    
    def to_k(h):
        return h.float() @ basis.float()
    
    def get_h(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        return out.hidden_states[-1][0, -1, :].float()
    
    jury_times = []
    verify_times = []
    
    test_texts = [
        "The answer is 42.",
        "In conclusion, the evidence supports the hypothesis.",
        "Therefore, we can deduce that the system is stable.",
        "To summarize, the main points are as follows.",
    ]
    
    for text in test_texts:
        for _ in range(n_drafts // len(test_texts)):
            h = get_h(text)
            hk = to_k(safe_h(h))
            
            # Time jury gate only
            t0 = time.time()
            J, sim, label = engine.jury_gate.jury_confidence(hk)
            engine.jury_gate.should_accept(hk)
            jury_times.append((time.time() - t0) * 1000)
            
            # Time full verification (coherence + novelty + stability)
            t0 = time.time()
            if trajectories:
                traj_projs = torch.stack([t["proj"].float() for t in trajectories[-32:]])
                traj_projs = traj_projs.to(hk.device)
                hk_norm = F.normalize(hk.unsqueeze(0).float(), dim=1)
                traj_norm = F.normalize(traj_projs, dim=1)
                sims = (traj_norm @ hk_norm.T).squeeze(-1)
                coherence = sims.max().item()
                dists = torch.norm(hk.unsqueeze(0).float() - traj_projs, dim=1)
                novelty = min(1.0, dists.min().item() / 5.0)
            else:
                coherence = 0.5; novelty = 1.0
            norm = torch.norm(hk).item()
            stability = 1.0 / (1.0 + abs(norm - 0.5))
            score = 0.4 * coherence + 0.3 * novelty + 0.3 * stability
            verify_times.append((time.time() - t0) * 1000)
    
    jury_mean = sum(jury_times) / len(jury_times)
    verify_mean = sum(verify_times) / len(verify_times)
    
    results = {
        "n_samples": len(jury_times),
        "jury_gate": {
            "mean_ms": round(jury_mean, 4),
            "min_ms": round(min(jury_times), 4),
            "max_ms": round(max(jury_times), 4),
        },
        "full_verify": {
            "mean_ms": round(verify_mean, 4),
            "min_ms": round(min(verify_times), 4),
            "max_ms": round(max(verify_times), 4),
        },
        "speedup_factor": round(verify_mean / jury_mean, 1) if jury_mean > 0 else "N/A",
    }
    
    print(f"    Jury gate:    {results['jury_gate']['mean_ms']:.4f}ms mean")
    print(f"    Full verify:  {results['full_verify']['mean_ms']:.4f}ms mean")
    print(f"    Jury speedup: {results['speedup_factor']}× per draft")
    
    return results


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OTT Engine Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model ID (default: Qwen2.5-0.5B)")
    parser.add_argument("--n", type=int, default=20, help="Number of trial turns")
    parser.add_argument("--quick", action="store_true",
                        help="Synthetic-only benchmark (no model load)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("  OTT ENGINE BENCHMARK")
    print(f"  Model: {args.model}")
    print(f"  Device: {DEVICE}")
    print(f"  Mode: {'quick (synthetic)' if args.quick else 'full (real model)'}")
    print("=" * 70)
    
    if args.quick:
        # Synthetic only
        K = 64; d_model = 512
        basis = torch.randn(d_model, K)
        Q, _ = torch.linalg.qr(basis)
        basis = Q[:, :K]
        
        trajectories = [{"proj": torch.randn(K) * 0.3, "label": f"topic_{i%4}"}
                       for i in range(128)]
        
        engine = OTTEngine(basis, d_model, jury_threshold=0.85, n_drafts=3)
        engine.calibrate_from_trajectories(trajectories)
        
        # Test jury gate timing only
        jury = engine.jury_gate
        t0 = time.time()
        for _ in range(1000):
            q = torch.randn(K) * 0.5
            jury.jury_confidence(q)
            jury.should_accept(q)
        jury_ms = (time.time() - t0) * 1000 / 1000
        
        print(f"\n  Synthetic benchmark:")
        print(f"    Jury gate:     {jury_ms:.4f}ms per query")
        print(f"    Jurors active: {len(jury._jurors)}")
        print(f"    Coverage R:    {jury.R:.4f}")
        
        report = {
            "mode": "synthetic",
            "jury_gate_ms": round(jury_ms, 4),
            "n_jurors": len(jury._jurors),
            "coverage_radius": round(jury.R, 4),
        }
    else:
        # Real model benchmark
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("\n  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True,
        )
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        
        d_model = model.config.hidden_size
        
        # Build UGT basis from calibration prompts
        print("  Building UGT basis...")
        cal_prompts = [
            "The mitochondria is the powerhouse of the cell.",
            "Newton's second law: F equals ma.",
            "Quantum mechanics describes particles through waves.",
            "The Pythagorean theorem: a squared plus b squared equals c squared.",
            "A transformer model uses self-attention.",
            "Gradient descent minimizes the loss.",
            "The Riemann zeta function encodes primes.",
            "DNA replication is semiconservative.",
        ]
        hs_list = []
        for p in cal_prompts:
            enc = tok(p, return_tensors="pt", truncation=True, max_length=64).to(model.device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            hs_list.append(out.hidden_states[-1][0, -1, :].float())
        
        hs_tensor = torch.stack(hs_list)
        hs_centered = hs_tensor - hs_tensor.mean(dim=0)
        U, S, _ = torch.linalg.svd(hs_centered.T, full_matrices=False)
        K = min(len(cal_prompts), 64)
        basis = U[:, :K].float().to(model.device)
        print(f"    Basis: {basis.shape}")
        
        # Build trajectory pool
        print("  Building trajectory pool...")
        traj_prompts = [
            "The capital of France is Paris.", "Photosynthesis produces oxygen.",
            "The derivative of x squared is 2x.", "Shakespeare wrote Hamlet.",
            "Water boils at 100 degrees Celsius.", "The Earth orbits the Sun.",
            "Python is a programming language.", "The human genome has 23 chromosomes.",
            "Gravity accelerates at 9.8 m/s².", "The first law of thermodynamics.",
            "Machine learning uses gradient descent.", "The moon affects ocean tides.",
            "Quantum entanglement is nonlocal.", "The industrial revolution.",
            "Cells divide through mitosis.", "The speed of light is c.",
            "Natural selection drives evolution.", "The Pythagorean theorem.",
        ]
        trajectories = []
        for p in traj_prompts:
            enc = tok(p, return_tensors="pt", truncation=True, max_length=64).to(model.device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :].float()
            hk = h.float() @ basis.float()
            trajectories.append({"proj": hk.cpu(), "label": p[:40]})
        
        # Run benchmarks
        geo_results = benchmark_geodesic(model, tok, basis, d_model, n_trials=min(args.n, 8))
        jury_results = benchmark_jury_gate(model, tok, basis, d_model, n_trajectories=64)
        speedup_results = benchmark_ott_speedup(model, tok, basis, d_model,
                                                 trajectories, n_turns=min(args.n, 10))
        ablation_results = benchmark_ablation(model, tok, basis, d_model,
                                               trajectories, n_drafts=args.n * 5)
        
        report = {
            "model": args.model,
            "device": DEVICE,
            "d_model": d_model,
            "K": K,
            "n_turns": args.n,
            "geodesic_quality": geo_results,
            "jury_gate": jury_results,
            "speedup": speedup_results,
            "ablation": ablation_results,
        }
    
    # Save report
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = OUT / f"benchmark_{args.model.replace('/', '_')}_{stamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Report saved to: {report_path}")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    main()
