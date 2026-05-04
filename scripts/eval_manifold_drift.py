#!/usr/bin/env python3
"""
+==================================================================+
|  EVALUATION 2: Manifold Drift Tracker                           |
|                                                                 |
|  Track trajectory coordinates over thousands of turns.          |
|  If the model genuinely evolves, its behavioral coordinates     |
|  should map a continuous journey or cluster around new domains,  |
|  NOT constantly snap back to a static baseline.                 |
|                                                                 |
|  Metrics:                                                       |
|  - Drift rate: mean distance moved per N turns                  |
|  - Cluster count: how many distinct regions does it visit?      |
|  - Home-base pull: does it snap back to origin?                 |
|  - Domain diversity: entropy of visited topic regions           |
|                                                                 |
|  Usage:                                                         |
|    python eval_manifold_drift.py --load state.miku               |
|    python eval_manifold_drift.py --load state.miku --plot       |
+==================================================================+
"""
import json, sys, os, argparse, math
import numpy as np

def analyze_manifold_drift(miku_path, plot=False, output_path="benchmarks/drift_report.json"):
    """Analyze trajectory coordinate evolution from a .miku file."""
    print("=" * 70)
    print("  MANIFOLD DRIFT TRACKER")
    print("=" * 70)
    
    with open(miku_path) as f:
        state = json.load(f)
    
    trajectories = state.get("trajectories", [])
    conv_log = state.get("conversation_log", [])
    
    if not trajectories:
        print("  No trajectories found in this .miku file.")
        return
    
    # Extract coordinate vectors
    coords = np.array([t["proj"] for t in trajectories])
    n, k = coords.shape
    
    print(f"\n  Trajectories: {n}")
    print(f"  Manifold dimension (k): {k}")
    print(f"  Conversation turns: {len(conv_log)}")
    
    # -- Metric 1: Drift Rate --
    # How far does the model move per turn?
    step_dists = []
    for i in range(1, n):
        d = np.linalg.norm(coords[i] - coords[i-1])
        step_dists.append(d)
    
    mean_step = np.mean(step_dists) if step_dists else 0
    total_drift = np.linalg.norm(coords[-1] - coords[0]) if n > 1 else 0
    
    print(f"\n  -- DRIFT --")
    print(f"  Mean step distance: {mean_step:.4f}")
    print(f"  Total drift (first->last): {total_drift:.4f}")
    
    # -- Metric 2: Home-Base Pull --
    # Does the model snap back to the calibration centroid?
    centroid = coords.mean(axis=0)
    dists_from_home = [np.linalg.norm(c - centroid) for c in coords]
    
    # Compute autocorrelation: if it snaps back, early and late distances are similar
    if n > 10:
        early_mean = np.mean(dists_from_home[:n//3])
        late_mean = np.mean(dists_from_home[2*n//3:])
        home_pull_ratio = late_mean / max(early_mean, 1e-10)
    else:
        home_pull_ratio = 1.0
    
    print(f"\n  -- HOME-BASE PULL --")
    print(f"  Early distance from centroid: {np.mean(dists_from_home[:max(1,n//3)]):.4f}")
    print(f"  Late distance from centroid:  {np.mean(dists_from_home[2*n//3:]):.4f}" if n > 10 else "  (Need >10 trajectories)")
    print(f"  Home-pull ratio: {home_pull_ratio:.3f}")
    if home_pull_ratio > 1.5:
        print(f"  [UP] EXPANDING --- model is moving AWAY from baseline over time")
    elif home_pull_ratio < 0.5:
        print(f"   COLLAPSING --- model is snapping BACK to baseline (not learning)")
    else:
        print(f"  ↔️  STABLE --- model hovers near baseline")
    
    # -- Metric 3: Cluster Analysis --
    # How many distinct regions does the model visit?
    if n > 3:
        from sklearn.cluster import DBSCAN
        eps = np.percentile(step_dists, 75) if step_dists else 0.1
        clustering = DBSCAN(eps=max(eps, 0.05), min_samples=2).fit(coords)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        noise_pct = sum(1 for l in clustering.labels_ if l == -1) / n * 100
        
        print(f"\n  -- CLUSTERS --")
        print(f"  Distinct regions visited: {n_clusters}")
        print(f"  Noise points (isolated): {noise_pct:.1f}%")
        if n_clusters >= 3:
            print(f"   DIVERSE EXPLORATION --- model visits multiple distinct conceptual regions")
        elif n_clusters == 2:
            print(f"   BIMODAL --- model oscillates between two regions")
        else:
            print(f"  -> FOCUSED --- model stays in one conceptual region")
    else:
        n_clusters = 1
        noise_pct = 0
    
    # -- Metric 4: Domain Entropy --
    # Calculate entropy of topic distribution from conversation labels
    from collections import Counter
    labels = [t.get("label", "unknown") for t in trajectories]
    # Extract domain from label (first few words)
    domains = []
    for label in labels:
        words = label.lower().split()
        if any(w in words for w in ["math", "riemann", "calculus", "algebra", "geometry", "prime"]):
            domains.append("mathematics")
        elif any(w in words for w in ["quantum", "physics", "biology", "chem", "dna", "science", "cell"]):
            domains.append("science")
        elif any(w in words for w in ["ai", "neural", "model", "compute", "algorithm", "transformer"]):
            domains.append("computing")
        elif any(w in words for w in ["philosophy", "conscious", "ethics", "mind", "godel"]):
            domains.append("philosophy")
        elif any(w in words for w in ["history", "war", "revolution", "ancient", "roman"]):
            domains.append("history")
        elif any(w in words for w in ["art", "music", "shakespeare", "literature", "poem"]):
            domains.append("arts")
        else:
            domains.append("general")
    
    domain_counts = Counter(domains)
    total = sum(domain_counts.values())
    entropy = -sum((c/total) * math.log2(c/total) for c in domain_counts.values())
    max_entropy = math.log2(len(domain_counts)) if domain_counts else 1
    
    print(f"\n  -- DOMAIN DIVERSITY --")
    print(f"  Domains visited: {len(domain_counts)} --- {dict(domain_counts)}")
    print(f"  Entropy: {entropy:.3f} / {max_entropy:.3f} max")
    print(f"  Diversity: {entropy/max_entropy*100:.1f}% of maximum" if max_entropy > 0 else "")
    
    # -- Overall Assessment --
    print(f"\n  === OVERALL ASSESSMENT ===")
    
    drift_score = min(1.0, mean_step / 0.5) * 30
    expansion_score = min(1.0, max(0, home_pull_ratio - 1.0)) * 25
    cluster_score = min(1.0, n_clusters / 5) * 25
    diversity_score = (entropy / max(max_entropy, 1)) * 20
    
    total_score = drift_score + expansion_score + cluster_score + diversity_score
    
    if total_score >= 70:
        verdict = "LIVING --- Model is genuinely evolving: exploring diverse domains, not snapping back to baseline."
    elif total_score >= 40:
        verdict = "GROWING --- Model shows signs of adaptation but is still anchored to baseline."
    elif total_score >= 20:
        verdict = "STATIC --- Model moves but always returns to origin. Limited learning."
    else:
        verdict = "FROZEN --- Model is essentially static. Living memory not engaging."
    
    print(f"  Score: {total_score:.1f}/100")
    print(f"  Verdict: {verdict}")
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    report = {
        "framework": "Manifold Drift",
        "miku_file": miku_path,
        "n_trajectories": n,
        "k_manifold": k,
        "drift": {
            "mean_step_distance": round(float(mean_step), 4),
            "total_drift": round(float(total_drift), 4),
        },
        "home_pull": {
            "ratio": round(float(home_pull_ratio), 3),
            "interpretation": "EXPANDING" if home_pull_ratio > 1.5 else ("COLLAPSING" if home_pull_ratio < 0.5 else "STABLE"),
        },
        "clusters": {
            "n_clusters": n_clusters,
            "noise_pct": round(float(noise_pct), 1),
        },
        "domains": {k: v for k, v in domain_counts.items()},
        "entropy": round(float(entropy), 3),
        "overall_score": round(float(total_score), 1),
        "verdict": verdict,
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, required=True, help=".miku state file")
    parser.add_argument("--plot", action="store_true", help="Generate drift plot")
    parser.add_argument("--output", type=str, default="benchmarks/drift_report.json")
    args = parser.parse_args()
    analyze_manifold_drift(args.load, args.plot, args.output)
