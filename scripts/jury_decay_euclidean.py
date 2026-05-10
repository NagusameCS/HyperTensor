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
FIXED J-DECAY BENCHMARK — Proper Euclidean distance metric
===========================================================
The jury gate uses Euclidean distance for calibration but cosine
distance for queries. This benchmark unifies both: we measure J as
a function of Euclidean k-space distance, which is the metric the
formula d_h = R * (-ln(1-0.5^(1/N))) operates in.

The key: d_h is in EUCLIDEAN k-space units, same as R.
We walk a query point AWAY from the manifold center in Euclidean
space, measure the Euclidean distance to the nearest trajectory,
and compute J using the jury formula with proper Euclidean metric.

python scripts/jury_decay_euclidean.py --synthetic
"""
import torch, json, time, math, sys
from pathlib import Path
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from ott_engine import JuryDraftGate

torch.manual_seed(42)

N_JURORS = 7
N_BINS = 40
N_SAMPLES = 100
OUT = Path("benchmarks/jury_decay")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  J-DECAY vs EUCLIDEAN DISTANCE")
print("  Using consistent Euclidean metric throughout")
print("=" * 70)

# Build manifold with small-norm trajectories so Euclidean and
# cosine distances scale proportionally
K = 32
n_traj = 128

# Trajectories cluster around the origin with small variance
traj_stack = torch.randn(n_traj, K) * 0.5
trajectories = [{"proj": traj_stack[i], "label": f"topic_{i%4}"}
               for i in range(n_traj)]

# Calibrate: R from Euclidean nearest-neighbor distances
jury = JuryDraftGate(threshold=0.85, n_jurors=N_JURORS)
jury.calibrate(trajectories)

# Get the stored Euclidean R
# The jury stores trajectories as-is (unnormalized k-space vectors)
# The jury_confidence function normalizes them for cosine computation
# We need to measure J vs Euclidean distance consistently

# For each test point, compute:
# 1. Euclidean distance to nearest trajectory
# 2. Jury confidence J (using the jury's own formula)
print(f"  Manifold: {n_traj} trajectories, k={K}")
print(f"  Coverage radius R = {jury.R:.4f} (Euclidean)")
d_h = jury.R * (-math.log(1.0 - 0.5 ** (1.0 / N_JURORS)))
print(f"  Instinct horizon d_h = {d_h:.4f} (J=0.5)")
print()

# Walk outward: generate queries at increasing Euclidean distances
print(f"  {'euclidean_d':>12s} | {'Jury J':>8s} | {'Status':>25s}")
print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*25}")

results = []

for dist_mult in torch.linspace(0.0, 4.0, N_BINS):
    dist_target = dist_mult.item() * jury.R  # target Euclidean distance in units of R
    J_vals = []
    euclidean_dists = []
    
    for _ in range(N_SAMPLES):
        # Pick random anchor
        anchor_idx = torch.randint(0, n_traj, (1,)).item()
        anchor = traj_stack[anchor_idx]
        
        # Generate a direction that maximizes Euclidean distance from the cloud
        # Use the direction to the farthest trajectory
        farthest_idx = torch.argmax(torch.norm(traj_stack - anchor, dim=1)).item()
        farthest = traj_stack[farthest_idx]
        direction = farthest - anchor
        direction = direction / (torch.norm(direction) + 1e-10)
        
        # Step outward
        query = anchor + dist_target * direction
        
        # Compute Euclidean distance to nearest trajectory
        all_dists = torch.norm(query.unsqueeze(0) - traj_stack, dim=1)
        nearest_dist = all_dists.min().item()
        
        # Compute J directly using EUCLIDEAN distances (consistent metric)
        # J = 1 - prod(1 - exp(-d_i/R)) where d_i are Euclidean NN distances
        top_dists, _ = torch.topk(all_dists, k=N_JURORS, largest=False)
        confidences = torch.exp(-top_dists / jury.R)
        J = (1.0 - torch.prod(1.0 - confidences)).item()
        
        J_vals.append(J)
        euclidean_dists.append(nearest_dist)
    
    if J_vals:
        mean_J = sum(J_vals) / len(J_vals)
        mean_euc = sum(euclidean_dists) / len(euclidean_dists)
        units_R = mean_euc / jury.R  # in units of R
        
        if mean_J >= 0.99: status = "DEEPLY FAMILIAR"
        elif mean_J >= 0.85: status = "INSIDE (grounded)"
        elif mean_J >= 0.50: status = "INSIDE (weakening)"
        elif mean_J >= 0.25: status = "NEAR HORIZON"
        elif mean_J >= 0.05: status = "OUTSIDE"
        else: status = "DEEPLY UNFAMILIAR"
        
        marker = ""
        if abs(mean_J - 0.5) < 0.15:
            marker = " <-- INSTINCT HORIZON"
        
        results.append({
            "euclidean_dist": round(mean_euc, 4),
            "dist_units_of_R": round(units_R, 2),
            "jury_J": round(mean_J, 4),
            "status": status,
        })
        
        # Print every 5th
        if len(results) % max(1, N_BINS//8) == 0 or marker:
            print(f"  {mean_euc:10.4f} ({units_R:.1f}R) | {mean_J:8.4f} | {status:>25s}{marker}")

# Show the horizon point if found
horizon_euc = None
for r in results:
    if abs(r["jury_J"] - 0.5) < 0.1:
        horizon_euc = r["euclidean_dist"]
        break

print()
print(f"  KEY METRICS:")
print(f"  R (Euclidean):             {jury.R:.4f}")
print(f"  d_h (theoretical):         {d_h:.4f}")
if horizon_euc:
    print(f"  d_h (measured, J~0.5):     {horizon_euc:.4f}")
    print(f"  d_h/R (measured):          {horizon_euc/jury.R:.2f}")
print(f"  d_h/R (theoretical):       {d_h/jury.R:.2f}")

# Save
report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "K": K, "n_jurors": N_JURORS, "n_trajectories": n_traj,
    "R_euclidean": round(jury.R, 4),
    "d_h_theoretical": round(d_h, 4),
    "d_h_measured": round(horizon_euc, 4) if horizon_euc else None,
    "results": results,
}
stamp = time.strftime("%Y%m%d_%H%M%S")
with open(OUT / f"jury_decay_euclidean_{stamp}.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"  Saved: benchmarks/jury_decay/jury_decay_euclidean_{stamp}.json")
print("=" * 70)
