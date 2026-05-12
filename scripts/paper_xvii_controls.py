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
PAPER XVII --- Outstanding controls for the Analytic Continuation Manifold.

This script implements the three controls called for by the Paper XVII
abstract self-falsification block:

    (1) HELD-OUT TEST. Train h on a strict subset of the 105 critical
        zeros and 60 off-critical points; report fixed-point deviation
        and TEH detection rate on a disjoint test split. The headline
        0.008 vs 0.81 numbers reported in v1 of the paper are in-sample;
        a held-out gap that is comparable would support generalisation,
        a large gap would be the falsification.

    (2) OVERPARAMETERISATION SWEEP. Train embedders with latent
        dimension D in {16, 32, 64, 128, 256, 768} and report training
        and held-out separation as a function of D. With only 165
        training points, a 768-dim embedder has roughly four orders of
        magnitude more parameters than constraints; this sweep tests
        whether the separation persists at modest D or whether it is
        an artefact of overparameterisation.

    (3) RANDOM-ZERO-SET ABLATION. Replace the 60 off-critical points
        with 60 uniformly random points sampled in the same critical
        strip but otherwise structureless. If the embedder still
        achieves comparable critical-vs-other separation, the
        separation reflects only that critical zeros lie on a learned
        cluster (relative to ANY second class), not that the off-
        critical points carry distinguishing structure. This is the
        equivalent of an A/A test for the negative class.

Outputs
-------
    benchmarks/paper_xvii_controls_results.json

Notes
-----
- Self-contained. Uses only numpy / torch. No HF model load required.
- The zero list in this script is a small canonical seed (the first
  ~30 zeros from the literature, accurate to 6 decimal places); the
  full 105-zero list, if available locally, can be passed via --zeros.
- This script does not attempt to reproduce the exact figures in the
  current Paper XVII; it is built so that the new figures it produces
  can directly support or falsify the abstract's in-sample claims.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = ROOT / "benchmarks" / "paper_xvii_controls_results.json"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ----- the canonical first ~30 nontrivial zeros of zeta on the critical line.
# Source: Andrew Odlyzko's tables; values rounded to 6 dp here. The user
# should supply a longer list via --zeros to match Paper XVII's 105.
DEFAULT_CRITICAL_T = [
    14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
    37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
    52.970321, 56.446248, 59.347044, 60.831779, 65.112544,
    67.079811, 69.546402, 72.067158, 75.704690, 77.144840,
    79.337375, 82.910381, 84.735493, 87.425275, 88.809111,
    92.491899, 94.651344, 95.870634, 98.831194, 101.317851,
]


# ---------------------------------------------------------------------- #
# Embedder
# ---------------------------------------------------------------------- #

class ComplexToReal(nn.Module):
    """Simple MLP s = sigma + i t  ->  R^D.

    Note: this is intentionally minimal. The aim is to reproduce the
    Paper XVII setup faithfully enough that the controls below answer the
    question they are designed to ask (in-sample fit vs held-out fit).
    A more elaborate architecture would not change what the controls
    measure.
    """

    def __init__(self, D: int):
        super().__init__()
        H = max(D, 32)
        self.net = nn.Sequential(
            nn.Linear(2, H),
            nn.SiLU(),
            nn.Linear(H, H),
            nn.SiLU(),
            nn.Linear(H, D),
        )

    def forward(self, sr: torch.Tensor, si: torch.Tensor) -> torch.Tensor:
        x = torch.stack([sr, si], dim=-1)
        return self.net(x)


def involution(sigma: torch.Tensor, t: torch.Tensor):
    """The functional-equation involution iota(s) = 1 - s."""
    return 1.0 - sigma, -t


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #

def train_embedder(
    crit_pts: list[tuple[float, float]],
    neg_pts: list[tuple[float, float]],
    D: int,
    steps: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> ComplexToReal:
    torch.manual_seed(seed)
    np.random.seed(seed)
    h = ComplexToReal(D).to(device)
    opt = torch.optim.AdamW(h.parameters(), lr=lr)

    crit = torch.tensor(crit_pts, dtype=torch.float32, device=device)   # (N_c, 2): (sigma, t)
    neg = torch.tensor(neg_pts, dtype=torch.float32, device=device)     # (N_n, 2)

    for step in range(steps):
        opt.zero_grad()

        # Critical: encourage h(s) ~ h(iota(s)), i.e. fixed point.
        sr, si = crit[:, 0], crit[:, 1]
        ir, ii = involution(sr, si)
        h_s = h(sr, si)
        h_is = h(ir, ii)
        crit_loss = ((h_s - h_is) ** 2).mean()

        # Off-critical: push h(s) and h(iota(s)) apart so iota acts non-trivially.
        nsr, nsi = neg[:, 0], neg[:, 1]
        nir, nii = involution(nsr, nsi)
        h_n = h(nsr, nsi)
        h_in = h(nir, nii)
        neg_sep = ((h_n - h_in) ** 2).mean()
        neg_loss = torch.exp(-neg_sep)  # bounded margin-style penalty

        # Cluster critical points together (so off-critical separation has reference).
        c_mean = h_s.mean(dim=0, keepdim=True)
        cluster = ((h_s - c_mean) ** 2).mean()

        loss = crit_loss + 0.5 * neg_loss + 0.05 * cluster
        loss.backward()
        opt.step()

    return h


# ---------------------------------------------------------------------- #
# Metrics
# ---------------------------------------------------------------------- #

@torch.no_grad()
def fixed_point_deviation(h: ComplexToReal, pts: list[tuple[float, float]],
                          device: torch.device) -> float:
    if not pts:
        return float("nan")
    p = torch.tensor(pts, dtype=torch.float32, device=device)
    sr, si = p[:, 0], p[:, 1]
    ir, ii = involution(sr, si)
    h_s = h(sr, si)
    h_is = h(ir, ii)
    norms = torch.linalg.norm(h_s, dim=-1).clamp(min=1e-8)
    devs = torch.linalg.norm(h_s - h_is, dim=-1) / norms
    return float(devs.mean().item())


@torch.no_grad()
def teh_detection_rate(h: ComplexToReal,
                       crit_train: list[tuple[float, float]],
                       test_neg: list[tuple[float, float]],
                       test_crit: list[tuple[float, float]],
                       device: torch.device) -> dict:
    """Build forbidden subspace Q_f from training criticals, then measure
    TEH energy on held-out positive and negative points."""
    if not crit_train or not test_neg:
        return {"detected": None, "false_positive": None, "n_neg": len(test_neg), "n_crit": len(test_crit)}

    crit_t = torch.tensor(crit_train, dtype=torch.float32, device=device)
    H = h(crit_t[:, 0], crit_t[:, 1])  # (N, D)
    # Centre and SVD-decompose to recover the "critical-line subspace"
    Hc = H - H.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Hc, full_matrices=False)
    # Forbidden subspace = directions of LOW variance among critical training pts
    # Take the bottom half of right singular vectors as Q_f.
    k_keep = max(1, S.numel() // 2)
    Qf = Vh[k_keep:, :].T  # (D, D - k_keep)

    def teh_ratio(pts: list[tuple[float, float]]) -> torch.Tensor:
        p = torch.tensor(pts, dtype=torch.float32, device=device)
        x = h(p[:, 0], p[:, 1])  # (M, D)
        proj = x @ Qf @ Qf.T
        num = torch.linalg.norm(proj, dim=-1)
        den = torch.linalg.norm(x, dim=-1).clamp(min=1e-8)
        return num / den

    teh_neg = teh_ratio(test_neg)
    teh_crit = teh_ratio(test_crit) if test_crit else torch.tensor([0.0], device=device)

    # A simple threshold midway between mean(crit) and mean(neg)
    thr = 0.5 * (float(teh_crit.mean()) + float(teh_neg.mean()))
    detected = float((teh_neg > thr).float().mean())
    fpr = float((teh_crit > thr).float().mean()) if test_crit else float("nan")
    return {
        "detected": detected,
        "false_positive": fpr,
        "teh_neg_mean": float(teh_neg.mean()),
        "teh_crit_mean": float(teh_crit.mean()),
        "n_neg": len(test_neg),
        "n_crit": len(test_crit),
    }


# ---------------------------------------------------------------------- #
# Point-set construction
# ---------------------------------------------------------------------- #

def make_critical(t_values: list[float]) -> list[tuple[float, float]]:
    return [(0.5, float(t)) for t in t_values]


def make_off_critical(t_values: list[float], offsets: list[float], rng: np.random.Generator) -> list[tuple[float, float]]:
    out = []
    for t in t_values:
        sig = float(rng.choice(offsets))
        out.append((sig, float(t)))
    return out


def make_random_strip(n: int, t_max: float, rng: np.random.Generator) -> list[tuple[float, float]]:
    """Uniformly random points in (0, 1) x (0, t_max)."""
    sig = rng.uniform(0.05, 0.95, size=n)
    t = rng.uniform(0.0, t_max, size=n)
    return [(float(s), float(tt)) for s, tt in zip(sig, t)]


# ---------------------------------------------------------------------- #
# Three controls
# ---------------------------------------------------------------------- #

def control_held_out(t_critical: list[float], offsets: list[float], D: int,
                     steps: int, seeds: int, train_frac: float, device: torch.device) -> dict:
    rng = np.random.default_rng(0)
    results = []
    for s in range(seeds):
        rng_s = np.random.default_rng(1000 + s)
        # Split critical zeros
        n_c = len(t_critical)
        idx = np.arange(n_c)
        rng_s.shuffle(idx)
        n_train = max(2, int(round(train_frac * n_c)))
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        crit_train = make_critical([t_critical[i] for i in train_idx])
        crit_test = make_critical([t_critical[i] for i in test_idx])

        # Off-critical: build matched train/test pools
        n_neg_train = max(2, n_train // 2)
        n_neg_test = max(2, (n_c - n_train) // 2)
        neg_t_train = rng_s.uniform(min(t_critical), max(t_critical), size=n_neg_train).tolist()
        neg_t_test = rng_s.uniform(min(t_critical), max(t_critical), size=n_neg_test).tolist()
        neg_train = make_off_critical(neg_t_train, offsets, rng_s)
        neg_test = make_off_critical(neg_t_test, offsets, rng_s)

        h = train_embedder(crit_train, neg_train, D=D, steps=steps,
                           lr=1e-3, seed=42 + s, device=device)
        fp_dev_train = fixed_point_deviation(h, crit_train, device)
        fp_dev_test = fixed_point_deviation(h, crit_test, device)
        neg_dev_train = fixed_point_deviation(h, neg_train, device)
        neg_dev_test = fixed_point_deviation(h, neg_test, device)
        teh = teh_detection_rate(h, crit_train, neg_test, crit_test, device)

        results.append({
            "seed": int(42 + s),
            "n_train_crit": int(len(crit_train)),
            "n_test_crit": int(len(crit_test)),
            "n_train_neg": int(len(neg_train)),
            "n_test_neg": int(len(neg_test)),
            "fp_dev_crit_train": fp_dev_train,
            "fp_dev_crit_test": fp_dev_test,
            "fp_dev_neg_train": neg_dev_train,
            "fp_dev_neg_test": neg_dev_test,
            "in_sample_separation": neg_dev_train - fp_dev_train,
            "held_out_separation": neg_dev_test - fp_dev_test,
            "held_out_gap": (neg_dev_train - fp_dev_train) - (neg_dev_test - fp_dev_test),
            "teh_held_out": teh,
        })
    return {"D": D, "train_frac": train_frac, "seeds_run": results,
            "in_sample_sep_mean": float(np.mean([r["in_sample_separation"] for r in results])),
            "held_out_sep_mean": float(np.mean([r["held_out_separation"] for r in results])),
            "held_out_gap_mean": float(np.mean([r["held_out_gap"] for r in results]))}


def control_overparam(t_critical: list[float], offsets: list[float],
                      Ds: list[int], steps: int, seed: int,
                      train_frac: float, device: torch.device) -> dict:
    out = []
    for D in Ds:
        r = control_held_out(t_critical, offsets, D=D, steps=steps,
                             seeds=1, train_frac=train_frac, device=device)
        # Single-seed condensation
        s = r["seeds_run"][0]
        out.append({
            "D": D,
            "params_estimate": _estimate_params(D),
            "in_sample_separation": s["in_sample_separation"],
            "held_out_separation": s["held_out_separation"],
            "held_out_gap": s["held_out_gap"],
            "teh_detected": s["teh_held_out"]["detected"],
            "teh_fpr": s["teh_held_out"]["false_positive"],
        })
    return {"sweep": out, "interpretation":
            "If held_out_separation grows roughly with D, the v1 separation "
            "is likely an overparam fit. If it plateaus at small D, the "
            "separation reflects structure rather than parameter count."}


def _estimate_params(D: int) -> int:
    H = max(D, 32)
    return 2 * H + H + H * H + H + H * D + D


def control_random_zero_set(t_critical: list[float], D: int, steps: int,
                            seeds: int, device: torch.device) -> dict:
    """Replace structured off-critical points with uniformly random strip points.

    H_artefact for this control: if the embedder still achieves the same
    critical-cluster vs not-on-cluster separation, the v1 separation only
    reflects the existence of a cluster, not that the off-critical class
    encodes anything zeta-specific.
    """
    rng = np.random.default_rng(7)
    crit_pts = make_critical(t_critical)
    out = []
    for s in range(seeds):
        rng_s = np.random.default_rng(2000 + s)
        rand_neg = make_random_strip(len(crit_pts), max(t_critical) * 1.05, rng_s)
        h = train_embedder(crit_pts, rand_neg, D=D, steps=steps,
                           lr=1e-3, seed=42 + s, device=device)
        fp_crit = fixed_point_deviation(h, crit_pts, device)
        fp_neg = fixed_point_deviation(h, rand_neg, device)
        out.append({
            "seed": int(42 + s),
            "fp_dev_crit_train": fp_crit,
            "fp_dev_random_neg_train": fp_neg,
            "separation": fp_neg - fp_crit,
        })
    return {"D": D, "n_pts": len(crit_pts), "seeds_run": out,
            "mean_separation_with_random_neg": float(np.mean([r["separation"] for r in out])),
            "interpretation":
            "Compare to held-out control's in_sample_sep_mean. If they are "
            "comparable, the v1 separation does not depend on the negative "
            "class carrying real off-critical structure."}


# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zeros", default="", help="Optional path to a text file with one zero (imaginary part) per line. Defaults to a 30-zero seed.")
    ap.add_argument("--D-default", type=int, default=128)
    ap.add_argument("--D-sweep", default="16,32,64,128,256,768")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args()

    if args.zeros and Path(args.zeros).is_file():
        t_critical = []
        for line in Path(args.zeros).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                t_critical.append(float(line.split()[-1]))
            except ValueError:
                pass
    else:
        t_critical = list(DEFAULT_CRITICAL_T)

    offsets = [0.3, 0.4, 0.45, 0.55, 0.6, 0.7]  # off-critical sigma values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# Paper XVII controls: device={device}, n_zeros={len(t_critical)}")

    t0 = time.time()

    print("\n== Control 1: HELD-OUT TEST ==")
    held_out = control_held_out(t_critical, offsets, D=args.D_default,
                                steps=args.steps, seeds=args.seeds,
                                train_frac=args.train_frac, device=device)
    print(f"   in-sample sep (mean): {held_out['in_sample_sep_mean']:.4f}")
    print(f"   held-out  sep (mean): {held_out['held_out_sep_mean']:.4f}")
    print(f"   held-out  gap (mean): {held_out['held_out_gap_mean']:+.4f}  "
          f"(positive = in-sample > held-out, i.e. partial overfit)")

    print("\n== Control 2: OVERPARAMETERISATION SWEEP ==")
    Ds = [int(x) for x in args.D_sweep.split(",")]
    overparam = control_overparam(t_critical, offsets, Ds=Ds, steps=args.steps,
                                  seed=42, train_frac=args.train_frac, device=device)
    for row in overparam["sweep"]:
        print(f"   D={row['D']:>4}  params~{row['params_estimate']:>7}  "
              f"in_sample={row['in_sample_separation']:.4f}  "
              f"held_out={row['held_out_separation']:.4f}  "
              f"gap={row['held_out_gap']:+.4f}")

    print("\n== Control 3: RANDOM-ZERO-SET ABLATION ==")
    random_neg = control_random_zero_set(t_critical, D=args.D_default,
                                         steps=args.steps, seeds=args.seeds, device=device)
    print(f"   sep (random negatives): {random_neg['mean_separation_with_random_neg']:.4f}")
    print("   compare to in-sample sep above; if comparable, off-critical structure")
    print("   was not driving the separation in v1.")

    summary = {
        "meta": {
            "n_zeros": len(t_critical),
            "D_default": args.D_default,
            "D_sweep": Ds,
            "steps": args.steps,
            "seeds": args.seeds,
            "train_frac": args.train_frac,
            "wallclock_seconds": round(time.time() - t0, 2),
            "torch_version": torch.__version__,
            "device": str(device),
        },
        "control_1_held_out": held_out,
        "control_2_overparam_sweep": overparam,
        "control_3_random_zero_set": random_neg,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n# wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
