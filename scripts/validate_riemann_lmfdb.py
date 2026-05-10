#!/usr/bin/env python3
"""End-to-end validation of AGT/ACM detection on real Odlyzko + LMFDB data.

Runs three at-scale experiments:

  (A) Odlyzko zeros1 — first 10,000 ζ-zeros (out of 100k cached).
  (B) Odlyzko zeros4 — 10,000 ζ-zeros near height 1.44 × 10²⁰
      (zero numbers 10²¹ + 1 .. 10²¹ + 10⁴ — covers the "10²⁰ regime").
  (C) LMFDB L-functions — every degree-1 and degree-2 L-function with
      published positive_zeros, scored individually + aggregated.

For each experiment we report:
  • # zeros tested
  • mean / max ‖D(s)‖
  • on-critical fraction (D_norm < tol)
  • aggregate jury confidence (extended precision)
  • wall time

Results written to benchmarks/riemann_lmfdb_validation.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from riemann_pipeline import (  # noqa: E402
    PipelineConfig,
    RiemannPipeline,
)
from data_sources import (  # noqa: E402
    load_riemann_zeros,
    load_lmfdb_lfunctions,
)

OUT = ROOT / "benchmarks" / "riemann_lmfdb_validation.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Header offset for Odlyzko zeros4 file (zero #10^21 + 1 ≈ 1.44e20 + 538.498)
ZEROS4_BASE = 144_176_897_509_546_973_000
# Header offset for zeros3 (zero #10^12+1)
ZEROS3_BASE = 267_653_395_647


def _summarise(label: str, points, pipe: RiemannPipeline, tol: float = 1e-6) -> dict:
    """Run one batch through the pipeline and produce a JSON-friendly record."""
    t0 = time.perf_counter()
    res = pipe.process_batch(points)
    elapsed = time.perf_counter() - t0
    return {
        "label": label,
        "n_total": res.n_total,
        "n_on_critical": res.n_on_critical,
        "n_off_critical": res.n_off_critical,
        "on_critical_fraction": res.n_on_critical / max(res.n_total, 1),
        "mean_D_norm": res.mean_D_norm,
        "max_D_norm": res.max_D_norm,
        "jury_confidence": res.jury_confidence,
        "wall_time_s": round(elapsed, 3),
    }


def experiment_a(pipe: RiemannPipeline, n: int = 10_000) -> dict:
    print(f"[A] Odlyzko zeros1 — first {n:,} ζ-zeros (height ~14..1e4)")
    ts = load_riemann_zeros(n=n, source="zeros1")
    points = [(0.5, t) for t in ts]
    rec = _summarise("odlyzko_zeros1", points, pipe)
    print(f"    on-critical: {rec['n_on_critical']}/{rec['n_total']}  "
          f"max ‖D‖={rec['max_D_norm']:.3e}  jury={rec['jury_confidence']}  "
          f"({rec['wall_time_s']}s)")
    return rec


def experiment_b(pipe: RiemannPipeline) -> dict:
    print("[B] Odlyzko zeros4 — 10⁴ ζ-zeros near height 1.44 × 10²⁰")
    text = (ROOT / "data" / "odlyzko" / "zeros4").read_text(encoding="ascii", errors="ignore")
    offsets: list[float] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or any(c.isalpha() for c in s):
            continue
        try:
            offsets.append(float(s.split()[0]))
        except ValueError:
            continue
    # Reconstruct true heights at extra precision-friendly form (float OK for
    # the AGT feature map, which only uses log/sin/cos of the magnitude).
    points = [(0.5, ZEROS4_BASE + delta) for delta in offsets]
    rec = _summarise("odlyzko_zeros4_height_1.44e20", points, pipe)
    rec["base_height"] = ZEROS4_BASE
    rec["n_offsets_loaded"] = len(offsets)
    print(f"    on-critical: {rec['n_on_critical']}/{rec['n_total']}  "
          f"max ‖D‖={rec['max_D_norm']:.3e}  jury={rec['jury_confidence']}  "
          f"({rec['wall_time_s']}s)")
    return rec


def experiment_c(pipe: RiemannPipeline) -> dict:
    print("[C] LMFDB L-functions at scale — degrees 1 + 2")
    records = load_lmfdb_lfunctions(degree=None)
    points: list[tuple[float, float]] = []
    per_label: dict[str, int] = {}
    for r in records:
        zs = r.get("positive_zeros") or []
        if not zs:
            continue
        per_label[r.get("label", "?")] = len(zs)
        for z in zs:
            try:
                points.append((0.5, float(z)))
            except (TypeError, ValueError):
                pass
    print(f"    {len(per_label)} L-functions, {len(points):,} total zeros")
    rec = _summarise("lmfdb_deg1_deg2_zeros", points, pipe)
    rec["n_lfunctions"] = len(per_label)
    rec["zeros_per_lfunction_min"] = min(per_label.values()) if per_label else 0
    rec["zeros_per_lfunction_max"] = max(per_label.values()) if per_label else 0
    print(f"    on-critical: {rec['n_on_critical']}/{rec['n_total']}  "
          f"max ‖D‖={rec['max_D_norm']:.3e}  jury={rec['jury_confidence']}  "
          f"({rec['wall_time_s']}s)")
    return rec


def experiment_d(pipe: RiemannPipeline, n_real: int = 5_000,
                 sigma_offsets=(0.05, 0.1, 0.2, 0.4)) -> dict:
    """Discrimination test: real ζ-zeros vs σ-perturbed off-critical points."""
    print("[D] Discrimination — real on-critical vs off-critical perturbations")
    ts = load_riemann_zeros(n=n_real, source="zeros1")

    real_pts = [(0.5, t) for t in ts]
    real_norms = [pipe.feature_map.difference_operator(s, t)
                  for (s, t) in real_pts]
    import numpy as np
    real_norm_vals = [float(np.linalg.norm(D)) for D in real_norms]

    per_offset = []
    for d in sigma_offsets:
        off_pts = [(0.5 + d, t) for t in ts]
        off_norms = [float(np.linalg.norm(
            pipe.feature_map.difference_operator(s, t))) for (s, t) in off_pts]
        # Detection rule: D_norm > tol => off-critical
        tol = 1e-6
        tp = sum(1 for v in off_norms if v > tol)        # off correctly flagged
        fp = sum(1 for v in real_norm_vals if v > tol)   # real wrongly flagged
        per_offset.append({
            "sigma_offset": d,
            "n": len(off_pts),
            "off_critical_mean_D": float(np.mean(off_norms)),
            "off_critical_min_D":  float(np.min(off_norms)),
            "off_critical_max_D":  float(np.max(off_norms)),
            "true_positive_rate":  tp / len(off_pts),
            "false_positive_rate": fp / len(real_norm_vals),
        })
        print(f"    σ-offset {d:>4}:  TPR={tp/len(off_pts):.4f}  "
              f"FPR={fp/len(real_norm_vals):.4f}  "
              f"mean ‖D‖={np.mean(off_norms):.3e}")

    return {
        "label": "discrimination_real_vs_perturbed",
        "n_real": len(real_pts),
        "real_max_D": float(np.max(real_norm_vals)),
        "per_offset": per_offset,
    }


def main() -> None:
    cfg = PipelineConfig(precision_dps=50, num_primes=2_000, batch_size=10_000)
    pipe = RiemannPipeline(cfg)

    summary = {
        "config": {
            "precision_dps": cfg.precision_dps,
            "num_primes": cfg.num_primes,
            "feature_dim": cfg.feature_dim,
        },
        "experiments": [],
    }

    summary["experiments"].append(experiment_a(pipe, n=10_000))
    summary["experiments"].append(experiment_b(pipe))
    summary["experiments"].append(experiment_c(pipe))
    summary["experiments"].append(experiment_d(pipe))

    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
