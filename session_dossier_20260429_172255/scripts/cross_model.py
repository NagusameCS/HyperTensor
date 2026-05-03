"""
curvature_warp/cross_model.py
=============================

Run the v1 (Christoffel-warp) and v2 (covariant warp) protocols on every
cached Phase-1/Phase-3 manifold under ``legacy/axiom_vis/`` to test
whether the negative result on smollm2-135m generalises across models.

Output: ``docs/figures/curvature_warp/cross_model_summary.json``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "gtc"))

from inject import run_protocol            # type: ignore
from v2 import run_protocol_v2             # type: ignore
from _phase_io import REPO, AXVIS          # type: ignore


def discover_models() -> list[str]:
    out = []
    for d in sorted(AXVIS.iterdir()):
        if not d.is_dir():
            continue
        if (d / "phase1_manifold.json").exists() and (d / "phase3_curvature.json").exists():
            out.append(d.name)
    return out


def main():
    models = discover_models()
    print(f"[curvature_warp/cross_model] models: {models}")

    rows = []
    t_total = time.time()
    for m in models:
        for n_intrinsic in (6, 8):
            print(f"  -> {m}  n_intrinsic={n_intrinsic}  v1...")
            t0 = time.time()
            try:
                r1 = run_protocol(model=m, n_intrinsic=n_intrinsic,
                                  n_facts=30, strength=0.7, sigma=1.2,
                                  T=16, dl=0.1, seed=20260429)
                r1["variant"] = "v1_christoffel"
                rows.append(r1)
                print(f"     v1: improvement={r1['improvement']:.3%}  "
                      f"spillover_mean={r1['spillover_mean'] if r1['spillover_mean'] is not None else 'NA'}  "
                      f"OK={r1['success_50pct_reduction'] and r1['success_spillover_under_5pct']}  "
                      f"({time.time()-t0:.1f}s)")
            except Exception as e:  # noqa: BLE001
                print(f"     v1 FAIL: {e}")
                rows.append({"model": m, "n_intrinsic": n_intrinsic, "variant": "v1_christoffel",
                             "error": str(e)})

            print(f"  -> {m}  n_intrinsic={n_intrinsic}  v2...")
            t0 = time.time()
            try:
                r2 = run_protocol_v2(model=m, n_intrinsic=n_intrinsic,
                                     n_facts=30, alpha=0.35, radius=1.2,
                                     T=16, dl=0.1, seed=20260429)
                r2["variant"] = "v2_covariant"
                rows.append(r2)
                print(f"     v2: improvement={r2['improvement']:.3%}  "
                      f"spillover_mean={r2['spillover_mean']:.4f}  "
                      f"OK={r2['success']}  ({time.time()-t0:.1f}s)")
            except Exception as e:  # noqa: BLE001
                print(f"     v2 FAIL: {e}")
                rows.append({"model": m, "n_intrinsic": n_intrinsic, "variant": "v2_covariant",
                             "error": str(e)})

    wall = time.time() - t_total
    out_path = REPO / "docs" / "figures" / "curvature_warp" / "cross_model_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"models": models, "wall_s": wall, "rows": rows}, indent=2),
                        encoding="utf-8")

    print(f"\n[curvature_warp/cross_model] wrote {out_path}")
    print(f"  total wall: {wall:.1f}s,  {len(rows)} rows")
    n_pass = sum(1 for r in rows
                 if r.get("success_50pct_reduction") and r.get("success_spillover_under_5pct"))
    print(f"  v1 protocols passing 50% + <5% spillover: {n_pass}/{sum(1 for r in rows if r.get('variant')=='v1_christoffel' and 'error' not in r)}")
    n_pass2 = sum(1 for r in rows if r.get("success"))
    print(f"  v2 protocols passing combined success:  {n_pass2}/{sum(1 for r in rows if r.get('variant')=='v2_covariant' and 'error' not in r)}")


if __name__ == "__main__":
    main()
