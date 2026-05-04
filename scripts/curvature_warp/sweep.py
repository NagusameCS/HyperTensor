"""
curvature_warp/sweep.py
========================

Sweep (strength, sigma, dl) for the warp protocol; find the parameter
region where mean geodesic error to target drops at least 50 % without
spillover exceeding 5 % at unrelated points.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "gtc"))

from inject import run_protocol  # type: ignore
from _phase_io import REPO  # type: ignore


def main():
    grid = []
    for strength in (0.4, 0.7, 0.9, 0.99):
        for sigma in (0.3, 0.6, 1.2, 2.4):
            for dl in (0.05, 0.1):
                grid.append((strength, sigma, dl))

    rows = []
    t0 = time.time()
    for (s, sig, dl) in grid:
        out = run_protocol(strength=s, sigma=sig, dl=dl, n_facts=20)
        rows.append({
            "strength": s, "sigma": sig, "dl": dl,
            "improvement": out["improvement"],
            "pre_err_mean": out["pre_err_mean"],
            "post_err_mean": out["post_err_mean"],
            "spillover_mean": out["spillover_mean"],
            "spillover_p95": out["spillover_p95"],
            "wall_s": out["wall_s"],
            "ok": out["success_50pct_reduction"] and out["success_spillover_under_5pct"],
        })
        flag = "OK" if rows[-1]["ok"] else "  "
        print(f"  [{flag}] strength={s:.2f}  sigma={sig:.2f}  dl={dl:.2f}  "
              f"improvement={out['improvement']:>6.2%}  "
              f"spillover={out['spillover_mean'] or 0:.3f}  "
              f"wall={out['wall_s']:.1f}s")
    wall = time.time() - t0

    out_path = REPO / "docs" / "figures" / "curvature_warp" / "smollm2-135m_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows, "wall_s": wall}, indent=2),
                        encoding="utf-8")
    oks = [r for r in rows if r["ok"]]
    print(f"\n[curvature_warp/sweep] {len(oks)}/{len(rows)} configs pass; "
          f"total wall {wall:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
