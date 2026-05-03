"""Grid sweep for curvature warp v2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from v2 import run_protocol_v2


def main():
    alphas = [0.10, 0.20, 0.30, 0.40]
    radii = [0.6, 0.9, 1.2, 1.8]
    dls = [0.05, 0.1]

    rows = []
    for alpha in alphas:
        for radius in radii:
            for dl in dls:
                out = run_protocol_v2(alpha=alpha, radius=radius, dl=dl, n_facts=32)
                row = {
                    "alpha": alpha,
                    "radius": radius,
                    "dl": dl,
                    "improvement": out["improvement"],
                    "spillover_mean": out["spillover_mean"],
                    "success": out["success"],
                    "post_err_mean": out["post_err_mean"],
                    "pre_err_mean": out["pre_err_mean"],
                }
                rows.append(row)
                print(
                    f"[v2] a={alpha:.2f} r={radius:.2f} dl={dl:.2f} "
                    f"impr={row['improvement']:.1%} sp={row['spillover_mean']:.2%} "
                    f"ok={row['success']}"
                )

    rows_sorted = sorted(rows, key=lambda r: (r["success"], r["improvement"]), reverse=True)
    best = rows_sorted[0]

    out = {
        "grid_size": len(rows),
        "success_count": int(sum(1 for r in rows if r["success"])),
        "best": best,
        "all": rows_sorted,
    }

    repo = Path(__file__).resolve().parents[2]
    out_path = repo / "docs" / "figures" / "curvature_warp" / "smollm2-135m_v2_grid.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[v2] wrote {out_path}")


if __name__ == "__main__":
    main()
