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
"""Experiment A — Linear-probe AUROC on B-projected vs B'-projected hidden states.

For each (seed, layer) and basis ∈ {B (UGT), B' (random)}:
  1. Project mean-pooled residual hidden state onto basis: x_proj = x @ B
  2. Train a one-vs-rest logistic regression to classify zone identity
     (syntax / algorithmic / factual)
  3. Report 5-fold-CV macro AUROC

A "geometric UGT" reading would predict AUROC(B) >> AUROC(B') because B is
trained to align zone columns with categories, while B' is Haar-random.

Output: benchmarks/expA_linear_probe_auroc.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ugt_exp_common as common  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--zones", default="12,24,32")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--top-lambda", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--layers", default="0,5,10,15,20,25,30",
                    help="comma-list of layer indices (0=embed, L=final)")
    ap.add_argument("--out", default=str(common.OUT / "expA_linear_probe_auroc.json"))
    args = ap.parse_args()

    zones = [int(x) for x in args.zones.split(",")]
    layer_idx = [int(x) for x in args.layers.split(",")]
    rb = common.get_rb_mod()
    suite = rb.EXTENDED_TEST_SUITE  # 30 prompts (10 syn, 10 alg, 10 fac)

    print(f"[A] model={args.model}  k={args.k}  zones={zones}  seeds={args.seeds}")
    model, tok, dev = common.load_model(args.model)
    model_tag = args.model.split("/")[-1].replace("-Instruct", "").lower()

    # Cap layer indices to model size
    try:
        L = len(model.model.layers)
    except AttributeError:
        L = len(model.transformer.h)
    layer_idx = [li for li in layer_idx if 0 <= li <= L]
    print(f"    transformer has {L} blocks; collecting layers {layer_idx}")

    prompts = [p["prompt"] for p in suite]
    labels = [p["category"] for p in suite]

    print("    collecting hidden states (one forward pass per prompt)...")
    t0 = time.time()
    hs_by_layer = common.collect_hidden_states(
        model, tok, prompts, dev, layer_indices=layer_idx)
    print(f"    collected in {time.time()-t0:.1f}s; "
          f"d={next(iter(hs_by_layer.values())).shape[1]}")

    d = next(iter(hs_by_layer.values())).shape[1]

    # Train basis B for each seed (cached)
    B_cache: dict[int, torch.Tensor] = {}
    Bp_cache: dict[int, torch.Tensor] = {}
    for s in range(args.seeds):
        seed = 42 + s
        print(f"    [seed={seed}] train/load B…")
        B = common.get_or_train_basis(
            model, tok, args.k, zones, args.steps, args.top_lambda,
            seed, model_tag).to("cpu")
        B_cache[seed] = B
        Bp_cache[seed] = common.make_random_basis(d, args.k, seed, "cpu")

    # AUROC sweep
    results = {"meta": {"model": args.model, "k": args.k, "zones": zones,
                          "seeds": args.seeds, "n_prompts": len(prompts),
                          "layers": layer_idx},
               "rows": []}

    for li in layer_idx:
        X = hs_by_layer[li]  # (n, d)
        # raw AUROC (no projection)
        raw = common.auroc_one_vs_rest(X, labels)
        for s in range(args.seeds):
            seed = 42 + s
            B = B_cache[seed].numpy()
            Bp = Bp_cache[seed].numpy()
            X_B = X @ B
            X_Bp = X @ Bp
            auc_B = common.auroc_one_vs_rest(X_B, labels)
            auc_Bp = common.auroc_one_vs_rest(X_Bp, labels)
            row = {
                "layer": li,
                "seed": seed,
                "auroc_raw_macro": raw["_macro"],
                "auroc_B_macro": auc_B["_macro"],
                "auroc_Bp_macro": auc_Bp["_macro"],
                "delta": auc_B["_macro"] - auc_Bp["_macro"],
                "auroc_B_per_class": {k: v for k, v in auc_B.items() if k != "_macro"},
                "auroc_Bp_per_class": {k: v for k, v in auc_Bp.items() if k != "_macro"},
            }
            results["rows"].append(row)
            print(f"    layer={li:>2} seed={seed}  AUROC_raw={raw['_macro']:.3f}  "
                  f"B={auc_B['_macro']:.3f}  B'={auc_Bp['_macro']:.3f}  "
                  f"Δ={row['delta']:+.3f}")

    # Aggregate over seeds per layer
    agg = []
    for li in layer_idx:
        rows = [r for r in results["rows"] if r["layer"] == li]
        agg.append({
            "layer": li,
            "auroc_raw_macro": float(np.mean([r["auroc_raw_macro"] for r in rows])),
            "auroc_B_macro_mean": float(np.mean([r["auroc_B_macro"] for r in rows])),
            "auroc_B_macro_std":  float(np.std([r["auroc_B_macro"] for r in rows])),
            "auroc_Bp_macro_mean":float(np.mean([r["auroc_Bp_macro"] for r in rows])),
            "auroc_Bp_macro_std": float(np.std([r["auroc_Bp_macro"] for r in rows])),
            "delta_mean": float(np.mean([r["delta"] for r in rows])),
            "delta_std":  float(np.std([r["delta"] for r in rows])),
        })
    results["aggregated"] = agg

    common.safe_dump(Path(args.out), results)
    print("\n# layer-by-layer aggregate (B vs B' macro AUROC):")
    print("layer | raw | B_mean (sd) | B'_mean (sd) | Δ (B − B')")
    for a in agg:
        print(f"  {a['layer']:>3} | {a['auroc_raw_macro']:.3f} | "
              f"{a['auroc_B_macro_mean']:.3f} ({a['auroc_B_macro_std']:.3f}) | "
              f"{a['auroc_Bp_macro_mean']:.3f} ({a['auroc_Bp_macro_std']:.3f}) | "
              f"{a['delta_mean']:+.3f}")


if __name__ == "__main__":
    main()
