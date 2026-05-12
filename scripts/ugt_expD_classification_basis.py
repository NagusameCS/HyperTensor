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
"""Experiment D — Replace TOP regulariser with zone-classification loss.

Hypothesis: the negative-sign result of run v (layer-wise UGT ablation) might
be a TOP-loss artifact: TOP rewards orthogonality of zone columns w.r.t.
gradients of the LM loss, which under sufficient training pushes B columns
into low-impact residual directions to keep LM loss flat. If we train B
explicitly to *classify* zones from hidden states, the resulting basis is
forced to land on category-discriminative directions; if H_meaningful is
right, ablating its zone columns should hurt the corresponding category.

Procedure
---------
  1. For each prompt in EXTENDED_TEST_SUITE, gather mean-pooled hidden state
     at the residual stream of layer L (final block).
  2. Trainable B ∈ R^{d×k} with zones [12, 24, 32] giving three column slices.
     Optimise:
         logits_z = ‖x @ B[:, z_lo:z_hi]‖²              (R^3)
         loss = CE(logits, zone_label) + 0.01·||B^T B − I||²
     The orthogonality penalty keeps the basis well-conditioned without
     forcing TOP-style independence from LM gradients.
  3. Run the layer-wise residual-stream ablation using this new B and the
     existing matched random B'.

Output: benchmarks/expD_zone_classification_basis.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ugt_exp_common as common  # noqa: E402


def collect_final_layer_means(model, tok, prompts, dev, max_length=256):
    """Return (n, d) numpy array of mean-pooled final hidden states."""
    out = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        try:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            txt = p
        enc = tok(txt, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            o = model(**enc, output_hidden_states=True)
        h = o.hidden_states[-1][0].float()  # (T, d)
        attn = enc.get("attention_mask")
        if attn is not None:
            h = h[attn[0].bool()]
        out.append(h.mean(0).cpu().numpy())
    return np.stack(out, 0)


def train_classification_basis(X: np.ndarray, y: np.ndarray, k: int,
                               zones: list[int], steps: int, seed: int,
                               device, lr: float = 5e-3,
                               ortho_lambda: float = 0.01) -> torch.Tensor:
    """Train B ∈ R^{d×k} so that ‖x B_z‖² is the logit for class z.

    Uses Adam + post-step orthonormalisation of the entire basis via QR.
    """
    torch.manual_seed(seed); np.random.seed(seed)
    d = X.shape[1]
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    B = torch.empty(d, k, device=device, dtype=torch.float32)
    torch.nn.init.orthogonal_(B)
    B = torch.nn.Parameter(B)
    opt = torch.optim.AdamW([B], lr=lr)
    slices = []
    prev = 0
    for ze in zones:
        slices.append((prev, ze))
        prev = ze

    for step in range(steps):
        proj = Xt @ B  # (n, k)
        # zone-norm logits: logit_z = ||proj[:, z_lo:z_hi]||^2
        logits = torch.stack([
            (proj[:, lo:hi] ** 2).sum(dim=1) for (lo, hi) in slices
        ], dim=1)  # (n, 3)
        ce = F.cross_entropy(logits, yt)
        # Soft orthogonality on full B
        BTB = B.T @ B
        I = torch.eye(k, device=device)
        ortho = ((BTB - I) ** 2).sum()
        loss = ce + ortho_lambda * ortho
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Hard re-orthonormalise via QR every 25 steps to prevent drift
        if step % 25 == 24:
            with torch.no_grad():
                q, _ = torch.linalg.qr(B, mode="reduced")
                B.data.copy_(q)
        if (step + 1) % 100 == 0:
            with torch.no_grad():
                acc = (logits.argmax(1) == yt).float().mean().item()
            print(f"    classif-train step {step+1:>4}: ce={ce.item():.4f}  "
                  f"ortho={ortho.item():.2e}  acc={acc:.3f}")
    # final orthonormalisation
    with torch.no_grad():
        q, _ = torch.linalg.qr(B, mode="reduced")
        B.data.copy_(q)
    return B.detach().cpu().contiguous()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--zones", default="12,24,32")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--out", default=str(common.OUT / "expD_zone_classification_basis.json"))
    args = ap.parse_args()

    zones = [int(x) for x in args.zones.split(",")]
    print(f"[D] model={args.model}  k={args.k}  seeds={args.seeds}")
    model, tok, dev = common.load_model(args.model)
    rb = common.get_rb_mod()
    suite = rb.EXTENDED_TEST_SUITE
    cat_names = ["syntax", "algorithmic", "factual"]

    # Hidden states for classification training
    prompts = [p["prompt"] for p in suite]
    y = np.array([cat_names.index(p["category"]) for p in suite])
    print("    collecting final-layer mean states for classification training...")
    X = collect_final_layer_means(model, tok, prompts, dev)
    print(f"    X shape: {X.shape}, y unique: {np.unique(y, return_counts=True)}")

    # Layerwise hooks
    lw = common.get_lw_mod()

    summary = {"meta": {"model": args.model, "k": args.k, "zones": zones,
                          "seeds": args.seeds, "steps": args.steps},
               "per_seed": []}

    for s in range(args.seeds):
        seed = 42 + s
        print(f"\n  [seed={seed}] training classification-loss basis B_class")
        B_class = train_classification_basis(X, y, args.k, zones, args.steps, seed, dev)
        Bp = common.make_random_basis(d=B_class.shape[0], k=args.k, seed=seed, device=dev).cpu()

        # Run layer-wise ablation under B_class and B'
        print(f"  [seed={seed}] layer-wise ablation under B_class")
        res_B = lw.run_layerwise_ablation(model, tok, B_class, zones, suite, dev)
        print(f"  [seed={seed}] layer-wise ablation under B'")
        res_Bp = lw.run_layerwise_ablation(model, tok, Bp, zones, suite, dev)

        summary["per_seed"].append({
            "seed": seed,
            "B": lw.summarise("B_class", res_B),
            "B_random": lw.summarise("B_random", res_Bp),
        })
        del B_class, Bp, res_B, res_Bp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary["aggregated"] = lw.aggregate(summary["per_seed"])
    common.safe_dump(Path(args.out), summary)

    # Print contrast in same format as run v
    print("\n# Layer-wise B_class vs B' contrast:")
    paired = rb._paired_stats
    agg = summary["aggregated"]["by_category_then_ablated_zone"]
    for cat, zb in agg.items():
        for z, row in zb.items():
            d = row["B_minus_Brand"]
            ps = row.get("paired_stats", {}) or {}
            tp = ps.get("t_p_two_sided"); wp = ps.get("wilcoxon_p_two_sided")
            sig = (tp is not None and tp < 0.05)
            mark = " *" if (cat == z and d is not None and d > 0 and sig) else "  "
            tp_str = f"t_p={tp:.3f}" if tp is not None else "t_p=--   "
            wp_str = f"w_p={wp:.3f}" if wp is not None else "w_p=--   "
            print(f"   cat={cat:<12}  z={z:<12}  delta={d:+.4f}{mark}  {tp_str}  {wp_str}")


if __name__ == "__main__":
    main()
