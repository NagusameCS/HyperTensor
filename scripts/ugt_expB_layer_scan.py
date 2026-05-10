#!/usr/bin/env python3
"""Experiment B — Layer-window scan ablation.

For each window size k ∈ {1, 3, 7} and starting index i ∈ [0, L-k]:
  * Subtract zone projection at layers [i, i+k) only (instead of all layers).
  * Measure damage = baseline_prob − ablated_prob, averaged over predicted-
    diagonal probes (cat == zone).
  * Plot/record damage(B) − damage(B') vs (i, k, zone).

A "zones live in specific layers" reading would predict a localized window
where damage(B) − damage(B') > 0 with statistical significance.

Output: benchmarks/expB_layer_scan.json (3D array)
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


class WindowAblator:
    """Hook subset of decoder layers, project out zone subspace at residual stream."""

    def __init__(self, model, basis: torch.Tensor, zone_slice: tuple[int, int],
                 layer_lo: int, layer_hi: int):
        self.model = model
        self.handles: list = []
        lo, hi = zone_slice
        Bz = basis[:, lo:hi].contiguous()
        q, _ = torch.linalg.qr(Bz, mode="reduced")
        self.proj = (q @ q.T)
        self.layer_lo = layer_lo
        self.layer_hi = layer_hi

    def _hook(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
            P = self.proj.to(dtype=h.dtype, device=h.device)
            return (h - h @ P,) + outputs[1:]
        h = outputs
        P = self.proj.to(dtype=h.dtype, device=h.device)
        return h - h @ P

    def __enter__(self):
        try:
            layers = self.model.model.layers
        except AttributeError:
            layers = self.model.transformer.h
        for li, layer in enumerate(layers):
            if self.layer_lo <= li < self.layer_hi:
                self.handles.append(layer.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def measure_prob(model, tok, probe: dict, device) -> tuple[float, list[int]]:
    msgs = [{"role": "user", "content": probe["prompt"]}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(txt, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    last = out.logits[0, -1, :]
    probs = F.softmax(last.float(), -1)
    cids = []
    for chk in probe["check"]:
        ids = tok.encode(chk, add_special_tokens=False)
        if ids:
            cids.append(ids[0])
    if not cids:
        return 0.0, []
    return float(probs[cids].sum().item()), cids


@torch.no_grad()
def measure_prob_under_hook(model, tok, probe: dict, cids: list[int], device) -> float:
    if not cids:
        return 0.0
    msgs = [{"role": "user", "content": probe["prompt"]}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(txt, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    last = out.logits[0, -1, :]
    probs = F.softmax(last.float(), -1)
    return float(probs[cids].sum().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--zones", default="12,24,32")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--top-lambda", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--windows", default="1,3,7", help="window sizes")
    ap.add_argument("--stride", type=int, default=2,
                    help="stride for window starting position (1=every layer)")
    ap.add_argument("--out", default=str(common.OUT / "expB_layer_scan.json"))
    args = ap.parse_args()

    zones = [int(x) for x in args.zones.split(",")]
    windows = [int(x) for x in args.windows.split(",")]
    rb = common.get_rb_mod()
    suite = rb.EXTENDED_TEST_SUITE
    zone_names = ["syntax", "algorithmic", "factual"]

    print(f"[B] model={args.model}  windows={windows}  stride={args.stride}  seeds={args.seeds}")
    model, tok, dev = common.load_model(args.model)
    model_tag = args.model.split("/")[-1].replace("-Instruct", "").lower()
    try:
        L = len(model.model.layers)
    except AttributeError:
        L = len(model.transformer.h)

    # Baselines: P_correct for each probe (no hook)
    print("    measuring baselines...")
    baselines = {}
    for p in suite:
        b, cids = measure_prob(model, tok, p, dev)
        baselines[p["id"]] = (b, cids)

    # Train bases per seed (cached)
    bases_B: dict[int, torch.Tensor] = {}
    bases_Bp: dict[int, torch.Tensor] = {}
    for s in range(args.seeds):
        seed = 42 + s
        B = common.get_or_train_basis(model, tok, args.k, zones, args.steps,
                                      args.top_lambda, seed, model_tag).to(dev)
        bases_B[seed] = B
        d = B.shape[0]
        bases_Bp[seed] = common.make_random_basis(d, args.k, seed, dev)

    # Zone slices
    slices = []
    prev = 0
    for ze in zones:
        slices.append((prev, ze))
        prev = ze

    # Sweep
    rows: list[dict] = []
    for w in windows:
        for i in range(0, L - w + 1, args.stride):
            for zi, zname in enumerate(zone_names):
                # Restrict to predicted-diagonal probes (cat == zone)
                diag_probes = [p for p in suite if p["category"] == zname]
                deltas_B_per_seed = []
                deltas_Bp_per_seed = []
                for s in range(args.seeds):
                    seed = 42 + s
                    # B
                    with WindowAblator(model, bases_B[seed], slices[zi], i, i + w):
                        ds = []
                        for p in diag_probes:
                            base, cids = baselines[p["id"]]
                            after = measure_prob_under_hook(model, tok, p, cids, dev)
                            ds.append(base - after)
                        deltas_B_per_seed.append(float(np.mean(ds)))
                    # B'
                    with WindowAblator(model, bases_Bp[seed], slices[zi], i, i + w):
                        ds = []
                        for p in diag_probes:
                            base, cids = baselines[p["id"]]
                            after = measure_prob_under_hook(model, tok, p, cids, dev)
                            ds.append(base - after)
                        deltas_Bp_per_seed.append(float(np.mean(ds)))
                rows.append({
                    "window": w, "i_start": i, "i_end": i + w, "zone": zname,
                    "B_minus_Bp_mean": float(np.mean(deltas_B_per_seed) - np.mean(deltas_Bp_per_seed)),
                    "B_mean_damage": float(np.mean(deltas_B_per_seed)),
                    "Bp_mean_damage": float(np.mean(deltas_Bp_per_seed)),
                    "B_per_seed": deltas_B_per_seed,
                    "Bp_per_seed": deltas_Bp_per_seed,
                })
            print(f"    w={w} i={i:>2}…{i+w:>2}  done")

    # Find the predicted-positive maximum cell
    best = max(rows, key=lambda r: r["B_minus_Bp_mean"])
    print(f"\n  best (B−B') predicted-positive: w={best['window']} layers=[{best['i_start']},{best['i_end']})  "
          f"zone={best['zone']}  Δ={best['B_minus_Bp_mean']:+.4f}")

    common.safe_dump(Path(args.out), {
        "meta": {"model": args.model, "k": args.k, "zones": zones,
                  "seeds": args.seeds, "windows": windows, "L": L,
                  "stride": args.stride},
        "rows": rows,
        "best_predicted_positive": best,
    })


if __name__ == "__main__":
    main()
