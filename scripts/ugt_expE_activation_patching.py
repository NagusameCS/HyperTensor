#!/usr/bin/env python3
"""Experiment E — Activation patching at the residual stream.

Standard mech-interp causal test: instead of zeroing the zone subspace at the
residual stream, *replace* the zone-projected component with the same zone
component from a different prompt of the same category. This isolates the
"category info carried by the zone subspace" from "ablating that subspace
breaks the LM in any way".

Procedure
---------
For each (probe p, zone z, basis_kind ∈ {B, B'}):
    1. Run a *donor* prompt p_donor of the same category as p, capturing the
       residual-stream activation at every layer (shape (T_donor, d)).
       Reduce to a per-layer "zone donation" vector by mean-pooling and
       projecting onto B_z B_z^T:    v_layer = mean_T h @ P_z   (shape (d,))
    2. Run p with a forward hook at every layer that performs:
           h ← h - h @ P_z + v_layer        (broadcast over T)
       i.e. swap p's zone component for the donor's zone component.
    3. Measure baseline_prob(p) and patched_prob(p) and report
       drift = patched − baseline. Positive ≈ donor info carries category
       relevant to p; we expect this to be small / random for non-zone
       categories, and largest when zone matches p's category if H_meaningful.

Output: benchmarks/expE_activation_patching.json
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


@torch.no_grad()
def collect_per_layer_residual(model, tok, prompt, dev, max_length=256):
    """Return (L+1, d) tensor of mean-pooled hidden states across all layers."""
    msgs = [{"role": "user", "content": prompt}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(txt, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(dev) for k, v in enc.items()}
    o = model(**enc, output_hidden_states=True)
    attn = enc["attention_mask"][0].bool()
    out = []
    for h in o.hidden_states:
        h0 = h[0].float()[attn]
        out.append(h0.mean(0))
    return torch.stack(out, 0)  # (L+1, d)


class PatchInjector:
    """At every block, replace zone component of h with the donor zone vector."""

    def __init__(self, model, basis: torch.Tensor, zone_slice, donor_per_layer):
        """
        donor_per_layer: tensor (L_blocks, d) — zone-projected donor vector per
        block output. We use index 1..L (skip embedding output).
        """
        self.model = model
        self.handles: list = []
        lo, hi = zone_slice
        Bz = basis[:, lo:hi].contiguous()
        q, _ = torch.linalg.qr(Bz, mode="reduced")
        self.proj = (q @ q.T)
        self.donor = donor_per_layer  # (L_blocks, d)
        self._idx = 0

    def _make_hook(self, li):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                h = outputs[0]
                P = self.proj.to(dtype=h.dtype, device=h.device)
                d_vec = self.donor[li].to(dtype=h.dtype, device=h.device)  # (d,)
                # Replace zone component with donor's zone-projected vector
                h_no_zone = h - h @ P
                # broadcast d_vec over (B, T)
                h_new = h_no_zone + d_vec.view(1, 1, -1).expand_as(h)
                # Project that to keep it within zone span (donor already is, but defensive)
                return (h_new,) + outputs[1:]
            h = outputs
            P = self.proj.to(dtype=h.dtype, device=h.device)
            d_vec = self.donor[li].to(dtype=h.dtype, device=h.device)
            return h - h @ P + d_vec.view(1, 1, -1).expand_as(h)
        return hook

    def __enter__(self):
        try:
            layers = self.model.model.layers
        except AttributeError:
            layers = self.model.transformer.h
        for li, layer in enumerate(layers):
            self.handles.append(layer.register_forward_hook(self._make_hook(li)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def measure_correct_prob(model, tok, probe, dev, max_length=256):
    msgs = [{"role": "user", "content": probe["prompt"]}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(txt, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(dev) for k, v in enc.items()}
    o = model(**enc)
    last = o.logits[0, -1, :]
    probs = F.softmax(last.float(), -1)
    cids = []
    for chk in probe["check"]:
        ids = tok.encode(chk, add_special_tokens=False)
        if ids:
            cids.append(ids[0])
    if not cids:
        return 0.0, []
    return float(probs[cids].sum().item()), cids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--zones", default="12,24,32")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--top-lambda", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(common.OUT / "expE_activation_patching.json"))
    args = ap.parse_args()

    zones = [int(x) for x in args.zones.split(",")]
    print(f"[E] model={args.model}  seeds={args.seeds}")
    model, tok, dev = common.load_model(args.model)
    model_tag = args.model.split("/")[-1].replace("-Instruct", "").lower()
    rb = common.get_rb_mod()
    suite = rb.EXTENDED_TEST_SUITE
    cat_names = ["syntax", "algorithmic", "factual"]

    try:
        L = len(model.model.layers)
    except AttributeError:
        L = len(model.transformer.h)
    print(f"    {L} transformer blocks")

    # Slices and basis cache
    slices = []
    prev = 0
    for ze in zones:
        slices.append((prev, ze))
        prev = ze

    bases_B: dict[int, torch.Tensor] = {}
    bases_Bp: dict[int, torch.Tensor] = {}
    for s in range(args.seeds):
        seed = 42 + s
        B = common.get_or_train_basis(model, tok, args.k, zones, args.steps,
                                      args.top_lambda, seed, model_tag)
        bases_B[seed] = B
        bases_Bp[seed] = common.make_random_basis(d=B.shape[0], k=args.k,
                                                  seed=seed, device="cpu")

    # Baselines per probe
    print("    measuring baselines...")
    baselines = {}
    for p in suite:
        b, cids = measure_correct_prob(model, tok, p, dev)
        baselines[p["id"]] = (b, cids)

    rows: list[dict] = []
    for s in range(args.seeds):
        seed = 42 + s
        B = bases_B[seed]
        Bp = bases_Bp[seed]
        for zi, zname in enumerate(cat_names):
            zlo, zhi = slices[zi]
            # For each probe, pick a same-category donor (different prompt)
            for p in suite:
                cat = p["category"]
                # donor: another probe of same category (deterministic)
                same_cat = [q for q in suite if q["category"] == cat and q["id"] != p["id"]]
                if not same_cat:
                    continue
                donor = same_cat[(seed + suite.index(p)) % len(same_cat)]
                # Per-layer mean residual for donor
                donor_h = collect_per_layer_residual(model, tok, donor["prompt"], dev)
                # Skip embedding (index 0); use indices 1..L for the L block hooks
                donor_block_h = donor_h[1:]  # (L, d)
                # Project onto zone subspace using the chosen basis
                for basis_kind, basis in (("B", B), ("Bp", Bp)):
                    Bz = basis[:, zlo:zhi].contiguous().to(dev)
                    q, _ = torch.linalg.qr(Bz, mode="reduced")
                    P = (q @ q.T)
                    donor_proj = donor_block_h @ P  # (L, d)
                    # patch
                    base, cids = baselines[p["id"]]
                    if not cids:
                        continue
                    with PatchInjector(model, basis.to(dev), (zlo, zhi), donor_proj):
                        msgs = [{"role": "user", "content": p["prompt"]}]
                        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                        enc = tok(txt, return_tensors="pt", truncation=True, max_length=256)
                        enc = {k: v.to(dev) for k, v in enc.items()}
                        with torch.no_grad():
                            o = model(**enc)
                        last = o.logits[0, -1, :]
                        probs = F.softmax(last.float(), -1)
                        patched = float(probs[cids].sum().item())
                    rows.append({
                        "seed": seed, "probe_id": p["id"], "probe_cat": cat,
                        "zone": zname, "donor_id": donor["id"],
                        "basis": basis_kind,
                        "baseline_prob": base, "patched_prob": patched,
                        "drift": patched - base,
                    })
        print(f"    seed={seed}: {sum(1 for r in rows if r['seed']==seed)} patches")

    # Aggregate: contrast B vs B' on diagonal probes
    summary = {"meta": {"model": args.model, "seeds": args.seeds,
                          "k": args.k, "zones": zones},
               "rows": rows}
    agg: dict[str, dict[str, dict]] = {}
    for cat in cat_names:
        agg[cat] = {}
        for z in cat_names:
            B_drifts = [r["drift"] for r in rows if r["basis"] == "B"
                        and r["probe_cat"] == cat and r["zone"] == z]
            Bp_drifts = [r["drift"] for r in rows if r["basis"] == "Bp"
                         and r["probe_cat"] == cat and r["zone"] == z]
            agg[cat][z] = {
                "B_drift_mean":  float(np.mean(B_drifts)) if B_drifts else None,
                "Bp_drift_mean": float(np.mean(Bp_drifts)) if Bp_drifts else None,
                "abs_B_minus_abs_Bp": (
                    float(np.mean(np.abs(B_drifts))) - float(np.mean(np.abs(Bp_drifts)))
                ) if B_drifts and Bp_drifts else None,
                "n_B": len(B_drifts), "n_Bp": len(Bp_drifts),
            }
    summary["aggregated"] = agg

    common.safe_dump(Path(args.out), summary)

    print("\n# Activation-patching: |B drift| − |B' drift| (positive ⇒ B carries more cat-info)")
    print("cat / zone:                B_drift_mean  Bp_drift_mean   |B|−|B'|")
    for cat in cat_names:
        for z in cat_names:
            r = agg[cat][z]
            mark = " *" if (cat == z and r["abs_B_minus_abs_Bp"] is not None
                              and r["abs_B_minus_abs_Bp"] > 0) else "  "
            b = r["B_drift_mean"]; bp = r["Bp_drift_mean"]; d = r["abs_B_minus_abs_Bp"]
            b_s = f"{b:+.4f}" if b is not None else "  --   "
            bp_s = f"{bp:+.4f}" if bp is not None else "  --   "
            d_s = f"{d:+.4f}" if d is not None else "  --   "
            print(f"  cat={cat:<12}  z={z:<12}{mark}  {b_s:<11}  {bp_s:<11}  {d_s}")


if __name__ == "__main__":
    main()
