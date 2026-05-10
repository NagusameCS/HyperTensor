#!/usr/bin/env python3
"""
UGT random-basis layer-wise ablation
=====================================

Stronger, layer-wise causal intervention test for Papers XII--XV. The earlier
``scripts/ugt_random_basis_ablation.py`` only zeroed the zone projection at the
final hidden state (one matrix multiply before the LM head), which intervenes
on at most ~k/d ~ 5--6% of the final-layer norm. This script registers a
forward hook on every transformer block and subtracts the zone projection at
every layer, which is the standard mechanistic-interpretability intervention
(RepE, abliteration, ROME, MEMIT all operate at every layer or at a chosen
mid-stack layer).

Predicted-positive cells under H_meaningful: cat=zone diagonal in the same
3x3 grid as before.

Output: benchmarks/ugt_random_basis_layerwise_<tag>.json with per-cell paired
stats (B vs B_rand) over (seed, probe) pairs.

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\ugt_random_basis_layerwise.py \\
        --seeds 5 --steps 800 --extended-suite

The script reuses TEST_SUITE / EXTENDED_TEST_SUITE / train_ugt_basis from
ugt_random_basis_ablation.py so the basis-training procedure is identical.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the existing scaffolding.
rb_mod = _load_module("ugt_random_basis_ablation",
                     ROOT / "scripts" / "ugt_random_basis_ablation.py")
adapter_for = rb_mod.adapter_for if hasattr(rb_mod, "adapter_for") else None  # may not exist
gather_corpus = rb_mod.gather_corpus
make_random_orthonormal_basis = rb_mod.make_random_orthonormal_basis
TEST_SUITE = rb_mod.TEST_SUITE
EXTENDED_TEST_SUITE = rb_mod.EXTENDED_TEST_SUITE
train_ugt_basis = rb_mod.train_ugt_basis
TOPLoss = rb_mod.TOPLoss
UGTAdapter = rb_mod.UGTAdapter

ugt_inf = _load_module("ugt_infrastructure",
                       ROOT / "scripts" / "ugt_infrastructure.py")


# ----- layer-wise ablation hook system ------------------------------------ #

class LayerwiseZoneAblator:
    """Registers forward hooks on every transformer block to subtract a zone
    projection at every layer.

    For basis B in R^{d x k} with column slice B_z = B[:, z_lo:z_hi] for the
    target zone, the hook performs (per layer output h):
        h_new = h - h @ B_z @ B_z.T
    i.e. zero out the component of h within span(B_z) at the residual stream.
    Because B_z columns are orthonormal (as columns of a Stiefel-manifold
    point), B_z @ B_z.T is the orthogonal projector onto span(B_z).
    """

    def __init__(self, model, basis: torch.Tensor, zone_slice: tuple[int, int] | None):
        self.model = model
        self.handles: list[Any] = []
        self.basis = basis
        if zone_slice is None:
            self.proj = None
        else:
            lo, hi = zone_slice
            Bz = basis[:, lo:hi].contiguous()  # (d, n_z)
            # Orthonormalise the slice defensively (B_z columns are already
            # orthonormal as columns of a Stiefel-manifold point, but a slice
            # may not be exactly orthonormal under finite-precision drift).
            q, _ = torch.linalg.qr(Bz, mode="reduced")
            self.proj = (q @ q.T).to(basis.dtype)  # (d, d)

    def _hook(self, module, inputs, outputs):
        if self.proj is None:
            return outputs
        # Llama decoder layer returns either a Tensor or a tuple
        if isinstance(outputs, tuple):
            h = outputs[0]
            P = self.proj.to(dtype=h.dtype, device=h.device)
            h_new = h - h @ P
            return (h_new,) + outputs[1:]
        h = outputs
        P = self.proj.to(dtype=h.dtype, device=h.device)
        return h - h @ P

    def __enter__(self):
        if self.proj is None:
            return self
        # Llama-style: model.model.layers
        try:
            layers = self.model.model.layers
        except AttributeError:
            layers = self.model.transformer.h  # GPT-2 fallback (unused)
        for layer in layers:
            self.handles.append(layer.register_forward_hook(self._hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


# ----- evaluation --------------------------------------------------------- #

def correct_token_ids(probe: dict, tokenizer) -> list[int]:
    out = []
    for chk in probe["check"]:
        ids = tokenizer.encode(chk, add_special_tokens=False)
        if ids:
            out.append(ids[0])
    return out


def measure_probe_logit_drop(model, tokenizer, probe: dict, device) -> dict:
    """Returns baseline log-prob of correct-answer tokens at the answer position
    (last input token's logits)."""
    msgs = [{"role": "user", "content": probe["prompt"]}]
    formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        last_logits = out.logits[0, -1, :]  # (vocab,)
        probs = F.softmax(last_logits.float(), dim=-1)
    cids = correct_token_ids(probe, tokenizer)
    if not cids:
        return {"prob": 0.0, "correct_token_ids": [], "top_token_id": int(last_logits.argmax().item())}
    p = float(probs[cids].sum().item())
    return {
        "prob": p,
        "correct_token_ids": cids,
        "top_token_id": int(last_logits.argmax().item()),
    }


def run_layerwise_ablation(model, tokenizer, basis, zones, test_suite, device) -> dict:
    """For each probe: compute baseline P_correct, then for each zone subtract
    the zone projection at EVERY transformer block via forward hooks and
    re-measure P_correct. Returns per-probe deltas."""
    zone_names = ["syntax", "algorithmic", "factual"]
    zone_slices = []
    prev = 0
    for z_end in zones:
        zone_slices.append((prev, z_end))
        prev = z_end

    out = {z: {"details": []} for z in ["none"] + zone_names}
    probe_baselines: dict[str, dict] = {}
    print("\n  [baseline]")
    for probe in test_suite:
        b = measure_probe_logit_drop(model, tokenizer, probe, device)
        probe_baselines[probe["id"]] = b
        out["none"]["details"].append({
            "id": probe["id"], "category": probe["category"],
            "baseline_prob": b["prob"],
        })

    for zi, z_name in enumerate(zone_names):
        print(f"  [ablate-zone-all-layers] {z_name} (slice {zone_slices[zi]})")
        with LayerwiseZoneAblator(model, basis, zone_slices[zi]):
            for probe in test_suite:
                b = probe_baselines[probe["id"]]
                # Recompute under hook
                msgs = [{"role": "user", "content": probe["prompt"]}]
                formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                enc = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=256)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    o = model(**enc)
                    last = o.logits[0, -1, :]
                    p_after = float(F.softmax(last.float(), -1)[b["correct_token_ids"]].sum().item()) if b["correct_token_ids"] else 0.0
                delta = b["prob"] - p_after  # positive => ablation hurts
                out[z_name]["details"].append({
                    "id": probe["id"], "category": probe["category"],
                    "baseline_prob": b["prob"], "ablated_prob": p_after,
                    "delta": delta,
                })
    return out


def summarise(label: str, results: dict) -> dict:
    """Reduce per-zone details into the (cat, zone) -> [deltas] structure used
    by the existing aggregate machinery in ugt_random_basis_ablation.py."""
    by_cat_zone: dict[str, dict[str, list[float]]] = {}
    by_cat_zone_ids: dict[str, dict[str, list[str]]] = {}
    for z_name in ["syntax", "algorithmic", "factual"]:
        for d in results.get(z_name, {}).get("details", []):
            cat = d["category"]
            by_cat_zone.setdefault(cat, {}).setdefault(z_name, []).append(d["delta"])
            by_cat_zone_ids.setdefault(cat, {}).setdefault(z_name, []).append(d["id"])
    return {
        "label": label,
        "raw_deltas_by_category_then_zone": {
            c: {z: list(map(float, vs)) for z, vs in zd.items()}
            for c, zd in by_cat_zone.items()
        },
        "probe_ids_by_category_then_zone": by_cat_zone_ids,
        "mean_delta_by_category_then_zone": {
            c: {z: float(np.mean(vs)) for z, vs in zd.items()}
            for c, zd in by_cat_zone.items()
        },
        "n_probes_by_category": {
            c: len(next(iter(zd.values()))) if zd else 0
            for c, zd in by_cat_zone.items()
        },
    }


def adapter_for(model, k: int, zones: list[int]):
    return UGTAdapter(model, k=k, zones=zones, top_lambda=0.05)


def run_one_seed(model, tok, k: int, zones: list[int], steps: int, top_lambda: float,
                 seed: int, test_suite, max_length: int = 256) -> dict:
    device = next(model.parameters()).device
    torch.manual_seed(seed); np.random.seed(seed)

    print(f"\n=== seed {seed}: training UGT basis B (steps={steps}, max_length={max_length}) ===")
    adapter = adapter_for(model, k=k, zones=zones)
    # In 4-bit quantised mode the wrapped model is already on cuda via
    # device_map='auto' and `.to(device)` on the adapter would error. Move
    # only the trainable basis parameter to the model's device.
    try:
        adapter.taxonomic_basis.data = adapter.taxonomic_basis.data.to(device)
    except Exception:
        pass
    if not any(p.dtype == torch.uint8 for p in model.parameters()):
        # non-quantised path: move the whole adapter (cheap, just registers
        # the parameter on the right device)
        adapter.to(device)
    purity_B = train_ugt_basis(adapter, tok, gather_corpus(), steps=steps, top_lambda=top_lambda, max_length=max_length)
    B = adapter.taxonomic_basis.detach().clone().contiguous()
    d = B.shape[0]
    print(f"   final purity(B) = {purity_B:.4f}")

    print(f"=== seed {seed}: building random orthonormal B' (d={d}, k={k}) ===")
    Bp = make_random_orthonormal_basis(d=d, k=k, seed=seed, device=device)

    print(f"=== seed {seed}: layer-wise ablation under B ===")
    res_B = run_layerwise_ablation(adapter.model, tok, B, zones, test_suite, device)
    print(f"=== seed {seed}: layer-wise ablation under B' ===")
    res_Bp = run_layerwise_ablation(adapter.model, tok, Bp, zones, test_suite, device)

    out = {
        "seed": int(seed),
        "purity_B": float(purity_B),
        "B": summarise("B", res_B),
        "B_random": summarise("B_random", res_Bp),
    }
    del adapter, B, Bp, res_B, res_Bp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def aggregate(seed_results: list[dict]) -> dict:
    paired = rb_mod._paired_stats
    cats = ["syntax", "algorithmic", "factual"]
    zones_set = ["syntax", "algorithmic", "factual"]
    out = {}
    for c in cats:
        out[c] = {}
        for z in zones_set:
            diffs: list[float] = []
            vals_B, vals_Bp = [], []
            for r in seed_results:
                rb = r["B"]["raw_deltas_by_category_then_zone"].get(c, {}).get(z, [])
                rbp = r["B_random"]["raw_deltas_by_category_then_zone"].get(c, {}).get(z, [])
                vals_B.append(float(np.mean(rb)) if rb else 0.0)
                vals_Bp.append(float(np.mean(rbp)) if rbp else 0.0)
                for i in range(min(len(rb), len(rbp))):
                    diffs.append(float(rb[i] - rbp[i]))
            out[c][z] = {
                "B_mean_delta": float(np.mean(vals_B)) if vals_B else None,
                "Brand_mean_delta": float(np.mean(vals_Bp)) if vals_Bp else None,
                "B_minus_Brand": (float(np.mean(vals_B)) - float(np.mean(vals_Bp))),
                "n_seeds": len(vals_B),
                "paired_stats": paired(diffs),
            }
    return {"by_category_then_ablated_zone": out, "categories": cats, "zones": zones_set}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--zones", default="12,24,32")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--top-lambda", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--extended-suite", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Use bitsandbytes 4-bit quantisation (for 7B+ models on small VRAM)")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="Enable activation checkpointing to reduce VRAM during basis training")
    ap.add_argument("--max-seqlen", type=int, default=256,
                    help="Max sequence length during basis training (reduce for big models)")
    args = ap.parse_args()

    zones = [int(x) for x in args.zones.split(",")]
    if zones[-1] != args.k:
        raise SystemExit(f"final zone boundary ({zones[-1]}) must equal --k ({args.k})")
    suite = EXTENDED_TEST_SUITE if args.extended_suite else TEST_SUITE
    suite_tag = "ext" if args.extended_suite else "def"
    model_tag = args.model.split("/")[-1].replace("-Instruct", "").lower()
    out_path = Path(args.out) if args.out else OUT_DIR / f"ugt_random_basis_layerwise_{model_tag}_{suite_tag}_n{args.seeds}.json"

    print(f"# UGT layer-wise random-basis ablation\n# model={args.model} k={args.k} zones={zones} suite={suite_tag} n={len(suite)}  4bit={args.load_in_4bit}  grad_ckpt={args.gradient_checkpointing}  seqlen={args.max_seqlen}")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_compute_dtype=torch.bfloat16,
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_quant_type="nf4")
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=qcfg, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto").to(dev)
    # Freeze all base parameters; only the adapter's basis trains.
    for p in model.parameters():
        p.requires_grad = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # transformers requires use_cache=False when gradient checkpointing is on
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.eval()

    seed_results = []
    for s in range(args.seeds):
        r = run_one_seed(model, tok, args.k, zones, args.steps, args.top_lambda,
                         seed=42 + s, test_suite=suite, max_length=args.max_seqlen)
        seed_results.append(r)
        # incremental partial save
        try:
            partial = {
                "meta": {
                    "model": args.model, "k": args.k, "zones": zones,
                    "steps": args.steps, "top_lambda": args.top_lambda,
                    "seeds_completed": s + 1, "seeds_target": args.seeds,
                    "probe_suite": suite_tag, "n_probes": len(suite),
                    "intervention": "layerwise_residual_subtraction",
                    "incremental": True,
                },
                "per_seed": seed_results,
                "aggregated": aggregate(seed_results),
            }
            partial_path = out_path.with_suffix(".partial.json")
            partial_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")
            print(f"  [partial save] {partial_path} ({s+1}/{args.seeds})")
        except Exception as e:
            print(f"  [partial save FAILED] {e!r}")

    summary = {
        "meta": {
            "model": args.model, "k": args.k, "zones": zones,
            "steps": args.steps, "top_lambda": args.top_lambda,
            "seeds": args.seeds, "wallclock_seconds": round(time.time() - t0, 2),
            "probe_suite": suite_tag, "n_probes": len(suite),
            "intervention": "layerwise_residual_subtraction",
            "torch_version": torch.__version__,
            "cuda": torch.cuda.is_available(),
        },
        "per_seed": seed_results,
        "aggregated": aggregate(seed_results),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n# wrote {out_path}")
    print("# Layer-wise B-vs-B' contrast (paired diff over seeds x probes; * = predicted-positive cell with p<0.05):")
    agg = summary["aggregated"]["by_category_then_ablated_zone"]
    for cat, zb in agg.items():
        for z, row in zb.items():
            d = row["B_minus_Brand"]
            ps = row.get("paired_stats", {}) or {}
            n = ps.get("n", 0)
            tp = ps.get("t_p_two_sided")
            wp = ps.get("wilcoxon_p_two_sided")
            ci_lo = ps.get("ci95_low"); ci_hi = ps.get("ci95_high")
            sig = (tp is not None and tp < 0.05)
            mark = " *" if (cat == z and d is not None and d > 0 and sig) else "  "
            ci_str = f"[{ci_lo:+.4f},{ci_hi:+.4f}]" if (ci_lo is not None and ci_hi is not None) else "[--,--]"
            tp_str = f"t_p={tp:.3f}" if tp is not None else "t_p=--   "
            wp_str = f"w_p={wp:.3f}" if wp is not None else "w_p=--   "
            print(f"   cat={cat:<12}  z={z:<12}  delta={d:+.4f}{mark}  CI95={ci_str}  {tp_str}  {wp_str}  (n_pairs={n})")


if __name__ == "__main__":
    sys.exit(main())
