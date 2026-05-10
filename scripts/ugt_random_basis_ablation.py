#!/usr/bin/env python3
"""
UGT RANDOM-BASIS CONTROL --- Falsification test for Papers XI/XII/XIII/XIV/XV.

Question
--------
Is the UGT basis B semantically meaningful, or is it merely a low-rank
parameter-efficient subspace whose internal structure does not matter?

Test
----
Re-run the same zone-ablation probe battery from `ugt_ablation.py` under
two basis conditions:

  (A) B  = UGT-trained basis (TOP-loss converged to high purity).
  (B) B' = uniformly random orthonormal basis of identical shape (d, k),
          partitioned into the same three zone slices (12, 24, 32).

Hypotheses
----------
H_meaningful : zone ablation under B produces the asymmetric pattern
               predicted by the UGT papers (zone 1 breaks syntax, zone 3
               breaks facts), while the same ablation under B' produces
               either uniform degradation or no degradation. This would
               support the cascading claims of Papers XII--XV.

H_artefact   : both bases produce indistinguishable per-zone effects.
               This would mean the cascade is a low-rank parameter
               efficiency story, not a UGT-as-meaningful-basis story,
               and the framing of Papers XII--XV must be revised.

The script does not pre-judge the outcome. It records both arms and a
between-condition contrast for each probe category. Negative results are
as informative as positive ones --- this is the central credibility
experiment for half the volume.

Usage
-----
    # Default (SmolLM2-135M, ~5 min on RTX 4070 Laptop, 3 seeds)
    python scripts/ugt_random_basis_ablation.py

    # Larger model (Qwen 0.5B, ~15 min)
    python scripts/ugt_random_basis_ablation.py --model Qwen/Qwen2.5-0.5B-Instruct

    # More seeds for tighter intervals
    python scripts/ugt_random_basis_ablation.py --seeds 5 --steps 800

Output
------
    benchmarks/ugt_random_basis_ablation_results.json

Notes
-----
- All compute runs locally; per repository policy, no model weights or
  results are written to remote storage.
- This file is intentionally self-contained except for reusing the
  TOPLoss / UGTAdapter / probe-suite scaffolding already in
  ugt_infrastructure.py and ugt_ablation.py.
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

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "benchmarks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "ugt_random_basis_ablation_results.json"


# ----- import shared scaffolding from existing UGT scripts ---------------- #

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ugt_mod = _load_module("ugt_infrastructure", ROOT / "scripts" / "ugt_infrastructure.py")
ablation_mod = _load_module("ugt_ablation", ROOT / "scripts" / "ugt_ablation.py")

TOPLoss = ugt_mod.TOPLoss
UGTAdapter = ugt_mod.UGTAdapter
TEST_SUITE = ablation_mod.TEST_SUITE
train_ugt_basis = ablation_mod.train_ugt_basis
run_ablation = ablation_mod.run_ablation


# ----- extended probe battery -------------------------------------------- #
# The default TEST_SUITE imported from ugt_ablation has 9 probes (3 per
# category). Per-cell n=24 with n=8 seeds is still underpowered. The
# extended suite has 30 probes (10 per category) for n_pairs=80 at n=8
# seeds, which gives ~3-4x detection power for small effects.

EXTENDED_TEST_SUITE = [
    # --- syntax (10) ---
    {"id": "syn_01", "category": "syntax", "prompt": "Write a grammatically correct sentence about a cat.",
     "check": ["the", "a", "cat", "is"]},
    {"id": "syn_02", "category": "syntax", "prompt": "Complete this sentence with proper English: The weather today is",
     "check": ["warm", "cold", "sunny", "rainy", "nice", "beautiful", "cool", "hot"]},
    {"id": "syn_03", "category": "syntax", "prompt": "Rewrite this in proper English: \"he go store yesterday\"",
     "check": ["went", "the", "to", "He"]},
    {"id": "syn_04", "category": "syntax", "prompt": "Fix the grammar: \"She don't know nothing.\" Answer with the corrected sentence.",
     "check": ["She", "doesn't", "anything"]},
    {"id": "syn_05", "category": "syntax", "prompt": "Insert the correct article: ___ apple a day keeps the doctor away.",
     "check": ["An", "an"]},
    {"id": "syn_06", "category": "syntax", "prompt": "What is the past tense of \"run\"? Answer with one word.",
     "check": ["ran"]},
    {"id": "syn_07", "category": "syntax", "prompt": "What is the plural of \"child\"? Answer with one word.",
     "check": ["children"]},
    {"id": "syn_08", "category": "syntax", "prompt": "Complete: I have ___ to the store. (Use the correct verb form.)",
     "check": ["been", "gone"]},
    {"id": "syn_09", "category": "syntax", "prompt": "What word is missing? \"The cat ___ on the mat.\" Answer with one word.",
     "check": ["sat", "is", "sits", "lay", "lies"]},
    {"id": "syn_10", "category": "syntax", "prompt": "Identify the verb in: \"The dog barks loudly.\" Answer with one word.",
     "check": ["barks"]},

    # --- algorithmic (10) ---
    {"id": "alg_01", "category": "algorithmic", "prompt": "What is 12 * 7? Answer with just the number.",
     "check": ["84"]},
    {"id": "alg_02", "category": "algorithmic", "prompt": "What is 15 + 27? Answer with just the number.",
     "check": ["42"]},
    {"id": "alg_03", "category": "algorithmic", "prompt": "If a train travels 60 miles in 2 hours, what is its speed in mph? Answer with just the number.",
     "check": ["30"]},
    {"id": "alg_04", "category": "algorithmic", "prompt": "What is 9 * 9? Answer with just the number.",
     "check": ["81"]},
    {"id": "alg_05", "category": "algorithmic", "prompt": "What is 100 - 37? Answer with just the number.",
     "check": ["63"]},
    {"id": "alg_06", "category": "algorithmic", "prompt": "What is the next number in the sequence 2, 4, 6, 8, ...? Answer with just the number.",
     "check": ["10"]},
    {"id": "alg_07", "category": "algorithmic", "prompt": "What is half of 50? Answer with just the number.",
     "check": ["25"]},
    {"id": "alg_08", "category": "algorithmic", "prompt": "What is 2 to the power of 5? Answer with just the number.",
     "check": ["32"]},
    {"id": "alg_09", "category": "algorithmic", "prompt": "If x + 5 = 12, what is x? Answer with just the number.",
     "check": ["7"]},
    {"id": "alg_10", "category": "algorithmic", "prompt": "How many sides does a hexagon have? Answer with just the number.",
     "check": ["6", "six"]},

    # --- factual (10) ---
    {"id": "fac_01", "category": "factual", "prompt": "What is the capital of France? Answer in one word.",
     "check": ["Paris"]},
    {"id": "fac_02", "category": "factual", "prompt": "What is the chemical symbol for water? Answer with just the symbol.",
     "check": ["H2O", "H\u2082O", "h2o"]},
    {"id": "fac_03", "category": "factual", "prompt": "Who wrote Romeo and Juliet? Answer with just the name.",
     "check": ["Shakespeare", "William"]},
    {"id": "fac_04", "category": "factual", "prompt": "What is the capital of Japan? Answer in one word.",
     "check": ["Tokyo"]},
    {"id": "fac_05", "category": "factual", "prompt": "Which planet is known as the Red Planet? Answer in one word.",
     "check": ["Mars"]},
    {"id": "fac_06", "category": "factual", "prompt": "What is the largest ocean? Answer in one word.",
     "check": ["Pacific"]},
    {"id": "fac_07", "category": "factual", "prompt": "Who painted the Mona Lisa? Answer with just the name.",
     "check": ["Leonardo", "Vinci", "da"]},
    {"id": "fac_08", "category": "factual", "prompt": "What is the chemical symbol for gold? Answer with just the symbol.",
     "check": ["Au"]},
    {"id": "fac_09", "category": "factual", "prompt": "In what year did World War II end? Answer with just the year.",
     "check": ["1945"]},
    {"id": "fac_10", "category": "factual", "prompt": "What is the tallest mountain on Earth? Answer with the name.",
     "check": ["Everest", "Mount"]},
]


# ----- basis constructors ------------------------------------------------- #

def make_random_orthonormal_basis(d: int, k: int, seed: int, device: torch.device) -> torch.Tensor:
    """Uniformly random k-dim orthonormal basis in R^d (columns orthonormal).

    Drawn by QR-decomposing a Gaussian matrix; the resulting Q is Haar-uniform
    on the Stiefel manifold V_k(R^d). Same shape and norm as the UGT basis.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    a = torch.randn(d, k, generator=g)
    q, _ = torch.linalg.qr(a, mode="reduced")  # (d, k), columns orthonormal
    return q.to(device=device, dtype=torch.float32)


# ----- main experiment ---------------------------------------------------- #

def load_model_and_tokenizer(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    model.eval()
    return model, tok


def adapter_for(model, k: int, zones: list[int]) -> Any:
    """Wrap a base CausalLM in the UGT adapter from ugt_infrastructure.

    The adapter exposes a `taxonomic_basis` parameter of shape (d, k) which
    we override to either a TOP-trained basis (B) or a random orthonormal
    basis (B').
    """
    return UGTAdapter(model=model, k=k, zones=zones)


def gather_corpus() -> list[str]:
    """A small mixed-domain text corpus for TOP training. Kept tiny so the
    script runs in minutes; not the production curriculum."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "import numpy as np; x = np.linspace(0, 1, 100); y = x**2",
        "The capital of France is Paris.",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)",
        "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen.",
        "for i in range(10):\n    print(i * i)",
        "Romeo and Juliet was written by William Shakespeare in the late sixteenth century.",
        "The mitochondrion is the powerhouse of the cell.",
        "let result = arr.map(x => x * 2).filter(x => x > 5);",
        "An integer n is prime if its only positive divisors are 1 and n itself.",
        "The Pythagorean theorem states that a^2 + b^2 = c^2 for a right triangle.",
        "She walked to the store and bought a loaf of fresh sourdough bread.",
        "SELECT name, age FROM users WHERE active = TRUE ORDER BY created_at DESC;",
        "The Treaty of Westphalia was signed in 1648, ending the Thirty Years' War.",
        "Newton's second law: F equals m times a.",
        "The mitochondrial membrane houses the electron transport chain.",
        "Use a hash table to achieve amortised constant-time lookup.",
        "The chemical formula for water is H2O.",
        "git commit -m 'Initial scaffolding' && git push origin main",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
    ]


def summarise_results(label: str, results: dict) -> dict:
    """Reduce ugt_ablation.run_ablation output into the slices we need.

    `results` from ugt_ablation.run_ablation has shape:
        results["none"|"syntax"|"algorithmic"|"factual"]["details"] = [probe_result, ...]
    where each probe_result carries:
        category in {syntax, algorithmic, factual}
        baseline_correct: bool
        zone_energies: {syntax, algorithmic, factual} -> float
        zone_deltas:    {syntax, algorithmic, factual} -> float (positive = ablation hurts)

    The "none" arm holds baseline correctness. The per-zone arms hold identical
    probe_result lists, so we read deltas from "none" only and avoid double-counting.
    """
    details = results.get("none", {}).get("details", [])
    by_cat: dict[str, dict[str, list[float]]] = {}
    by_cat_probe_ids: dict[str, dict[str, list[str]]] = {}
    baseline_correct: dict[str, list[int]] = {}
    for pr in details:
        cat = pr.get("category", "?")
        pid = pr.get("id", "?")
        baseline_correct.setdefault(cat, []).append(1 if pr.get("baseline_correct") else 0)
        deltas = pr.get("zone_deltas", {})
        slot = by_cat.setdefault(cat, {})
        slot_ids = by_cat_probe_ids.setdefault(cat, {})
        for z_name, d in deltas.items():
            slot.setdefault(z_name, []).append(float(d))
            slot_ids.setdefault(z_name, []).append(pid)

    return {
        "label": label,
        "baseline_correct_rate": {c: float(np.mean(v)) for c, v in baseline_correct.items()},
        "mean_delta_by_category_then_zone": {
            c: {z: float(np.mean(vs)) for z, vs in zd.items()} for c, zd in by_cat.items()
        },
        "raw_deltas_by_category_then_zone": {
            c: {z: list(map(float, vs)) for z, vs in zd.items()} for c, zd in by_cat.items()
        },
        "probe_ids_by_category_then_zone": by_cat_probe_ids,
        "n_probes_by_category": {c: len(v) for c, v in baseline_correct.items()},
    }


def run_one_seed(model, tok, k: int, zones: list[int], steps: int, top_lambda: float,
                 seed: int, test_suite=None) -> dict:
    """Run the full B-vs-B' contrast for a single seed."""
    if test_suite is None:
        test_suite = TEST_SUITE

    device = next(model.parameters()).device
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n=== seed {seed}: training UGT basis B (steps={steps}) ===")
    adapter = adapter_for(model, k=k, zones=zones)
    adapter.to(device)
    purity_B = train_ugt_basis(adapter, tok, gather_corpus(), steps=steps, top_lambda=top_lambda)
    B = adapter.taxonomic_basis.detach().clone()
    d = B.shape[0]
    print(f"   final purity(B) = {purity_B:.4f}")

    print(f"=== seed {seed}: building random orthonormal basis B' (d={d}, k={k}) ===")
    Bp = make_random_orthonormal_basis(d=d, k=k, seed=seed, device=device)
    purity_Bp = float(TOPLoss(k=k, zones=zones).purity_score(Bp))
    print(f"   purity(B') = {purity_Bp:.4f}  (expected near chance)")

    print(f"=== seed {seed}: ablation under B ===")
    results_B = run_ablation(model, tok, B, zones, test_suite)

    print(f"=== seed {seed}: ablation under B' (random) ===")
    results_Bp = run_ablation(model, tok, Bp, zones, test_suite)

    out = {
        "seed": int(seed),
        "purity": {"B": float(purity_B), "B_random": float(purity_Bp)},
        "B": summarise_results("B", results_B),
        "B_random": summarise_results("B_random", results_Bp),
    }
    # Free per-seed tensors so multi-seed runs don't accumulate VRAM
    del adapter, B, Bp, results_B, results_Bp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def _paired_stats(diffs: list[float]) -> dict:
    """Paired t-test and Wilcoxon signed-rank on a list of (B - B') paired diffs.

    Returns mean, std, sem, t-statistic, two-sided p-value, Wilcoxon p, and n.
    Falls back gracefully if scipy is unavailable or n < 2.
    """
    n = len(diffs)
    if n == 0:
        return {"n": 0}
    a = np.asarray(diffs, dtype=np.float64)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    out = {"n": n, "mean": mean, "std": std, "sem": sem}
    if n < 2:
        return out
    try:
        from scipy import stats as _st  # type: ignore
        t_stat, t_p = _st.ttest_1samp(a, 0.0)
        out["t_stat"] = float(t_stat)
        out["t_p_two_sided"] = float(t_p)
        # Wilcoxon needs at least one non-zero diff
        if np.any(a != 0):
            try:
                w_stat, w_p = _st.wilcoxon(a, alternative="two-sided", zero_method="wilcox")
                out["wilcoxon_stat"] = float(w_stat)
                out["wilcoxon_p_two_sided"] = float(w_p)
            except Exception:
                pass
    except Exception:
        # Fallback: ad-hoc t via numpy
        t_stat = mean / sem if sem > 0 else float("nan")
        out["t_stat"] = float(t_stat) if np.isfinite(t_stat) else None
        out["t_p_two_sided"] = None
    # Bootstrap 95% CI of the mean (5000 resamples)
    rng = np.random.default_rng(20260508)
    if n >= 2:
        boots = rng.choice(a, size=(5000, n), replace=True).mean(axis=1)
        out["ci95_low"] = float(np.percentile(boots, 2.5))
        out["ci95_high"] = float(np.percentile(boots, 97.5))
    return out


def aggregate(seed_results: list[dict]) -> dict:
    """Aggregate per-seed numbers into the B vs B' contrast.

    Per-cell paired statistics: gather (B - B') paired by (seed, probe_id) for
    each (category, ablated zone) cell and run a paired t-test / Wilcoxon. The
    paired analysis dominates underpowering concerns from N small in any one
    seed because pairing kills probe-level idiosyncratic variance.
    """
    cats = sorted({c for r in seed_results for c in r["B"]["mean_delta_by_category_then_zone"]})
    zones_set = sorted({
        z for r in seed_results
          for c in r["B"]["mean_delta_by_category_then_zone"]
          for z in r["B"]["mean_delta_by_category_then_zone"][c]
    })

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for c in cats:
        out[c] = {}
        for z in zones_set:
            vals_B, vals_Bp = [], []          # mean-by-seed (legacy)
            paired_diffs: list[float] = []     # per (seed, probe) raw diffs
            for r in seed_results:
                # legacy per-seed cell means
                try:
                    vals_B.append(r["B"]["mean_delta_by_category_then_zone"][c][z])
                except KeyError:
                    pass
                try:
                    vals_Bp.append(r["B_random"]["mean_delta_by_category_then_zone"][c][z])
                except KeyError:
                    pass
                # raw per-probe pairing
                raw_B = r["B"].get("raw_deltas_by_category_then_zone", {}).get(c, {}).get(z, [])
                raw_Bp = r["B_random"].get("raw_deltas_by_category_then_zone", {}).get(c, {}).get(z, [])
                ids_B = r["B"].get("probe_ids_by_category_then_zone", {}).get(c, {}).get(z, [])
                ids_Bp = r["B_random"].get("probe_ids_by_category_then_zone", {}).get(c, {}).get(z, [])
                if ids_B and ids_Bp and ids_B == ids_Bp:
                    for i in range(min(len(raw_B), len(raw_Bp))):
                        paired_diffs.append(float(raw_B[i] - raw_Bp[i]))
                else:
                    # fall back to position-pairing (probe order is fixed in TEST_SUITE)
                    for i in range(min(len(raw_B), len(raw_Bp))):
                        paired_diffs.append(float(raw_B[i] - raw_Bp[i]))
            cell = {
                "B_mean_delta": float(np.mean(vals_B)) if vals_B else None,
                "B_std_delta": float(np.std(vals_B, ddof=1)) if len(vals_B) > 1 else 0.0,
                "Brand_mean_delta": float(np.mean(vals_Bp)) if vals_Bp else None,
                "Brand_std_delta": float(np.std(vals_Bp, ddof=1)) if len(vals_Bp) > 1 else 0.0,
                "B_minus_Brand": (float(np.mean(vals_B)) - float(np.mean(vals_Bp))) if (vals_B and vals_Bp) else None,
                "n_seeds": min(len(vals_B), len(vals_Bp)),
                "paired_stats": _paired_stats(paired_diffs),
            }
            out[c][z] = cell
    return {"by_category_then_ablated_zone": out, "categories": cats, "zones": zones_set}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--k", type=int, default=32, help="UGT basis rank")
    ap.add_argument("--zones", default="12,24,32",
                    help="Cumulative zone boundaries (must end at k)")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--top-lambda", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--extended-suite", action="store_true",
                    help="Use the 30-probe EXTENDED_TEST_SUITE instead of the 9-probe default")
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args()

    zones = [int(x) for x in args.zones.split(",")]
    if zones[-1] != args.k:
        raise SystemExit(f"final zone boundary ({zones[-1]}) must equal --k ({args.k})")

    print(f"\n# UGT random-basis ablation\n# model={args.model}  k={args.k}  zones={zones}  seeds={args.seeds}")
    chosen_suite = EXTENDED_TEST_SUITE if args.extended_suite else TEST_SUITE
    print(f"# probe_suite={'extended' if args.extended_suite else 'default'}  n_probes={len(chosen_suite)}")
    t0 = time.time()
    model, tok = load_model_and_tokenizer(args.model)

    seed_results = []
    for s in range(args.seeds):
        r = run_one_seed(model, tok, args.k, zones, args.steps, args.top_lambda,
                         seed=42 + s, test_suite=chosen_suite)
        seed_results.append(r)
        # release any per-seed adapter references
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Incremental partial save so a mid-run crash doesn't lose work
        try:
            partial = {
                "meta": {
                    "model": args.model,
                    "k": args.k,
                    "zones": zones,
                    "steps": args.steps,
                    "top_lambda": args.top_lambda,
                    "seeds_completed": s + 1,
                    "seeds_target": args.seeds,
                    "probe_suite": "extended" if args.extended_suite else "default",
                    "n_probes": len(chosen_suite),
                    "incremental": True,
                },
                "per_seed": seed_results,
                "aggregated": aggregate(seed_results),
            }
            partial_path = Path(args.out).with_suffix(".partial.json")
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            with partial_path.open("w", encoding="utf-8") as f:
                json.dump(partial, f, indent=2)
            print(f"  [partial save] {partial_path} ({s+1}/{args.seeds} seeds)")
        except Exception as e:  # pragma: no cover
            print(f"  [partial save FAILED] {e!r}")

    summary = {
        "meta": {
            "model": args.model,
            "k": args.k,
            "zones": zones,
            "steps": args.steps,
            "top_lambda": args.top_lambda,
            "seeds": args.seeds,
            "wallclock_seconds": round(time.time() - t0, 2),
            "torch_version": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "probe_suite": "extended" if args.extended_suite else "default",
            "n_probes": len(chosen_suite),
        },
        "per_seed": seed_results,
        "aggregated": aggregate(seed_results),
        "interpretation_key": {
            "delta_meaning": "Per-probe delta is (P_correct_baseline - P_correct_after_zone_ablation). Positive delta = ablation hurts the answer-token probability for that probe.",
            "B_minus_Brand_positive_for_predicted_pair": "If, e.g., ablating zone1 (syntax-named) hurts the syntax category more under B than under B_random, that supports H_meaningful: the UGT basis carries the predicted semantic structure. Predicted positive cells: (cat=syntax, z=syntax), (cat=algorithmic, z=algorithmic), (cat=factual, z=factual).",
            "B_minus_Brand_near_zero": "B and B_random behave the same --- supports H_artefact: the cascade of Papers XII-XV must be reframed as a low-rank parameter-efficiency story, not a UGT-as-meaningful-basis story.",
            "B_minus_Brand_negative": "B is less affected than B_random --- diagnostic anomaly; check basis training, probe leakage, or that the projector + residual reconstruction in run_ablation is well-conditioned.",
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n# wrote {out_path}")
    print("# B-vs-B' contrast (per-cell paired diff over seeds x probes; * = predicted-positive cell with p<0.05):")
    agg = summary["aggregated"]["by_category_then_ablated_zone"]
    for cat, zones_block in agg.items():
        for z, row in zones_block.items():
            d = row["B_minus_Brand"]
            ps = row.get("paired_stats", {}) or {}
            n = ps.get("n", 0)
            tp = ps.get("t_p_two_sided")
            wp = ps.get("wilcoxon_p_two_sided")
            ci_lo = ps.get("ci95_low")
            ci_hi = ps.get("ci95_high")
            sig = (tp is not None and tp < 0.05)
            marker = " *" if (cat == z and d is not None and d > 0.0 and sig) else "  "
            ci_str = f"[{ci_lo:+.4f},{ci_hi:+.4f}]" if (ci_lo is not None and ci_hi is not None) else "[--,--]"
            tp_str = f"t_p={tp:.3f}" if tp is not None else "t_p=--   "
            wp_str = f"w_p={wp:.3f}" if wp is not None else "w_p=--   "
            if d is None:
                print(f"   cat={cat:<12}  z={z:<12}  delta=  N/A   (n_pairs={n})")
            else:
                print(f"   cat={cat:<12}  z={z:<12}  delta={d:+.4f}{marker}  CI95={ci_str}  {tp_str}  {wp_str}  (n_pairs={n})")


if __name__ == "__main__":
    sys.exit(main())
