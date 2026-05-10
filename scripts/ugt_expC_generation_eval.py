#!/usr/bin/env python3
"""Experiment C — Generation evaluation under zone ablations.

For 50 problems each from {HumanEval, GSM8K, TriviaQA, CoLA}, compare
greedy-decode quality under three conditions:
    baseline (no ablation), B layer-wise zone ablation, B' layer-wise zone
    ablation.

Hypothesis: if UGT zones carry the categorical information they claim, then
the *predicted-positive* zone should hurt the matching dataset more than B'.

  - HumanEval  ↔ algorithmic  (code/structure)
  - GSM8K      ↔ algorithmic
  - TriviaQA   ↔ factual
  - CoLA       ↔ syntax (binary acceptable / unacceptable)

Output: benchmarks/expC_generation_eval.json
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ugt_exp_common as common  # noqa: E402


# --------------------------------------------------------------------- #
# Dataset loaders (50 problems each)
# --------------------------------------------------------------------- #

def load_humaneval(n: int = 50):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    out = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        out.append({
            "id": f"humaneval_{ex['task_id']}",
            "prompt": ex["prompt"],
            "test": ex["test"],
            "entry_point": ex["entry_point"],
        })
    return out


def load_gsm8k(n: int = 50):
    path = ROOT / "data" / "gsm8k_test.jsonl"
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            ex = json.loads(line)
            ans = ex["answer"]
            m = re.search(r"####\s*(-?[0-9.,]+)", ans)
            gold = m.group(1).replace(",", "").strip() if m else ""
            out.append({
                "id": f"gsm8k_{i}",
                "prompt": ex["question"],
                "gold": gold,
            })
    return out


def load_triviaqa(n: int = 50):
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation",
                      trust_remote_code=False)
    out = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        aliases = ex["answer"].get("aliases", []) + [ex["answer"]["value"]]
        aliases = [a for a in aliases if a]
        out.append({
            "id": f"triviaqa_{i}",
            "prompt": ex["question"],
            "gold_aliases": aliases,
        })
    return out


def load_cola(n: int = 50):
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "cola", split="validation")
    out = []
    for i in range(min(n, len(ds))):
        ex = ds[i]
        # label 1=acceptable, 0=unacceptable
        out.append({
            "id": f"cola_{i}",
            "prompt": (
                "Is the following sentence grammatically acceptable in English? "
                f"Answer 'yes' or 'no'.\n\nSentence: {ex['sentence']}"
            ),
            "gold": "yes" if ex["label"] == 1 else "no",
        })
    return out


# --------------------------------------------------------------------- #
# Layer-wise zone ablator (residual-stream subtraction)
# --------------------------------------------------------------------- #

class ZoneAblator:
    def __init__(self, model, basis: torch.Tensor | None, zone_slice):
        self.model = model
        self.basis = basis
        self.zone_slice = zone_slice
        self.handles = []

    def _hook(self):
        if self.basis is None:
            return None
        lo, hi = self.zone_slice
        Bz = self.basis[:, lo:hi].contiguous()
        q, _ = torch.linalg.qr(Bz, mode="reduced")
        P = q @ q.T

        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple):
                h = outputs[0]
                Pl = P.to(dtype=h.dtype, device=h.device)
                return (h - h @ Pl,) + outputs[1:]
            h = outputs
            Pl = P.to(dtype=h.dtype, device=h.device)
            return h - h @ Pl
        return hook

    def __enter__(self):
        if self.basis is None:
            return self
        try:
            layers = self.model.model.layers
        except AttributeError:
            layers = self.model.transformer.h
        h = self._hook()
        for layer in layers:
            self.handles.append(layer.register_forward_hook(h))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


# --------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------- #

@torch.no_grad()
def greedy_generate(model, tok, prompt: str, dev, max_new_tokens: int,
                    use_chat: bool = True) -> str:
    if use_chat:
        msgs = [{"role": "user", "content": prompt}]
        try:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            txt = prompt
    else:
        txt = prompt
    enc = tok(txt, return_tensors="pt", truncation=True, max_length=1024)
    enc = {k: v.to(dev) for k, v in enc.items()}
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    new_ids = out[0, enc["input_ids"].shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True)


# --------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------- #

def score_humaneval(items, completions: list[str]) -> float:
    import multiprocessing as mp
    n_pass = 0
    for it, comp in zip(items, completions):
        # Extract first contiguous indented block plus completion glob
        program = it["prompt"] + comp + "\n" + it["test"] + f"\ncheck({it['entry_point']})\n"
        try:
            ok = _run_with_timeout(program, timeout=5.0)
        except Exception:
            ok = False
        n_pass += int(ok)
    return n_pass / max(1, len(items))


def _run_with_timeout(program: str, timeout: float) -> bool:
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(program)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, text=True)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            Path(path).unlink()
        except Exception:
            pass


def score_gsm8k(items, completions: list[str]) -> float:
    n_correct = 0
    for it, comp in zip(items, completions):
        # Find last number in completion
        nums = re.findall(r"-?\d+(?:\.\d+)?", comp.replace(",", ""))
        pred = nums[-1] if nums else ""
        gold = it["gold"]
        try:
            n_correct += int(abs(float(pred) - float(gold)) < 1e-6)
        except ValueError:
            n_correct += int(pred == gold)
    return n_correct / max(1, len(items))


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def score_triviaqa(items, completions: list[str]) -> float:
    n_correct = 0
    for it, comp in zip(items, completions):
        nc = _norm(comp.split("\n")[0][:200])
        for alias in it["gold_aliases"]:
            if _norm(alias) and _norm(alias) in nc:
                n_correct += 1
                break
    return n_correct / max(1, len(items))


def score_cola(items, completions: list[str]) -> float:
    n_correct = 0
    for it, comp in zip(items, completions):
        c = comp.lower()
        if "yes" in c[:30] and "no" not in c[:30]:
            pred = "yes"
        elif "no" in c[:30] and "yes" not in c[:30]:
            pred = "no"
        else:
            pred = "yes" if "yes" in c[:50] else "no"
        n_correct += int(pred == it["gold"])
    return n_correct / max(1, len(items))


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--zones", default="12,24,32")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--top-lambda", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=1,
                    help="generation eval is expensive — default 1 seed")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", default=str(common.OUT / "expC_generation_eval.json"))
    args = ap.parse_args()

    zones = [int(x) for x in args.zones.split(",")]
    print(f"[C] model={args.model}  n={args.n}  seeds={args.seeds}")

    print("    loading datasets...")
    DATASETS = {
        "humaneval": (load_humaneval(args.n), score_humaneval, 384, False, "algorithmic"),
        "gsm8k":     (load_gsm8k(args.n),     score_gsm8k,     256, True,  "algorithmic"),
        "triviaqa":  (load_triviaqa(args.n),  score_triviaqa,   64, True,  "factual"),
        "cola":      (load_cola(args.n),      score_cola,       16, True,  "syntax"),
    }
    for name, (items, *_rest) in DATASETS.items():
        print(f"      {name}: {len(items)} items")

    model, tok, dev = common.load_model(args.model)
    model_tag = args.model.split("/")[-1].replace("-Instruct", "").lower()

    cat_names = ["syntax", "algorithmic", "factual"]
    slices = []; prev = 0
    for ze in zones:
        slices.append((prev, ze))
        prev = ze
    cat_to_slice = dict(zip(cat_names, slices))

    summary = {"meta": vars(args), "per_seed": []}

    for s in range(args.seeds):
        seed = 42 + s
        print(f"\n  [seed={seed}] training/loading B")
        B = common.get_or_train_basis(model, tok, args.k, zones, args.steps,
                                      args.top_lambda, seed, model_tag).to(dev)
        Bp = common.make_random_basis(d=B.shape[0], k=args.k, seed=seed,
                                      device=dev)

        seed_results: dict[str, dict] = {}
        for name, (items, scorer, mnt, use_chat, predicted_cat) in DATASETS.items():
            print(f"\n  === {name} ({len(items)}, predicted-positive zone={predicted_cat}) ===")
            mnt_use = min(mnt, args.max_new_tokens)
            entry: dict = {"predicted_positive_zone": predicted_cat,
                            "max_new_tokens": mnt_use, "scores": {}}
            t0 = time.time()

            # Baseline (no ablation)
            print(f"    [baseline]")
            comps = [greedy_generate(model, tok, it["prompt"], dev,
                                      mnt_use, use_chat) for it in items]
            entry["scores"]["baseline"] = scorer(items, comps)
            print(f"      score={entry['scores']['baseline']:.3f}  "
                  f"({time.time()-t0:.1f}s)")

            # Per-zone ablation under B and B'
            for kind, basis in (("B", B), ("Bp", Bp)):
                for zname in cat_names:
                    zlo, zhi = cat_to_slice[zname]
                    t1 = time.time()
                    with ZoneAblator(model, basis, (zlo, zhi)):
                        comps = [greedy_generate(model, tok, it["prompt"], dev,
                                                  mnt_use, use_chat)
                                 for it in items]
                    sc = scorer(items, comps)
                    key = f"{kind}_zone_{zname}"
                    entry["scores"][key] = sc
                    print(f"    [{key}]  score={sc:.3f}  ({time.time()-t1:.1f}s)")
            seed_results[name] = entry
            common.safe_dump(Path(args.out),
                             {"meta": summary["meta"],
                              "per_seed_partial": [{"seed": seed, "datasets": seed_results}]})
        summary["per_seed"].append({"seed": seed, "datasets": seed_results})
        del B, Bp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate: focus on predicted-positive cell vs B' on same zone
    print("\n# Generation contrast — predicted-positive cell (B-Bp on matching zone)")
    print(f"{'dataset':<12} {'pred_cat':<14} {'baseline':>9}  "
          f"{'B_match':>9}  {'Bp_match':>9}  {'Δ_match':>9}")
    seed0 = summary["per_seed"][0]["datasets"]
    for name, entry in seed0.items():
        zc = entry["predicted_positive_zone"]
        b = entry["scores"]["baseline"]
        bm = entry["scores"][f"B_zone_{zc}"]
        pm = entry["scores"][f"Bp_zone_{zc}"]
        d = bm - pm
        print(f"{name:<12} {zc:<14} {b:>9.3f}  {bm:>9.3f}  {pm:>9.3f}  {d:>+9.3f}")

    common.safe_dump(Path(args.out), summary)


if __name__ == "__main__":
    main()
