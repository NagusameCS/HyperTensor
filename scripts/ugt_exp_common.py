"""Shared helpers for the UGT mech-interp experiment battery (A–E).

These five experiments stress-test the UGT random-basis ablation pipeline
beyond the original final-state and layerwise residual-stream interventions.
They share:

  * model / tokenizer loading
  * a cache of trained taxonomic bases B (and matched random Q' bases)
  * residual-stream collection at all transformer blocks
  * a small set of seed-stable training corpora
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCR = ROOT / "scripts"
OUT = ROOT / "benchmarks"
CACHE = OUT / "ugt_basis_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Lazy access to existing modules so callers can import this without paying
# the cost when running tools that only need the registry helpers.
def get_rb_mod():
    return _load("ugt_random_basis_ablation", SCR / "ugt_random_basis_ablation.py")


def get_lw_mod():
    return _load("ugt_random_basis_layerwise", SCR / "ugt_random_basis_layerwise.py")


# --------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------- #

def load_model(model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to(dev)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model, tok, dev


# --------------------------------------------------------------------- #
# Basis cache
# --------------------------------------------------------------------- #

def get_or_train_basis(model, tok, k: int, zones: list[int], steps: int,
                       top_lambda: float, seed: int,
                       model_tag: str, max_length: int = 256,
                       force_retrain: bool = False) -> torch.Tensor:
    """Train a UGT taxonomic basis (or load from cache) for one seed."""
    rb = get_rb_mod()
    cache_path = CACHE / (
        f"{model_tag}_k{k}_z{'-'.join(map(str, zones))}_"
        f"steps{steps}_lam{top_lambda}_seed{seed}.pt"
    )
    if cache_path.exists() and not force_retrain:
        B = torch.load(cache_path, map_location="cpu")
        return B.contiguous()
    torch.manual_seed(seed); np.random.seed(seed)
    adapter = rb.UGTAdapter(model=model, k=k, zones=zones, top_lambda=top_lambda)
    dev = next(model.parameters()).device
    adapter.taxonomic_basis.data = adapter.taxonomic_basis.data.to(dev)
    rb.train_ugt_basis(adapter, tok, rb.gather_corpus(),
                       steps=steps, top_lambda=top_lambda, max_length=max_length)
    B = adapter.taxonomic_basis.detach().clone().contiguous().cpu()
    torch.save(B, cache_path)
    del adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return B


def make_random_basis(d: int, k: int, seed: int, device) -> torch.Tensor:
    return get_rb_mod().make_random_orthonormal_basis(d=d, k=k, seed=seed, device=device)


# --------------------------------------------------------------------- #
# Hidden-state collection
# --------------------------------------------------------------------- #

@torch.no_grad()
def collect_hidden_states(model, tok, prompts: list[str], device,
                          layer_indices: list[int] | None = None,
                          max_length: int = 256) -> dict[int, np.ndarray]:
    """For each prompt, return mean-pooled hidden state at requested layers.

    Returns dict: layer_idx -> ndarray of shape (n_prompts, d_model).
    Layer 0 is the embedding output; layer L is after block L-1.
    """
    try:
        layers = model.model.layers  # llama-style
    except AttributeError:
        layers = model.transformer.h
    L = len(layers)
    if layer_indices is None:
        layer_indices = list(range(L + 1))

    capture: dict[int, list[np.ndarray]] = {l: [] for l in layer_indices}

    for prompt in prompts:
        msgs = [{"role": "user", "content": prompt}]
        try:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            txt = prompt
        enc = tok(txt, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states  # tuple len L+1, each (1, T, d)
        attn = enc.get("attention_mask")
        for li in layer_indices:
            h = hs[li][0].float()  # (T, d)
            if attn is not None:
                m = attn[0].bool()
                h = h[m]
            v = h.mean(dim=0).cpu().numpy()
            capture[li].append(v)
    return {li: np.stack(vs, axis=0) for li, vs in capture.items()}


# --------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------- #

def auroc_one_vs_rest(X: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Train one logistic-regression per class on X, return per-class
    macro AUROC via 5-fold CV."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])
    out: dict[str, float] = {}
    n = len(y)
    for ci, c in enumerate(classes):
        yb = (y == ci).astype(int)
        # bail out if only one class present in some fold
        if yb.sum() < 2 or (n - yb.sum()) < 2:
            out[c] = float("nan")
            continue
        nfolds = min(5, yb.sum(), n - yb.sum())
        skf = StratifiedKFold(n_splits=int(nfolds), shuffle=True, random_state=0)
        scores = []
        for tr, te in skf.split(X, yb):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(X[tr], yb[tr])
            scores.append(roc_auc_score(yb[te], clf.predict_proba(X[te])[:, 1]))
        out[c] = float(np.mean(scores))
    out["_macro"] = float(np.nanmean(list(out.values())))
    return out


def safe_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)} ({path.stat().st_size:,} B)")
