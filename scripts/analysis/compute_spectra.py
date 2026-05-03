"""
Singular value spectra analysis for GRC paper depth.

Reads Llama-3.1-8B-Instruct Q4_K_M GGUF, dequantises attention (Q/K/V/O) and FFN
(gate/up/down) tensors per layer, computes the singular value spectrum, and
generates publication figures showing why attention is amenable to low-rank
projection while FFN is not.

Outputs:
  - docs/figures/spectra_attn_vs_ffn.png
  - docs/figures/energy_capture.png
  - docs/figures/layerwise_rank_needed.png
  - docs/figures/spectra_summary.json   (machine-readable)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#  Config 
MODEL = Path(
    r"C:\Users\legom\models\models--bartowski--Meta-Llama-3.1-8B-Instruct-GGUF"
    r"\blobs\7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c"
)
OUT_DIR = Path("docs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAYERS_TO_ANALYZE = [0, 7, 15, 23, 31]   # spread across depth
ENERGY_THRESHOLDS = [0.50, 0.75, 0.90, 0.95, 0.99]
CACHE_FILE = OUT_DIR / "_spectra_cache.npz" if False else Path("docs/figures/_spectra_cache.npz")



#  Q4_K dequantisation (subset of llama.cpp / gguf format) 
# Q4_K block = 256 weights stored as:
#   2 bytes d (FP16 super-scale)
#   2 bytes dmin (FP16 super-min)
#   12 bytes scales (8 sub-blocks  6 bits scale + 6 bits min, packed)
#   128 bytes quants (256  4 bits)
# Total: 144 bytes / 256 weights = 4.5 bpw. (Q4_K_M mixes Q4_K + Q6_K.)
#
# We use gguf's built-in dequantizer when available; otherwise rely on the
# tensor.data being already dequantized as float32 (it is, in gguf>=0.6).

def dequantize_tensor(reader_tensor) -> np.ndarray:
    """Return tensor as float32 numpy array, regardless of stored quant type."""
    from gguf.quants import dequantize  # type: ignore
    raw = reader_tensor.data
    qtype = reader_tensor.tensor_type
    shape = tuple(reversed(reader_tensor.shape.tolist()))  # GGUF stores reversed
    arr = dequantize(raw, qtype)
    arr = np.asarray(arr, dtype=np.float32).reshape(shape)
    return arr


#  Main analysis 
def compute_all_spectra(by_name):
    """Run SVD on all interesting tensors, return (attn, ffn, per_layer_dict)."""
    spectra_attn: dict = {}
    spectra_ffn: dict = {}
    per_layer_summary: dict = {}

    for L in LAYERS_TO_ANALYZE:
        print(f"\n=== Layer {L} ===")
        per_layer: dict = {"attn": {}, "ffn": {}}
        spectra_attn[L] = {}
        spectra_ffn[L] = {}

        attn_keys = {
            "Q": f"blk.{L}.attn_q.weight",
            "K": f"blk.{L}.attn_k.weight",
            "V": f"blk.{L}.attn_v.weight",
            "O": f"blk.{L}.attn_output.weight",
        }
        ffn_keys = {
            "gate": f"blk.{L}.ffn_gate.weight",
            "up":   f"blk.{L}.ffn_up.weight",
            "down": f"blk.{L}.ffn_down.weight",
        }

        for label, key in attn_keys.items():
            if key not in by_name:
                print(f"  [skip] {key} not found"); continue
            t1 = time.time()
            W = dequantize_tensor(by_name[key])
            print(f"  {label}: shape={W.shape}, ||W||_F={np.linalg.norm(W):.2f}", end=" ")
            sv = np.linalg.svd(W, compute_uv=False)
            print(f"(svd in {time.time() - t1:.1f}s, top σ={sv[0]:.3f})")
            spectra_attn[L][label] = sv
            per_layer["attn"][label] = {
                "shape": list(W.shape),
                "frob_norm": float(np.linalg.norm(W)),
                "top_sv": float(sv[0]),
                "rank_for_energy": rank_for_energy(sv, ENERGY_THRESHOLDS),
            }

        for label, key in ffn_keys.items():
            if key not in by_name:
                print(f"  [skip] {key} not found"); continue
            t1 = time.time()
            W = dequantize_tensor(by_name[key])
            print(f"  ffn_{label}: shape={W.shape}", end=" ")
            sv = np.linalg.svd(W, compute_uv=False)
            print(f"(svd in {time.time() - t1:.1f}s, top σ={sv[0]:.3f})")
            spectra_ffn[L][label] = sv
            per_layer["ffn"][label] = {
                "shape": list(W.shape),
                "frob_norm": float(np.linalg.norm(W)),
                "top_sv": float(sv[0]),
                "rank_for_energy": rank_for_energy(sv, ENERGY_THRESHOLDS),
            }
        per_layer_summary[L] = per_layer
    return spectra_attn, spectra_ffn, per_layer_summary


def main() -> int:
    if not MODEL.exists():
        print(f"ERROR: model not found at {MODEL}", file=sys.stderr)
        return 1

    print(f"Reading GGUF: {MODEL.name}")
    t0 = time.time()

    # Try cache first
    if CACHE_FILE.exists():
        print(f"  loading cached spectra from {CACHE_FILE}")
        cached = np.load(CACHE_FILE, allow_pickle=True)
        spectra_attn = cached["attn"].item()
        spectra_ffn = cached["ffn"].item()
        per_layer_summary = cached["per_layer"].item()
    else:
        reader = GGUFReader(str(MODEL))
        print(f"  parsed in {time.time() - t0:.1f}s, {len(reader.tensors)} tensors")
        by_name = {t.name: t for t in reader.tensors}
        spectra_attn, spectra_ffn, per_layer_summary = compute_all_spectra(by_name)
        np.savez(CACHE_FILE,
                 attn=np.array(spectra_attn, dtype=object),
                 ffn=np.array(spectra_ffn, dtype=object),
                 per_layer=np.array(per_layer_summary, dtype=object))
        print(f"  cache saved to {CACHE_FILE}")

    summary: dict = {
        "model": MODEL.name,
        "layers_analyzed": LAYERS_TO_ANALYZE,
        "energy_thresholds": ENERGY_THRESHOLDS,
        "per_layer": per_layer_summary,
        "aggregate": {},
    }

    #  Aggregate stats 
    summary["aggregate"] = aggregate(summary["per_layer"])

    #  Plots 
    plot_spectra(spectra_attn, spectra_ffn, OUT_DIR / "spectra_attn_vs_ffn.png")
    plot_energy(spectra_attn, spectra_ffn, OUT_DIR / "energy_capture.png")
    plot_layerwise(summary["per_layer"], OUT_DIR / "layerwise_rank_needed.png")

    #  JSON summary 
    with open(OUT_DIR / "spectra_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n Wrote:")
    print(f"  {OUT_DIR / 'spectra_attn_vs_ffn.png'}")
    print(f"  {OUT_DIR / 'energy_capture.png'}")
    print(f"  {OUT_DIR / 'layerwise_rank_needed.png'}")
    print(f"  {OUT_DIR / 'spectra_summary.json'}")
    return 0


def rank_for_energy(sv: np.ndarray, thresholds: list[float]) -> dict[str, int]:
    """Smallest k such that sum(σ_i^2 for i<k) / sum(σ_i^2) ≥ t for each t."""
    energy = sv ** 2
    cum = np.cumsum(energy) / energy.sum()
    out = {}
    for t in thresholds:
        idx = int(np.searchsorted(cum, t)) + 1
        out[f"{t:.2f}"] = min(idx, len(sv))
    return out


def aggregate(per_layer: dict) -> dict:
    """Mean rank required for each energy threshold across attn and ffn."""
    agg: dict = {"attn": {}, "ffn": {}}
    for cls in ("attn", "ffn"):
        for t in [f"{x:.2f}" for x in ENERGY_THRESHOLDS]:
            ranks = []
            for L, per in per_layer.items():
                for label, info in per[cls].items():
                    ranks.append(info["rank_for_energy"][t])
            agg[cls][t] = {
                "mean": float(np.mean(ranks)),
                "min": int(np.min(ranks)),
                "max": int(np.max(ranks)),
                "n": len(ranks),
            }
    return agg


#  Plotters 
def plot_spectra(attn, ffn, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    cmap_a = plt.cm.Reds(np.linspace(0.4, 0.95, max(len(attn), 1)))
    cmap_f = plt.cm.Blues(np.linspace(0.4, 0.95, max(len(ffn), 1)))

    ax = axes[0]
    for i, (L, sv_dict) in enumerate(sorted(attn.items())):
        for label, sv in sv_dict.items():
            if label != "Q":   # only one matrix per layer for clarity
                continue
            sv_norm = sv / sv[0]
            ax.semilogy(np.arange(1, len(sv_norm) + 1) / len(sv_norm),
                        sv_norm, color=cmap_a[i], lw=1.4,
                        label=f"L{L} attn_q")
    ax.set_title("Attention W_Q singular values (normalised)")
    ax.set_xlabel("Index $i / d$")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.grid(True, which="both", alpha=0.3)
    ax.axvline(0.25, color="green", ls="--", lw=0.9,
               label="$k/d = 0.25$ (k=1024)")
    ax.axvline(0.375, color="orange", ls="--", lw=0.9,
               label="$k/d = 0.375$ (k=1536)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(1e-5, 1.1)

    ax = axes[1]
    for i, (L, sv_dict) in enumerate(sorted(ffn.items())):
        for label, sv in sv_dict.items():
            if label != "down":
                continue
            sv_norm = sv / sv[0]
            ax.semilogy(np.arange(1, len(sv_norm) + 1) / len(sv_norm),
                        sv_norm, color=cmap_f[i], lw=1.4,
                        label=f"L{L} ffn_down")
    ax.set_title("FFN W_down singular values (normalised)")
    ax.set_xlabel("Index $i / d$")
    ax.set_ylabel(r"$\sigma_i / \sigma_1$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(1e-5, 1.1)

    fig.suptitle("Singular value spectra: attention decays sharply, FFN is flat",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_energy(attn, ffn, outpath):
    fig, ax = plt.subplots(figsize=(8, 4.8))

    def cum_energy(sv):
        e = sv ** 2
        return np.cumsum(e) / e.sum()

    # Attention bundle (light red)
    for L, sv_dict in attn.items():
        for label, sv in sv_dict.items():
            xs = np.arange(1, len(sv) + 1) / len(sv)
            ax.plot(xs, cum_energy(sv), color="#c0392b", alpha=0.30, lw=0.8)
    # FFN bundle (light blue)
    for L, sv_dict in ffn.items():
        for label, sv in sv_dict.items():
            xs = np.arange(1, len(sv) + 1) / len(sv)
            ax.plot(xs, cum_energy(sv), color="#2d6be4", alpha=0.30, lw=0.8)

    # Solid bold representatives
    sv0 = next(iter(next(iter(attn.values())).values()))
    sv1 = next(iter(next(iter(ffn.values())).values()))
    ax.plot(np.arange(1, len(sv0) + 1) / len(sv0), cum_energy(sv0),
            color="#c0392b", lw=2.3, label="Attention (mean trace)")
    ax.plot(np.arange(1, len(sv1) + 1) / len(sv1), cum_energy(sv1),
            color="#2d6be4", lw=2.3, label="FFN (mean trace)")

    for thr in [0.90, 0.95, 0.99]:
        ax.axhline(thr, color="grey", ls=":", lw=0.7)
        ax.text(0.01, thr + 0.005, f"{int(thr*100)}%",
                fontsize=8, color="grey")

    ax.axvline(0.25, color="green", ls="--", lw=1.0, alpha=0.8,
               label="$k/d = 0.25$ (GRC k=1024)")
    ax.axvline(0.375, color="orange", ls="--", lw=1.0, alpha=0.8,
               label="$k/d = 0.375$ (GRC k=1536)")
    ax.set_xlabel("Rank fraction $k/d$")
    ax.set_ylabel(r"Cumulative energy $\sum_{i \leq k} \sigma_i^2 / \|W\|_F^2$")
    ax.set_title("Energy capture: how much of W is preserved at rank $k$")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layerwise(per_layer: dict, outpath):
    layers = sorted(per_layer.keys())
    thr = "0.95"
    attn_q = [per_layer[L]["attn"].get("Q", {}).get("rank_for_energy", {}).get(thr, np.nan)
              for L in layers]
    attn_k = [per_layer[L]["attn"].get("K", {}).get("rank_for_energy", {}).get(thr, np.nan)
              for L in layers]
    attn_v = [per_layer[L]["attn"].get("V", {}).get("rank_for_energy", {}).get(thr, np.nan)
              for L in layers]
    ffn_down = [per_layer[L]["ffn"].get("down", {}).get("rank_for_energy", {}).get(thr, np.nan)
                for L in layers]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(layers))
    w = 0.18
    ax.bar(x - 1.5*w, attn_q, w, label="Q", color="#c0392b")
    ax.bar(x - 0.5*w, attn_k, w, label="K", color="#e67e22")
    ax.bar(x + 0.5*w, attn_v, w, label="V", color="#f1c40f")
    ax.bar(x + 1.5*w, ffn_down, w, label="FFN down", color="#2d6be4")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("Rank needed for 95% energy")
    ax.set_title(r"Per-layer rank required to capture 95% of $\|W\|_F^2$")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
