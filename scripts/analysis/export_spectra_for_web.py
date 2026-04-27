"""Export downsampled SV spectra + per-layer ranks to JSON for the website's
interactive Plotly figures on Paper 1.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

CACHE = Path("docs/figures/_spectra_cache.npz")
SUMMARY = Path("docs/figures/spectra_summary.json")
OUT = Path("docs/figures/spectra_web.json")

ATTN_SLOTS = ("Q", "K", "V", "O")
FFN_SLOTS = ("gate", "up", "down")


def downsample(sv: np.ndarray, n: int = 200) -> tuple[list[int], list[float]]:
    """Log-space downsample: keep first 30 points, then log-spaced through end."""
    L = len(sv)
    if L <= n:
        idx = np.arange(L)
    else:
        head = np.arange(min(30, n // 4))
        tail = np.unique(
            np.round(np.geomspace(max(head[-1] + 1, 1), L - 1, n - len(head))).astype(int)
        )
        idx = np.unique(np.concatenate([head, tail]))
    return idx.tolist(), sv[idx].astype(float).tolist()


def cumulative(sv: np.ndarray) -> list[float]:
    """Cumulative energy fraction at every k. Downsample to ~200 points."""
    e = sv ** 2
    c = np.cumsum(e) / e.sum()
    L = len(c)
    if L <= 200:
        idx = np.arange(L)
    else:
        idx = np.unique(np.round(np.geomspace(1, L - 1, 200)).astype(int))
    return idx.tolist(), c[idx].astype(float).tolist()


def main() -> int:
    cache = np.load(CACHE, allow_pickle=True)
    attn = cache["attn"].item()
    ffn = cache["ffn"].item()
    summary = json.load(SUMMARY.open())

    out: dict = {
        "model": "Llama-3.1-8B-Instruct (Q4_K_M dequantised)",
        "layers": sorted(int(L) for L in attn.keys()),
        "spectra": {"attn": {}, "ffn": {}},
        "energy": {"attn": {}, "ffn": {}},
        "rank95_per_layer": {"attn": {}, "ffn": {}},
    }

    for L in sorted(attn.keys()):
        out["spectra"]["attn"][str(L)] = {}
        out["energy"]["attn"][str(L)] = {}
        for slot in ATTN_SLOTS:
            sv = np.asarray(attn[L][slot], dtype=float)
            idx, vals = downsample(sv)
            out["spectra"]["attn"][str(L)][slot] = {"k": idx, "sigma": vals}
            cidx, cvals = cumulative(sv)
            out["energy"]["attn"][str(L)][slot] = {"k": cidx, "cum": cvals}
        out["spectra"]["ffn"][str(L)] = {}
        out["energy"]["ffn"][str(L)] = {}
        for slot in FFN_SLOTS:
            sv = np.asarray(ffn[L][slot], dtype=float)
            idx, vals = downsample(sv)
            out["spectra"]["ffn"][str(L)][slot] = {"k": idx, "sigma": vals}
            cidx, cvals = cumulative(sv)
            out["energy"]["ffn"][str(L)][slot] = {"k": cidx, "cum": cvals}

    # per-layer ranks for 95% energy
    for cls in ("attn", "ffn"):
        for slot, info in summary["per_layer"]["0"][cls].items():
            slots = list(info.keys()) if isinstance(info, dict) else []
        # easier: iterate full structure
    for cls in ("attn", "ffn"):
        per_slot = {}
        slot_names = ATTN_SLOTS if cls == "attn" else FFN_SLOTS
        for slot in slot_names:
            per_slot[slot] = []
            for L_str in sorted(summary["per_layer"].keys(), key=int):
                r = summary["per_layer"][L_str][cls][slot]["rank_for_energy"]["0.95"]
                per_slot[slot].append(int(r))
        out["rank95_per_layer"][cls] = {
            "layers": sorted(int(L) for L in summary["per_layer"].keys()),
            "by_slot": per_slot,
        }

    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"Wrote {OUT}: {OUT.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
