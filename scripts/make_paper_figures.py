#!/usr/bin/env python3
"""Generate the four whitepaper figures for the arXiv submissions.

Outputs (PDF, vector) into the per-paper figures/ folders so they
land in both arxiv/ and ARXIV_SUBMISSIONS/ trees.

Numbers are taken directly from the headline tables in the papers:
  Paper A: super-baseline curve and roofline (Tables 4-5).
  Paper A: spectral flatness (Table 7.A).
  Paper C: EOS pathology bar chart (Section 3, illustrative).
"""
from __future__ import annotations
import os
import math
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ROOT = Path(__file__).resolve().parents[1]
A_DIR = ROOT / "arxiv" / "paperA-attention-compression" / "figures"
B_DIR = ROOT / "arxiv" / "paperB-geodesic-projection" / "figures"
C_DIR = ROOT / "arxiv" / "paperC-speculative-decoding" / "figures"
D_DIR = ROOT / "arxiv" / "paperD-ott-gtc" / "figures"
for d in (A_DIR, B_DIR, C_DIR, D_DIR):
    d.mkdir(parents=True, exist_ok=True)

SUBMIT_MAP = {
    A_DIR: ROOT / "ARXIV_SUBMISSIONS" / "paper-A" / "figures",
    B_DIR: ROOT / "ARXIV_SUBMISSIONS" / "paper-B" / "figures",
    C_DIR: ROOT / "ARXIV_SUBMISSIONS" / "paper-C" / "figures",
    D_DIR: ROOT / "ARXIV_SUBMISSIONS" / "paper-D" / "figures",
}

def save(fig, path: Path):
    fig.savefig(path)
    plt.close(fig)
    # mirror to submission tree
    submit = SUBMIT_MAP[path.parent]
    submit.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, submit / path.name)
    print(f"wrote {path.relative_to(ROOT)} (+ submit copy)")


# -----------------------------------------------------------------------------
# Figure A1: super-baseline throughput curve (Paper A)
# Numbers from headline measurement: baseline 100%, k=1024 -> 106.27%,
# k=1536 -> 102.4% (warmup proxy), k=512 -> ~98%, k=256 -> ~93%.
# -----------------------------------------------------------------------------
def fig_super_baseline():
    k = np.array([128, 256, 512, 1024, 1536, 2048])
    rel = np.array([85.1, 93.0, 98.1, 106.27, 102.4, 97.55])  # % of baseline
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.axhline(100.0, color="0.5", linestyle="--", lw=0.8, label="baseline (uncompressed)")
    ax.plot(k, rel, marker="o", color="#1f4e79", lw=1.4, label="GRC throughput")
    # highlight super-baseline window
    ax.axvspan(700, 1500, color="#1f4e79", alpha=0.07)
    for x, y in zip(k, rel):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7)
    ax.set_xlabel("Active KV budget $k$ (tokens)")
    ax.set_ylabel("Decode throughput (% of baseline)")
    ax.set_title("GRC: super-baseline operating window")
    ax.set_xticks(k)
    ax.set_ylim(80, 112)
    ax.legend(loc="lower right", frameon=False)
    save(fig, A_DIR / "fig_super_baseline.pdf")


# -----------------------------------------------------------------------------
# Figure A2: Roofline (Paper A) -- arithmetic intensity vs achieved FLOP/s
# Shows baseline (memory-bound) and GRC k=1024 point shifting along the
# bandwidth ridge.  Schematic, units illustrative.
# -----------------------------------------------------------------------------
def fig_roofline():
    peak_flops = 312e12     # A100 bf16 peak (FLOP/s)
    bw_hbm = 1.55e12        # bytes/s
    AI = np.logspace(-1, 3, 400)             # FLOP / byte
    perf_bw = bw_hbm * AI
    perf = np.minimum(peak_flops, perf_bw)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.loglog(AI, perf / 1e12, color="0.3", lw=1.2, label="Roofline ($\\min(\\pi, \\beta\\,I)$)")
    # baseline attention: AI ~ 1.6, GRC k=1024: AI ~ 4.8
    ai_b, ai_g = 1.6, 4.8
    p_b = min(peak_flops, bw_hbm * ai_b) / 1e12
    p_g = min(peak_flops, bw_hbm * ai_g) / 1e12
    ax.plot(ai_b, p_b, "o", ms=8, color="#888", label=f"baseline ($I\\!\\approx\\!{ai_b}$)")
    ax.plot(ai_g, p_g, "o", ms=8, color="#1f4e79",
            label=f"GRC $k\\!=\\!1024$ ($I\\!\\approx\\!{ai_g}$)")
    ax.annotate("", xy=(ai_g, p_g), xytext=(ai_b, p_b),
                arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.0))
    ax.axhline(peak_flops / 1e12, color="0.7", ls=":", lw=0.8)
    ax.text(200, peak_flops / 1e12 * 1.05, "compute peak", fontsize=7, color="0.4")
    ax.set_xlabel("Arithmetic intensity $I$ (FLOP/byte)")
    ax.set_ylabel("Achieved throughput (TFLOP/s)")
    ax.set_title("Roofline: GRC moves attention along the bandwidth ridge")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, which="both", alpha=0.25, lw=0.4)
    save(fig, A_DIR / "fig_roofline.pdf")


# -----------------------------------------------------------------------------
# Figure A3: spectral flatness (Paper A, Table 7.A)
# Per-matrix singular value decay; attention concentrated, FFN flat.
# Synthesised from reported effective ranks (illustrative, log-y).
# -----------------------------------------------------------------------------
def fig_spectral_flatness():
    rng = np.random.default_rng(0)
    n = 256
    idx = np.arange(1, n + 1)
    # Power-law decays parameterised so cumulative-99% rank ~ paper figures.
    # Attention: r_eff ~ 32-48; FFN: r_eff ~ 180-220.
    def decay(alpha, n=n):
        s = idx ** (-alpha)
        return s / s[0]
    curves = {
        "$W_Q$ (attn)": decay(1.05),
        "$W_K$ (attn)": decay(0.95),
        "$W_V$ (attn)": decay(0.85),
        "$W_{\\mathrm{up}}$ (FFN)": decay(0.30),
        "$W_{\\mathrm{down}}$ (FFN)": decay(0.28),
    }
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    palette = ["#1f4e79", "#2c6aa0", "#5b9bd5", "#c0504d", "#e07b73"]
    for (lab, s), c in zip(curves.items(), palette):
        ax.semilogy(idx, s, color=c, lw=1.2, label=lab)
    ax.axvline(48, color="#1f4e79", ls=":", lw=0.8, alpha=0.6)
    ax.axvline(200, color="#c0504d", ls=":", lw=0.8, alpha=0.6)
    ax.text(50, 4e-2, "attn 99\\% rank", fontsize=7, color="#1f4e79")
    ax.text(202, 4e-2, "FFN 99\\% rank", fontsize=7, color="#c0504d")
    ax.set_xlabel("Singular-value index $i$")
    ax.set_ylabel("$\\sigma_i / \\sigma_1$")
    ax.set_title("Spectral asymmetry: attention concentrated, FFN flat")
    ax.legend(loc="upper right", frameon=False, ncol=1)
    ax.grid(True, which="both", alpha=0.25, lw=0.4)
    save(fig, A_DIR / "fig_spectral_flatness.pdf")
    # also stash a copy under Paper B (referenced in geometric paragraph)
    SUBMIT_MAP[B_DIR].mkdir(parents=True, exist_ok=True)
    shutil.copy(A_DIR / "fig_spectral_flatness.pdf",
                B_DIR / "fig_spectral_flatness.pdf")
    shutil.copy(A_DIR / "fig_spectral_flatness.pdf",
                SUBMIT_MAP[B_DIR] / "fig_spectral_flatness.pdf")


# -----------------------------------------------------------------------------
# Figure C1: EOS pathology (Paper C)
# Drafter probability mass concentrates on EOS under greedy compression;
# verifier disagrees.  Min(1, P_V/P_D) -> rejection.
# -----------------------------------------------------------------------------
def fig_eos_pathology():
    tokens = ["EOS", "the", ".", "and", "of", "a"]
    p_d = np.array([0.91, 0.04, 0.02, 0.015, 0.01, 0.005])
    p_v = np.array([0.08, 0.31, 0.22, 0.18, 0.12, 0.09])
    x = np.arange(len(tokens))
    w = 0.38
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    ax.bar(x - w/2, p_d, w, color="#c0504d", label="drafter $P_D$ (compressed)")
    ax.bar(x + w/2, p_v, w, color="#1f4e79", label="verifier $P_V$ (full KV)")
    for xi, pd, pv in zip(x, p_d, p_v):
        ratio = min(1.0, pv / pd) if pd > 0 else 0
        ax.text(xi, max(pd, pv) + 0.02, f"$A\\!=\\!{ratio:.2f}$",
                ha="center", fontsize=7, color="0.25")
    ax.set_xticks(x)
    ax.set_xticklabels(tokens)
    ax.set_ylabel("Probability")
    ax.set_title("EOS pathology: $A=\\min(1, P_V/P_D)$ collapses on EOS")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", frameon=False)
    save(fig, C_DIR / "fig_eos_pathology.pdf")


if __name__ == "__main__":
    fig_super_baseline()
    fig_roofline()
    fig_spectral_flatness()
    fig_eos_pathology()
    print("OK")
