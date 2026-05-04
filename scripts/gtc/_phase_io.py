"""
gtc/_phase_io.py
================

Tiny loader for the Phase-1 / Phase-3 JSON exports under
``legacy/axiom_vis/<model>/``.

We deliberately keep this module minimal --- no pandas, no torch, just numpy +
stdlib --- so that the GTC prototype can run on the same `.venv` already in use
by `scripts/analysis/`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
AXVIS = REPO / "legacy" / "axiom_vis"


@dataclass
class Phase1:
    intrinsic_dim: int
    cloud: np.ndarray  # (N, 3) projected scatter cloud
    eigenvalues: np.ndarray
    explained_ratio: float


@dataclass
class Phase3Point:
    pos: np.ndarray  # (3,)
    R: float  # scalar curvature
    g_diag: np.ndarray  # (3,) diag of g in the 3-D vis projection
    christoffel_norm: float


@dataclass
class Phase3:
    fisher_trace_mean: float
    fisher_det_log_mean: float
    points: List[Phase3Point]
    mean_R: float
    dim: int


def load_phase1(model: str) -> Phase1:
    p = AXVIS / model / "phase1_manifold.json"
    j = json.loads(p.read_text(encoding="utf-8"))
    return Phase1(
        intrinsic_dim=int(j["intrinsic_dim"]),
        cloud=np.asarray(j["cloud"], dtype=np.float64),
        eigenvalues=np.asarray(j["eigenvalues"], dtype=np.float64),
        explained_ratio=float(j["explained_ratio"]),
    )


def load_phase3(model: str) -> Phase3:
    p = AXVIS / model / "phase3_curvature.json"
    j = json.loads(p.read_text(encoding="utf-8"))
    pts = [
        Phase3Point(
            pos=np.asarray(pt["pos"], dtype=np.float64),
            R=float(pt["R"]),
            g_diag=np.asarray(pt["g_diag"], dtype=np.float64),
            christoffel_norm=float(pt["christoffel_norm"]),
        )
        for pt in j["points"]
    ]
    return Phase3(
        fisher_trace_mean=float(j["fisher_trace_mean"]),
        fisher_det_log_mean=float(j["fisher_det_log_mean"]),
        points=pts,
        mean_R=float(j["mean_scalar_curvature"]),
        dim=int(j["dim"]),
    )


def nearest_point(p3: Phase3, x: np.ndarray) -> Phase3Point:
    """Nearest neighbour in the 3-D visualisation cloud."""
    diffs = np.stack([pt.pos for pt in p3.points]) - x[None, :]
    idx = int(np.argmin(np.einsum("ij,ij->i", diffs, diffs)))
    return p3.points[idx]
