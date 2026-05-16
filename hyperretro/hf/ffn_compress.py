#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................:::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............:::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............:::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""FFN (MLP) compression for HuggingFace transformers.

Companion to ``hyperretro.hf.compress`` (which handles attention Q/K/V).

At 0.5B scale, attention contributes only ~4% of parameters; the FFN
block (gate_proj/up_proj/down_proj for Llama-family, or c_fc/c_proj for
GPT-2) dominates parameter count and FLOPs. Compressing the FFN is
therefore the biggest lever for both memory and drafter wall-clock in
speculative decoding.

Strategy
--------
For Llama / Qwen / Mistral / Gemma (SwiGLU MLPs)::

    gate_proj :  W_gate [intermediate, hidden]
    up_proj   :  W_up   [intermediate, hidden]
    down_proj :  W_down [hidden, intermediate]

``gate_proj`` and ``up_proj`` share input space (hidden_size), so we
build a GRC-style shared basis ``P`` from their joint Gram and project
both onto ``P[:, :k]``::

    K = W_gate.T @ W_gate + W_up.T @ W_up
    P = top-k eigenvectors of K
    W_gate' = W_gate @ P @ P.T
    W_up'   = W_up   @ P @ P.T

``down_proj`` has a different input space (intermediate_size). We use
truncated SVD on its rows (or equivalently the right singular vectors)
to reduce the effective intermediate-axis rank::

    U, S, Vt = svd(W_down)
    W_down'  = U[:, :r] @ diag(S[:r]) @ Vt[:r, :]

For GPT-2-style two-layer MLPs (c_fc, c_proj), we apply the same SVD
trick to each independently since there's no shared geometry.

Storage Format
--------------
We write the projected weights back at *full* shape — same as the
attention compress pipeline. This means:
  * vanilla ``AutoModelForCausalLM.from_pretrained`` loads them fine
  * the rank reduction shows up as Frobenius error, not shape change
  * runtime savings come from a rank-aware kernel (TODO) or from the
    fact that low-rank weights compress well under Q8_0 / Q4

This keeps HyperRetro's "drop-in safetensors" contract intact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# FFN tensor-name patterns
# ---------------------------------------------------------------------------

# Patterns: (marker, group, {"gate": suffix, "up": suffix, "down": suffix})
# or for GPT-2 style: (marker, group, {"fc": suffix, "proj": suffix})
_FFN_PATTERNS: list[tuple[str, int, dict[str, str]]] = [
    # Llama / Qwen2 / Mistral / Gemma — SwiGLU MLP (gate + up + down)
    (".layers.", 0, {"gate": ".mlp.gate_proj.weight",
                      "up":   ".mlp.up_proj.weight",
                      "down": ".mlp.down_proj.weight"}),
    # GPT-2 — two-layer MLP (c_fc + c_proj)
    ("transformer.h.", 1, {"fc":   ".mlp.c_fc.weight",
                            "proj": ".mlp.c_proj.weight"}),
    # GPT-NeoX / Pythia
    ("gpt_neox.layers.", 1, {"fc":   ".mlp.dense_h_to_4h.weight",
                              "proj": ".mlp.dense_4h_to_h.weight"}),
    # Falcon
    ("transformer.h.", 1, {"fc":   ".mlp.dense_h_to_4h.weight",
                            "proj": ".mlp.dense_4h_to_h.weight"}),
]


def _parse_layer_idx(name: str, marker: str) -> int | None:
    idx = name.find(marker)
    if idx < 0:
        return None
    rest = name[idx + len(marker):]
    end = 0
    while end < len(rest) and rest[end].isdigit():
        end += 1
    if end == 0:
        return None
    try:
        return int(rest[:end])
    except ValueError:
        return None


def _group_ffn_by_layer(state_dict: dict) -> dict[int, dict[str, str]]:
    """Discover FFN weight keys by layer index.

    Returns:
        dict mapping layer_idx → {"gate"|"up"|"down": key} (SwiGLU)
        or layer_idx → {"fc"|"proj": key} (GPT-2 style).
    """
    layers: dict[int, dict[str, str]] = {}
    for name in state_dict.keys():
        for marker, _group, patterns in _FFN_PATTERNS:
            li = _parse_layer_idx(name, marker)
            if li is None:
                continue
            for slot, suffix in patterns.items():
                if name.endswith(suffix):
                    existing = layers.setdefault(li, {})
                    if slot not in existing:
                        existing[slot] = name
                    break
            else:
                continue
            break
    return layers


# ---------------------------------------------------------------------------
# Compression primitives
# ---------------------------------------------------------------------------

def build_ffn_shared_basis(W_gate: np.ndarray, W_up: np.ndarray,
                            n_iter: int = 3) -> np.ndarray:
    """Build a shared input-space basis for gate+up (SwiGLU).

    Mirrors ``scripts.grc_distill.build_shared_basis`` but for two
    matrices that share the input dimension.
    """
    K = W_gate.T @ W_gate + W_up.T @ W_up
    K = K / np.linalg.norm(K, "fro")
    A = K.copy()
    for _ in range(n_iter):
        A = A @ K
        A = A / np.linalg.norm(A, "fro")
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]


def project_basis(W: np.ndarray, P: np.ndarray, k: int) -> np.ndarray:
    """Project W onto the top-k columns of basis P: W' = W @ P_k @ P_k.T."""
    Pk = P[:, :k]
    return W @ Pk @ Pk.T


def truncated_svd_project(W: np.ndarray, r: int) -> tuple[np.ndarray, float]:
    """Truncate W to rank r via SVD.

    Returns (W_r, sigma_r_plus_1) where sigma_r_plus_1 is the largest
    discarded singular value (= spectral norm of the truncation error).
    """
    r = min(r, min(W.shape))
    # Economy SVD — np handles arbitrary aspect ratios.
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    W_r = (U[:, :r] * S[:r]) @ Vt[:r, :]
    sigma_next = float(S[r]) if r < len(S) else 0.0
    return W_r, sigma_next


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FFNCompressConfig:
    """Configuration for FFN compression.

    ``rank_in``: SVD rank for gate_proj and up_proj (each independently).
        For SwiGLU MLPs (Llama-family).
    ``rank_out``: SVD rank for down_proj (also for GPT-2-style c_fc/c_proj).
    ``mode``: "svd" (default; independent SVD on each matrix) or
        "shared" (joint gate+up shared basis — known to break SwiGLU
        gating, kept only for ablation).
    """
    rank_in: int = 0       # 0 = no compression on gate/up
    rank_out: int = 0      # 0 = no compression on down (or c_fc/c_proj)
    layers: list[int] = field(default_factory=list)  # [] = all
    mode: str = "svd"      # "svd" or "shared"


# ---------------------------------------------------------------------------
# Main compression routine
# ---------------------------------------------------------------------------

def _to_np(t) -> np.ndarray:
    if hasattr(t, "detach"):
        return t.detach().to("cpu").float().numpy()
    return np.asarray(t, dtype=np.float32)


def _write_back(state_dict: dict, name: str, arr: np.ndarray) -> None:
    orig = state_dict[name]
    if hasattr(orig, "detach"):
        import torch
        t = torch.from_numpy(arr).to(orig.dtype).to(orig.device)
        state_dict[name] = t
    else:
        state_dict[name] = arr.astype(getattr(orig, "dtype", np.float32))


def _frob_relerr(W: np.ndarray, Wh: np.ndarray) -> float:
    n = np.linalg.norm(W)
    return float(np.linalg.norm(W - Wh) / n) if n > 0 else 0.0


def compress_ffn_state_dict(state_dict: dict,
                             cfg: FFNCompressConfig) -> dict[str, dict]:
    """Apply FFN compression in place. Returns per-layer stats."""
    stats: dict[str, dict] = {}
    layer_map = _group_ffn_by_layer(state_dict)
    layers_to_do: Iterable[int] = sorted(layer_map.keys())
    if cfg.layers:
        layers_to_do = [i for i in layers_to_do if i in cfg.layers]

    for li in layers_to_do:
        names = layer_map[li]
        layer_stats: dict[str, float | int | bool] = {"layer": int(li)}

        # --- SwiGLU path: gate + up + down ---
        if "gate" in names and "up" in names and "down" in names:
            W_gate = _to_np(state_dict[names["gate"]])
            W_up = _to_np(state_dict[names["up"]])
            W_down = _to_np(state_dict[names["down"]])

            hidden = W_gate.shape[1]      # input dim of gate/up
            intermediate = W_gate.shape[0]  # output dim

            # gate + up: independent SVD by default (mode="svd").
            # Shared-basis mode (mode="shared") is empirically broken for
            # SwiGLU because gate and up are multiplied element-wise through
            # the gating nonlinearity; projecting both onto the same low-dim
            # subspace destroys the gating relationship and gives garbage
            # PPL (~4800 vs ~7 baseline on Qwen2.5-0.5B at 50% rank).
            if cfg.rank_in > 0:
                if cfg.mode == "svd":
                    r_g = min(cfg.rank_in, min(W_gate.shape))
                    W_gate_new, sig_g_next = truncated_svd_project(W_gate, r_g)
                    r_u = min(cfg.rank_in, min(W_up.shape))
                    W_up_new, sig_u_next = truncated_svd_project(W_up, r_u)
                    _write_back(state_dict, names["gate"], W_gate_new)
                    _write_back(state_dict, names["up"], W_up_new)
                    layer_stats["rank_in"] = int(r_g)
                    layer_stats["hidden"] = int(hidden)
                    layer_stats["frob_relerr_gate"] = _frob_relerr(W_gate, W_gate_new)
                    layer_stats["frob_relerr_up"] = _frob_relerr(W_up, W_up_new)
                    layer_stats["sigma_gate_next"] = float(sig_g_next)
                    layer_stats["sigma_up_next"] = float(sig_u_next)
                    layer_stats["mode"] = "svd"
                elif cfg.mode == "shared" and cfg.rank_in < hidden:
                    P = build_ffn_shared_basis(W_gate, W_up)
                    k = min(cfg.rank_in, hidden)
                    W_gate_new = project_basis(W_gate, P, k)
                    W_up_new = project_basis(W_up, P, k)
                    _write_back(state_dict, names["gate"], W_gate_new)
                    _write_back(state_dict, names["up"], W_up_new)
                    layer_stats["rank_in"] = int(k)
                    layer_stats["hidden"] = int(hidden)
                    layer_stats["frob_relerr_gate"] = _frob_relerr(W_gate, W_gate_new)
                    layer_stats["frob_relerr_up"] = _frob_relerr(W_up, W_up_new)
                    layer_stats["mode"] = "shared"
                else:
                    raise ValueError(f"unknown FFN mode {cfg.mode!r}")

            # down_proj SVD truncation
            if cfg.rank_out > 0 and cfg.rank_out < min(W_down.shape):
                r = min(cfg.rank_out, min(W_down.shape))
                W_down_new, sigma_next = truncated_svd_project(W_down, r)
                _write_back(state_dict, names["down"], W_down_new)
                layer_stats["rank_out"] = int(r)
                layer_stats["intermediate"] = int(intermediate)
                layer_stats["frob_relerr_down"] = _frob_relerr(W_down, W_down_new)
                layer_stats["sigma_out_next"] = float(sigma_next)

            layer_stats["arch"] = "swiglu"

        # --- GPT-2 style: fc + proj ---
        elif "fc" in names and "proj" in names:
            W_fc = _to_np(state_dict[names["fc"]])
            W_proj = _to_np(state_dict[names["proj"]])

            # Note: GPT-2 stores c_fc / c_proj as transposed Conv1D weights
            # of shape [in, out] (not the usual [out, in]). The math works
            # the same for SVD truncation — we just truncate to rank r.
            if cfg.rank_out > 0:
                r_fc = min(cfg.rank_out, min(W_fc.shape))
                W_fc_new, s_fc_next = truncated_svd_project(W_fc, r_fc)
                _write_back(state_dict, names["fc"], W_fc_new)
                layer_stats["rank_fc"] = int(r_fc)
                layer_stats["frob_relerr_fc"] = _frob_relerr(W_fc, W_fc_new)
                layer_stats["sigma_fc_next"] = float(s_fc_next)

                r_pj = min(cfg.rank_out, min(W_proj.shape))
                W_proj_new, s_pj_next = truncated_svd_project(W_proj, r_pj)
                _write_back(state_dict, names["proj"], W_proj_new)
                layer_stats["rank_proj"] = int(r_pj)
                layer_stats["frob_relerr_proj"] = _frob_relerr(W_proj, W_proj_new)
                layer_stats["sigma_proj_next"] = float(s_pj_next)

            layer_stats["arch"] = "gpt2_mlp"

        else:
            continue

        stats[f"layer_{li}"] = layer_stats  # type: ignore[assignment]

    return stats


__all__ = [
    "FFNCompressConfig",
    "compress_ffn_state_dict",
    "build_ffn_shared_basis",
    "project_basis",
    "truncated_svd_project",
]
