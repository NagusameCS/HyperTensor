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

"""Factored on-disk storage for HyperRetro-compressed attention weights.

After ``compress_hf_model`` projects ``W_q, W_k, W_v`` onto a shared rank-k
basis ``P_k``, the resulting weights have *effective* rank k, but the
compress pipeline rematerialises them as dense ``d×d`` matrices to keep the
output 100% HuggingFace-compatible. That means *zero on-disk shrink* despite
the rank reduction (the central audit finding of round 9).

This module fixes that by writing the projections in factored form:

    W_q ≈ B_q @ A          shape(A) = (k, d_in)     # shared per layer
    W_k ≈ B_k @ A          shape(B_*) = (d_out, k)
    W_v ≈ B_v @ A          A is the *shared* GRC basis P_k^T

Storage per attention layer drops from ``3·d_out·d_in`` to
``d_in·k + 3·d_out·k = k·(d_in + 3·d_out)``, which for square attn with
d=1536, k=640 is **4dk/(3d²) ≈ 55.5%** of the original — i.e. ~45% attn
savings, ~10–11% total params on Qwen2.5-1.5B.

Format on disk
--------------
The safetensors file keeps standard HF keys for every non-factored tensor.
For each factored layer ``L`` we add three keys::

    model.layers.{L}.self_attn.factored_A   shape (k, d_in)
    model.layers.{L}.self_attn.factored_Bq  shape (d_out, k)
    model.layers.{L}.self_attn.factored_Bk  shape (d_out, k)
    model.layers.{L}.self_attn.factored_Bv  shape (d_out, k)

and *omit* the original dense ``q_proj.weight``, ``k_proj.weight``,
``v_proj.weight``. A manifest ``hyperretro_factored.json`` records the
list of factored layers, the per-layer rank, and original Linear shapes
so the loader can rebuild the architecture.

Loaders use :func:`load_factored_hf_model` which:
  1. ``AutoModelForCausalLM.from_pretrained`` of the *config only*, building
     an uninitialised skeleton.
  2. Loads dense weights with ``strict=False`` so missing q/k/v keys are
     ignored.
  3. Replaces each factored Linear with :class:`FactoredLinear` populated
     from the (A, B_*) tensors.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

def _torch_module():
    """Return the FactoredLinear class, importing torch lazily."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class FactoredLinear(nn.Module):
        """Two-stage Linear: ``y = (x @ A^T) @ B^T + bias``.

        ``A`` has shape ``(k, in_features)`` and ``B`` has shape
        ``(out_features, k)``. The intermediate activation is rank-k.

        Equivalent to a standard ``nn.Linear(in, out)`` whose weight matrix
        equals ``B @ A``, but stored and computed in factored form so the
        memory footprint is ``k·(in+out)`` instead of ``in·out``.
        """

        in_features: int
        out_features: int
        rank: int

        def __init__(self, A: torch.Tensor, B: torch.Tensor,
                     bias: torch.Tensor | None = None):
            super().__init__()
            assert A.dim() == 2 and B.dim() == 2
            assert A.shape[0] == B.shape[1], (
                f"rank mismatch: A is {tuple(A.shape)}, B is {tuple(B.shape)}"
            )
            self.in_features = A.shape[1]
            self.out_features = B.shape[0]
            self.rank = A.shape[0]
            self.A = nn.Parameter(A, requires_grad=False)
            self.B = nn.Parameter(B, requires_grad=False)
            if bias is not None:
                self.bias = nn.Parameter(bias, requires_grad=False)
            else:
                self.register_parameter("bias", None)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
            # Two-stage matmul. Promote to fp32 for the intermediate to avoid
            # bf16/fp16 catastrophic cancellation when rank is moderate.
            in_dtype = x.dtype
            if in_dtype in (torch.float16, torch.bfloat16):
                x32 = x.to(torch.float32)
                A32 = self.A.to(torch.float32)
                B32 = self.B.to(torch.float32)
                h = F.linear(x32, A32)            # (..., k) in fp32
                y = F.linear(h, B32,
                             self.bias.to(torch.float32) if self.bias is not None else None)
                return y.to(in_dtype)
            h = F.linear(x, self.A)
            return F.linear(h, self.B, self.bias)

        def extra_repr(self) -> str:
            return (f"in_features={self.in_features}, "
                    f"out_features={self.out_features}, rank={self.rank}, "
                    f"bias={self.bias is not None}")

    return FactoredLinear


# ---------------------------------------------------------------------------
# Factor extraction
# ---------------------------------------------------------------------------

def _svd_factor(W: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, B) such that B @ A ≈ W with rank ≤ k.

    Using truncated SVD: W = U S V^T → A = S V^T (k, n), B = U (m, k).
    """
    m, n = W.shape
    k_eff = min(k, m, n)
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    U = U[:, :k_eff]
    S = S[:k_eff]
    Vt = Vt[:k_eff, :]
    A = (S[:, None] * Vt).astype(W.dtype)   # (k, n)
    B = U.astype(W.dtype)                    # (m, k)
    return A, B


def _adaptive_rank(W: np.ndarray, *, max_k: int,
                    rel_tol: float = 1e-4) -> int:
    """Detect effective rank: smallest k s.t. trailing singular values are <= rel_tol * S[0]."""
    s = np.linalg.svd(W, compute_uv=False)
    if s.size == 0:
        return 0
    thresh = rel_tol * s[0]
    eff = int(np.searchsorted(-s, -thresh))  # number of s_i > thresh
    return max(1, min(eff, max_k, *W.shape))


def _svd_factor_adaptive(W: np.ndarray, max_k: int,
                          rel_tol: float = 1e-4) -> tuple[np.ndarray, np.ndarray, int]:
    """SVD factor with rank auto-clipped to effective rank.

    Returns (A, B, k_used). Only stores k_used singular components — if the
    matrix is exactly low-rank this is exact.
    """
    k_use = _adaptive_rank(W, max_k=max_k, rel_tol=rel_tol)
    A, B = _svd_factor(W, k_use)
    return A, B, k_use


def _svd_factor_aware(W: np.ndarray, col_norms: np.ndarray, k: int
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Activation-aware truncated SVD (Wanda/AWQ-style).

    Given column-wise input activation 2-norms ``s`` (shape ``(n,)`` for
    W of shape ``(m, n)``), factor ``W ≈ B @ A`` such that the truncation
    error is minimised in the *output* sense ``||(W − B A) diag(s)||_F``,
    not the raw weight sense ``||W − B A||_F``. This shifts approximation
    error into low-activation input channels, which Wanda/AWQ showed
    recovers significant PPL at fixed rank vs vanilla SVD.

    Returns ``(A, B)`` with the same shapes as :func:`_svd_factor`.
    """
    m, n = W.shape
    s = np.clip(col_norms.astype(np.float64), 1e-8, None)
    Ws = W.astype(np.float64) * s[None, :]
    U, S, Vt = np.linalg.svd(Ws, full_matrices=False)
    k_eff = min(k, m, n)
    U = U[:, :k_eff]
    S = S[:k_eff]
    Vt = Vt[:k_eff, :]
    # B @ A ≈ Ws  →  B @ (A / s) ≈ W
    A = ((S[:, None] * Vt) / s[None, :]).astype(W.dtype)   # (k, n)
    B = U.astype(W.dtype)                                    # (m, k)
    return A, B


def _adaptive_rank_aware(W: np.ndarray, col_norms: np.ndarray, *,
                          max_k: int, rel_tol: float = 1e-4) -> int:
    s = np.clip(col_norms.astype(np.float64), 1e-8, None)
    Ws = W.astype(np.float64) * s[None, :]
    sv = np.linalg.svd(Ws, compute_uv=False)
    if sv.size == 0:
        return 0
    thresh = rel_tol * sv[0]
    eff = int(np.searchsorted(-sv, -thresh))
    return max(1, min(eff, max_k, *W.shape))


def _shared_basis_factor(Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                          k: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Joint SVD on stacked [Wq; Wk; Wv] → shared right basis A.

    Returns ``(A, {"q": Bq, "k": Bk, "v": Bv})`` with::

        Wq ≈ Bq @ A
        Wk ≈ Bk @ A
        Wv ≈ Bv @ A
        A  shape (k, d_in)
        B* shape (d_out, k)
    """
    stacked = np.concatenate([Wq, Wk, Wv], axis=0)  # (3*d_out, d_in)
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    k_eff = min(k, Vt.shape[0])
    Vt_k = Vt[:k_eff, :]                  # (k, d_in)
    # Project each onto the shared basis (right-multiplication by V_k V_k^T
    # is the rank-k approximation in the row space).
    A = Vt_k.astype(stacked.dtype)
    Bq = (Wq @ A.T).astype(stacked.dtype)
    Bk = (Wk @ A.T).astype(stacked.dtype)
    Bv = (Wv @ A.T).astype(stacked.dtype)
    return A, {"q": Bq, "k": Bk, "v": Bv}


# ---------------------------------------------------------------------------
# State-dict rewrite
# ---------------------------------------------------------------------------

@dataclass
class FactoredEntry:
    layer_idx: int
    rank: int
    in_features: int
    out_features: int
    keys: dict[str, str]   # "q"/"k"/"v" -> original dense weight key
    bias_keys: dict[str, str]  # "q"/"k"/"v" -> bias key (may be empty)


def factor_attn_state_dict(
    state_dict: dict,
    *,
    rank: int,
    layer_keys: dict[int, dict[str, str]],
    rel_tol: float = 1e-2,
) -> tuple[dict, list[FactoredEntry]]:
    """Rewrite a state dict in-place so attention Q/K/V are factored.

    ``layer_keys`` maps layer index -> {"q": "model.layers.0.self_attn.q_proj.weight", ...}
    (this is the output of :func:`hyperretro.hf.compress._group_attn_by_layer`).

    Returns the modified state dict and a list of :class:`FactoredEntry`
    suitable for the on-disk manifest.
    """
    import torch

    entries: list[FactoredEntry] = []
    for li in sorted(layer_keys.keys()):
        names = layer_keys[li]
        if not all(s in names for s in ("q", "k", "v")):
            continue  # skip fused-QKV layers; out of scope for v1
        Wq = state_dict[names["q"]].detach().to("cpu").float().numpy()
        Wk = state_dict[names["k"]].detach().to("cpu").float().numpy()
        Wv = state_dict[names["v"]].detach().to("cpu").float().numpy()
        # Shared-basis works as long as in_features (axis=1) matches across
        # Q/K/V — this is true for both MHA and GQA. Out_features may differ.
        if not (Wq.shape[1] == Wk.shape[1] == Wv.shape[1]):
            # Truly heterogeneous: fall back to independent factoring
            Aq, Bq = _svd_factor(Wq, rank)
            Ak, Bk = _svd_factor(Wk, rank)
            Av, Bv = _svd_factor(Wv, rank)
            shared_A = None
            indep_As = {"q": Aq, "k": Ak, "v": Av}
            indep_Bs = {"q": Bq, "k": Bk, "v": Bv}
        else:
            shared_A, Bs = _shared_basis_factor(Wq, Wk, Wv, rank)
            indep_As = None
            indep_Bs = Bs

        # Adaptive rank: clip to effective rank of the dense matrix so we
        # don't pad zeros (and so exact-low-rank inputs reconstruct exactly).
        if shared_A is not None:
            stacked = np.concatenate([Wq, Wk, Wv], axis=0)
            k_use = _adaptive_rank(stacked, max_k=rank, rel_tol=rel_tol)
            if k_use < shared_A.shape[0]:
                shared_A = shared_A[:k_use, :]
                for slot in ("q", "k", "v"):
                    indep_Bs[slot] = indep_Bs[slot][:, :k_use]

        # Original dtype/device to preserve
        orig = state_dict[names["q"]]
        dtype = orig.dtype
        device = orig.device

        prefix = names["q"].rsplit(".q_proj.weight", 1)[0]
        if shared_A is not None:
            state_dict[f"{prefix}.factored_A"] = (
                torch.from_numpy(shared_A).to(dtype=dtype, device=device)
            )
            for slot in ("q", "k", "v"):
                state_dict[f"{prefix}.factored_B{slot}"] = (
                    torch.from_numpy(indep_Bs[slot]).to(dtype=dtype, device=device)
                )
        else:
            # Independent factors
            for slot in ("q", "k", "v"):
                state_dict[f"{prefix}.factored_A{slot}"] = (
                    torch.from_numpy(indep_As[slot]).to(dtype=dtype, device=device)
                )
                state_dict[f"{prefix}.factored_B{slot}"] = (
                    torch.from_numpy(indep_Bs[slot]).to(dtype=dtype, device=device)
                )

        # Bias keys (may not exist)
        bias_keys: dict[str, str] = {}
        for slot in ("q", "k", "v"):
            bk = names[slot].replace(".weight", ".bias")
            if bk in state_dict:
                bias_keys[slot] = bk

        # Remove the dense weight keys
        for slot in ("q", "k", "v"):
            state_dict.pop(names[slot], None)

        entries.append(FactoredEntry(
            layer_idx=li,
            rank=int(shared_A.shape[0] if shared_A is not None else indep_As["q"].shape[0]),
            in_features=int(Wq.shape[1]),
            out_features=int(Wq.shape[0]),
            keys={s: names[s] for s in ("q", "k", "v")},
            bias_keys=bias_keys,
        ))

    return state_dict, entries


def build_manifest(entries: list[FactoredEntry], *, shared: bool = True,
                    ffn_entries: list[dict] | None = None) -> dict:
    return {
        "version": 1,
        "shared_basis": shared,
        "layers": [
            {
                "layer_idx": e.layer_idx,
                "rank": e.rank,
                "in_features": e.in_features,
                "out_features": e.out_features,
                "q_key": e.keys["q"],
                "k_key": e.keys["k"],
                "v_key": e.keys["v"],
                "biases": e.bias_keys,
            }
            for e in entries
        ],
        "ffn": ffn_entries or [],
    }


# ---------------------------------------------------------------------------
# FFN factoring (independent SVD per Linear)
# ---------------------------------------------------------------------------

_FFN_SUFFIXES = (
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def factor_ffn_state_dict(
    state_dict: dict,
    *,
    max_rank: int,
    rel_tol: float = 1e-4,
    activation_col_norms: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """Adaptively factor any FFN Linear weights that are low-rank.

    For each gate/up/down weight, computes the effective rank ``k_eff`` via
    SVD; if ``k_eff < min(W.shape)`` *and* factoring saves bytes
    (``k_eff·(m+n) < m·n``) we replace the dense key with ``factored_A``,
    ``factored_B``. Returns a list of manifest entries for each factored
    Linear.

    If ``activation_col_norms`` is provided (mapping ``weight_key →
    col-norm vector of shape (in_features,)``), activation-aware SVD is
    used for that key (Wanda/AWQ-style). Keys absent from the mapping fall
    back to vanilla SVD.
    """
    import torch

    entries: list[dict] = []
    factored_keys: list[str] = []
    # Snapshot keys so we don't mutate during iteration
    for key in list(state_dict.keys()):
        if not any(key.endswith(suf) for suf in _FFN_SUFFIXES):
            continue
        W_t = state_dict[key]
        W = W_t.detach().to("cpu").float().numpy()
        m, n = W.shape
        col_norms = (activation_col_norms or {}).get(key)
        if col_norms is not None and col_norms.shape == (n,):
            k_use = _adaptive_rank_aware(W, col_norms,
                                          max_k=max_rank, rel_tol=rel_tol)
            A, B = _svd_factor_aware(W, col_norms, k_use)
        else:
            A, B, k_use = _svd_factor_adaptive(W, max_k=max_rank, rel_tol=rel_tol)
        # Only commit if it actually saves bytes
        if k_use * (m + n) >= m * n:
            continue
        prefix = key[:-len(".weight")]
        dtype = W_t.dtype
        device = W_t.device
        state_dict[f"{prefix}.factored_A"] = (
            torch.from_numpy(A).to(dtype=dtype, device=device)
        )
        state_dict[f"{prefix}.factored_B"] = (
            torch.from_numpy(B).to(dtype=dtype, device=device)
        )
        # Optional bias
        bias_key = prefix + ".bias"
        has_bias = bias_key in state_dict
        # Remove dense weight
        state_dict.pop(key)
        entries.append({
            "weight_key": key,
            "rank": int(k_use),
            "in_features": int(n),
            "out_features": int(m),
            "bias_key": bias_key if has_bias else None,
        })
        factored_keys.append(key)
    return entries


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_factored_hf_model(model_dir: str | Path, *, dtype: str = "bfloat16"):
    """Load a HyperRetro-factored model directory into an HF model object.

    Replaces each factored Linear with a :class:`FactoredLinear` whose A/B
    parameters come from the safetensors shards.
    """
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM

    model_dir = Path(model_dir)
    manifest_path = model_dir / "hyperretro_factored.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} missing — this dir is not a factored checkpoint")
    manifest = json.loads(manifest_path.read_text())

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    cfg = AutoConfig.from_pretrained(model_dir)
    # Build skeleton without loading any weights yet
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch_dtype)

    # Load all available tensors (may span multiple shards)
    sd: dict[str, torch.Tensor] = {}
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no safetensors shards in {model_dir}")
    for shard in shards:
        sd.update(load_file(str(shard)))

    # Re-tie weights: if config has tie_word_embeddings and lm_head.weight is
    # absent, the checkpoint deliberately omitted it. Restore by aliasing.
    if getattr(cfg, "tie_word_embeddings", False) and "lm_head.weight" not in sd:
        embed_key = None
        for k in sd:
            if k.endswith(".embed_tokens.weight") or k == "transformer.wte.weight":
                embed_key = k
                break
        if embed_key is not None:
            sd["lm_head.weight"] = sd[embed_key]

    # Load dense weights non-strictly (q/k/v originals are absent by design)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # Patch factored modules
    FactoredLinear = _torch_module()
    shared = bool(manifest.get("shared_basis", True))
    patched = 0
    for entry in manifest["layers"]:
        for slot, weight_key in (("q", entry["q_key"]),
                                  ("k", entry["k_key"]),
                                  ("v", entry["v_key"])):
            prefix = weight_key.rsplit(".weight", 1)[0]
            parent_path = prefix.rsplit(".", 1)[0]
            leaf = prefix.rsplit(".", 1)[1]
            # Walk the model
            parent = model
            for tok in parent_path.split("."):
                if tok.isdigit():
                    parent = parent[int(tok)]
                else:
                    parent = getattr(parent, tok)

            attn_prefix = prefix.rsplit(".", 1)[0]
            if shared:
                A = sd[f"{attn_prefix}.factored_A"]
            else:
                A = sd[f"{attn_prefix}.factored_A{slot}"]
            B = sd[f"{attn_prefix}.factored_B{slot}"]

            bias_key = entry["biases"].get(slot)
            bias = sd[bias_key] if bias_key and bias_key in sd else None

            new_mod = FactoredLinear(
                A.to(torch_dtype),
                B.to(torch_dtype),
                bias.to(torch_dtype) if bias is not None else None,
            )
            setattr(parent, leaf, new_mod)
            patched += 1

    # Patch factored FFN linears (if any)
    for entry in manifest.get("ffn", []):
        weight_key = entry["weight_key"]
        prefix = weight_key[:-len(".weight")]
        parent_path = prefix.rsplit(".", 1)[0]
        leaf = prefix.rsplit(".", 1)[1]
        parent = model
        for tok in parent_path.split("."):
            if tok.isdigit():
                parent = parent[int(tok)]
            else:
                parent = getattr(parent, tok)
        A = sd[f"{prefix}.factored_A"]
        B = sd[f"{prefix}.factored_B"]
        bias = None
        if entry.get("bias_key") and entry["bias_key"] in sd:
            bias = sd[entry["bias_key"]]
        new_mod = FactoredLinear(
            A.to(torch_dtype),
            B.to(torch_dtype),
            bias.to(torch_dtype) if bias is not None else None,
        )
        setattr(parent, leaf, new_mod)
        patched += 1

    return model, {
        "patched_linears": patched,
        "missing_dense": len(missing),
        "unexpected": len(unexpected),
        "manifest_layers": len(manifest["layers"]),
        "manifest_ffn": len(manifest.get("ffn", [])),
    }


# ---------------------------------------------------------------------------
# Unified factored checkpoint writer
# ---------------------------------------------------------------------------

def save_factored_checkpoint(
    state_dict: dict,
    attn_entries: list,
    ffn_entries: list,
    *,
    out_dir: "str | Path",
    hf_config,
    dtype: str = "bfloat16",
    tokenizer=None,
) -> dict:
    """Write a HyperRetro-factored checkpoint to disk.

    ``state_dict`` must already have factored_A / factored_B* keys in place
    of dense q/k/v (and optionally FFN) weights — call
    :func:`factor_attn_state_dict` (or :func:`compress_state_dict_factored`)
    and :func:`factor_ffn_state_dict` first.

    Honors ``hf_config.tie_word_embeddings`` by dropping
    ``lm_head.weight`` on disk (the loader re-aliases on load).
    """
    import torch
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    target_dtype = dtype_map.get(dtype, torch.float32)

    tied = bool(getattr(hf_config, "tie_word_embeddings", False))

    out_sd: dict = {}
    for k_, v in state_dict.items():
        if tied and k_ == "lm_head.weight":
            continue
        if hasattr(v, "to"):
            out_sd[k_] = v.to(target_dtype).contiguous().cpu()
        else:
            out_sd[k_] = torch.from_numpy(np.ascontiguousarray(v)).to(target_dtype)
    save_file(out_sd, str(out / "model.safetensors"))

    # Normalise attn_entries: accept either FactoredEntry objects or dicts
    entry_objs: list[FactoredEntry] = []
    for e in attn_entries:
        if isinstance(e, FactoredEntry):
            entry_objs.append(e)
        else:
            entry_objs.append(FactoredEntry(
                layer_idx=e["layer_idx"], rank=e["rank"],
                in_features=e["in_features"], out_features=e["out_features"],
                keys={"q": e["q_key"], "k": e["k_key"], "v": e["v_key"]},
                bias_keys=e.get("biases", {}),
            ))

    manifest = build_manifest(entry_objs, shared=True,
                               ffn_entries=ffn_entries)
    (out / "hyperretro_factored.json").write_text(
        json.dumps(manifest, indent=2)
    )

    hf_config.save_pretrained(out)
    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(out)
        except Exception:
            pass

    return {
        "out_dir": str(out),
        "attn_factored": len(entry_objs),
        "ffn_factored": len(ffn_entries),
        "tied_lm_head_dropped": tied and "lm_head.weight" in state_dict,
        "dtype": dtype,
    }
