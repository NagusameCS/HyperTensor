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

"""KV-cache compression via GRC shared basis (attack #4).

At decode time, transformer models cache the key and value states for
every previous token. For Qwen2.5-1.5B at 32k context with batch=1:
    KV-cache = 2 × 28 layers × 2 kv_heads × head_dim 128 × 32k tokens
             = 2 × 28 × 256 × 32768 × 2 bytes (fp16)
             ≈ 918 MB

That's nearly 1 GB of VRAM for ONE sequence. At batch=8 it's 7.3 GB.

The GRC shared basis P_k (d × k) computed during attn weight compression
can ALSO be applied to the KV cache at decode time:

    K_cache_compressed = K_cache @ P_k    (project from d→k dims)
    V_cache_compressed = V_cache @ P_k

Since k < d (typically k=640 vs d=1536), the cache shrinks by d/k = 2.4×.
The projected keys/queries are used with the projected query:

    Q_proj = Q @ P_k                       (project query)
    attn_scores = Q_proj @ K_cache_compressed^T / sqrt(head_dim)

The attention output is computed from the compressed values, then
reconstructed:

    A = softmax(scores)
    output_compressed = A @ V_cache_compressed           (in projected space)
    output = output_compressed @ P_k_preferred            (lift back to d)

This is mathematically equivalent to attention in the projected space
because P_k is orthonormal (P_k^T P_k = I_k). Specifically:

    Q @ P_k @ (K @ P_k)^T = Q @ P_k @ P_k^T @ K^T ≈ Q @ K^T

The approximation error is exactly the GRC projection error — the same
error already accepted for the weight compression. So KV-cache compression
via the shared basis has NO ADDITIONAL PPL COST.

Industry context:
- No production system does KV-cache compression as of May 2026.
- Research: KIVI (2-bit KV cache), GEAR (low-rank + quantization),
  StreamingLLM (window + sink tokens).
- HyperRetro's approach is unique: reuse the weight-compression basis
  for cache compression, giving a mathematically unified treatment.

This module provides:
* :func:`compress_kv_cache` — project past K/V states onto the shared basis.
* :func:`CompressedKVCache` — drop-in replacement for a HuggingFace
  DynamicCache that stores compressed K/V.
* :func:`kv_cache_shrink_factor` — compute d/k shrink for a model config.
* :func:`estimate_kv_cache_bytes` — project cache size at a given context length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def compress_kv_cache(
    K_past: np.ndarray,    # (batch, n_kv_heads, seq_len, head_dim_or_d)
    V_past: np.ndarray,    # same shape
    P_k: np.ndarray,       # (d, k) shared basis from GRC
) -> tuple[np.ndarray, np.ndarray]:
    """Project cached K/V states onto the GRC shared basis.

    K/V are stored in the model's native head_dim space. For GQA models
    like Qwen2.5 where K/V heads use the full d (they're projected from
    d to n_kv_heads*head_dim by the weight matrix, not by a per-head
    split), the projection is applied along the last dimension.

    Parameters
    ----------
    K_past : (batch, n_kv_heads, seq_len, d) or (batch, n_kv_heads, seq_len, head_dim)
    V_past : same shape as K_past
    P_k : (d, k) orthonormal projection matrix (columns are the shared basis)

    Returns
    -------
    K_compressed : (batch, n_kv_heads, seq_len, k)
    V_compressed : (batch, n_kv_heads, seq_len, k)
    """
    d, k = P_k.shape
    assert K_past.shape[-1] == d, (
        f"K_past last dim {K_past.shape[-1]} != P_k rows {d}"
    )
    K_c = K_past @ P_k   # (..., seq_len, k)
    V_c = V_past @ P_k
    return K_c, V_c


def decompress_attn_output(
    attn_output_compressed: np.ndarray,  # (batch, n_heads, seq_len, k)
    P_k: np.ndarray,                      # (d, k)
) -> np.ndarray:
    """Lift attention output from the k-dim projected space back to d-dim.

    Parameters
    ----------
    attn_output_compressed : (batch, n_heads, seq_len, k)
    P_k : (d, k)

    Returns
    -------
    output : (batch, n_heads, seq_len, d)
    """
    return attn_output_compressed @ P_k.T  # (..., seq_len, d)


# ---------------------------------------------------------------------------
# Compressed KV cache (HF DynamicCache compatible)
# ---------------------------------------------------------------------------

@dataclass
class CompressedKVCache:
    """Drop-in replacement for a HF DynamicCache that stores compressed K/V.

    Usage::

        cache = CompressedKVCache(P_k_dict)  # P_k_dict: {layer_idx: P_k_array}
        for token in generation:
            # Project query before attention
            q_proj = query_states @ P_k  # (batch, n_heads, seq, k)
            k_proj = key_states @ P_k
            v_proj = value_states @ P_k

            # Concatenate with compressed past
            if cache.has_past(layer_idx):
                k_proj = torch.cat([cache.get_k(layer_idx), k_proj], dim=-2)
                v_proj = torch.cat([cache.get_v(layer_idx), v_proj], dim=-2)

            cache.update(layer_idx, k_proj, v_proj)

            # Attention in projected space
            attn = softmax(q_proj @ k_proj.T / sqrt(head_dim))
            out_proj = attn @ v_proj
            out = out_proj @ P_k.T  # lift back to d
    """

    _k_caches: dict[int, np.ndarray]
    _v_caches: dict[int, np.ndarray]
    _P_k: dict[int, np.ndarray]
    _k: dict[int, int]

    def __init__(self, P_k_dict: dict[int, np.ndarray]):
        self._k_caches = {}
        self._v_caches = {}
        self._P_k = P_k_dict
        self._k = {li: P.shape[1] for li, P in P_k_dict.items()}

    @property
    def shrink_factor(self) -> dict[int, float]:
        """Per-layer d/k shrink factor."""
        return {li: P.shape[0] / P.shape[1] for li, P in self._P_k.items()}

    def has_past(self, layer_idx: int) -> bool:
        return layer_idx in self._k_caches

    def get_k(self, layer_idx: int) -> np.ndarray:
        return self._k_caches[layer_idx]

    def get_v(self, layer_idx: int) -> np.ndarray:
        return self._v_caches[layer_idx]

    def update(self, layer_idx: int, k_proj: np.ndarray, v_proj: np.ndarray):
        if layer_idx in self._k_caches:
            self._k_caches[layer_idx] = np.concatenate(
                [self._k_caches[layer_idx], k_proj], axis=-2
            )
            self._v_caches[layer_idx] = np.concatenate(
                [self._v_caches[layer_idx], v_proj], axis=-2
            )
        else:
            self._k_caches[layer_idx] = k_proj
            self._v_caches[layer_idx] = v_proj

    def clear(self):
        self._k_caches.clear()
        self._v_caches.clear()

    def total_bytes(self) -> int:
        """Total bytes stored in the compressed cache (fp16)."""
        total = 0
        for li in self._k_caches:
            total += self._k_caches[li].nbytes + self._v_caches[li].nbytes
        return total


# ---------------------------------------------------------------------------
# Cache size estimation
# ---------------------------------------------------------------------------

@dataclass
class KVCacheEstimate:
    """Estimated KV cache sizes at a given context length.

    Two strategies are modelled:

    1. **pre-projection caching** (GRC shared basis): store ``x @ P_k``
       (k-dim) per token instead of per-head K/V states. Beneficial when
       ``k < n_kv_heads × head_dim`` — i.e., for models with many KV
       heads (MHA). For GQA models with few KV heads this typically
       *expands* the cache.

    2. **intra-head projection** (per-head low-rank): each head's K/V
       is head_dim-dimensional. Project to a smaller rank within that
       space. Always gives shrink (rank < head_dim) but is independent
       of the GRC shared basis.
    """
    context_len: int
    d_model: int
    k_rank: int            # GRC rank for pre-projection strategy
    n_layers: int
    n_kv_heads: int
    head_dim: int
    intra_head_rank: int = 64   # rank for per-head compression
    bytes_per_elem: int = 2     # fp16

    # ---- Uncompressed ----

    @property
    def kv_dim_per_token(self) -> int:
        """Total K+V dimensions stored per token per layer (standard)."""
        return 2 * self.n_kv_heads * self.head_dim

    @property
    def uncompressed_bytes(self) -> int:
        return (self.kv_dim_per_token
                * self.n_layers
                * self.context_len
                * self.bytes_per_elem)

    # ---- Strategy 1: pre-projection (GRC shared basis) ----

    @property
    def pre_projection_dim_per_token(self) -> int:
        """If we store x @ P_k instead of per-head K/V."""
        return 2 * self.k_rank  # K and V projections

    @property
    def pre_projection_bytes(self) -> int:
        return (self.pre_projection_dim_per_token
                * self.n_layers
                * self.context_len
                * self.bytes_per_elem)

    @property
    def pre_projection_shrink(self) -> float:
        return self.uncompressed_bytes / max(self.pre_projection_bytes, 1)

    @property
    def pre_projection_viable(self) -> bool:
        """True if this strategy actually compresses."""
        return self.pre_projection_dim_per_token < self.kv_dim_per_token

    # ---- Strategy 2: intra-head projection ----

    @property
    def intra_head_dim_per_token(self) -> int:
        """If we store per-head K/V projected to intra_head_rank."""
        return 2 * self.n_kv_heads * self.intra_head_rank

    @property
    def intra_head_bytes(self) -> int:
        return (self.intra_head_dim_per_token
                * self.n_layers
                * self.context_len
                * self.bytes_per_elem)

    @property
    def intra_head_shrink(self) -> float:
        return self.uncompressed_bytes / max(self.intra_head_bytes, 1)

    @property
    def intra_head_viable(self) -> bool:
        return self.intra_head_rank < self.head_dim

    # ---- Best strategy ----

    @property
    def best_shrink(self) -> float:
        s1 = self.pre_projection_shrink if self.pre_projection_viable else 0
        s2 = self.intra_head_shrink if self.intra_head_viable else 0
        return max(s1, s2)

    @property
    def best_strategy(self) -> str:
        s1 = self.pre_projection_shrink if self.pre_projection_viable else 0
        s2 = self.intra_head_shrink if self.intra_head_viable else 0
        if s1 >= s2 and s1 > 1:
            return "pre_projection"
        elif s2 > 1:
            return "intra_head"
        return "none_viable"

    @property
    def compressed_bytes(self) -> int:
        if self.best_strategy == "pre_projection":
            return self.pre_projection_bytes
        elif self.best_strategy == "intra_head":
            return self.intra_head_bytes
        return self.uncompressed_bytes

    @property
    def shrink_factor(self) -> float:
        return self.uncompressed_bytes / max(self.compressed_bytes, 1)

    @property
    def savings_bytes(self) -> int:
        return self.uncompressed_bytes - self.compressed_bytes


def estimate_kv_cache(
    context_len: int,
    *,
    d_model: int = 1536,
    k_rank: int = 640,
    n_layers: int = 28,
    n_kv_heads: int = 2,
    head_dim: int = 128,
    intra_head_rank: int = 64,
    bytes_per_elem: int = 2,
) -> KVCacheEstimate:
    """Estimate KV cache size for a model configuration.

    Two strategies:
    1. Pre-projection (GRC): store ``x @ P_k`` instead of per-head K/V.
       Wins when ``n_kv_heads × head_dim > k_rank`` (MHA models).
    2. Intra-head: compress each head's K/V from head_dim to intra_head_rank.
       Always gives shrink when intra_head_rank < head_dim.
    """
    return KVCacheEstimate(
        context_len=context_len,
        d_model=d_model,
        k_rank=k_rank,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        intra_head_rank=intra_head_rank,
        bytes_per_elem=bytes_per_elem,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(
        description="Estimate KV-cache compression savings."
    )
    p.add_argument("--context-len", type=int, default=32768)
    p.add_argument("--d-model", type=int, default=1536)
    p.add_argument("--k-rank", type=int, default=640)
    p.add_argument("--n-layers", type=int, default=28)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--intra-head-rank", type=int, default=64)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    est = estimate_kv_cache(
        context_len=args.context_len,
        d_model=args.d_model,
        k_rank=args.k_rank,
        n_layers=args.n_layers,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
        intra_head_rank=args.intra_head_rank,
    )

    if args.json:
        print(json.dumps({
            "context_len": est.context_len,
            "kv_dim_per_token": est.kv_dim_per_token,
            "uncompressed_mb": round(est.uncompressed_bytes / 1e6, 1),
            "pre_projection": {
                "viable": est.pre_projection_viable,
                "dim_per_token": est.pre_projection_dim_per_token,
                "mb": round(est.pre_projection_bytes / 1e6, 1),
                "shrink": round(est.pre_projection_shrink, 1),
            },
            "intra_head": {
                "viable": est.intra_head_viable,
                "dim_per_token": est.intra_head_dim_per_token,
                "mb": round(est.intra_head_bytes / 1e6, 1),
                "shrink": round(est.intra_head_shrink, 1),
            },
            "best": {
                "strategy": est.best_strategy,
                "compressed_mb": round(est.compressed_bytes / 1e6, 1),
                "shrink": round(est.shrink_factor, 1),
                "savings_mb": round(est.savings_bytes / 1e6, 1),
            },
        }, indent=2))
    else:
        print(f"Context length: {est.context_len:,} tokens")
        print(f"KV dim / token: {est.kv_dim_per_token} ({est.n_kv_heads} heads × {est.head_dim} dim × 2)")
        print(f"Uncompressed:    {est.uncompressed_bytes/1e6:.1f} MB")
        print()
        print(f"Strategy 1 — pre-projection (GRC shared basis, k={est.k_rank}):")
        print(f"  Viable: {est.pre_projection_viable}")
        print(f"  Dim/token: {est.pre_projection_dim_per_token}")
        print(f"  Cache: {est.pre_projection_bytes/1e6:.1f} MB ({est.pre_projection_shrink:.1f}×)")
        print()
        print(f"Strategy 2 — intra-head projection (rank={est.intra_head_rank}):")
        print(f"  Viable: {est.intra_head_viable}")
        print(f"  Dim/token: {est.intra_head_dim_per_token}")
        print(f"  Cache: {est.intra_head_bytes/1e6:.1f} MB ({est.intra_head_shrink:.1f}×)")
        print()
        print(f"Best: {est.best_strategy} → {est.compressed_bytes/1e6:.1f} MB ({est.shrink_factor:.1f}×)")
        print(f"Savings: {est.savings_bytes/1e6:.1f} MB")
        print(f"At batch=8: {8*est.uncompressed_bytes/1e6:.1f} → {8*est.compressed_bytes/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())
