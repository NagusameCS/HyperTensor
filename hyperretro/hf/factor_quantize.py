"""Factor × int4 quantize composition (attack #7).

HyperRetro's low-rank factoring and 4-bit quantization are *orthogonal*
levers. Factoring alone caps at ~1.5× shrink on Qwen2.5-1.5B (FFN is
75% of params but rank≥768 needed for viability). Applying int4 to the
factored (A, B) matrices multiplies the shrink: at r=768, factor+int4
gives 6.8× shrink on a gate_proj vs 4× for dense-int4 alone.

This module provides:
* :func:`quantize_matrix_int4` — symmetric per-channel int4 quant/dequant.
* :func:`factor_then_quantize` — factor (vanilla or aware) then int4-quantize
  A and B separately; returns dequantised reconstruction and size stats.
* :func:`estimate_full_model_shrink` — project total model size given a
  per-matrix strategy (factor+int4, direct-int4, fp16).

Industry context (May 2026):
- bnb nf4: 4-bit block-wise (64-weight blocks), ~4× shrink, +3-7% PPL.
- AWQ/GPTQ: similar.
- This module explores whether factor+int4 can beat 4× shrink at
  comparable PPL by pre-conditioning weights into low-rank form before
  quantizing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Int4 symmetric per-channel quantize / dequantize
# ---------------------------------------------------------------------------

def quantize_matrix_int4(
    W: np.ndarray,
    *,
    axis: int = 0,
    n_bits: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-channel (row-wise) int4 quantization.

    Parameters
    ----------
    W : shape (m, n)
    axis : 0 for per-row, 1 for per-column.
    n_bits : 4 (int4) or 8 (int8).

    Returns
    -------
    W_q : np.int8 array of same shape — values in [-2^{n_bits-1}+1, 2^{n_bits-1}-1].
    scales : fp32 array of shape (m,) if axis=0 else (n,).
    """
    assert n_bits in (4, 8), f"n_bits must be 4 or 8, got {n_bits}"
    max_val = (1 << (n_bits - 1)) - 1  # 7 for int4, 127 for int8
    if axis == 0:
        scales = np.max(np.abs(W), axis=1).astype(np.float32) / max_val
        scales = np.maximum(scales, 1e-12)
        W_q = np.clip(np.round(W / scales[:, None]), -max_val, max_val).astype(np.int8)
    elif axis == 1:
        scales = np.max(np.abs(W), axis=0).astype(np.float32) / max_val
        scales = np.maximum(scales, 1e-12)
        W_q = np.clip(np.round(W / scales[None, :]), -max_val, max_val).astype(np.int8)
    else:
        raise ValueError(f"axis must be 0 or 1, got {axis}")
    return W_q, scales


def dequantize_matrix_int4(
    W_q: np.ndarray,
    scales: np.ndarray,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Reverse of :func:`quantize_matrix_int4`."""
    if axis == 0:
        return W_q.astype(np.float32) * scales[:, None]
    else:
        return W_q.astype(np.float32) * scales[None, :]


# ---------------------------------------------------------------------------
# Block-wise int4 quantization (Q4_0 style, ~llama.cpp)
# ---------------------------------------------------------------------------

def quantize_blockwise_int4(
    W: np.ndarray,
    *,
    block_size: int = 128,
    n_bits: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Block-wise symmetric int4 quantization along axis=1 (per-row blocks).

    Each row is divided into blocks of `block_size` contiguous elements.
    Each block gets its own fp16 scale. This is equivalent to llama.cpp's
    Q4_0 format and dramatically reduces quantization error vs per-row
    scaling when values within a row have varying magnitudes.

    Parameters
    ----------
    W : shape (m, n)
    block_size : elements per quantization block (default 128).
    n_bits : 4 (int4) or 8 (int8).

    Returns
    -------
    W_q : np.int8 array, shape (m, n) — values in [-2^{n_bits-1}+1, 2^{n_bits-1}-1].
    scales : fp32 array, shape (m, n_blocks) where n_blocks = ceil(n/block_size).
    """
    assert n_bits in (4, 8), f"n_bits must be 4 or 8, got {n_bits}"
    max_val = (1 << (n_bits - 1)) - 1
    m, n = W.shape
    n_blocks = (n + block_size - 1) // block_size

    W_q = np.zeros((m, n), dtype=np.int8)
    scales = np.zeros((m, n_blocks), dtype=np.float32)

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = W[:, start:end]
        # Symmetric quantization: scale = max(abs(block)) / max_val
        block_max = np.max(np.abs(block), axis=1)
        block_max = np.maximum(block_max, 1e-12)
        scale = block_max / max_val
        q = np.clip(np.round(block / scale[:, None]), -max_val, max_val)
        W_q[:, start:end] = q
        scales[:, b] = scale

    return W_q, scales


def dequantize_blockwise_int4(
    W_q: np.ndarray,
    scales: np.ndarray,
    *,
    block_size: int = 128,
) -> np.ndarray:
    """Reverse of :func:`quantize_blockwise_int4`."""
    m, n = W_q.shape
    result = np.zeros((m, n), dtype=np.float32)
    n_blocks = scales.shape[1]

    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        result[:, start:end] = W_q[:, start:end].astype(np.float32) * scales[:, b:b+1]

    return result


def blockwise_storage_bytes(m: int, n: int, block_size: int = 128, n_bits: int = 4) -> int:
    """On-disk bytes for block-wise int4 storage."""
    n_blocks = (n + block_size - 1) // block_size
    packed_bytes = int(np.ceil(m * n * n_bits / 8.0))
    scale_bytes = m * n_blocks * 2  # fp16 per-block scales
    return packed_bytes + scale_bytes


# ---------------------------------------------------------------------------
# AWQ-style activation-aware quantization
# ---------------------------------------------------------------------------

def quantize_matrix_int4_aware(
    W: np.ndarray,
    col_norms: np.ndarray,
    *,
    axis: int = 0,
    n_bits: int = 4,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Activation-aware int4 quantization (AWQ-style).

    Given column-wise activation norms ``s`` (shape ``(n,)`` for W of shape
    ``(m, n)``), scales W *before* quantization so that error is shifted
    into low-activation columns. This is the same principle as
    :func:`_svd_factor_aware` but applied to quantization instead of SVD.

    Algorithm (AWQ, Lin et al. 2023):
      1. Compute scaling factors: ``s_j = (activation_norm_j)^alpha``
      2. Scale weights: ``W' = W * diag(s)``
      3. Quantize W' symmetrically
      4. Dequantize: ``W_hat = dequant(W'_q) / diag(s)``

    This protects high-activation channels from quantization noise.

    Parameters
    ----------
    W : shape (m, n)
    col_norms : shape (n,) — activation column 2-norms.
    axis : 0 for per-row quantization.
    n_bits : 4 or 8.
    alpha : AWQ scaling exponent (0.5 = sqrt, as in the paper).

    Returns
    -------
    W_q : int8 array, shape (m, n) — quantized *scaled* weights.
    scales : fp32 array, shape per axis — per-row dequant scales.
    awq_scales : fp32 array, shape (n,) — the AWQ column scaling factors
        (must be applied on dequant: W_deq = dequant(W_q, scales) / awq_scales).
    """
    assert n_bits in (4, 8), f"n_bits must be 4 or 8, got {n_bits}"
    max_val = (1 << (n_bits - 1)) - 1
    m, n = W.shape

    # Compute AWQ scaling factors
    s = np.clip(col_norms.astype(np.float64), 1e-8, None)
    awq_scales = (s ** alpha).astype(np.float32)
    awq_scales = np.maximum(awq_scales, 1e-12)

    # Scale weights
    Ws = W.astype(np.float64) * awq_scales[None, :].astype(np.float64)
    Ws = Ws.astype(np.float32)

    # Quantize scaled weights
    if axis == 0:
        scales = np.max(np.abs(Ws), axis=1).astype(np.float32) / max_val
        scales = np.maximum(scales, 1e-12)
        W_q = np.clip(np.round(Ws / scales[:, None]), -max_val, max_val).astype(np.int8)
    else:
        scales = np.max(np.abs(Ws), axis=0).astype(np.float32) / max_val
        scales = np.maximum(scales, 1e-12)
        W_q = np.clip(np.round(Ws / scales[None, :]), -max_val, max_val).astype(np.int8)

    return W_q, scales, awq_scales


def dequantize_matrix_int4_aware(
    W_q: np.ndarray,
    scales: np.ndarray,
    awq_scales: np.ndarray,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Reverse of :func:`quantize_matrix_int4_aware`.

    Dequantizes then divides by AWQ scaling factors.
    """
    if axis == 0:
        Ws_deq = W_q.astype(np.float32) * scales[:, None]
    else:
        Ws_deq = W_q.astype(np.float32) * scales[None, :]
    return Ws_deq / awq_scales[None, :].astype(np.float32)


# ---------------------------------------------------------------------------
# Best-of-breed: block-wise + AWQ-aware quantization
# ---------------------------------------------------------------------------

def quantize_matrix_int4_best(
    W: np.ndarray,
    *,
    col_norms: np.ndarray | None = None,
    block_size: int = 128,
    n_bits: int = 4,
    awq_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Best available int4 quantization: block-wise + optional AWQ-aware.

    Parameters
    ----------
    W : shape (m, n)
    col_norms : optional (n,) activation column norms for AWQ.
    block_size : block size for block-wise quantization (128 = Q4_0 style).
    n_bits : 4 or 8.
    awq_alpha : AWQ exponent (0.5 = sqrt).

    Returns
    -------
    W_q : int8, shape (m, n).
    scales : fp32, shape (m, n_blocks).
    awq_scales : fp32, shape (n,) or None if col_norms is None.
    """
    if col_norms is not None:
        # AWQ scale
        s = np.clip(col_norms.astype(np.float64), 1e-8, None)
        awq = (s ** awq_alpha).astype(np.float32)
        awq = np.maximum(awq, 1e-12)
        Ws = W.astype(np.float64) * awq[None, :].astype(np.float64)
        Ws = Ws.astype(np.float32)
        W_q, scales = quantize_blockwise_int4(Ws, block_size=block_size, n_bits=n_bits)
        return W_q, scales, awq
    else:
        W_q, scales = quantize_blockwise_int4(W, block_size=block_size, n_bits=n_bits)
        return W_q, scales, None


def dequantize_matrix_int4_best(
    W_q: np.ndarray,
    scales: np.ndarray,
    awq_scales: np.ndarray | None = None,
    *,
    block_size: int = 128,
) -> np.ndarray:
    """Reverse of :func:`quantize_matrix_int4_best`."""
    result = dequantize_blockwise_int4(W_q, scales, block_size=block_size)
    if awq_scales is not None:
        result = result / awq_scales[None, :].astype(np.float32)
    return result


def int4_storage_bytes(m: int, n: int, n_bits: int = 4) -> int:
    """Theoretical on-disk bytes for an (m,n) matrix stored as packed int4/8
    plus fp16 per-row scales.

    Packed: ceil(n_bits/8) * m * n for the quantized values
    (realistically 0.5 bytes per element for int4 packed as uint8 pairs).
    Scales: m * 2 bytes (fp16).
    """
    bytes_per_elem = n_bits / 8.0
    return int(np.ceil(m * n * bytes_per_elem)) + m * 2


# ---------------------------------------------------------------------------
# Factor-then-quantize
# ---------------------------------------------------------------------------

@dataclass
class FactorQuantizeResult:
    """Result of factoring then quantizing a single weight matrix."""
    shape: tuple[int, int]
    rank: int
    # Frobenius errors (relative to ||W||_F)
    frob_relerr_factor: float          # fp16 factor vs W
    frob_relerr_int4: float            # int4-factor vs W
    frob_relerr_dense_int4: float      # direct int4 of W vs W
    # Storage sizes (bytes)
    dense_fp16_bytes: int
    dense_int4_bytes: int
    factor_fp16_bytes: int
    factor_int4_bytes: int


def factor_then_quantize(
    W: np.ndarray,
    rank: int,
    *,
    col_norms: np.ndarray | None = None,
    n_bits: int = 4,
) -> FactorQuantizeResult:
    """Factor a weight matrix, then int4-quantize A and B independently.

    Parameters
    ----------
    W : (m, n) fp16/fp32 weight matrix.
    rank : target rank k.
    col_norms : optional (n,) activation column norms for aware SVD.
    n_bits : 4 for int4, 8 for int8.

    Returns
    -------
    FactorQuantizeResult with relative errors and byte counts.
    """
    from hyperretro.hf.factored import _svd_factor, _svd_factor_aware

    m, n = W.shape
    frob_W = float(np.linalg.norm(W, "fro"))
    if frob_W == 0:
        frob_W = 1.0

    # --- 1. Factor ---
    if col_norms is not None:
        A, B = _svd_factor_aware(W, col_norms, rank)
    else:
        A, B = _svd_factor(W, rank)
    # A: (k, n), B: (m, k)
    k = A.shape[0]

    W_factor = (B.astype(np.float64) @ A.astype(np.float64)).astype(W.dtype)
    frob_relerr_factor = float(np.linalg.norm(W - W_factor, "fro") / frob_W)

    # --- 2. Int4-quantize A and B ---
    A_q, A_scales = quantize_matrix_int4(A, axis=0, n_bits=n_bits)
    B_q, B_scales = quantize_matrix_int4(B, axis=0, n_bits=n_bits)
    A_dq = dequantize_matrix_int4(A_q, A_scales, axis=0)
    B_dq = dequantize_matrix_int4(B_q, B_scales, axis=0)
    W_int4_factor = (B_dq.astype(np.float64) @ A_dq.astype(np.float64)).astype(W.dtype)
    frob_relerr_int4 = float(np.linalg.norm(W - W_int4_factor, "fro") / frob_W)

    # --- 3. Direct int4 of dense W ---
    W_q, W_scales = quantize_matrix_int4(W, axis=0, n_bits=n_bits)
    W_dq = dequantize_matrix_int4(W_q, W_scales, axis=0)
    frob_relerr_dense_int4 = float(np.linalg.norm(W - W_dq, "fro") / frob_W)

    # --- 4. Byte counts ---
    dense_fp16_bytes = m * n * 2
    dense_int4_bytes = int4_storage_bytes(m, n, n_bits)
    factor_fp16_bytes = k * (m + n) * 2
    factor_int4_bytes = int4_storage_bytes(k, n, n_bits) + int4_storage_bytes(m, k, n_bits)

    return FactorQuantizeResult(
        shape=(m, n),
        rank=k,
        frob_relerr_factor=frob_relerr_factor,
        frob_relerr_int4=frob_relerr_int4,
        frob_relerr_dense_int4=frob_relerr_dense_int4,
        dense_fp16_bytes=dense_fp16_bytes,
        dense_int4_bytes=dense_int4_bytes,
        factor_fp16_bytes=factor_fp16_bytes,
        factor_int4_bytes=factor_int4_bytes,
    )


# ---------------------------------------------------------------------------
# Full-model projection
# ---------------------------------------------------------------------------

@dataclass
class ModelShrinkEstimate:
    """Estimated on-disk size for a whole model under a mixed strategy."""
    total_fp16_mb: float
    total_int4_mb: float           # direct nf4-like of everything
    total_factor_int4_mb: float    # factor+int4 for large matmuls, direct-int4 for rest
    shrink_vs_fp16: float
    shrink_vs_dense_int4: float
    breakdown: dict  # per-matrix-type breakdown


def estimate_full_model_shrink(
    hidden_dim: int,
    intermediate_dim: int,
    n_layers: int,
    vocab_size: int,
    *,
    ffm_rank: int = 1024,
    attn_rank: int = 640,
    n_bits: int = 4,
    tie_word_embeddings: bool = True,
    n_kv_heads: int = 2,
    head_dim: int = 128,
) -> ModelShrinkEstimate:
    """Estimate total model bytes under factor+int4 vs dense-int4 vs fp16.

    Strategy:
    - FFN gate/up/down: factor then int4-quantize A, B
    - Q-proj: factor then int4-quantize
    - K-proj, V-proj: direct int4 (too small to factor profitably)
    - O-proj: direct int4
    - Embedding: direct int4 (int8 for vocab if needed)
    - Layer norms: fp16 (negligible)
    """
    d = hidden_dim
    f = intermediate_dim
    L = n_layers
    V = vocab_size
    n_q_heads = d // head_dim
    n_kv = n_kv_heads

    bytes_fp16 = 0.0
    bytes_int4 = 0.0
    bytes_factor_int4 = 0.0
    breakdown = {}

    # Per-layer components
    # Q-proj: d × d
    bytes_fp16 += L * d * d * 2
    bytes_int4 += L * int4_storage_bytes(d, d, n_bits)
    bytes_factor_int4 += L * (
        int4_storage_bytes(attn_rank, d, n_bits) + int4_storage_bytes(d, attn_rank, n_bits)
    )
    breakdown["q_proj"] = {
        "fp16_mb": L * d * d * 2 / 1e6,
        "int4_mb": L * int4_storage_bytes(d, d, n_bits) / 1e6,
        "factor_int4_mb": L * (
            int4_storage_bytes(attn_rank, d, n_bits) + int4_storage_bytes(d, attn_rank, n_bits)
        ) / 1e6,
        "strategy": f"factor r={attn_rank} + int4",
    }

    # K-proj: n_kv*head_dim × d
    k_out = n_kv * head_dim
    bytes_fp16 += L * k_out * d * 2
    bytes_int4 += L * int4_storage_bytes(k_out, d, n_bits)
    bytes_factor_int4 += L * int4_storage_bytes(k_out, d, n_bits)  # direct int4
    breakdown["k_proj"] = {
        "fp16_mb": L * k_out * d * 2 / 1e6,
        "int4_mb": L * int4_storage_bytes(k_out, d, n_bits) / 1e6,
        "factor_int4_mb": L * int4_storage_bytes(k_out, d, n_bits) / 1e6,
        "strategy": "direct int4 (too small to factor)",
    }

    # V-proj: same as K
    bytes_fp16 += L * k_out * d * 2
    bytes_int4 += L * int4_storage_bytes(k_out, d, n_bits)
    bytes_factor_int4 += L * int4_storage_bytes(k_out, d, n_bits)
    breakdown["v_proj"] = {
        "fp16_mb": L * k_out * d * 2 / 1e6,
        "int4_mb": L * int4_storage_bytes(k_out, d, n_bits) / 1e6,
        "factor_int4_mb": L * int4_storage_bytes(k_out, d, n_bits) / 1e6,
        "strategy": "direct int4",
    }

    # O-proj: d × d
    bytes_fp16 += L * d * d * 2
    bytes_int4 += L * int4_storage_bytes(d, d, n_bits)
    bytes_factor_int4 += L * int4_storage_bytes(d, d, n_bits)  # direct int4
    breakdown["o_proj"] = {
        "fp16_mb": L * d * d * 2 / 1e6,
        "int4_mb": L * int4_storage_bytes(d, d, n_bits) / 1e6,
        "factor_int4_mb": L * int4_storage_bytes(d, d, n_bits) / 1e6,
        "strategy": "direct int4",
    }

    # FFN gate_proj: f × d
    bytes_fp16 += L * f * d * 2
    bytes_int4 += L * int4_storage_bytes(f, d, n_bits)
    bytes_factor_int4 += L * (
        int4_storage_bytes(ffm_rank, d, n_bits) + int4_storage_bytes(f, ffm_rank, n_bits)
    )
    breakdown["gate_proj"] = {
        "fp16_mb": L * f * d * 2 / 1e6,
        "int4_mb": L * int4_storage_bytes(f, d, n_bits) / 1e6,
        "factor_int4_mb": L * (
            int4_storage_bytes(ffm_rank, d, n_bits) + int4_storage_bytes(f, ffm_rank, n_bits)
        ) / 1e6,
        "strategy": f"factor r={ffm_rank} + int4",
    }

    # FFN up_proj: f × d
    bytes_fp16 += L * f * d * 2
    bytes_int4 += L * int4_storage_bytes(f, d, n_bits)
    bytes_factor_int4 += L * (
        int4_storage_bytes(ffm_rank, d, n_bits) + int4_storage_bytes(f, ffm_rank, n_bits)
    )
    breakdown["up_proj"] = {
        "fp16_mb": L * f * d * 2 / 1e6,
        "int4_mb": L * int4_storage_bytes(f, d, n_bits) / 1e6,
        "factor_int4_mb": L * (
            int4_storage_bytes(ffm_rank, d, n_bits) + int4_storage_bytes(f, ffm_rank, n_bits)
        ) / 1e6,
        "strategy": f"factor r={ffm_rank} + int4",
    }

    # FFN down_proj: d × f
    bytes_fp16 += L * d * f * 2
    bytes_int4 += L * int4_storage_bytes(d, f, n_bits)
    bytes_factor_int4 += L * (
        int4_storage_bytes(ffm_rank, f, n_bits) + int4_storage_bytes(d, ffm_rank, n_bits)
    )
    breakdown["down_proj"] = {
        "fp16_mb": L * d * f * 2 / 1e6,
        "int4_mb": L * int4_storage_bytes(d, f, n_bits) / 1e6,
        "factor_int4_mb": L * (
            int4_storage_bytes(ffm_rank, f, n_bits) + int4_storage_bytes(d, ffm_rank, n_bits)
        ) / 1e6,
        "strategy": f"factor r={ffm_rank} + int4",
    }

    # Embedding: V × d
    bytes_fp16 += V * d * 2
    bytes_int4 += int4_storage_bytes(V, d, n_bits)
    bytes_factor_int4 += int4_storage_bytes(V, d, n_bits)
    breakdown["embedding"] = {
        "fp16_mb": V * d * 2 / 1e6,
        "int4_mb": int4_storage_bytes(V, d, n_bits) / 1e6,
        "factor_int4_mb": int4_storage_bytes(V, d, n_bits) / 1e6,
        "strategy": "direct int4",
    }
    # lm_head is tied, skip
    if not tie_word_embeddings:
        bytes_fp16 += V * d * 2
        bytes_int4 += int4_storage_bytes(V, d, n_bits)
        bytes_factor_int4 += int4_storage_bytes(V, d, n_bits)
        breakdown["lm_head"] = {
            "fp16_mb": V * d * 2 / 1e6,
            "int4_mb": int4_storage_bytes(V, d, n_bits) / 1e6,
            "factor_int4_mb": int4_storage_bytes(V, d, n_bits) / 1e6,
            "strategy": "direct int4 (untied)",
        }

    # Layer norms (~2 per layer: input_layernorm + post_attention_layernorm, fp16, tiny)
    bytes_fp16 += L * d * 4 * 2  # rough: 2 norms, weight+bias each
    bytes_int4 += L * d * 4 * 2  # keep fp16
    bytes_factor_int4 += L * d * 4 * 2  # keep fp16
    breakdown["norms"] = {
        "fp16_mb": L * d * 4 * 2 / 1e6,
        "int4_mb": L * d * 4 * 2 / 1e6,
        "factor_int4_mb": L * d * 4 * 2 / 1e6,
        "strategy": "fp16 (negligible)",
    }

    shrink_vs_fp16 = bytes_fp16 / bytes_factor_int4 if bytes_factor_int4 > 0 else 0
    shrink_vs_dense_int4 = bytes_int4 / bytes_factor_int4 if bytes_factor_int4 > 0 else 0

    return ModelShrinkEstimate(
        total_fp16_mb=bytes_fp16 / 1e6,
        total_int4_mb=bytes_int4 / 1e6,
        total_factor_int4_mb=bytes_factor_int4 / 1e6,
        shrink_vs_fp16=shrink_vs_fp16,
        shrink_vs_dense_int4=shrink_vs_dense_int4,
        breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Estimate factor×int4 shrink for a model architecture."
    )
    p.add_argument("--hidden-dim", type=int, default=1536)
    p.add_argument("--intermediate-dim", type=int, default=8960)
    p.add_argument("--n-layers", type=int, default=28)
    p.add_argument("--vocab-size", type=int, default=151936)
    p.add_argument("--ffn-rank", type=int, default=1024)
    p.add_argument("--attn-rank", type=int, default=640)
    p.add_argument("--n-bits", type=int, default=4)
    p.add_argument("--n-kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    est = estimate_full_model_shrink(
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        ffm_rank=args.ffn_rank,
        attn_rank=args.attn_rank,
        n_bits=args.n_bits,
        n_kv_heads=args.n_kv_heads,
        head_dim=args.head_dim,
    )

    if args.json:
        print(json.dumps({
            "total_fp16_mb": round(est.total_fp16_mb, 1),
            "total_int4_mb": round(est.total_int4_mb, 1),
            "total_factor_int4_mb": round(est.total_factor_int4_mb, 1),
            "shrink_vs_fp16": round(est.shrink_vs_fp16, 2),
            "shrink_vs_dense_int4": round(est.shrink_vs_dense_int4, 2),
            "breakdown": {k: {kk: round(vv, 1) if isinstance(vv, float) else vv
                             for kk, vv in v.items()}
                         for k, v in est.breakdown.items()},
        }, indent=2))
    else:
        print(f"fp16 baseline:        {est.total_fp16_mb:.1f} MB")
        print(f"dense int{n_bits} (nf4-like):   {est.total_int4_mb:.1f} MB  ({est.total_fp16_mb/est.total_int4_mb:.2f}×)")
        print(f"factor + int{n_bits}:       {est.total_factor_int4_mb:.1f} MB  ({est.shrink_vs_fp16:.2f}× vs fp16, {est.shrink_vs_dense_int4:.2f}× vs dense-int4)")
        print()
        for name, bd in est.breakdown.items():
            print(f"  {name}: {bd['strategy']} → {bd['factor_int4_mb']:.1f} MB")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli_main())
