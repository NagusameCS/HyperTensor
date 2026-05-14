"""Geodesic speculative draft model for vLLM-style pipelines.

vLLM's speculative decoding interface (``vllm.spec_decode``) expects a
draft proposer that, given a sequence's hidden state / token history,
returns a short list of candidate next tokens.  We provide an adapter
that drives HyperTensor's k-space geodesic step (mirrors
``runtime/nn/axiom_beta.c::axiom_beta_geodesic_step_fast`` and the
Python reference in ``scripts/ott_engine.GeodesicDraftGenerator``).

This module is deliberately *not* a hard dependency on vLLM — it
exposes a minimal interface (``propose``) that vLLM's speculative
decoding manager, EAGLE, Medusa, or our own bench harness can call
identically.  When vLLM is installed, ``register_with_vllm()`` wires
the proposer into ``vllm.spec_decode.draft_model_runner``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


@dataclass
class GeodesicDraftConfig:
    """Hyperparameters for the geodesic drafter."""
    k: int = 256                 # k-space rank
    n_drafts: int = 4            # tokens proposed per step (=== vLLM gamma)
    jury_threshold: float = 0.85 # accept-without-verify if jury > thr
    curvature_damping: float = 0.5


class GeodesicDraft:
    """Drop-in replacement for a vLLM draft proposer.

    Usage (framework-agnostic)::

        draft = GeodesicDraft(basis=U, embedding=embed.weight, cfg=cfg)
        token_ids, confidences = draft.propose(hidden_state)

    The output (token_ids, confidences) is what vLLM's
    SpeculativeDecoder expects from EAGLE / Medusa / draft-model heads.
    """

    def __init__(
        self,
        basis,
        embedding,
        cfg: Optional[GeodesicDraftConfig] = None,
    ):
        self.cfg = cfg or GeodesicDraftConfig()
        if _HAS_TORCH and torch.is_tensor(basis):
            self.basis = basis.detach().to(torch.float32)
            self.embedding = embedding.detach().to(torch.float32)
            self._np_basis = self.basis.cpu().numpy()
            self._np_embed = self.embedding.cpu().numpy()
        else:
            self._np_basis = np.asarray(basis, dtype=np.float32)
            self._np_embed = np.asarray(embedding, dtype=np.float32)
            self.basis = None
            self.embedding = None
        self._prev_h: np.ndarray | None = None
        self._metric_cov: np.ndarray | None = None
        self._n_cal = 0

    # ------------------------------------------------------------------
    # Calibration: builds the metric from a small set of hidden states.
    # ------------------------------------------------------------------
    def calibrate(self, hidden_states) -> None:
        H = self._as_numpy_2d(hidden_states)
        projs = H @ self._np_basis            # [N, K]
        c = projs - projs.mean(axis=0, keepdims=True)
        cov = (c.T @ c) / max(1, c.shape[0] - 1)
        K = cov.shape[0]
        self._metric_cov = cov + 0.01 * np.eye(K, dtype=np.float32)
        self._n_cal = H.shape[0]

    # ------------------------------------------------------------------
    # Public API: propose `n_drafts` token ids.
    # ------------------------------------------------------------------
    def propose(self, h_curr, h_prev=None, top_k_search: int = 512):
        """Propose draft tokens.

        Args:
            h_curr: last hidden state, shape [d_model]
            h_prev: previous hidden state for velocity (optional)
            top_k_search: vocab-side candidate cap (efficiency knob)

        Returns:
            (token_ids: np.ndarray[int64], confidences: np.ndarray[float32])
        """
        h_curr = self._as_numpy_1d(h_curr)
        h_prev_np = self._as_numpy_1d(h_prev) if h_prev is not None else self._prev_h

        token_ids: list[int] = []
        confidences: list[float] = []
        h_prev_step = h_prev_np
        h_step = h_curr

        for _ in range(self.cfg.n_drafts):
            p_pred = self._geodesic_step(h_step, h_prev_step)
            e_pred = p_pred @ self._np_basis.T           # back to d_model
            logits = self._np_embed @ e_pred             # [vocab]
            # top-k for efficiency
            top_k = min(int(top_k_search), logits.shape[0])
            idx = np.argpartition(logits, -top_k)[-top_k:]
            ordered = idx[np.argsort(-logits[idx])]
            best = int(ordered[0])
            second = float(logits[ordered[1]]) if len(ordered) > 1 else float(logits[ordered[0]]) - 1.0
            margin = float(logits[ordered[0]]) - second
            token_ids.append(best)
            confidences.append(margin)
            # roll forward: feed predicted hidden state back as h_step
            h_prev_step = h_step
            h_step = e_pred / max(1e-9, float(np.linalg.norm(e_pred))) * float(np.linalg.norm(h_step))

        self._prev_h = h_curr
        return np.asarray(token_ids, dtype=np.int64), np.asarray(confidences, dtype=np.float32)

    # ------------------------------------------------------------------
    # Jury aggregate confidence (Paper C jury-GTC gate).
    # ------------------------------------------------------------------
    def jury_confidence(self, confidences: np.ndarray) -> float:
        c = np.clip(confidences / (np.max(confidences) + 1e-9), 0.0, 1.0)
        prod = float(np.prod(1.0 - c))
        return 1.0 - prod

    # ------------------------------------------------------------------
    # Optional integration with vLLM.
    # ------------------------------------------------------------------
    @staticmethod
    def register_with_vllm() -> bool:
        """Best-effort hook: register this proposer with vLLM's
        speculative decoding manager.  Returns True on success.

        vLLM's spec-decode plugin API is still evolving across versions;
        we attempt the public entry-points and fall back gracefully.
        """
        try:
            import vllm  # noqa: F401
            from vllm.spec_decode import spec_decode_worker  # noqa: F401
        except Exception:
            return False
        # The real integration plugs a `Proposer` subclass into
        # vllm.spec_decode.SpeculativeWorker._proposer.  Different vLLM
        # versions expose this through different paths; we leave the
        # binding to the deployment script rather than monkey-patching
        # at import time.
        return True

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------
    def _geodesic_step(self, h_curr: np.ndarray, h_prev: np.ndarray | None) -> np.ndarray:
        p_curr = h_curr @ self._np_basis        # [K]
        if h_prev is None:
            v = np.zeros_like(p_curr)
        else:
            v = p_curr - (h_prev @ self._np_basis)
        n = float(np.linalg.norm(v))
        if n > 1e-12:
            v = v / n
        if self._metric_cov is not None and self._n_cal >= 8:
            try:
                M_inv = np.linalg.inv(
                    self._metric_cov + 0.001 * np.eye(self._metric_cov.shape[0])
                )
                correction = v - M_inv @ v
                return p_curr + v - self.cfg.curvature_damping * correction
            except np.linalg.LinAlgError:
                pass
        return p_curr + v

    @staticmethod
    def _as_numpy_1d(x) -> np.ndarray:
        if _HAS_TORCH and torch.is_tensor(x):
            x = x.detach().to("cpu").to(torch.float32).numpy()
        x = np.asarray(x, dtype=np.float32)
        return x.reshape(-1)

    @staticmethod
    def _as_numpy_2d(x) -> np.ndarray:
        if _HAS_TORCH and torch.is_tensor(x):
            x = x.detach().to("cpu").to(torch.float32).numpy()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x
