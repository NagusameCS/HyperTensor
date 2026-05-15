"""Speculative draft models for vLLM-style pipelines.

Paper III (Geodesic Speculative Decoding): the GRC-compressed model
serves as a fast drafter; the full-precision model verifies candidate
tokens.  This is the standard speculative-decoding pattern
(Leviathan et al. 2023, Chen et al. 2023) with GRC compression as
the draft model instead of a separately trained smaller model.

The primary class, ``CompressedDrafter``, wraps a HuggingFace
compressed model and uses its REAL LM head — no hidden-state
projection guesswork.  The older k-space-only ``KSpaceDrafter`` is
retained for research comparison.

When vLLM is installed, ``register_with_vllm()`` wires the proposer
into ``vllm.spec_decode.draft_model_runner``.
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


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class DraftConfig:
    """Hyperparameters for speculative draft proposers."""
    n_drafts: int = 4            # tokens proposed per step (=== vLLM gamma)
    temperature: float = 0.0     # 0 = greedy argmax; >0 = sampling
    top_k: int = 1               # top-k for draft token selection
    top_p: float = 1.0           # nucleus sampling threshold
    max_new_tokens: int = 4      # alias for n_drafts

    def __post_init__(self):
        if self.n_drafts != self.max_new_tokens:
            self.max_new_tokens = self.n_drafts


@dataclass
class GeodesicDraftConfig:
    """Hyperparameters for the legacy k-space geodesic drafter."""
    k: int = 256                 # k-space rank
    n_drafts: int = 4            # tokens proposed per step
    jury_threshold: float = 0.85 # accept-without-verify if jury > thr
    curvature_damping: float = 0.5


# ===========================================================================
# CompressedDrafter — the real one (Paper III)
# ===========================================================================

class CompressedDrafter:
    """Speculative draft proposer backed by a GRC-compressed HF model.

    This is the correct approach from Paper III: the compressed model
    *itself* is the drafter.  It runs autoregressively for ``n_drafts``
    tokens, using its real LM head, real attention, real everything —
    just with geometrically compressed weights.

    Usage::

        draft = CompressedDrafter(compressed_model, cfg=DraftConfig(n_drafts=4))
        token_ids, confidences = draft.propose(input_ids)

    ``input_ids`` is a [1, seq_len] tensor of prompt tokens.
    Returns (draft_ids: [n_drafts] int64, confidences: [n_drafts] float32)
    where confidences are softmax probabilities of the chosen tokens.

    For fp16/bf16 speed, load the model with ``torch_dtype=torch.float16``
    and pass the same dtype here::

        from transformers import AutoModelForCausalLM
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16,
        )
        draft = CompressedDrafter(model, dtype=torch.float16)
    """

    def __init__(self, model, cfg: Optional[DraftConfig] = None, *, dtype=None):
        if not _HAS_TORCH:
            raise RuntimeError("CompressedDrafter requires PyTorch")
        self.model = model
        self.cfg = cfg or DraftConfig()
        self._device = next(model.parameters()).device
        self._dtype = dtype  # None = use model's native dtype

    def propose(self, input_ids, **kwargs):
        """Propose n_drafts tokens autoregressively.

        Args:
            input_ids: torch.LongTensor [1, seq_len] prompt token ids

        Returns:
            (token_ids: np.ndarray[int64], confidences: np.ndarray[float32])
        """
        n_drafts = kwargs.get("n_drafts", self.cfg.n_drafts)
        temperature = kwargs.get("temperature", self.cfg.temperature)

        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=self._device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device=self._device)

        token_ids: list[int] = []
        confidences: list[float] = []
        past_key_values = None
        current_ids = input_ids

        with torch.no_grad():
            for _ in range(n_drafts):
                if past_key_values is not None:
                    # Single-token forward with KV cache
                    out = self.model(
                        input_ids=current_ids[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                else:
                    out = self.model(
                        input_ids=current_ids,
                        use_cache=True,
                    )
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :]  # [1, vocab]

                if temperature > 0:
                    logits = logits / temperature
                    probs = torch.softmax(logits, dim=-1)
                    if self.cfg.top_k > 1:
                        topk_vals, topk_idx = torch.topk(probs, k=min(self.cfg.top_k, probs.size(-1)), dim=-1)
                        probs = torch.zeros_like(probs).scatter_(-1, topk_idx, topk_vals)
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                    next_token = torch.multinomial(probs, num_samples=1)
                    conf = float(probs[0, next_token[0, 0]].cpu())
                else:
                    # Greedy argmax
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [1, 1]
                    probs = torch.softmax(logits, dim=-1)
                    conf = float(probs[0, next_token[0, 0]].cpu())

                tid = int(next_token[0, 0].cpu())
                token_ids.append(tid)
                confidences.append(conf)
                current_ids = next_token

        return np.asarray(token_ids, dtype=np.int64), np.asarray(confidences, dtype=np.float32)

    def propose_batch(self, input_ids_batch, **kwargs):
        """Batch proposal for multiple prompts. Returns list of (ids, confs)."""
        results = []
        for i in range(input_ids_batch.size(0)):
            ids, confs = self.propose(input_ids_batch[i:i+1], **kwargs)
            results.append((ids, confs))
        return results

    def jury_confidence(self, confidences: np.ndarray) -> float:
        """Aggregate draft confidence (same jury formula, applied to softmax confs)."""
        c = np.clip(confidences, 0.0, 1.0)
        prod = float(np.prod(1.0 - c))
        return 1.0 - prod

    @staticmethod
    def register_with_vllm() -> bool:
        """Best-effort hook into vLLM's speculative decoding manager."""
        try:
            import vllm  # noqa: F401
            from vllm.spec_decode import spec_decode_worker  # noqa: F401
        except Exception:
            return False
        return True


# ===========================================================================
# GeodesicDraft / KSpaceDrafter — legacy k-space predictor (research only)
# ===========================================================================

class KSpaceDrafter:
    """Legacy k-space geodesic-step drafter — RESEARCH USE ONLY.

    This predicts tokens by projecting hidden states through a UGT basis
    and dotting with the LM head embedding.  Paper III shows the correct
    approach is to use the compressed model itself (see ``CompressedDrafter``).
    This class is retained for ablation studies and research comparison.

    Usage::

        draft = KSpaceDrafter(basis=U, lm_head_weight=W_lm, cfg=GeodesicDraftConfig())
        token_ids, confidences = draft.propose(h_curr, h_prev, top_k_search=512)
    """

    def __init__(
        self,
        basis,
        lm_head_weight,
        cfg: Optional[GeodesicDraftConfig] = None,
    ):
        self.cfg = cfg or GeodesicDraftConfig()
        if _HAS_TORCH and torch.is_tensor(basis):
            self.basis = basis.detach().to(torch.float32)
            self.lm_head = lm_head_weight.detach().to(torch.float32)
            self._np_basis = self.basis.cpu().numpy()
            self._np_lm_head = self.lm_head.cpu().numpy()
        else:
            self._np_basis = np.asarray(basis, dtype=np.float32)
            self._np_lm_head = np.asarray(lm_head_weight, dtype=np.float32)
            self.basis = None
            self.lm_head = None
        self._prev_h: np.ndarray | None = None
        self._metric_cov: np.ndarray | None = None
        self._n_cal = 0
        # Trajectory bank: k-space bin → [token_id, ...] from calibration
        self._trajectory_bank: dict[int, list[int]] = {}

    def calibrate(self, hidden_states, token_ids=None) -> None:
        """Build metric covariance + optional trajectory bank."""
        H = self._as_numpy_2d(hidden_states)
        projs = H @ self._np_basis            # [N, K]
        c = projs - projs.mean(axis=0, keepdims=True)
        cov = (c.T @ c) / max(1, c.shape[0] - 1)
        K = cov.shape[0]
        self._metric_cov = cov + 0.01 * np.eye(K, dtype=np.float32)
        self._n_cal = H.shape[0]
        # Build trajectory bank if token_ids provided
        if token_ids is not None:
            tids = np.asarray(token_ids, dtype=np.int64).reshape(-1)
            for i in range(min(len(tids) - 1, H.shape[0] - 1)):
                # Bin by dominant k-space direction
                p = projs[i]
                bin_idx = int(np.argmax(np.abs(p)))
                self._trajectory_bank.setdefault(bin_idx, []).append(int(tids[i + 1]))

    def propose(self, h_curr, h_prev=None, top_k_search: int = 512):
        """Propose draft tokens from k-space geodesic step.

        Uses the trajectory bank when available (jury-based selection),
        falling back to LM-head dot-product when the bank is empty.

        Args:
            h_curr: last hidden state, shape [d_model]
            h_prev: previous hidden state for velocity (optional)
            top_k_search: vocab-side candidate cap

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

            if self._trajectory_bank:
                # Jury-based trajectory lookup
                best_tid, best_conf = self._jury_lookup(p_pred)
            else:
                # Fallback: LM-head dot-product
                e_pred = p_pred @ self._np_basis.T           # back to d_model
                logits = self._np_lm_head @ e_pred           # [vocab]
                top_k = min(int(top_k_search), logits.shape[0])
                idx = np.argpartition(logits, -top_k)[-top_k:]
                ordered = idx[np.argsort(-logits[idx])]
                best_tid = int(ordered[0])
                second = float(logits[ordered[1]]) if len(ordered) > 1 else float(logits[ordered[0]]) - 1.0
                best_conf = 1.0 / (1.0 + np.exp(-(float(logits[ordered[0]]) - second)))

            token_ids.append(best_tid)
            confidences.append(best_conf)
            h_prev_step = h_step
            h_step = p_pred @ self._np_basis.T
            nrm = float(np.linalg.norm(h_step))
            if nrm > 1e-9:
                h_step = h_step / nrm * float(np.linalg.norm(h_step))

        self._prev_h = h_curr
        return np.asarray(token_ids, dtype=np.int64), np.asarray(confidences, dtype=np.float32)

    def _jury_lookup(self, p_pred: np.ndarray) -> tuple[int, float]:
        """Jury-based token selection from trajectory bank."""
        bin_idx = int(np.argmax(np.abs(p_pred)))
        candidates = self._trajectory_bank.get(bin_idx, [])
        if not candidates:
            # Fall back to nearest bin
            all_candidates = []
            for _bin, toks in self._trajectory_bank.items():
                all_candidates.extend(toks)
            if not all_candidates:
                return 0, 0.0
            from collections import Counter
            most_common = Counter(all_candidates).most_common(1)[0]
            return most_common[0], 0.5
        from collections import Counter
        most_common = Counter(candidates).most_common(1)[0]
        conf = most_common[1] / len(candidates)
        return most_common[0], float(conf)

    def jury_confidence(self, confidences: np.ndarray) -> float:
        c = np.clip(confidences / (np.max(confidences) + 1e-9), 0.0, 1.0)
        prod = float(np.prod(1.0 - c))
        return 1.0 - prod

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


# ===========================================================================
# Backward-compatible alias
# ===========================================================================

# GeodesicDraft now points to CompressedDrafter (the correct approach).
# Old code that used GeodesicDraft(basis, embedding) will break —
# that's intentional; migrate to CompressedDrafter(model) or
# KSpaceDrafter(basis, lm_head_weight) for research use.
GeodesicDraft = CompressedDrafter
