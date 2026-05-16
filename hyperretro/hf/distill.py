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

"""GRC Light Distillation (Paper V).

Recovers PPL lost to aggressive GRC compression by fitting small
rank-r LoRA adapters on the teacher-student logit residual.

Protocol (from Paper V / scripts/distill_runner.py):

  1. Load teacher model (original, full precision)
  2. Apply GRC compression to attention Q/K/V → student base weights
  3. Wrap each compressed attention layer with trainable LoRA adapters
  4. Minimise teacher-student logit MSE on a calibration corpus
  5. Merge LoRA deltas back into compressed weights
  6. Save as standard safetensors (loadable by AutoModelForCausalLM)

Typical hyperparams: lora_rank=8, lr=1e-4, steps=200-500, batch=4-8,
seq_len=256-512, calibration corpus 5k-10k tokens of WikiText-2.

CLI: ``hyperretro-distill`` (registered in pyproject.toml).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Config
# ===========================================================================


@dataclass
class DistillConfig:
    """Distillation hyperparameters."""
    rank_k: int = 1024              # GRC compression rank
    sink_T: int = 0                 # 0 = vanilla GRC; >0 = sink-aware
    lora_rank: int = 8              # rank of LoRA correction adapters
    lora_alpha: float = 16.0        # LoRA scaling factor
    lora_dropout: float = 0.0
    learning_rate: float = 1e-4
    steps: int = 200                # ~5-10k tokens at batch=4 seq=256
    batch_size: int = 4
    seq_len: int = 256
    warmup_steps: int = 50
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    device: str = "cuda"
    dtype: str = "float32"          # bfloat16 recommended when available
    layers: list[int] = field(default_factory=list)  # [] = all
    log_every: int = 50
    # Calibration corpus — if None, use synthetic Gaussian noise
    corpus_path: str | None = None
    corpus_max_samples: int = 200
    # Loss function: "mse" (logit MSE, fast), "kl" (KL divergence,
    # behavioral-residue objective — weights loss by teacher confidence),
    # "margin" (ranking loss — directly optimizes argmax match, best
    # for speculative decoding acceptance rate), or "behavioral_residue"
    # (confidence-weighted KL — upweights positions where the teacher
    # is decisive; ignores high-entropy/uncertain positions)
    loss_type: str = "mse"
    kl_temperature: float = 4.0     # temperature for KL distillation
    margin_value: float = 1.0       # margin for ranking loss
    margin_topk: int = 5            # top-k teacher tokens for margin loss
    br_confidence_exp: float = 1.0  # exponent on teacher max-prob weights
    br_min_confidence: float = 0.0  # ignore positions with max teacher prob < this
    # FFN compression (distillation-recovered).
    # SwiGLU: gate_proj/up_proj/down_proj. GPT-2: c_fc/c_proj.
    # Each gets SVD-truncated then wrapped with a LoRA adapter, same
    # protocol as attention. 0 = no FFN compression.
    ffn_rank_in: int = 0   # SVD rank for gate/up (or c_fc for GPT-2)
    ffn_rank_out: int = 0  # SVD rank for down (or c_proj for GPT-2)
    # Name patterns for attention Q/K/V (Llama/Qwen/Mistral)
    name_patterns: tuple[str, ...] = (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
    )


# ===========================================================================
# GRC helpers (vendored from compress.py for standalone use)
# ===========================================================================


def _build_shared_basis(Wq, Wk, Wv, n_iter: int = 3):
    """Joint Gram power-iteration basis (same as compress.py)."""
    K = Wq.T @ Wq + Wk.T @ Wk + Wv.T @ Wv
    K = K / np.linalg.norm(K, "fro")
    A = K.copy()
    for _ in range(n_iter):
        A = A @ K
        A = A / np.linalg.norm(A, "fro")
    eigvals, eigvecs = np.linalg.eigh(A)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]


def _project(W, P, k):
    Pk = P[:, :k]
    return W @ Pk @ Pk.T


def _sink_indices(Wq, Wk, Wv, T: int):
    if T <= 0:
        return np.array([], dtype=np.int64)
    mag = (np.linalg.norm(Wq, axis=0) ** 2
           + np.linalg.norm(Wk, axis=0) ** 2
           + np.linalg.norm(Wv, axis=0) ** 2)
    return np.argsort(mag)[::-1][:T]


# ===========================================================================
# LoRA adapter
# ===========================================================================


class LoRALinear(nn.Module):
    """Rank-r LoRA adapter wrapping a frozen linear layer."""

    def __init__(self, base: nn.Linear, r: int, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0

        out_features, in_features = base.weight.shape
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling

    def get_delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def merge(self) -> None:
        """Merge LoRA delta into base weight and zero out LoRA factors."""
        delta = self.get_delta_weight()
        self.base.weight.data.add_(delta)
        with torch.no_grad():
            self.lora_A.zero_()
            self.lora_B.zero_()


# ===========================================================================
# Calibration corpus
# ===========================================================================


class _TextChunkDataset(torch.utils.data.IterableDataset):
    """Stream text chunks from a file or generate synthetic noise."""

    def __init__(self, tokenizer, seq_len: int = 256,
                 corpus_path: str | None = None,
                 max_samples: int = 200):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.max_samples = max_samples

    def __iter__(self):
        if self.corpus_path and Path(self.corpus_path).exists():
            text = Path(self.corpus_path).read_text(encoding="utf-8")
            tokens = self.tokenizer.encode(text)
            stride = self.seq_len // 2
            count = 0
            for i in range(0, len(tokens) - self.seq_len, stride):
                chunk = tokens[i:i + self.seq_len]
                if len(chunk) < self.seq_len:
                    break
                yield torch.tensor(chunk, dtype=torch.long)
                count += 1
                if count >= self.max_samples:
                    break
        else:
            # Synthetic: random token IDs in vocab range
            vocab_size = getattr(self.tokenizer, "vocab_size", 50257)
            for _ in range(self.max_samples):
                ids = torch.randint(0, min(vocab_size, 50000),
                                    (self.seq_len,), dtype=torch.long)
                yield ids


# ===========================================================================
# Model surgery
# ===========================================================================


def _find_attn_modules(model: nn.Module) -> dict[int, dict[str, nn.Linear]]:
    """Discover attention Q/K/V Linear layers by layer index."""
    import re
    attn_modules: dict[int, dict[str, nn.Linear]] = {}
    for name, module in model.named_modules():
        m = re.match(
            r".*layers?\.(\d+)\.(?:self_)?attn\.(q_proj|k_proj|v_proj)$", name
        )
        if m and isinstance(module, nn.Linear):
            layer_idx = int(m.group(1))
            slot = m.group(2)[0].upper()
            attn_modules.setdefault(layer_idx, {})[slot] = module
    return attn_modules


def _replace_module(model: nn.Module, dotted_name: str, new_module: nn.Module):
    """Replace a submodule by dotted path."""
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def _apply_grc_and_lora(
    teacher: nn.Module,
    cfg: DistillConfig,
) -> nn.Module:
    """Apply GRC projection to attention weights and wrap with LoRA.

    Modifies teacher IN PLACE.  Returns the same model with LoRA params
    registered and base weights compressed.
    """
    attn_modules = _find_attn_modules(teacher)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.dtype, torch.float32)
    device = next(teacher.parameters()).device

    target_layers = cfg.layers if cfg.layers else sorted(attn_modules.keys())
    lora_params = []
    stats: dict[str, dict] = {}

    for li in target_layers:
        if li not in attn_modules:
            continue
        slots = attn_modules[li]
        if not all(s in slots for s in ("Q", "K", "V")):
            continue

        # Extract weights as numpy
        Wq_np = slots["Q"].weight.detach().cpu().float().numpy()
        Wk_np = slots["K"].weight.detach().cpu().float().numpy()
        Wv_np = slots["V"].weight.detach().cpu().float().numpy()
        d = Wq_np.shape[1]
        k = min(cfg.rank_k, d)
        sink = _sink_indices(Wq_np, Wk_np, Wv_np, cfg.sink_T)

        # Apply GRC
        if cfg.sink_T > 0 and len(sink) > 0:
            Wq_R = Wq_np.copy(); Wq_R[:, sink] = 0.0
            Wk_R = Wk_np.copy(); Wk_R[:, sink] = 0.0
            Wv_R = Wv_np.copy(); Wv_R[:, sink] = 0.0
            P = _build_shared_basis(Wq_R, Wk_R, Wv_R)
            Wq_proj = _project(Wq_R, P, k); Wq_proj[:, sink] = Wq_np[:, sink]
            Wk_proj = _project(Wk_R, P, k); Wk_proj[:, sink] = Wk_np[:, sink]
            Wv_proj = _project(Wv_R, P, k); Wv_proj[:, sink] = Wv_np[:, sink]
        else:
            P = _build_shared_basis(Wq_np, Wk_np, Wv_np)
            Wq_proj = _project(Wq_np, P, k)
            Wk_proj = _project(Wk_np, P, k)
            Wv_proj = _project(Wv_np, P, k)

        def _relerr(W, Wh):
            n = np.linalg.norm(W)
            return float(np.linalg.norm(W - Wh) / n) if n > 0 else 0.0

        stats[f"layer_{li}"] = {
            "rank_k": k, "sink_T": cfg.sink_T, "d": int(d),
            "frob_relerr_q": _relerr(Wq_np, Wq_proj),
            "frob_relerr_k": _relerr(Wk_np, Wk_proj),
            "frob_relerr_v": _relerr(Wv_np, Wv_proj),
        }

        # Write compressed weights back + wrap with LoRA
        for slot, W_proj in [("Q", Wq_proj), ("K", Wk_proj), ("V", Wv_proj)]:
            base_linear = slots[slot]
            with torch.no_grad():
                base_linear.weight.copy_(
                    torch.from_numpy(W_proj).to(
                        device=device, dtype=base_linear.weight.dtype
                    )
                )
            lora_linear = LoRALinear(
                base_linear, r=cfg.lora_rank, alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
            )
            lora_linear = lora_linear.to(dtype=torch_dtype)
            parent_name = f"model.layers.{li}.self_attn.{slot.lower()}_proj"
            _replace_module(teacher, parent_name, lora_linear)
            attn_modules[li][slot] = lora_linear  # update reference
            lora_params.extend([lora_linear.lora_A, lora_linear.lora_B])

    total_lora = sum(p.numel() for p in lora_params)
    print(f"[distill] GRC applied to {len(stats)} layers, "
          f"total LoRA params: {total_lora:,}  "
          f"({total_lora * 2 / 1024 / 1024:.1f} MB in {torch_dtype})")

    # Optional: compress FFN matrices and wrap with LoRA too.
    if cfg.ffn_rank_in > 0 or cfg.ffn_rank_out > 0:
        ffn_stats, ffn_lora_params = _apply_ffn_compress_and_lora(
            teacher, cfg, target_layers, torch_dtype, device,
        )
        stats["ffn"] = ffn_stats
        total_ffn = sum(p.numel() for p in ffn_lora_params)
        print(f"[distill] FFN compress applied to {len(ffn_stats)} layers, "
              f"FFN LoRA params: {total_ffn:,}  "
              f"({total_ffn * 2 / 1024 / 1024:.1f} MB in {torch_dtype})")

    # Freeze all non-LoRA parameters — only lora_A/lora_B should be trainable
    for n, p in teacher.named_parameters():
        if "lora_A" not in n and "lora_B" not in n:
            p.requires_grad_(False)

    teacher._hyperretro_stats = stats
    return teacher


def _find_ffn_modules(model: nn.Module) -> dict[int, dict[str, nn.Linear]]:
    """Discover FFN Linear layers by layer index.

    Returns a per-layer dict with subset of keys:
      SwiGLU/Llama-style: "gate", "up", "down"
      GPT-2 style: "fc", "proj"
    """
    import re
    ffn_modules: dict[int, dict[str, nn.Linear]] = {}
    swiglu_re = re.compile(
        r".*layers?\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)$"
    )
    gpt2_re = re.compile(r".*\.h\.(\d+)\.mlp\.(c_fc|c_proj)$")
    name_map = {
        "gate_proj": "gate", "up_proj": "up", "down_proj": "down",
        "c_fc": "fc", "c_proj": "proj",
    }
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        m = swiglu_re.match(name) or gpt2_re.match(name)
        if m:
            li = int(m.group(1))
            ffn_modules.setdefault(li, {})[name_map[m.group(2)]] = module
    return ffn_modules


def _svd_truncate(W_np: np.ndarray, rank: int) -> np.ndarray:
    """Return rank-r SVD approximation of W (out_features x in_features)."""
    r = min(rank, min(W_np.shape))
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vt[:r, :]


def _apply_ffn_compress_and_lora(
    teacher: nn.Module,
    cfg: DistillConfig,
    target_layers: list[int],
    torch_dtype: torch.dtype,
    device,
):
    """SVD-truncate FFN matrices and wrap each with a LoRALinear adapter."""
    ffn_modules = _find_ffn_modules(teacher)
    stats: dict[str, dict] = {}
    lora_params: list[nn.Parameter] = []

    # Slot rank policy: "in" matrices (gate/up/fc) get ffn_rank_in,
    # "out" matrix (down/proj) gets ffn_rank_out. 0 = skip that slot.
    slot_rank = {
        "gate": cfg.ffn_rank_in, "up": cfg.ffn_rank_in,
        "fc":   cfg.ffn_rank_in,
        "down": cfg.ffn_rank_out, "proj": cfg.ffn_rank_out,
    }

    for li in target_layers:
        if li not in ffn_modules:
            continue
        slots = ffn_modules[li]
        layer_stat: dict[str, float] = {}
        for slot_name, base_linear in slots.items():
            r = slot_rank.get(slot_name, 0)
            if r <= 0:
                continue
            W_np = base_linear.weight.detach().cpu().float().numpy()
            W_proj = _svd_truncate(W_np, r)
            n = float(np.linalg.norm(W_np))
            relerr = float(np.linalg.norm(W_np - W_proj) / n) if n > 0 else 0.0
            layer_stat[f"frob_relerr_{slot_name}"] = relerr
            layer_stat[f"rank_{slot_name}"] = int(min(r, min(W_np.shape)))

            with torch.no_grad():
                base_linear.weight.copy_(
                    torch.from_numpy(W_proj).to(
                        device=device, dtype=base_linear.weight.dtype
                    )
                )
            lora_linear = LoRALinear(
                base_linear, r=cfg.lora_rank, alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
            ).to(dtype=torch_dtype)

            # Locate the parent dotted path
            slot_to_attr = {
                "gate": ("mlp", "gate_proj"), "up": ("mlp", "up_proj"),
                "down": ("mlp", "down_proj"),
                "fc": ("mlp", "c_fc"), "proj": ("mlp", "c_proj"),
            }
            parent_root, leaf = slot_to_attr[slot_name]
            # Walk: try model.layers.{li}.{parent_root}.{leaf} first
            candidates = [
                f"model.layers.{li}.{parent_root}.{leaf}",
                f"transformer.h.{li}.{parent_root}.{leaf}",
                f"layers.{li}.{parent_root}.{leaf}",
            ]
            replaced = False
            for path in candidates:
                try:
                    _replace_module(teacher, path, lora_linear)
                    replaced = True
                    break
                except AttributeError:
                    continue
            if not replaced:
                # Fallback: scan named_modules to find the matching path
                for nm, mod in teacher.named_modules():
                    if mod is base_linear:
                        _replace_module(teacher, nm, lora_linear)
                        replaced = True
                        break
            lora_params.extend([lora_linear.lora_A, lora_linear.lora_B])

        if layer_stat:
            stats[f"layer_{li}"] = layer_stat

    return stats, lora_params


# ===========================================================================
# Training loop
# ===========================================================================


def _train_distill(
    teacher: nn.Module,
    student: nn.Module,
    tokenizer,
    cfg: DistillConfig,
    out_dir: Path,
) -> nn.Module:
    """Run teacher-student distillation, minimising logit MSE."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.dtype, torch.float32)

    teacher = teacher.to(device=device, dtype=torch_dtype)
    student = student.to(device=device, dtype=torch_dtype)
    teacher.eval()

    # Freeze everything except LoRA adapters
    for p in student.parameters():
        p.requires_grad_(False)

    # Collect ONLY LoRA parameters (lora_A, lora_B)
    trainable = []
    for name, p in student.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad_(True)
            trainable.append(p)

    if not trainable:
        print("[distill] WARNING: no trainable LoRA parameters found. "
              "Did GRC+LoRA apply correctly?")
        return student

    print(f"[distill] trainable LoRA params: {sum(p.numel() for p in trainable):,}")

    dataset = _TextChunkDataset(
        tokenizer, seq_len=cfg.seq_len,
        corpus_path=cfg.corpus_path,
        max_samples=cfg.corpus_max_samples,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size,
    )

    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    def lr_lambda(step: int):
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_kl = cfg.loss_type == "kl"
    use_margin = cfg.loss_type == "margin"
    use_br = cfg.loss_type == "behavioral_residue"
    if use_kl:
        print(f"[distill] Using KL divergence loss (T={cfg.kl_temperature}) — "
              "behavioral-residue objective (Paper V addendum)")
    elif use_margin:
        print(f"[distill] Using margin ranking loss (margin={cfg.margin_value}, topk={cfg.margin_topk}) — "
              "directly optimizes argmax match for acceptance rate")
    elif use_br:
        print(f"[distill] Using behavioral-residue loss (T={cfg.kl_temperature}, "
              f"conf_exp={cfg.br_confidence_exp}, min_conf={cfg.br_min_confidence}) — "
              "confidence-weighted KL; focuses on deterministic teacher behavior")
    else:
        loss_fn = nn.MSELoss()

    data_iter = iter(dataloader)
    total_loss = 0.0
    total_steps = 0
    t0 = time.time()
    log_lines = []

    for step in range(cfg.steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(device=device)

        with torch.no_grad():
            teacher_out = teacher(batch).logits

        student_out = student(batch).logits

        if use_kl:
            # Behavioral-residue distillation: KL(teacher_soft || student_logsoft)
            T = cfg.kl_temperature
            teacher_probs = F.softmax(teacher_out / T, dim=-1)
            student_logprobs = F.log_softmax(student_out / T, dim=-1)
            kl = F.kl_div(student_logprobs, teacher_probs, reduction="batchmean", log_target=False)
            loss = kl * (T ** 2)
        elif use_br:
            # Behavioral-residue v2: confidence-weighted KL.
            #
            # Per-position weight = (teacher_max_prob)^conf_exp, optionally
            # gated by a minimum confidence threshold. Positions where the
            # teacher is uncertain (high entropy) contribute little; positions
            # where the teacher is decisive (one token clearly dominates)
            # contribute most. This is what "acceptance rate" actually
            # measures — the student needs to match the teacher exactly when
            # the teacher has a clear preference.
            T = cfg.kl_temperature
            with torch.no_grad():
                teacher_probs_full = F.softmax(teacher_out, dim=-1)  # for weights
                w_per_pos = teacher_probs_full.max(dim=-1).values  # [B, S]
                if cfg.br_confidence_exp != 1.0:
                    w_per_pos = w_per_pos ** cfg.br_confidence_exp
                if cfg.br_min_confidence > 0:
                    w_per_pos = torch.where(
                        w_per_pos >= cfg.br_min_confidence,
                        w_per_pos,
                        torch.zeros_like(w_per_pos),
                    )
                w_sum = w_per_pos.sum().clamp(min=1.0)
            teacher_probs = F.softmax(teacher_out / T, dim=-1)
            student_logprobs = F.log_softmax(student_out / T, dim=-1)
            # Per-position KL: sum_v teacher_probs[v] * (log teacher - log student)
            log_teacher = torch.log(teacher_probs.clamp(min=1e-20))
            kl_per_pos = (teacher_probs * (log_teacher - student_logprobs)).sum(dim=-1)  # [B, S]
            loss = (w_per_pos * kl_per_pos).sum() / w_sum * (T ** 2)
        elif use_margin:
            # Margin ranking loss: ensure student's top-1 matches teacher's top-1.
            # For each position, get teacher's top-k tokens.
            # Student's logit for teacher's #1 must exceed all other top-k
            # logits by at least `margin`.
            with torch.no_grad():
                _, teacher_topk_idx = torch.topk(teacher_out, k=cfg.margin_topk, dim=-1)
                teacher_top1 = teacher_topk_idx[:, :, 0]  # [B, S]

            # Gather student logits for teacher's top-1
            student_top1_logit = torch.gather(
                student_out, dim=-1,
                index=teacher_top1.unsqueeze(-1)
            ).squeeze(-1)  # [B, S]

            # Gather student logits for all teacher top-k tokens
            student_topk_logits = torch.gather(
                student_out, dim=-1, index=teacher_topk_idx
            )  # [B, S, K]

            # Margin: student_top1_logit must be > student_other_logit + margin
            # loss = Σ max(0, margin - (student_top1 - student_other))
            diffs = student_top1_logit.unsqueeze(-1) - student_topk_logits  # [B, S, K]
            # Don't penalise the top-1 vs itself
            margin_losses = torch.clamp(cfg.margin_value - diffs, min=0.0)
            # Zero out the top-1 position
            mask = torch.ones_like(margin_losses)
            mask[:, :, 0] = 0.0
            loss = (margin_losses * mask).sum() / max(1, mask.sum())
        else:
            loss = loss_fn(student_out, teacher_out)

        optimizer.zero_grad()
        loss.backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_steps += 1

        if (step + 1) % cfg.log_every == 0:
            avg_loss = total_loss / total_steps
            elapsed = time.time() - t0
            lr = scheduler.get_last_lr()[0]
            msg = (f"  step {step+1:4d}/{cfg.steps}  "
                   f"loss={avg_loss:.6f}  lr={lr:.2e}  "
                   f"elapsed={elapsed:.1f}s")
            print(msg, flush=True)
            log_lines.append(msg)
            total_loss = 0.0
            total_steps = 0

    elapsed = time.time() - t0
    print(f"[distill] training done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Merge LoRA deltas back into base weights + unwrap to plain nn.Linear
    _unwrap_and_merge_lora(student)

    return student


def _unwrap_and_merge_lora(model: nn.Module) -> None:
    """Merge all LoRA adapters and replace LoRALinear → nn.Linear.

    After this call the model is a plain transformers model with updated
    weights and no LoRA wrappers, so save_pretrained produces a standard
    checkpoint loadable by AutoModelForCausalLM.from_pretrained.
    """
    replacements: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge()  # ensure merged
            replacements.append((name, module.base))

    for dotted_name, linear in replacements:
        _replace_module(model, dotted_name, linear)

    n = len(replacements)
    if n > 0:
        print(f"[distill] Unwrapped {n} LoRA adapters → plain nn.Linear")


# ===========================================================================
# Public API
# ===========================================================================


def distill_hf_model(
    model_id_or_path: str,
    out_dir: str | Path,
    *,
    rank_k: int = 1024,
    sink_T: int = 0,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    steps: int = 200,
    batch_size: int = 4,
    seq_len: int = 256,
    learning_rate: float = 1e-4,
    corpus_path: str | None = None,
    layers: list[int] | None = None,
    device: str = "cuda",
    dtype: str = "float32",
    loss_type: str = "mse",
    kl_temperature: float = 4.0,
    margin_value: float = 1.0,
    margin_topk: int = 5,
    br_confidence_exp: float = 1.0,
    br_min_confidence: float = 0.0,
    ffn_rank_in: int = 0,
    ffn_rank_out: int = 0,
    revision: str | None = None,
    factored: bool = False,
    factored_ffn_rel_tol: float = 1e-4,
    save_dtype: str | None = None,
    activation_aware: bool = False,
    activation_corpus_path: str | None = None,
    activation_n_batches: int = 16,
    activation_seq_len: int = 512,
) -> dict:
    """Load a HF model, apply GRC + LoRA distillation, save.

    Returns a report dict with per-layer stats and output path.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = DistillConfig(
        rank_k=rank_k, sink_T=sink_T,
        lora_rank=lora_rank, lora_alpha=lora_alpha,
        steps=steps, batch_size=batch_size, seq_len=seq_len,
        learning_rate=learning_rate,
        corpus_path=corpus_path,
        layers=layers or [],
        device=device, dtype=dtype,
        loss_type=loss_type, kl_temperature=kl_temperature,
        margin_value=margin_value, margin_topk=margin_topk,
        br_confidence_exp=br_confidence_exp,
        br_min_confidence=br_min_confidence,
        ffn_rank_in=ffn_rank_in,
        ffn_rank_out=ffn_rank_out,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[distill] Loading teacher: {model_id_or_path}")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    teacher_dtype = dtype_map.get(cfg.dtype, torch.float32)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id_or_path, revision=revision,
        torch_dtype=teacher_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load a SEPARATE student copy — _apply_grc_and_lora modifies in-place,
    # so teacher must remain untouched for the logit-MSE objective.
    print("[distill] Loading student copy (uncompressed)...")
    student = AutoModelForCausalLM.from_pretrained(
        model_id_or_path, revision=revision,
        torch_dtype=teacher_dtype,
    )

    print("[distill] Applying GRC compression + LoRA adapters to student...")
    student = _apply_grc_and_lora(student, cfg)

    print(f"[distill] Training for {cfg.steps} steps...")
    student = _train_distill(teacher, student, tokenizer, cfg, out)

    # Save the distilled model
    print(f"[distill] Saving to {out}...")
    if factored:
        # Post-distill: factor attn (shared-basis) and FFN, then write the
        # HyperRetro factored format directly. Rank budget allows for the
        # LoRA correction's added subspace (k + lora_rank + sink).
        from hyperretro.hf.compress import _group_attn_by_layer
        from hyperretro.hf.factored import (
            factor_attn_state_dict, factor_ffn_state_dict,
            save_factored_checkpoint,
        )

        student.to("cpu")
        sd = student.state_dict()
        layer_keys = _group_attn_by_layer(sd)
        attn_rank = cfg.rank_k + cfg.sink_T + cfg.lora_rank
        sd, attn_entries = factor_attn_state_dict(
            sd, rank=attn_rank, layer_keys=layer_keys,
            rel_tol=1e-3,
        )
        ffn_entries: list = []
        if cfg.ffn_rank_in > 0 or cfg.ffn_rank_out > 0:
            ffn_max = max(cfg.ffn_rank_in, cfg.ffn_rank_out) + cfg.lora_rank
            activation_col_norms = None
            if activation_aware:
                from hyperretro.hf.activation import collect_ffn_input_norms
                print("[distill] Collecting activation norms for aware FFN factoring...")
                # Use a fresh dense copy so the merged student isn't disturbed.
                norms_model = AutoModelForCausalLM.from_pretrained(
                    model_id_or_path, revision=revision,
                    torch_dtype=teacher_dtype,
                )
                norms_device = "cuda" if torch.cuda.is_available() else "cpu"
                norms_model.to(norms_device)
                activation_col_norms = collect_ffn_input_norms(
                    norms_model, tokenizer,
                    corpus_path=activation_corpus_path or corpus_path or "",
                    n_batches=activation_n_batches,
                    seq_len=activation_seq_len,
                    device=norms_device,
                )
                del norms_model
                if norms_device == "cuda":
                    torch.cuda.empty_cache()
            ffn_entries = factor_ffn_state_dict(
                sd, max_rank=ffn_max, rel_tol=factored_ffn_rel_tol,
                activation_col_norms=activation_col_norms,
            )
        save_factored_checkpoint(
            sd, attn_entries, ffn_entries,
            out_dir=out, hf_config=student.config,
            dtype=(save_dtype or cfg.dtype), tokenizer=tokenizer,
        )
        print(f"[distill] Factored: attn={len(attn_entries)} ffn={len(ffn_entries)}")
    else:
        student.save_pretrained(out, safe_serialization=True)
        try:
            tokenizer.save_pretrained(out)
        except Exception:
            pass

    report = {
        "source": model_id_or_path,
        "out_dir": str(out),
        "config": {
            "rank_k": cfg.rank_k, "sink_T": cfg.sink_T,
            "lora_rank": cfg.lora_rank, "lora_alpha": cfg.lora_alpha,
            "steps": cfg.steps, "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len, "learning_rate": cfg.learning_rate,
            "dtype": cfg.dtype, "loss_type": cfg.loss_type,
            "kl_temperature": cfg.kl_temperature if cfg.loss_type == "kl" else None,
        },
        "per_layer": getattr(student, "_hyperretro_stats", {}),
        "n_layers_distilled": sum(
            1 for k in getattr(student, "_hyperretro_stats", {})
            if k.startswith("layer_")
        ),
    }
    (out / "hyperretro_distill_report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(f"[distill] Done. Report: {out / 'hyperretro_distill_report.json'}")
    return report


# ===========================================================================
# CLI
# ===========================================================================


def _cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hyperretro-distill",
        description="GRC Light Distillation: compress + LoRA-correct a HF model",
    )
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID or local path")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--rank", type=int, default=1024,
                   help="GRC rank k (default: 1024)")
    p.add_argument("--sink", type=int, default=0,
                   help="Sink channels to exempt (default: 0)")
    p.add_argument("--lora-rank", type=int, default=8,
                   help="LoRA adapter rank (default: 8)")
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--steps", type=int, default=200,
                   help="Distillation steps (default: 200)")
    p.add_argument("--batch", type=int, default=4,
                   help="Batch size (default: 4)")
    p.add_argument("--seq-len", type=int, default=256,
                   help="Sequence length (default: 256)")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate (default: 1e-4)")
    p.add_argument("--corpus", default=None,
                   help="Path to calibration corpus .txt (optional)")
    p.add_argument("--layers", type=str, default="",
                   help="Comma-separated layer indices (default: all)")
    p.add_argument("--device", default="cuda",
                   help="Device: cuda or cpu")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--loss", default="mse",
                   choices=["mse", "kl", "margin", "behavioral_residue"],
                   help="Loss: mse (logit MSE), kl (KL divergence), "
                   "margin (ranking — directly optimizes argmax match), "
                   "behavioral_residue (confidence-weighted KL — focuses on "
                   "positions where teacher is decisive)")
    p.add_argument("--kl-temperature", type=float, default=4.0,
                   help="Temperature for KL/behavioral-residue loss (default: 4.0)")
    p.add_argument("--margin", type=float, default=1.0,
                   help="Margin for ranking loss (default: 1.0)")
    p.add_argument("--margin-topk", type=int, default=5,
                   help="Top-k teacher tokens for margin loss (default: 5)")
    p.add_argument("--br-confidence-exp", type=float, default=1.0,
                   help="Exponent on teacher max-prob weights for behavioral_residue (default: 1.0)")
    p.add_argument("--br-min-confidence", type=float, default=0.0,
                   help="Ignore positions where teacher max-prob is below this (default: 0)")
    p.add_argument("--ffn-rank-in", type=int, default=0,
                   help="SVD rank for FFN input matrices (gate/up or c_fc). "
                   "0 disables FFN compression.")
    p.add_argument("--ffn-rank-out", type=int, default=0,
                   help="SVD rank for FFN output matrix (down or c_proj). "
                   "0 disables that slot.")
    p.add_argument("--revision", default=None)
    p.add_argument("--factored", action="store_true",
                   help="Save in HyperRetro factored format (post-distill SVD)")
    p.add_argument("--factored-ffn-rel-tol", type=float, default=1e-4,
                   help="Adaptive-rank relative tolerance for factored FFN")
    p.add_argument("--save-dtype", default=None,
                   help="Override save dtype (e.g. bfloat16). Defaults to --dtype.")
    p.add_argument("--activation-aware", action="store_true",
                   help="Use activation-aware SVD for factored FFN (round 13)")
    p.add_argument("--activation-corpus", default=None,
                   help="Calibration corpus for activation norms (defaults to --corpus)")
    p.add_argument("--activation-n-batches", type=int, default=16)
    p.add_argument("--activation-seq-len", type=int, default=512)
    args = p.parse_args(argv)

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    report = distill_hf_model(
        args.model, args.out,
        rank_k=args.rank, sink_T=args.sink,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        steps=args.steps, batch_size=args.batch, seq_len=args.seq_len,
        learning_rate=args.lr,
        corpus_path=args.corpus,
        layers=layers,
        device=args.device, dtype=args.dtype,
        loss_type=args.loss, kl_temperature=args.kl_temperature,
        margin_value=args.margin, margin_topk=args.margin_topk,
        br_confidence_exp=args.br_confidence_exp,
        br_min_confidence=args.br_min_confidence,
        ffn_rank_in=args.ffn_rank_in,
        ffn_rank_out=args.ffn_rank_out,
        revision=args.revision,
        factored=args.factored,
        factored_ffn_rel_tol=args.factored_ffn_rel_tol,
        save_dtype=args.save_dtype,
        activation_aware=args.activation_aware,
        activation_corpus_path=args.activation_corpus,
        activation_n_batches=args.activation_n_batches,
        activation_seq_len=args.activation_seq_len,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
