#!/usr/bin/env python3
"""
Phase 2 distillation runner for Paper E (GRC Light Distillation).

Loads a HuggingFace teacher model in BF16, applies GRC projection to
attention Q/K/V weights (via the NumPy scaffold in grc_distill.py),
wraps each projected attention layer with trainable rank-r LoRA adapters,
and minimises teacher-student logit MSE on a small calibration corpus.

Usage (local smoke test, SmolLM2-135M):
  python scripts/distill_runner.py \
    --teacher HfCas/SmolLM2-135M-Instruct \
    --gguf models/smollm2-135m-instruct-q8_0.gguf \
    --corpus data/wikitext2_train_5k.txt \
    --out benchmarks/paper_e_distill/smollm2_k256_r8 \
    --rank 256 --lora-rank 8 --steps 200 --batch 4 --seq-len 256

Usage (EC2, Llama-3.1-8B):
  python scripts/distill_runner.py \
    --teacher meta-llama/Llama-3.1-8B-Instruct \
    --gguf models/llama3.1-8b-instruct-q4_k_m.gguf \
    --corpus data/wikitext2_train_10k.txt \
    --out benchmarks/paper_e_distill/llama8b_k1536_r8 \
    --rank 1536 --lora-rank 8 --steps 500 --batch 8 --seq-len 512

Requires: torch, transformers, datasets, safetensors, numpy, gguf
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

# ---------------------------------------------------------------------------
# Import GRC utilities from the scaffold (same repo)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from grc_distill import (
    build_shared_basis,
    project as grc_project,
    sink_indices,
    _load_attn_weights_gguf,
    _n_layers_gguf,
)


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class DistillConfig:
    """Distillation hyperparameters."""
    rank: int = 1024              # GRC compression rank k
    sink_T: int = 0               # 0 = vanilla GRC; >0 = sink-aware
    lora_rank: int = 8            # rank of LoRA correction adapters
    learning_rate: float = 1e-4
    steps: int = 500
    batch_size: int = 8
    seq_len: int = 512
    warmup_steps: int = 100
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    device: str = "cuda"
    dtype: str = "bfloat16"
    # subset of layers to train (empty = all attention layers)
    layers: list[int] = field(default_factory=list)
    # LoRA init scale
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    # log interval
    log_every: int = 50
    # checkpoint
    save_every: int = 0           # 0 = only at end


# ===========================================================================
# GRC projection (NumPy, from GGUF reference)
# ===========================================================================

def compute_grc_projections(gguf_path: str, cfg: DistillConfig) -> dict:
    """
    Compute GRC-projected Q/K/V weights for every layer from the GGUF.
    Returns dict: layer_idx -> {"Wq_proj": np, "Wk_proj": np, "Wv_proj": np,
                                 "P": np, "sink": np}
    """
    n_layers = _n_layers_gguf(gguf_path)
    print(f"[grc] {gguf_path}: {n_layers} layers, k={cfg.rank}, sink_T={cfg.sink_T}")
    projections = {}
    for layer in range(n_layers):
        Wq, Wk, Wv = _load_attn_weights_gguf(gguf_path, layer)
        sink = sink_indices(Wq, Wk, Wv, cfg.sink_T)
        if cfg.sink_T > 0:
            Wq_R = Wq.copy(); Wq_R[:, sink] = 0.0
            Wk_R = Wk.copy(); Wk_R[:, sink] = 0.0
            Wv_R = Wv.copy(); Wv_R[:, sink] = 0.0
            P = build_shared_basis(Wq_R, Wk_R, Wv_R)
            Wq_proj = grc_project(Wq_R, P, cfg.rank); Wq_proj[:, sink] = Wq[:, sink]
            Wk_proj = grc_project(Wk_R, P, cfg.rank); Wk_proj[:, sink] = Wk[:, sink]
            Wv_proj = grc_project(Wv_R, P, cfg.rank); Wv_proj[:, sink] = Wv[:, sink]
        else:
            P = build_shared_basis(Wq, Wk, Wv)
            Wq_proj = grc_project(Wq, P, cfg.rank)
            Wk_proj = grc_project(Wk, P, cfg.rank)
            Wv_proj = grc_project(Wv, P, cfg.rank)
        projections[layer] = {
            "Wq_proj": Wq_proj.astype(np.float32),
            "Wk_proj": Wk_proj.astype(np.float32),
            "Wv_proj": Wv_proj.astype(np.float32),
            "P": P.astype(np.float32),
            "sink": sink,
        }
        if layer % 8 == 0:
            print(f"  [grc] layer {layer}/{n_layers}", flush=True)
    return projections


# ===========================================================================
# LoRA adapter module
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
        # Freeze base
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

        # LoRA factors
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
        """Return LoRA delta weight: B @ A (for merge/export)."""
        return (self.lora_B @ self.lora_A) * self.scaling


# ===========================================================================
# Student model builder
# ===========================================================================

def _find_attn_modules(model: nn.Module) -> dict[int, dict[str, nn.Linear]]:
    """Find attention Q/K/V projection Linear layers, keyed by layer index."""
    import re
    attn_modules: dict[int, dict[str, nn.Linear]] = {}

    for name, module in model.named_modules():
        # SmolLM2 / Llama pattern: model.layers.N.self_attn.{q_proj,k_proj,v_proj}
        m = re.match(
            r".*layers?\.(\d+)\.(?:self_)?attn\.(q_proj|k_proj|v_proj)$", name
        )
        if m and isinstance(module, nn.Linear):
            layer_idx = int(m.group(1))
            slot = m.group(2)[0].upper()  # q_proj -> Q
            attn_modules.setdefault(layer_idx, {})[slot] = module
    return attn_modules


def build_student_model(teacher: nn.Module, projections: dict,
                        cfg: DistillConfig) -> nn.Module:
    """
    Build the student model: replace attention Q/K/V with GRC-projected
    weights + trainable LoRA adapters. Returns the modified teacher
    (in-place) with LoRA parameters registered.
    """
    attn_modules = _find_attn_modules(teacher)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

    target_layers = cfg.layers if cfg.layers else sorted(projections.keys())
    lora_params = []

    for layer_idx in target_layers:
        if layer_idx not in projections:
            print(f"  [student] skipping layer {layer_idx}: no GRC projection")
            continue
        if layer_idx not in attn_modules:
            print(f"  [student] skipping layer {layer_idx}: no attn modules found")
            continue

        proj = projections[layer_idx]
        attn = attn_modules[layer_idx]

        for slot in ("Q", "K", "V"):
            if slot not in attn:
                continue
            base_linear = attn[slot]
            proj_key = f"W{slot.lower()}_proj"

            if proj_key not in proj:
                continue

            # Replace the base weight with GRC-projected weight
            W_proj_np = proj[proj_key]
            with torch.no_grad():
                base_linear.weight.copy_(
                    torch.from_numpy(W_proj_np).to(
                        device=base_linear.weight.device,
                        dtype=base_linear.weight.dtype,
                    )
                )

            # Create LoRA wrapper
            lora_linear = LoRALinear(
                base_linear, r=cfg.lora_rank, alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
            )
            lora_linear = lora_linear.to(dtype=torch_dtype)

            # Replace the module in-place
            # Navigate to parent and replace
            parent_name = f"model.layers.{layer_idx}.self_attn.{slot.lower()}_proj"
            _replace_module(teacher, parent_name, lora_linear)

            # Track LoRA params
            lora_params.extend([
                lora_linear.lora_A,
                lora_linear.lora_B,
            ])

        if layer_idx % 8 == 0:
            print(f"  [student] layer {layer_idx}: Q/K/V LoRA attached", flush=True)

    total_lora = sum(p.numel() for p in lora_params)
    print(f"[student] total LoRA params: {total_lora:,}  "
          f"({total_lora * 2 / 1024 / 1024:.1f} MB in {torch_dtype})")
    return teacher


def _replace_module(model: nn.Module, dotted_name: str, new_module: nn.Module):
    """Replace a submodule by dotted path like 'model.layers.0.self_attn.q_proj'."""
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


# ===========================================================================
# Calibration dataset
# ===========================================================================

class TextChunkDataset(IterableDataset):
    """Stream text chunks from a plain-text corpus file."""

    def __init__(self, corpus_path: str, tokenizer, seq_len: int = 512,
                 stride: int = 256, max_samples: Optional[int] = None):
        self.corpus_path = corpus_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.max_samples = max_samples

    def __iter__(self):
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Truncate to roughly the needed size
        if self.max_samples:
            chars_per_token = 4  # rough heuristic
            max_chars = self.max_samples * self.seq_len * chars_per_token
            text = text[:max_chars]

        tokens = self.tokenizer.encode(text)
        count = 0
        for i in range(0, len(tokens) - self.seq_len, self.stride):
            chunk = tokens[i : i + self.seq_len]
            if len(chunk) < self.seq_len:
                break
            yield torch.tensor(chunk, dtype=torch.long)
            count += 1
            if self.max_samples and count >= self.max_samples:
                break


# ===========================================================================
# Training loop
# ===========================================================================

def distill(teacher: nn.Module, student: nn.Module,
            dataloader, cfg: DistillConfig, out_dir: Path):
    """Run the teacher-student distillation loop."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

    teacher = teacher.to(device=device, dtype=torch_dtype)
    student = student.to(device=device, dtype=torch_dtype)
    teacher.eval()

    # Collect trainable params (LoRA factors only)
    trainable = [p for p in student.parameters() if p.requires_grad]
    print(f"[train] trainable params: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    # Cosine LR schedule
    def lr_lambda(step: int):
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss: MSE on logits
    loss_fn = nn.MSELoss()

    data_iter = iter(dataloader)
    total_loss = 0.0
    total_steps = 0
    t0 = time.time()

    out_dir.mkdir(parents=True, exist_ok=True)
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
            tok_per_s = (total_steps * cfg.batch_size * cfg.seq_len) / max(elapsed, 0.1)
            lr = scheduler.get_last_lr()[0]
            msg = (f"  step {step+1:4d}/{cfg.steps}  "
                   f"loss={avg_loss:.6f}  lr={lr:.2e}  "
                   f"tok/s={tok_per_s:.0f}")
            print(msg, flush=True)
            log_lines.append(msg)
            total_loss = 0.0
            total_steps = 0

        if cfg.save_every > 0 and (step + 1) % cfg.save_every == 0:
            _save_checkpoint(student, out_dir, step + 1, optimizer)

    elapsed = time.time() - t0
    print(f"[train] done in {elapsed:.1f}s  ({elapsed/60:.1f} min)", flush=True)

    # Final save
    _save_checkpoint(student, out_dir, cfg.steps, optimizer)

    # Write training log
    with open(out_dir / "train_log.txt", "w") as f:
        f.write("\n".join(log_lines))

    return student


def _save_checkpoint(student: nn.Module, out_dir: Path, step: int,
                     optimizer=None):
    """Save LoRA adapter weights as safetensors."""
    from safetensors.torch import save_file as safe_save

    lora_weights = {}
    for name, param in student.named_parameters():
        if param.requires_grad and "lora_" in name:
            safe_name = name.replace(".", "_")
            lora_weights[safe_name] = param.detach().cpu()

    if not lora_weights:
        print("[save] no LoRA weights found --- nothing to save")
        return

    path = out_dir / f"lora_adapters_step{step}.safetensors"
    safe_save(lora_weights, path)
    print(f"[save] wrote {path}  ({len(lora_weights)} tensors)", flush=True)


# ===========================================================================
# PPL evaluation helper
# ===========================================================================

def evaluate_ppl(model: nn.Module, dataloader, cfg: DistillConfig,
                 max_batches: int = 20) -> float:
    """Quick PPL evaluation on a held-out slice."""
    device = next(model.parameters()).device
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = batch.to(device)
            out = model(batch)
            logits = out.logits[:, :-1, :]  # predict next token
            targets = batch[:, 1:]
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_tokens += targets.numel()

    if total_tokens == 0:
        return float("inf")
    return np.exp(total_nll / total_tokens)


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="GRC Light Distillation --- Phase 2 PyTorch Runner"
    )
    # Model paths
    ap.add_argument("--teacher", required=True,
                    help="HuggingFace model ID or path for the TEACHER (uncompressed)")
    ap.add_argument("--gguf", required=True,
                    help="Path to GGUF file for GRC projection reference")
    ap.add_argument("--corpus", required=True,
                    help="Path to calibration corpus (.txt, WikiText-2 train split)")
    ap.add_argument("--eval-corpus", default=None,
                    help="Optional held-out corpus for PPL evaluation")
    ap.add_argument("--out", default="benchmarks/paper_e_distill/run",
                    help="Output directory")
    # GRC
    ap.add_argument("--rank", type=int, default=1024)
    ap.add_argument("--sink-T", type=int, default=0)
    # LoRA
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    # Training
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=8, dest="batch_size")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100, dest="warmup_steps")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--layers", type=str, default="",
                    help="Comma-separated layer indices (empty=all)")
    # Hardware
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    # Checkpoint
    ap.add_argument("--save-every", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    # Modes
    ap.add_argument("--eval-only", action="store_true",
                    help="Skip training, only run PPL eval on existing adapters")
    ap.add_argument("--dry-run", action="store_true",
                    help="Load model, apply GRC, build student, exit without training")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = DistillConfig(
        rank=args.rank, sink_T=args.sink_T, lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, learning_rate=args.lr,
        steps=args.steps, batch_size=args.batch_size, seq_len=args.seq_len,
        warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm, device=args.device, dtype=args.dtype,
        layers=[int(x) for x in args.layers.split(",") if x] if args.layers else [],
        log_every=args.log_every, save_every=args.save_every,
    )

    print(f"[distill] config: {json.dumps(cfg.__dict__, indent=2, default=str)}",
          flush=True)

    # ---- Phase 1: GRC projection ----
    print("\n=== Phase 1: GRC projection ===")
    projections = compute_grc_projections(args.gguf, cfg)

    # ---- Load teacher ----
    print(f"\n=== Loading teacher: {args.teacher} ===")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype={"bfloat16": torch.bfloat16, "float16": torch.float16,
                     "float32": torch.float32}[cfg.dtype],
        device_map="auto" if cfg.device == "cuda" else None,
    )
    print(f"[teacher] {sum(p.numel() for p in teacher.parameters()):,} params")

    # ---- Build student ----
    print("\n=== Building student ===")
    student = build_student_model(teacher, projections, cfg)

    if args.dry_run:
        print("[dry-run] done. Exiting without training.")
        manifest = {"config": cfg.__dict__, "stage": "dry_run_ok"}
        with open(out_dir / "distill_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return

    # ---- Calibration dataset ----
    print(f"\n=== Calibration corpus: {args.corpus} ===")
    if not os.path.exists(args.corpus):
        print(f"[distill] ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(2)

    num_batches = cfg.steps
    max_samples = num_batches * cfg.batch_size
    dataset = TextChunkDataset(
        args.corpus, tokenizer, seq_len=cfg.seq_len,
        max_samples=max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size)

    # ---- Baseline PPL (pre-distill) ----
    print("\n=== Pre-distillation PPL ===")
    pre_ppl = evaluate_ppl(student, dataloader, cfg, max_batches=10)

    # ---- Distill ----
    print(f"\n=== Distillation: {cfg.steps} steps ===")
    student = distill(teacher, student, dataloader, cfg, out_dir)

    # ---- Post-distill PPL ----
    print("\n=== Post-distillation PPL ===")
    post_ppl = evaluate_ppl(student, dataloader, cfg, max_batches=10)

    # ---- Summary ----
    summary = {
        "config": cfg.__dict__,
        "pre_distill_ppl": round(pre_ppl, 4),
        "post_distill_ppl": round(post_ppl, 4),
        "ppl_delta_pct": round((post_ppl - pre_ppl) / pre_ppl * 100, 2) if pre_ppl > 0 else None,
        "status": "complete",
    }
    with open(out_dir / "distill_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[done] pre-distill PPL={pre_ppl:.4f}  "
          f"post-distill PPL={post_ppl:.4f}  "
          f"delta={summary['ppl_delta_pct']:+.2f}%")
    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
