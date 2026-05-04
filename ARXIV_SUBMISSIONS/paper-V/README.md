# Paper E --- Light Distillation for Calibration-Permitted GRC

Status: Scaffold (v0.2). The Phase 1 reference implementation is
operational; the Phase 2 PyTorch runner is checked-in as a scaffold and
has not yet executed. Empirical numbers in `sec:status` are predictions,
not measurements. The paper is published in this scaffolded form so the
protocol can be reviewed before the run, not after.

Headline target. A 50--60% closure of the Paper A perplexity gap
(measured: +13.30% PPL at k=1536, 6.79 -> 7.69 on WikiText-2) using a
single opt-in distillation pass against a frozen teacher, at unchanged
runtime structure (kernel-fusion path of Paper A is preserved).

## Relationship to Paper A

This paper is a strict extension of Paper A. Disabling distillation
reproduces Paper A's GRC numbers exactly. The default merge strategy
(fused base + side-LoRA, see Sec. `sec:fusion-fit`) preserves the
+6.27% throughput gain at k=1024.

## Coverage map (for reviewers)

| Item | Section |
|---|---|
| Method (Phase 1: GRC projection) | §2.1 |
| Method (Phase 2: teacher--student LoRA) | §2.2 |
| Method (Phase 3: merge and re-quantise) | §2.3 |
| First-order PPL gap-closure bound | §3 |
| Fusion-path preservation | §4 |
| Falsifiable predictions P1, P2, P3 | §5 |
| Status / pending items | §6 |
| Threat model and limitations | §7 |
| Reproducibility (CPU-only and EC2 paths) | §8 |

## What is and is not in this paper

In scope:
- A protocol that augments calibration-free GRC with a small post-projection
  correction.
- A constructive bound on the achievable gap closure, computable at zero
  runtime cost (`scripts/grc_distill.py --print-rho`).
- Three documented merge strategies, each with a known throughput cost.
- Three pre-registered predictions that the empirical run will adjudicate.

Out of scope:
- Generalisation beyond WikiText-2 (single calibration corpus).
- Generalisation beyond Llama-3.1-8B (Phase 2 runner is architecture-specific).
- Generalisation beyond Q4_K_M (re-quantisation determinism is assumed).
- Comparison against full LoRA fine-tuning (different question, different cost).

## Build

```bash
latexmk -pdf -interaction=nonstopmode grc-light-distillation.tex
```

## arXiv prep (from the parent folder)

```bash
make submit-E   # produces ../dist/paper-E.tar.gz with .bbl and .sty
```

## Reproduction

CPU-only Phase 1 reference (60--120 s on a Ryzen 9 7940HS):

```powershell
python scripts/grc_distill.py `
  --model models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf `
  --rank 1024 --print-rho `
  --out distill_out/
```

GPU-required Phase 2 (24 GB VRAM, A10G/L4/A100; pending runner):

```bash
python scripts/grc_distill.py \
  --model models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --rank 1024 --distill --lora-rank 8 --steps 500 \
  --corpus calibration/wikitext_calib_2k.txt \
  --device cuda --out distill_out/
```

A pre-built W_proj cache from the matching Paper A configuration can be
fetched from the project Releases page to skip the calibration-free
projection step:

```powershell
gh release download wproj-cache-2405A3B6 --repo NagusameCS/HyperTensor --pattern '*.bin'
```
