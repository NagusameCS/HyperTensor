# HyperTensor Pipeline Dashboard --- May 1, 2026

## Models
| Model | Status | Steps | Time | Loss |
|-------|--------|-------|------|------|
| M (Pure Math) |  COMPLETE | 10,000 | 2h 26m | 2.215 |
| L (Pure Language) |  TRAINING | 6% | ~2h remaining | --- |

## Experiments (30 total: 8  + 1  + 11  + 10 )

### Verified (8)
| ID | Paper | Result |
|----|-------|--------|
| A1 | I | Safe frontier k≥512 (+14.4%), NOT k≥256 (+1486%) |
| E1 | V | Per-matrix SVD: +79% SmolLM2, +19% Llama-8B |
| G1 | VII | Cluster compression: 20.9-25.0% improvement |
| G2 | VII | FFN cluster PPL measured (EC2 L40S) |
| H1 | VIII | GTC 1121 faster than RAG (claimed 15.5) |
| I4 | IX | KV-cache: 15.8 VRAM savings at 32K context |
| I5 | IX | LoRA FFN fusion: +76.6% at k=256 |
| J1 | X | Within-model CECI: FALSIFIED (0/120 at k=32) |

### Running
| ID | Paper | What |
|----|-------|------|
| J2 | X | CECI cross-model splice (after Model L) |

### Pending (scripts ready)
| ID | Paper | What | Needs |
|----|-------|------|-------|
| E2 | V | Distillation PPL | Running on EC2 |
| F1 | VI | Task benchmarks | Queued on EC2 |
| F2-F4 | VI | GSM8K, HumanEval, safe frontier | After F1 |
| I1-I3,I6 | IX | Cross-GPU | Needs A100/H100 |

### Blueprints (Papers XI-XV)
| ID | Paper | Requires |
|----|-------|----------|
| K1-K2 | XI | UGT: taxonomic training (needs custom loss) |
| L1-L2 | XII | Native k-space: Riemannian optimizer |
| M1-M2 | XIII | OGD: UGT prerequisite |
| N1-N2 | XIV | Sniping: UGT prerequisite |
| O1-O2 | XV | COG: full stack prerequisite |

## Papers (I-XV)
| Part | Title | Status |
|------|-------|--------|
| I | GRC Attention Compression |  Compiled |
| II | Geodesic Projection Pipeline |  Compiled |
| III | Geodesic Speculative Decoding |  Compiled |
| IV | OTT/GTC Manifold Runtime |  Compiled |
| V | GRC Light Distillation |  Compiled |
| VI | Task-Level Impact |  Structural predictions |
| VII | FFN Cluster Compression |  PPL pending |
| VIII | GTC as Vector DB |  Deployment pending |
| IX | Super-Baseline Universality |  Cross-GPU pending |
| X | CECI Chimeric Splicing |  Data collected |
| XI | Universal Geodesic Taxonomy |  Blueprint |
| XII | Geodesic Compiler |  Blueprint |
| XIII | Orthogonal Geodesic Deviation |  Blueprint |
| XIV | Geodesic Sniping |  Blueprint |
| XV | Completely Organic Generation |  Blueprint |

## Resources
| Resource | Load | Task |
|----------|------|------|
| RTX 4070 (8.6GB) | 26% | Model L training |
| EC2 L40S (46GB) | Active | E2 + F1 experiments |
