# HyperTensor — Reproduction Quickstart

**Three paths.** Pick the one that matches what you want to verify. Each path takes <30 minutes from a clean checkout.

For the full canonical guide see [`../REPRODUCTION.md`](../REPRODUCTION.md). For hardware spec see [`HARDWARE.md`](HARDWARE.md). For the GRC throughput protocol see [`REPRODUCE.md`](REPRODUCE.md).

---

## Path A — CPU-only mathematical verification (15 minutes, no GPU)

Reproduces the **core mathematical claims** of papers XVI–XVIII (D(s) is rank-1, AGT detects critical-line zeros, jury aggregation is universal) plus the entire jury foundation. 27 Riemann tests + ~80 jury tests.

```bash
git clone https://github.com/NagusameCS/HyperTensor.git
cd HyperTensor
python -m venv .venv
.venv\Scripts\activate                    # Windows
source .venv/bin/activate                 # Linux/macOS
pip install numpy scipy mpmath sympy

# Riemann verification (4 scripts, ~15 s)
python scripts/faithfulness_rigorous.py
python scripts/riemann_comprehensive_verify.py
python scripts/riemann_adversarial_tests.py
python scripts/riemann_mega_verify.py

# Jury foundation (5 scripts, ~3 min)
python scripts/jury_discovery.py
python scripts/jury_solver.py
python scripts/jury_advance.py
python scripts/jury_bridge.py
python scripts/jury_gaps.py

# Papers I–XV audit (~3 min)
python scripts/bulletproof_audit.py
python scripts/benchmarks_quick.py
```

**Success criterion:** every script prints a `PASS` block. The faithfulness output ends with `SV1=8.94, SV2..SV12=0.000000`.

---

## Path B — Per-paper transformer analysis (T2, ~1 hour)

Reproduces the per-layer SVD spectra, UGT zone separation, bilateral subspace overlap, native-training compression ratios. Needs a CUDA GPU and HuggingFace `transformers`.

```bash
pip install torch transformers safetensors

# Paper I — analyse any model
python scripts/hypertensorize.py --model Qwen/Qwen2.5-1.5B-Instruct

# Paper II — real SVD spectra (Q/K/V/O × all layers)
python scripts/measure_real_spectra.py

# Paper XI — bilateral UGT at 1.5B
python scripts/close_xi_bilateral_ec2.py

# Paper XII — native geodesic training at 1.5B
python scripts/native_15b_v2.py
```

**Outputs:** under `benchmarks/`. Compare to the committed reference JSONs.

---

## Path C — End-to-end GRC throughput on Llama 3.1 8B (T2, ~75 minutes)

Reproduces the headline 106.27% throughput measurement and the +13.30% PPL delta. Needs Zig CC, CUDA toolkit, the GGUF model.

```powershell
# Build the C runtime
.\build_host.ps1                          # Windows
# expect build_host\geodessical.exe

# Get the model
# (Download Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf from
#  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)
$MODEL = "C:\path\to\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Run the full benchmark suite (40-70 min)
.\scripts\benchmark_whitepaper_finalize.ps1 -Model $MODEL -CooldownSec 30

# Validate gates
.\scripts\validation_cycle.ps1 -PackDir benchmarks\<pack_dir_from_above>
```

See [`REPRODUCE.md`](REPRODUCE.md) for the detailed step-by-step protocol with expected output ranges and the validation gate definitions.

---

## What each path proves

| Path | Hardware | Time | Proves |
|---|---|---|---|
| A | CPU | 15 min | Core math (papers XVI–XVIII), jury universality, papers I–XV claims |
| B | T2 GPU | 1 h | Per-layer geometry, UGT bilateral overlap, native-training ratios |
| C | T2 GPU + GGUF | 75 min | GRC throughput speedup, deterministic PPL delta |

**Comprehensive coverage** = all three paths = ~2.5 hours from clean checkout. After that you have re-derived every published HyperTensor result that does not require a datacenter GPU. The remaining T3 results (7B bilateral, 50K-prime AGT, 10K-COG) ship pre-computed in `benchmarks/`.
