# ht-repro — HyperTensor Reproduction CLI

**One command to reproduce any paper in the extended volume.**

[![PyPI](https://img.shields.io/badge/pypi-ht--repro-blue)](https://pypi.org/project/ht-repro/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

##  Install (any method)

```bash
# pip (recommended)
pip install ht-repro

# pipx (isolated)
pipx install ht-repro

# One-line (any OS)
curl -sSL https://nagusamecs.github.io/HyperTensor/install.sh | bash   # Linux/macOS
irm https://nagusamecs.github.io/HyperTensor/install.ps1 | iex         # Windows

# From source
git clone https://github.com/NagusameCS/HyperTensor
cd HyperTensor/ht_repro && pip install -e .
```

## � One-Click UI

```bash
# Double-click in File Explorer:
ht-repro-ui.bat

# Or from terminal:
python -m ht_repro serve

# Pin to desktop (run once):
powershell -File create-shortcuts.ps1
```

Opens `http://localhost:8765` in your browser — live test runner with streaming output, GPU config, stop button, and full history.
##  Quick Start

```bash
ht-repro setup          # auto-detect environment, install deps
ht-repro smoke          # 60-second Riemann core math test
ht-repro list           # show all available tests
ht-repro summary        # print verified results summary
```bash
ht-repro setup          # auto-detect environment, install deps
ht-repro smoke          # 60-second Riemann core math test
ht-repro list           # show all 16 available tests
ht-repro summary        # print verified results summary
```

##  Commands

| Command | Description |
|---------|-------------|
| `ht-repro smoke` | 60-second Riemann core math test |
| `ht-repro all-t1` | All CPU-only tests (~30 min) |
| `ht-repro paper-N` | Reproduce a specific paper (1–18) |
| `ht-repro jury` | All jury theorem verification |
| `ht-repro riemann` | All Riemann Hypothesis verification |
| `ht-repro safety` | All safety verification |
| `ht-repro runtime` | All runtime benchmarks |
| `ht-repro list` | Show all available tests |
| `ht-repro status` | Show last run results |
| `ht-repro summary` | Print verified results summary |
| `ht-repro setup` | Auto-detect environment + install deps |
| `ht-repro serve` | Start localhost web UI (http://localhost:8765) |
| `ht-repro tools` | List all 9 tool categories (60+ tools) |
| `ht-repro tools graft` | Model grafting/splicing tools |
| `ht-repro tools bench` | Benchmarking tools |
| `ht-repro tools train` | Training tools (NativeLinear, SHF, LoRA) |
| `ht-repro tools compress` | Compression tools (GRC, FFN cluster, SVD) |
| `ht-repro tools gtc` | GTC/manifold tools (trajectories, Jacobi) |
| `ht-repro tools safety` | Safety tools (OGD, TEH, Snipe, Red-team) |
| `ht-repro tools ugt` | UGT taxonomy tools (bilateral, zone mapping) |
| `ht-repro tools models` | Model download, HF token, Ollama, GPU check |
| `ht-repro tools models token-setup` | Configure HuggingFace token interactively |
| `ht-repro update` | Self-update to latest version |
| `ht-repro run <id>` | Run a specific test by ID |

##  Verified Results (2026-05-13)

| Test | Status |
|------|--------|
| Riemann Core Math |  SV1=8.944272, Z₂ EXACT, rank-1 proven |
| Jury Proof |  8 theorems, 174× speedup |
| Riemann LMFDB |  54,949 zeros on critical line |
| AGT v3 |  98% detection, 1392× separation |
| Safe OGD |  0% forbidden leakage |
| GTC vs RAG |  30.9 µs/q, 5.96 KB/record |
| BP/NS Bound |  160/160 trials pass |

##  Web Dashboard

```bash
ht-repro dashboard    # generates benchmarks/ht_repro_dashboard.html
```

Open the generated HTML file in any browser for a full visual report with run history, environment info, and test catalog.

##  License

MIT — see [LICENSE](https://github.com/NagusameCS/HyperTensor/blob/main/LICENSE)
