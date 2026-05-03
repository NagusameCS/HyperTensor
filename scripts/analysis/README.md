# hypertensor-analysis

Reproducibility scripts for the HyperTensor / Geodesic Runtime Compression (GRC) report
by **William Ken Ohara Stewart** (NagusameCS).

This is a small companion package that bundles the three analysis scripts used to produce
the spectral, statistical, and Eckart\u2013Young figures in the report.

## Install (editable, from the repo)

```bash
git clone https://github.com/NagusameCS/HyperTensor.git
cd HyperTensor/scripts/analysis
pip install -e .
```

## Console entry points

After installation the following commands are available on `PATH`:

| Command                | Source script              | Purpose |
| ---                    | ---                        | --- |
| `hypertensor-spectra`  | `compute_spectra.py`       | SVD all attention + FFN matrices for layers 0/7/15/23/31; emits 3 PNGs and `spectra_summary.json`. |
| `hypertensor-stats`    | `statistical_tests.py`     | Paired t-test, Wilcoxon, bootstrap CI on `rank_sweep_relative_to_baseline.csv`. |
| `hypertensor-eckart`   | `eckart_young_bound.py`    | Eckart\u2013Young oracle vs GRC shared-basis comparison; layers 0/15/31; ranks 512\u20132048. |

Each script also remains runnable as a plain module: `python compute_spectra.py`, etc.

## Inputs

The scripts expect the GGUF model file path and the benchmark CSVs at the locations described
in `repro/QUICKSTART.md` at the repo root. See the report's
[Reproduce section](https://nagusamecs.github.io/HyperTensor/#reproduce) for the full recipe.

## Licence

MIT. See repository root.
