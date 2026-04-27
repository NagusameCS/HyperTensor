# Quickstart: reproducing the HyperTensor / GRC analysis figures

Three commands. They produce the spectral plots, the statistical test JSON, and the
Eckart\u2013Young comparison from the data already shipped in this repository.

## 1. Install the analysis package

From a clean Python 3.10+ virtual environment:

```bash
pip install -e scripts/analysis
```

This makes `hypertensor-spectra`, `hypertensor-stats`, and `hypertensor-eckart`
available as console commands.

## 2. Run the three analyses

```bash
# Spectral analysis (needs the GGUF; see scripts/analysis/compute_spectra.py for path)
hypertensor-spectra

# Statistical tests on the rank sweep CSV (already in the repo)
hypertensor-stats

# Eckart\u2013Young oracle bound
hypertensor-eckart
```

Outputs land in `docs/figures/` and `repro/expected_outputs/` per the script defaults.

## 3. Compare against the published figures

The PNGs and JSON in `docs/figures/` correspond 1:1 to the figures embedded in
[the research site](https://nagusamecs.github.io/HyperTensor) and the
[whitepaper](https://nagusamecs.github.io/HyperTensor/whitepaper.html).

For the full reproduction protocol (including the GGUF download, the `geodessical.exe`
binary, and the throughput benchmark loop) see [`REPRODUCE.md`](REPRODUCE.md) in this
folder.
