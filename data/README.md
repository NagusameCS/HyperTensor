# External Number-Theory Data Cache

Cached snapshots of public datasets used by `scripts/riemann_pipeline.py`,
`scripts/l_functions.py`, and `scripts/data_sources.py`.

## `odlyzko/` — Andrew Odlyzko's tables of Riemann ζ zeros

Source: <https://www-users.cse.umn.edu/~odlyzko/zeta_tables/>

| File       | Contents                                                                    |
|------------|-----------------------------------------------------------------------------|
| `zeros1`   | First 100,000 zeros (imaginary part), accurate to ~3·10⁻⁹                   |
| `zeros1.gz`| Gzipped form of `zeros1`                                                    |
| `zeros2`   | First 100 zeros to >1000 decimal places (line-wrapped)                      |
| `zeros3`   | Zeros 10¹²+1 … 10¹²+10⁴, given as γ − 267 653 395 647                       |
| `zeros4`   | Zeros 10²¹+1 … 10²¹+10⁴ (offset noted in file header)                       |
| `zeros5`   | Zeros 10²²+1 … 10²²+10⁴ (offset noted in file header)                       |

`zeros6` (35 MB, 2 001 052 zeros) is **not** cached locally; download on demand.

Re-fetch with:

```powershell
$base = "https://www-users.cse.umn.edu/~odlyzko/zeta_tables"
foreach ($f in "zeros1.gz","zeros2","zeros3","zeros4","zeros5") {
    Invoke-WebRequest "$base/$f" -OutFile "data\odlyzko\$f"
}
python -c "import gzip,shutil; shutil.copyfileobj(gzip.open('data/odlyzko/zeros1.gz','rb'), open('data/odlyzko/zeros1','wb'))"
```

## `lmfdb/` — L-Functions and Modular Forms Database snapshots

Source: <https://www.lmfdb.org/api/>  (re-fetch with `python scripts/pull_lmfdb.py`)

| File                          | Records | Description                                  |
|-------------------------------|---------|----------------------------------------------|
| `lfunctions_deg1.json`        | 300     | Degree-1 L-functions incl. Riemann ζ + Dirichlet L; field `positive_zeros` holds zero ordinates |
| `lfunctions_deg2.json`        | 100     | Degree-2 L-functions (modular / GL(2))       |
| `dirichlet_small.json`        | 4       | Dirichlet characters of modulus 7            |
| `nf_fields_low_degree.json`   | 100     | Number fields of degree 2–5 (for Dedekind ζ) |
| `mf_gamma1_wt2.json`          | 100     | Γ₁ classical newforms, weight 2              |
| `maass_newforms.json`         | 100     | Maass newforms                               |

Loader API: `scripts/data_sources.py`

```python
from data_sources import load_riemann_zeros, load_lmfdb_zeros
load_riemann_zeros(100)              # first 100 ζ-zeros from Odlyzko
load_lmfdb_zeros(degree=1, limit=50) # flatten zeros across degree-1 L-fns
```

## License / attribution

* Odlyzko tables: courtesy of Andrew Odlyzko, freely redistributable for
  research use; cite *A. M. Odlyzko, "Tables of zeros of the Riemann zeta
  function".*
* LMFDB content: CC BY-SA 4.0 (per <https://www.lmfdb.org/license>); cite
  the LMFDB Collaboration, "The L-functions and modular forms database",
  <https://www.lmfdb.org>.
