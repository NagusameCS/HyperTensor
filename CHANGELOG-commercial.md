# Commercial-grade infrastructure — May 2026

This change-set fills the eight commercial-viability gaps identified in the
product audit. All pieces are pure Python (stdlib + optional `torch` /
`huggingface_hub`) and pass live smoke tests.

## What's new

| # | Gap | Delivered as |
|---|---|---|
| 1 | C runtime not pip-installable | New `hypertensor-runtime` package + `deploy/cibuildwheel.toml` + `deploy/build_native.sh` |
| 2 | No Docker image for the API | `deploy/Dockerfile.ht-repro` (lean image) + existing top-level `Dockerfile` (research stack) |
| 3 | No REST API | `ht_repro/api_v1.py` — 10 endpoints under `/api/v1/`, optional Bearer auth via `HT_REPRO_TOKEN` |
| 4 | No persistent GTC/manifold storage | `ht_repro/storage.py` — SQLite at `~/.ht-repro/store.db` with runs, GTC trajectories, manifold KV, model registry |
| 5 | No cloud recipe | `deploy/docker-compose.yml`, `deploy/nginx.conf`, `deploy/ht-repro.service`, `deploy/terraform/main.tf` |
| 6 | Manual model download | `ht_repro/models.py` — HF auto-fetch with disk cache + registry; `POST /api/v1/models/pull` |
| 7 | Grafting needs pre-downloaded models | `ht_repro/graft_wrapper.py` — new `ht-graft` console script that resolves donor + recipient automatically |
| 8 | Hard-coded CUDA | `ht_repro/gpu.py` — backend detection (CUDA / ROCm / MPS / CPU), `device()`, `@gpu_compatible`, `env_for_subprocess()` |

Plus `ht_repro/runtime_loader.py` to locate the geodessical binary across installs.

## Live verification

```text
$ python -m ht_repro.gpu
CUDA — NVIDIA GeForce RTX 4070 Laptop GPU (8.0 GB, 1 device(s))

$ curl http://127.0.0.1:8772/api/v1/health
{"ok": true, "ts": 1778722762.0, "backend": "cuda", "db": "...\\store.db"}

$ curl -X POST -H 'Content-Type: application/json' \
       -d '{"data":[3,1,4,1,5,9,2,6]}' http://127.0.0.1:8772/api/v1/sort
{"engine": "fallback:sorted", "result": [1,1,2,3,4,5,6,9], "n": 8}
```

## CLI additions

- `ht-repro serve --port 8765 --daemon --no-browser`
- `ht-graft DONOR_REPO RECIPIENT_REPO [--cpu] [...]` — autoresolves both models

## Versions

- `ht-repro` 1.0.0 → 1.1.0
- `hypertensor-runtime` 0.1.0 (new)
