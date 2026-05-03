# Session Dossier — 2026-04-29

Complete record of a single Copilot working session covering three new
empirical artefacts, four paper revisions, and a global "third-person
self-reference" cleanup. Provided here as a self-contained folder for
secondary analysis. Nothing in this folder is required by the runtime —
every file is either a copy of an in-repo artefact or a session-specific
log/diff.

## Two git commits land all changes

| SHA | Subject | Diff file |
|---|---|---|
| `d4fef86` | empiricalize: HJB stub + Thermal Rank trace + MCR weight-PCA finding + paper-D curvature-warp v1 negative subsection | [commit_d4fef86.txt](commit_d4fef86.txt), [commit_d4fef86_full_diff.txt](commit_d4fef86_full_diff.txt) |
| `7418a3e` | papers: first-person voice + curvature-warp 0/12 cross-model strengthening | [commit_7418a3e.txt](commit_7418a3e.txt), [commit_7418a3e_full_diff.txt](commit_7418a3e_full_diff.txt) |

`git_log_session.txt` is the raw `git log` for the session window.
`chat_transcript.jsonl` is the raw Copilot chat transcript (line-delimited
JSON, one record per turn) for full reconstruction.

## Folder map

```
hjb/             Paper-D Hamilton–Jacobi–Bellman feasibility stub
thermal/         Paper-B Thermal Rank sustained-decode trace + analysis
mcr/             Paper-B MCR (ε-Margin Compressive Rank) weight-PCA result
curvature_warp/  Paper-D negative result: 0/32 single-model + 0/12 cross-model
scripts/         The new/touched scripts that produced the data above
papers_diffs/    (intentionally empty — see the two commit_*_full_diff.txt files)
```

---

## 1. HJB feasibility stub (`hjb/`)

Goal: was the joint-training Hamilton–Jacobi–Bellman premise of
paper-D §sec:hjb-future operationally testable on existing exported
manifolds, even at zero training? Answer: yes, but the residual is
trivially dominated by the kinetic term (frozen networks export
nearly-flat manifolds), which is the publishable observation.

- `hjb_residual_magnitudes.json` — full per-trajectory residual decomposition for three models  three trajectory classes (baked geodesic via Magnus-3, NN-walk K=4, random straight line). Each row reports `(total, kin, curv)` of the discrete Jacobi residual `J_l(s) = Δ²s_l + R̂(s_l)·Δs_l`.
- `hjb_residual_summary.md` — human-readable summary; the headline number is **kinetic-to-curvature ratio > 10^17 in squared L2 across all three models**.
- Script: `scripts/hjb_feasibility.py` (defaults: `--n-intrinsic 8 --L 32 --n-traj 12 --seed 20260429`).
- Cited from: `ARXIV_SUBMISSIONS/paper-D/ott-gtc-manifold-runtime.tex` §sec:hjb-future "Empirical feasibility stub (2026-04-29)".

## 2. Thermal Rank (`thermal/`)

Goal: produce a real sustained-decode telemetry trace under the rank
controller’s actuation band (≥65 °C) so paper-B §9 Thermal Rank is
no longer purely speculative.

- `thermal_sustained_8b.csv` — nvidia-smi CSV (timestamp, temperature, util%, draw_W) for 4384-token back-to-back decode loop on Llama-3.1-8B Q4_K_M.
- `thermal_sustained_8b.log` — UTF-16 LE Tee-Object capture of the geodessical decode loop.
- `thermal_summary.json` — analyser output: T_max=75 °C, T_p90=74 °C, mean 27.5 tok/s @ 66.3 W → 0.415 tok/J, throughput drift +8.5 % (warm-up, not throttle).
- `thermal_summary.md` — human-readable rendering of the same.
- Script: `scripts/analyze_thermal.py` (handles UTF-16 LE Tee-Object output).
- Cited from: `ARXIV_SUBMISSIONS/paper-B/geodesic-projection-pipeline.tex` §9 "Measured sustained-decode trace (2026-04-29)" and the v0.2-status fbox.

## 3. MCR weight-PCA finding (`mcr/`)

Goal: empiricalize the ε-Margin Compressive Rank claim. The 8B
weight-PCA run was killed (~30–60 s/layer  32 layers), but the
short run captured the actually-paper-relevant negative observation:
flat activation variance after the early layers means weight-PCA
alone cannot supply the rank-selection signal.

- `mcr_ablation.csv`, `mcr_ablation_log.txt`, `mcr_ablation_summary.json` — clean activation-variance run.
- `mcr_ablation_run1_contaminated*.{csv,json}` — first run, contaminated by a cache state from a prior process; retained for reproducibility audit.
- `mcr_ablation_run2_log.txt` — replication log.
- `mcr_8b_ppl.log` (in `thermal/`) — co-located 8B PPL log captured during the same session.
- Cited from: `ARXIV_SUBMISSIONS/paper-B/geodesic-projection-pipeline.tex` §MCR "Weight-PCA flat-fallback finding (2026-04-29)".

## 4. Curvature-warp negative result (`curvature_warp/`)

Goal (per user’s 2026-04-29 directive: *"even if we disprove the
curvature warp that does demonstrate we had a thought, we tested it,
maybe it was wrong but someone can learn from it"*): preserve and
strengthen the negative result rather than hiding it.

- `smollm2-135m_sweep.json` — original 32-config single-model sweep on SmolLM2-135M. Best config 16 % target reduction at α=0.99 with NaN spillover. **0/32 satisfy the joint criterion** (≥50 % target reduction AND ≤5 % spillover).
- `cross_model_summary.json` — 12-config cross-model replication (3 models  2 intrinsic dims  2 protocols) at 146 s wall:
  - v1 Christoffel-Gaussian warp (α=0.7, σ=1.2, T=16, δl=0.1): best is gemma-4-e2b n=6 with +39.4 % target reduction but **60 % p95 spillover**.
  - v2 covariant Riemannian-exponential push (α=0.35, r=1.2): produces **negative** target improvements on phi-3.5-mini (target error grows by 9–24 %).
  - phi-3.5-mini v1 n=6: **NaN post-error** — the warp drives the geodesic solver out of its convergence basin.
  - **0/12 pass both criteria.**
- `cross_model_log.txt` — full runner log.
- `STATUS.md` — author-facing status note ("strengthens the negative conclusion").
- Script: `scripts/cross_model.py`.
- Cited from: `ARXIV_SUBMISSIONS/paper-D/ott-gtc-manifold-runtime.tex` §sec:curvature-warp-negative (rewritten in commit `7418a3e`).

---

## 5. First-person voice cleanup (no folder; see commit `7418a3e`)

Per user directive ("find anywhere wherein I the author am referred to
in 3rd person, e.g. 'the author' (in a context wherein that author is
me, you can refer to others as the author but not me), then delete
them"). Four occurrences eliminated across:

- `ARXIV_SUBMISSIONS/paper-A/grc-attention-compression.tex` — line 521 acknowledgements
- `ARXIV_SUBMISSIONS/paper-B/geodesic-projection-pipeline.tex` — line 748 acknowledgements
- `ARXIV_SUBMISSIONS/paper-C/geodesic-speculative-decoding.tex` — line 471 acknowledgements
- `docs/WHITEPAPER.md` — line 612

Sole-author byline metadata (e.g. `NagusameCS Independent Research`)
was retained — those are identifiers, not third-person prose.
Verification: two follow-up grep searches for `\b(the author|this
author|the present author|the author's)\b` and an extended pattern
including `the writer|the researcher|the maintainer|sole maintainer`
returned zero remaining matches across `**/*.{md,tex}`.

---

## Caveats / open follow-ups for secondary analysis

1. The Mistral-Nemo-12B Phase-1 TwoNN reading (`k_int = 5.36`) is undersampled at n=256 hidden states for d=5120 — paper-D explicitly flags this as preliminary.
2. The paper-C `llama8b_specdecode_summary.json` (8 prompts  64 tokens, mean 46.9 % acceptance, **stdev 0 %**) was inspected this session and **not** added to any paper. Every prompt yields identically `geo_accepted=30, xfmr=34, od_drafts=0`, which looks like a snapshot/regression test rather than a measurement. Provenance to be checked before citation.
3. The HJB residual reports the pre-training feasibility floor only — the joint-training premise itself is not yet evaluated and explicitly remains future work.
4. The thermal trace shows warm-up drift, not steady-state throttle. A back-to-back run reaching the 80 °C control band on the same chassis is the natural follow-on.
