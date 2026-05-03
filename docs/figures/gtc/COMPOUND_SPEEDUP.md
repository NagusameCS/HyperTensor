# Compound Speedup Projection (OTT x GRC x GTC)

Date: 2026-04-27

## Inputs used

- OTT headline factor (from prior OTT/Paper-1 tracking): `320x`
- GRC at practical quality point (Llama-3.1-8B, k=1536): decode retention `97.55%` (`0.9755x` speed factor)
- GTC live three-bucket projection on real exported clouds: about `1.97x`
- GTC batch Jacobi resonance at large batch: `60.0x`

## Two valid compositions

Not all multipliers apply to the same computational slice, so we report two
separate compositions instead of one inflated number.

### A) End-to-end conservative online composition

Use OTT + GRC + live decode-substitution projection:

`S_online = 320 * 0.9755 * 1.972 = 622.5x`

This is a conservative deployment-facing factor where all terms can coexist in
a decode loop accounting model.

### B) OTT + correction-path upper-bound composition

Use OTT + dense in-regime correction-path gain (per correctable step):

`S_correctable_path = 320 * 1.0 * 2704.6 = 865472x`

This is not an end-to-end claim. It is a per-correctable-step upper bound that
requires high local cloud density so that most steps are within rho.

### C) Legacy 4800x-style compound estimate (for continuity)

If we pair OTT with the earlier single-bucket live substitution estimate
(~14.4x from previous run), then:

`S_legacy = 320 * 14.4 = 4608x`

This is the origin of the "~4800x" narrative. Under the new three-bucket metric,
that estimate is optimistic unless cloud density is increased.

## Recommended headline now

Use `~620x` as the honest online compound estimate on current telemetry density,
and report `~4600x` only as a legacy optimistic estimate with explicit caveat.

## Source anchors

- docs/figures/ppl_sweep/llama31_8b_ppl_sweep.json
- docs/figures/gtc/smollm2-135m_batch_jacobi.json
- docs/figures/gtc/smollm2-135m_decode_substitution.json
- docs/figures/gtc/smollm2-135m_decode_substitution_dense.json
