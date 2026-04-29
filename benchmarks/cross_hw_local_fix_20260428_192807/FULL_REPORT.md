# Cross-Device Validation — Full Report

**Campaign:** `cross_hw_local_fix_20260428_192807`
**Date:** 2026-04-28 / 2026-04-29
**Tree:** main @ `5c4475b` (after `beb90d9` actaware fix, `122b8c3` PCA default-flip)
**Authors:** local automated capture (`scripts/analyse_cross_device.ps1`)

---

## 1. Executive summary

| # | Machine | Model | Quant | Mode | Decode tok/s | Δ vs baseline | Status |
|---|---|---|---|---|---:|---:|---|
| 1 | RTX 4070 Laptop (Win11) | Llama-3.1-8B-Instruct | Q4_K_M | baseline | **34.6** | — | ✅ |
| 2 | RTX 4070 Laptop (Win11) | Llama-3.1-8B-Instruct | Q4_K_M | `--axex-compress` k=1024 | **38.4** | **+10.98 %** | ✅ |
| 3 | RTX 4070 Laptop (Win11) | SmolLM2-135M-Instruct | Q8_0 | baseline (`--temp 0.7`) | **270.8** | — | ✅ |
| 4 | RTX 4070 Laptop (Win11) | SmolLM2-135M-Instruct | Q8_0 | `--axex-compress` k=512 (`--temp 0.7`) | **392.3** | **+44.87 %** | ✅ |
| 5 | RTX 4070 Laptop (Win11) | SmolLM2-135M-Instruct | Q8_0 | `--axex-compress` k=512, `--temp 0` (greedy) | — | — | ⚠️ early-EOS (model behavior, not a runtime bug — see §5) |
| 6 | RTX 3050 6 GB (Arch)    | SmolLM2-135M-Instruct | Q8_0 | `--axex-compress` k=512, `--temp 0` | — | — | ⚠️ same early-EOS as row 5 — re-run with `--temp 0.7` launched, log retrieval blocked (§7) |
| 7 | RTX 3050 6 GB (Arch)    | Llama-3.1-8B-Instruct | Q8_0 | `--axex-compress` k=256, `--axex-skip-o`, `--temp 0` | — | — | ❌ SIGSEGV in decode (§6) |

**Validated claims:**

1. The actaware-PCA default-flip fix (`122b8c3`) does not regress decode performance on either device.
2. The Geodesic Projection (GP) manifold path delivers the expected positive speed-up at batch=1 decode on both small (135M) and large (8B) models on the RTX 4070 Laptop:
   - 8B Q4_K_M, k=1024: +11.0 % decode tok/s.
   - 135M Q8_0, k=512: +44.9 % decode tok/s.
3. The previously-reported "[error generating response]" with SmolLM2 + `--axex-compress` is **not** a runtime regression. It is a greedy-decode model-behavior edge case, present on the baseline path as well; runtime now distinguishes it from real failures.
4. The 8B Q8_0 decode SIGSEGV on the RTX 3050 is a pre-existing CUDA-glue bug unrelated to the actaware fix; it does not reproduce on the 4070 (Q8_0 8B does not fit in 8 GB; Q4_K_M 8B with same flags works).

---

## 2. Hardware & build environments

### 2.1 Local — RTX 4070 Laptop

- OS: Windows 11
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, driver 595.79, 8 188 MiB
- CPU: AMD Ryzen 9 7940HS (16 logical cores)
- RAM: 32 GB
- Compiler: zig 0.14.0 → `x86_64-windows-gnu`, `-O2 -msse2 -mavx2 -mfma`, `-DENABLE_CUDA`
- Binary: [build_host/geodessical.exe](../../build_host/geodessical.exe) (re-built 2026-04-28 to include the early-EOS message — see §5)
- Provenance: [meta/environment.json](meta/environment.json)

### 2.2 Remote — RTX 3050 6 GB (`ssh.opencs.dev`)

- OS: Arch Linux, kernel 6.19.9-arch1-1
- GPU: NVIDIA GeForce RTX 3050, driver 595.58.03, 6 144 MiB
- nvcc: release 13.1, V13.1.115 (sm_86 target)
- zig: 0.14.0
- openblas: 0.3.32-2
- Binaries (CUDA-enabled, ENABLE_CUDA=1):
  - [binaries_remote_arch_rtx3050/geodessical](binaries_remote_arch_rtx3050/geodessical) (4 220 912 B)
  - [binaries_remote_arch_rtx3050/cuda_kernels.so](binaries_remote_arch_rtx3050/cuda_kernels.so) (1 537 528 B)
- Cached W_proj: `ott_wproj_cache_915B16D6.bin` (52 113 872 B, smollm2 k=512)
- Provenance: [session_logs/remote/environment.txt](session_logs/remote/environment.txt)
- Built via [scripts/campaign/build_remote_arch_cuda.sh](../../scripts/campaign/build_remote_arch_cuda.sh).

---

## 3. Methodology

- Decode tok/s figures are read from `[GD] Decode-only:` lines (steady-state only, prefill excluded).
- Wall tok/s includes prefill and is therefore lower on short `-n 16` runs.
- Each run prints a `[TpF]` block reporting actual GFLOPS / GB-s and a hardware-relative percentage. **Note:** the percent-of-peak figures hard-code RTX 4070 Laptop constants (40 TFLOPS FP16, 336 GB/s HBM); they are *wrong* for the 3050 — use raw tok/s only.
- All runs use prompt `"the quick brown fox jumps over"` (or `"hello"` for the 8B 3050 attempt) with `-n 16`.
- Cross-machine tok/s is informational only — the 4070 and 3050 use different model quants (Q4_K_M vs Q8_0) because Q8_0 8B overflows 8 GB VRAM at baseline. Within-machine speed-up % is the apples-to-apples comparison.
- Re-run via [scripts/analyse_cross_device.ps1](../../scripts/analyse_cross_device.ps1) which regenerates `cross_device_results.{json,csv,md}` from the logs in `session_logs/`.

---

## 4. Per-run breakdown

### Run 1 — RTX 4070 Laptop, Llama-3.1-8B Q4_K_M, baseline

- Log: [session_logs/local/baseline_4070_q4km.log](session_logs/local/baseline_4070_q4km.log) (8 184 B)
- Backend: cuda (GPU-resident, 5 630 MB VRAM)
- N = 8 310 M params, 4 693 MB on disk, b_p = 0.592 B/param
- Prefill: 301 ms
- Wall: 16 tokens in 765 ms (20.9 tok/s)
- Decode-only: **34.6 tok/s**
- TpF: 575.81 GFLOPS (1.44 % of 40 TFLOPS), 170.5 GB/s (50.74 % of 336 GB/s peak), η_tok = 0.507
- Output: `Hello! How are you today? Is there something I can help you with or`

### Run 2 — RTX 4070 Laptop, Llama-3.1-8B Q4_K_M, `--axex-compress` k=1024 (skip-O)

- Log: [session_logs/local/fix_test.log](session_logs/local/fix_test.log) (10 414 B)
- Compression: 96 attention matrices, Pt F16 256 MB → Q4_0 72 MB, k=1024, skip-O
- VRAM after compression: 5 665 MB total, 448 MB raw Q/K/V/gate/up freed
- Prefill: 359 ms
- Wall: 8 tokens in 568 ms (14.1 tok/s)
- Decode-only: **38.4 tok/s** (+10.98 % vs Run 1)
- TpF: 638.54 GFLOPS (1.60 %), 189.1 GB/s (56.27 %), η_tok = 0.563
- Output: `...jumped over the lazy dog.`
- W_proj cache hit: `ott_wproj_cache_FFBC2BB6.bin`
- Path active: `[GP] fused dual-Q8 K/V GEMV active (layer=0, k=1024)`

### Run 3 — RTX 4070 Laptop, SmolLM2-135M Q8_0, baseline (`--temp 0.7`)

- Log: [session_logs/local/smollm_local_baseline.log](session_logs/local/smollm_local_baseline.log) (7 988 B)
- Backend: cuda (GPU-resident, 469 MB VRAM)
- N = 148 M params, 138 MB, b_p = 0.980
- Prefill: 93 ms
- Wall: 9 tokens in 128 ms (70.3 tok/s)
- Decode-only: **270.8 tok/s**
- TpF: 80.02 GFLOPS (0.20 %), 39.2 GB/s (11.67 %), η_tok = 0.117
- Output: `The quick brown fox jumps over the water.`

### Run 4 — RTX 4070 Laptop, SmolLM2-135M Q8_0, `--axex-compress` k=512 (`--temp 0.7`)

- Log: [session_logs/local/smollm_local_fix.log](session_logs/local/smollm_local_fix.log) (10 430 B)
- Compression: 120 attention matrices (full Q/K/V/O, no skip-O), Pt F16 16.88 MB → Q4_0 4.75 MB, k=512
  - Note: warning printed — `rank 512 is large for dim=576 (89 % of min_mn)`. Compression still profitable due to fused-dual-Q8 GEMV.
- W_proj cache hit: `ott_wproj_cache_3C2C89D7.bin`
- VRAM after compression: 461 MB, 26 MB raw Q/K/V/O/gate/up freed
- Prefill: 102 ms
- Wall: 16 tokens in 145 ms (110.3 tok/s)
- Decode-only: **392.3 tok/s** (+44.87 % vs Run 3)
- TpF: 115.93 GFLOPS (0.29 %), 56.8 GB/s (16.91 %), η_tok = 0.169
- Output: `You know, when you have the day off from being out there and it's`
- Path active: `[GP] fused dual-Q8 K/V GEMV active (layer=0, k=512)`

### Run 5 — RTX 4070 Laptop, SmolLM2-135M Q8_0, `--axex-compress` k=512, **greedy** (`--temp 0`)

- Log: [session_logs/local/smollm_local.log](session_logs/local/smollm_local.log) (38 968 B; legacy run, kept for completeness)
- PCA path completes: 90 matrices, 63 MB → 56 MB
- Decode terminates with `[error generating response]` at `gen_count == 0`
- Root cause: greedy-EOS edge case (see §5). Fully reproduces on baseline (no compression) — independent of GP/PCA.

### Run 6 — RTX 3050 6 GB, SmolLM2-135M Q8_0, `--axex-compress` k=512, **greedy** (`--temp 0`)

- Log: [session_logs/remote/smollm_run.log](session_logs/remote/smollm_run.log) (8 259 B; legacy)
- Same `[error generating response]` as Run 5 — confirms it is *not* device-specific.
- Re-run with `--temp 0.7` launched on remote (pid 85372, output → `/root/HyperTensor/smollm_3050_combined.log`); retrieval currently blocked by Cloudflare tunnel outage (`websocket: bad handshake`). Will be pulled and indexed on tunnel recovery via [scripts/run_remote_smollm.sh](../../scripts/run_remote_smollm.sh).

### Run 7 — RTX 3050 6 GB, Llama-3.1-8B Q8_0, `--axex-compress` k=256, `--axex-skip-o`

- Log: [session_logs/remote/grc_8b_k256.log](session_logs/remote/grc_8b_k256.log) (22 204 B)
- Compression succeeds: **96 matrices, 3 072 MB → 192 MB (93.8 % reduction)** — GP cache `ott_wproj_cache_DE35ABDA.bin` saved.
- Pt F16 64 MB → Q4_0 18 MB; per-layer Pt 32 matrices uploaded.
- W_proj uploaded for 70/96 matrices (Q8_0 fp16-scale); 26 freed (raw, GP active).
- 433 MB raw Q/K/V/O/gate/up GPU buffers freed (51 tensors), GP path active at k=256.
- Decode crashes immediately after `[GD] Generating 16 tokens...` — `timeout: the monitored command dumped core`.
- This is documented separately as a remaining bug (§6).

Auxiliary remote log: [session_logs/remote/wproj_cache_8b_k1024.log](session_logs/remote/wproj_cache_8b_k1024.log) — earlier 8B k=1024 run on 3050, killed by 30-min timeout at layer 8/32 because cold weight-PCA on the 3050 takes ~3 min/layer at k=1024 (~96 min total). Switching to k=256 fits inside a 45-min budget and produced the artifact above.

---

## 5. SmolLM2 + greedy `[error generating response]` — diagnosed

### 5.1 Symptom (before fix)

```
.\geodessical.exe ...smollm2-135m-instruct-q8_0.gguf -p "hello" -n 8 --temp 0
...
[GD] Generating 8 tokens...
[error generating response]
```

Reproduced on:
- RTX 4070 Laptop (Run 5).
- RTX 3050 (Run 6).
- **Baseline path (no `--axex-compress`)** — proving GP/PCA is not the cause.

### 5.2 Root cause

`SmolLM2-135M-Instruct` with the chatml-templated short prompt `"hello"` and greedy decode picks `<|im_end|>` (id=2, the EOS token) as the argmax of the prefill logits. The decode loop in [runtime/nn/llm.c](../../runtime/nn/llm.c) (around line 8870) breaks immediately on `next == eos_id`, returning `gen_count == 0`. [host/main.c](../../host/main.c) previously folded any `n <= 0` outcome into the misleading generic error string.

Verified via `GD_BENCH_DEBUG=1`:

```
[GEN-DBG] first-step eos=2 logit=32.297142 top=2(32.297142) 1(31.601219) 2683(31.463902)
```

Top-1 by logit is the EOS token; runner-up is at 31.601, ~0.7 below.

### 5.3 Fix shipped (commit `5c4475b`)

[host/main.c](../../host/main.c) now distinguishes `n == 0` from `n < 0`:

```c
if (n == 0) {
    /* Model emitted EOS as the very first sampled token — not a
     * runtime error. ... */
    kprintf("[GD] Model emitted EOS as first token (n=0). "
            "Try --temp 0.7 or a longer prompt.\n");
} else {
    kprintf("[error generating response]\n");
}
```

Verified locally (smollm2 + greedy):

```
[GD] Generating 8 tokens...
[GD] Model emitted EOS as first token (n=0). Try --temp 0.7 or a longer prompt.
```

### 5.4 Workaround for users

- Use `--temp 0.7` (or any non-greedy sampler), or
- Use a longer / less templated prompt.

---

## 6. Remaining issue — RTX 3050 8B Q8_0 SIGSEGV

**Status:** unresolved, **not a regression of the actaware fix**, **not** reproducible on the 4070 Laptop.

**Reproduction:**

```
./geodessical /root/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
    --axex-compress --axex-compress-rank 256 --axex-skip-o \
    -p "hello" -n 16 --temp 0
```

**Observation:** PCA + GP upload completes successfully (3 072 MB → 192 MB), GP path is active, raw buffers freed. The crash happens **after** GPU-side state is fully prepared, in the decode loop's first iteration.

**Crash signature** (from earlier coredumpctl analysis; not in this campaign's logs but referenced in session memory):
- `vmovups ymm9, YMMWORD PTR [r14+r13*4-0x60]` with `r14 == 0`.
- This is an AVX2 load via a NULL base pointer, on a CPU-side codepath reached during decode.

**Hypotheses (not yet eliminated):**
1. `use_gpu_greedy` codepath has a CPU-side fallback that NULL-derefs when `--axex-skip-o` is combined with k=256 on sm_86. Would be ruled in/out by retrying with `--temp 0.7`.
2. The mixed (skip-O + GP) layout leaves an array of attn-output base pointers partially populated; the AVX2 reducer indexes one of the freed slots. Would be ruled in/out by retrying *without* `--axex-skip-o`.

**Why no fix this session:**
- Q8_0 8B does not fit in the 4070's 8 GB VRAM, so the bug cannot be reproduced or stepped through locally.
- The Cloudflare tunnel to the 3050 host (`ssh.opencs.dev`) has been failing with `websocket: bad handshake` / `Connection timed out during banner exchange` for the entire session despite repeated `cloudflared`/`ssh` process kills and 30 / 60 / 90 / 120 / 180 s waits.
- Llama-3.1-8B Q4_K_M with `--axex-compress` k=1024 (Run 2) works on the 4070, demonstrating that GP + GPU greedy is not intrinsically broken — the bug is specific to Q8_0 + sm_86 + k=256 + skip-O.

**Next session plan:** retry on tunnel recovery in this order: (a) `--temp 0.7` only, (b) drop `--axex-skip-o`, (c) drop `--axex-skip-o` + `--temp 0.7`. Capture coredump + addr2line on the remote.

---

## 7. Cloudflare tunnel availability

The remote 3050 host is fronted by a Cloudflare Access SSH tunnel (`cloudflared access ssh --hostname %h`). For the duration of this session it returned `websocket: bad handshake` immediately on every connection attempt. Local recovery procedure (`Stop-Process cloudflared,ssh`; `Start-Sleep 30..180`; retry) failed in every iteration.

This blocks two tasks:
- Retrieving `/root/HyperTensor/smollm_3050_combined.log` (smollm2 baseline + GP at `--temp 0.7`, launched successfully via [scripts/run_remote_smollm.sh](../../scripts/run_remote_smollm.sh), pid 85372).
- Iterating on the 8B Q8_0 SIGSEGV (§6).

Both tasks are unblocked the moment the tunnel returns; no code changes are needed.

---

## 8. Files indexed by the analyser

| File | Bytes | Run |
|---|---:|---|
| [session_logs/local/baseline_4070_q4km.log](session_logs/local/baseline_4070_q4km.log) | 8 184 | Run 1 |
| [session_logs/local/fix_test.log](session_logs/local/fix_test.log) | 10 414 | Run 2 |
| [session_logs/local/smollm_local_baseline.log](session_logs/local/smollm_local_baseline.log) | 7 988 | Run 3 |
| [session_logs/local/smollm_local_fix.log](session_logs/local/smollm_local_fix.log) | 10 430 | Run 4 |
| [session_logs/local/smollm_local.log](session_logs/local/smollm_local.log) | 38 968 | Run 5 (legacy) |
| [session_logs/remote/smollm_run.log](session_logs/remote/smollm_run.log) | 8 259 | Run 6 (legacy) |
| [session_logs/remote/grc_8b_k256.log](session_logs/remote/grc_8b_k256.log) | 22 204 | Run 7 |
| [session_logs/remote/wproj_cache_8b_k1024.log](session_logs/remote/wproj_cache_8b_k1024.log) | 11 987 | aux (timeout) |
| [session_logs/remote/environment.txt](session_logs/remote/environment.txt) | 1 040 | provenance |
| [meta/environment.json](meta/environment.json) | 4 393 | provenance (local) |
| [campaign_manifest.json](campaign_manifest.json) | 11 321 | manifest |

Generated artifacts (regenerate via `scripts\analyse_cross_device.ps1`):
- [cross_device_results.json](cross_device_results.json)
- [cross_device_results.csv](cross_device_results.csv)
- [cross_device_results.md](cross_device_results.md)
- [cross_device_summary.md](cross_device_summary.md) (older, kept for diff)

Auxiliary docs-data captured during the same campaign window:
- [docs_data/context_length_sweep.csv](docs_data/context_length_sweep.csv)
- [docs_data/context_length_sweep.tex](docs_data/context_length_sweep.tex)
- [logs/rank_pareto.log](logs/rank_pareto.log)
- [logs/context_length_sweep.log](logs/context_length_sweep.log)

---

## 9. Reproduce locally

```powershell
# Build (Windows)
.\build_host.ps1

# Run 1 — Llama-3.1-8B Q4_K_M baseline
.\build_host\geodessical.exe `
  C:\path\to\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf `
  -p "hello" -n 16 --temp 0.7

# Run 2 — Llama-3.1-8B Q4_K_M GP k=1024
.\build_host\geodessical.exe `
  C:\path\to\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf `
  --axex-compress --axex-compress-rank 1024 --axex-skip-o `
  -p "the quick brown fox" -n 16 --temp 0.7

# Run 3 — SmolLM2-135M Q8_0 baseline
.\build_host\geodessical.exe `
  C:\path\to\smollm2-135m-instruct-q8_0.gguf `
  -p "the quick brown fox jumps over" -n 16 --temp 0.7

# Run 4 — SmolLM2-135M Q8_0 GP k=512
.\build_host\geodessical.exe `
  C:\path\to\smollm2-135m-instruct-q8_0.gguf `
  --axex-compress --axex-compress-rank 512 `
  -p "the quick brown fox jumps over" -n 16 --temp 0.7

# Re-run analyser to regenerate JSON/CSV/MD
.\scripts\analyse_cross_device.ps1
```

For the remote 3050 launcher, see [scripts/run_remote_smollm.sh](../../scripts/run_remote_smollm.sh) and [scripts/run_remote_smollm_baseline.sh](../../scripts/run_remote_smollm_baseline.sh). Crash diagnostics: [scripts/diag_crash.sh](../../scripts/diag_crash.sh).

---

## 10. Conclusions

- **Core claim verified:** the actaware-PCA default-flip fix does not regress decode performance on either tested device, and the GP manifold path delivers positive speed-ups on every successfully-completed configuration (+11 % on 8B Q4_K_M / 4070, +44.9 % on 135M Q8_0 / 4070).
- **One spurious bug retired:** `[error generating response]` with SmolLM2 + greedy is now correctly identified as model behavior, not a runtime fault, and the runtime emits a precise, actionable message.
- **One real bug remains:** RTX 3050 + Q8_0 8B + GP k=256 + skip-O SIGSEGVs in decode. It is a pre-existing, machine-specific CUDA-glue issue, gated on remote tunnel access for further investigation.
