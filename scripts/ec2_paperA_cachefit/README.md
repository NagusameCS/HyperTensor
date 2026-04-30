# Paper-A Cache-Fit Reproduction on AWS EC2

Probes the **L2-capacity dimension** of the cache-fit hypothesis using GPUs
that span 6 MB → 96 MB L2 cache, well beyond the 4070-Laptop (32 MB) +
3050 (~3 MB) baseline. Runs on AWS, pulls all data back to the local box,
auto-terminates the instance.

> **Data residency:** all benchmark outputs live on the local machine
> only. EC2 is compute-only.

## What it produces (per GPU)

In `benchmarks\paperA_cachefit_<GPU>_<ts>\`:

- `paperA_baseline_<GPU>.txt` — 5× decode tok/s, baseline
- `paperA_grc_<GPU>.txt` — 5× decode tok/s, GRC k=1024 attn-only --skip-o
- `paperA_baseline_<GPU>_run<i>.log`, `paperA_grc_<GPU>_run<i>.log` — full per-run logs
- `paperA_grc_<GPU>_cold.log` — cold-cache GRC (wproj population)
- `paperA_ncu_baseline_<GPU>.{ncu-rep,csv}` — NCU L2 trace (baseline)
- `paperA_ncu_grc_<GPU>.{ncu-rep,csv}` — NCU L2 trace (GRC)
- `env.txt`, `gpu_query.csv`, `meta.json`, `summary.txt`
- `remote_session.log`, `paperA_remote.log` — full transcripts

## Supported instance types

| InstanceType    | GPU       | L2     | VRAM | sm   | $/hr   |
|-----------------|-----------|--------|------|------|--------|
| `g6e.xlarge`    | L40S      | 96 MB  | 48 GB | 8.9 | ~$1.86 |
| `g6.xlarge`     | L4        | 48 MB  | 24 GB | 8.9 | ~$0.80 |
| `g5.xlarge`     | A10G      | 6 MB   | 24 GB | 8.6 | ~$1.00 |
| `p4d.24xlarge`  | A100×8    | 40 MB  | 40 GB | 8.0 | ~$32   |
| `p5.48xlarge`   | H100×8    | 50 MB  | 80 GB | 9.0 | ~$98   |

**Recommended pairs:**

- `g6e.xlarge` (96 MB) + `g5.xlarge` (6 MB)
  → 16× span around our 32 MB datapoint; best L2-capacity coverage on AWS.
- `g6e.xlarge` alone is sufficient as the workstation-flagship analog
  (RTX 6000 Ada uses the same GL102 die / 96 MB L2 as L40S).

A100 / H100 single-GPU SKUs require Capacity Blocks. The 8-GPU variants
work but are >10× cost.

## Usage

```powershell
# Default: g6e.xlarge (L40S, 96 MB L2)
.\scripts\ec2_paperA_cachefit\launch.ps1

# Small-L2 control point (predicts win SHOULD shrink)
.\scripts\ec2_paperA_cachefit\launch.ps1 -InstanceType g5.xlarge

# Preview without launching
.\scripts\ec2_paperA_cachefit\launch.ps1 -DryRun
```

### Pre-reqs (local)

- AWS CLI v2 configured (`aws sts get-caller-identity` works)
- EC2 key pair `hypertensor-key` exists, PEM at `~/.ssh/hypertensor-key.pem`
- OpenSSH in PATH (default on Win11)
- `git`, `tar`
- Optional: `$env:HF_TOKEN` for the gated Llama-3.1-8B GGUF

### What happens

1. Look up latest **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)**.
2. Open SSH (port 22) only from your current public IP.
3. Launch instance with `InstanceInitiatedShutdownBehavior=terminate` and a
   user-data script that schedules `shutdown -h now` after `MaxRuntimeMinutes`
   (default 120) — defence-in-depth so the instance terminates even if the
   orchestrator dies.
4. `git archive HEAD` source → scp tarball + `build_ubuntu_cuda.sh` + `run_paperA.sh`.
5. Build `cuda_kernels.so` with the right `sm_*` for the GPU + host binary
   via zig 0.14 + openblas.
6. Download model (~4.9 GB) into `/opt/hypertensor/models/`.
7. Run the benchmark suite (`run_paperA.sh`) with the **working** flags
   (`--axex-weight-pca`, see `benchmark_runtime_status_20260428.md`).
8. NCU L2 trace pair on `kernel_gemv_q4_k` (skipped gracefully if `ncu` missing).
9. `scp` results back to local.
10. **Always** `aws ec2 terminate-instances` in the `finally` block (use
    `-KeepInstance` to override, then you must terminate manually).

### If something goes wrong

- The orchestrator's `finally` always issues a terminate.
- The instance's `shutdown -h +N` watchdog will terminate after `MaxRuntimeMinutes`
  (because we set `InstanceInitiatedShutdownBehavior=terminate`).
- Worst case manual cleanup:
  ```powershell
  aws ec2 describe-instances --region us-east-1 --filters Name=tag:Project,Values=HyperTensor Name=instance-state-name,Values=running --query 'Reservations[].Instances[].InstanceId' --output text
  aws ec2 terminate-instances --region us-east-1 --instance-ids <id>
  ```

### Quotas

`g6e` and `g5` are governed by the *Running On-Demand G and VT instances*
service quota (vCPU-based). 4 vCPU is well within the typical default
(8–32 vCPU), but a brand-new account may show 0 — request via Service Quotas
console if `run-instances` returns `VcpuLimitExceeded`.

`p4d` / `p5` need *Running On-Demand P instances* quota and are usually 0
by default; AWS approves these on a case-by-case basis.

## Post-processing (local)

The CSVs from `ncu --page raw --csv` give per-launch L2 metrics. To turn
them into a hit-rate pair:

```powershell
$base = Import-Csv .\benchmarks\paperA_cachefit_*\paperA_ncu_baseline_*.csv
$grc  = Import-Csv .\benchmarks\paperA_cachefit_*\paperA_ncu_grc_*.csv
# columns of interest:
#   lts__t_sectors_op_read_lookup_hit.sum
#   lts__t_sectors_op_read.sum
# hit_rate = hit / read
```

(A small post-processor is left as a follow-up — the raw CSVs are
preserved exactly so reviewers can re-derive.)
