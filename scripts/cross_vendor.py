#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
#  ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
#  ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
#  ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
#  ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
#  ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
#  :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
#  :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
#  ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
#  :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
#  ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
#  ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
#  :::::::::................................:@@@@@@@@@@%:...............................::::::
#  ::::::::..................................*@@@@@@@@@-................................::::::::
#  ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
#  :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
#  :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
#  :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
#  :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
#  :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
#  :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
#  :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
#  :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
#  ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
#  :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
#  ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
#  ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
#  :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
#  ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
#  :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
#  ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
#  :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
#  ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
#  ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
#  :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#  ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#!/usr/bin/env python3
"""
Cross-Vendor GPU Portability Layer (Paper I gap 7, Paper IX gap 1)

Provides a unified interface for detecting GPU capabilities and computing
optimal GRC compression rank across vendors:
  - NVIDIA (CUDA): NVML via pynvml
  - AMD (ROCm): ROCm SMI via subprocess
  - Apple Silicon (Metal): Metal device query via subprocess
  - TPU (Google): libtpu via environment

The layer abstracts hardware detection behind a single API:
    from cross_vendor import detect_gpu, recommend_kstar
    gpu = detect_gpu()
    kstar = recommend_kstar(gpu, model_dim=4096)

Reference: Stewart, "Cross-GPU Transfer," HyperTensor Paper IX, 2026.

Usage:
    python scripts/cross_vendor.py           # prints detected GPU + k* rec
    python scripts/cross_vendor.py --json    # JSON output
    python scripts/cross_vendor.py --all     # scan all available GPUs
"""

import os
import sys
import json
import subprocess
import re
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# GPU descriptor
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Detected GPU capabilities."""
    vendor: str                     # "nvidia" | "amd" | "apple" | "tpu" | "cpu"
    name: str                       # Human-readable GPU name
    vram_mb: int                    # Total VRAM in MB
    l2_cache_mb: int                # L2 cache size in MB
    l1_cache_kb: int                # L1 / shared memory per SM in KB
    hbm_bandwidth_gb_s: float       # Memory bandwidth in GB/s
    sm_count: int                   # Number of SMs / compute units
    clock_mhz: int                  # Boost clock in MHz
    compute_capability: str         # e.g., "8.9" (NVIDIA) or "gfx1100" (AMD)
    driver_version: str             # Driver version string
    detected: bool = True           # Whether detection succeeded
    raw_info: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# NVIDIA detection
# ---------------------------------------------------------------------------

def detect_nvidia() -> Optional[GPUInfo]:
    """Detect NVIDIA GPU via NVML."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8')

        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_mb = info.total // (1024 * 1024)

        # L2 cache: query via nvidia-smi if NVML doesn't expose directly
        l2_mb = 0
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=l2_cache_size', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                l2_mb = int(float(result.stdout.strip().split('\n')[0]) * 1024)
        except Exception:
            # Fallback: known values for common GPUs
            l2_fallback = {
                'RTX 4070 Laptop': 32,
                'RTX 4090': 72,
                'RTX 4080': 64,
                'RTX 4070': 36,
                'RTX 4060': 24,
                'RTX 3050': 16,
                'A100': 40,
                'A10G': 24,
                'L40S': 48,
                'H100': 50,
                'L4': 36,
            }
            for pattern, size in l2_fallback.items():
                if pattern.lower() in name.lower():
                    l2_mb = size
                    break

        # Clock
        try:
            clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        except Exception:
            clock_mhz = 0

        # Compute capability
        try:
            cc_major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            cc = f"{cc_major[0]}.{cc_major[1]}"
        except Exception:
            cc = "unknown"

        # SM count
        try:
            sm_count = pynvml.nvmlDeviceGetNumGpuCores(handle)
        except Exception:
            sm_count = 0

        # Bandwidth: estimate from memory type
        bw = _estimate_nvidia_bandwidth(name, vram_mb)

        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode('utf-8')

        pynvml.nvmlShutdown()

        return GPUInfo(
            vendor='nvidia', name=name, vram_mb=vram_mb,
            l2_cache_mb=l2_mb, l1_cache_kb=128,  # Typical: 128KB L1/SM
            hbm_bandwidth_gb_s=bw, sm_count=sm_count,
            clock_mhz=clock_mhz, compute_capability=cc,
            driver_version=driver,
        )
    except ImportError:
        return _fallback_nvidia()
    except Exception as e:
        return GPUInfo(vendor='nvidia', name=f'Detection failed: {e}',
                       vram_mb=0, l2_cache_mb=0, l1_cache_kb=0,
                       hbm_bandwidth_gb_s=0, sm_count=0, clock_mhz=0,
                       compute_capability='unknown', driver_version='',
                       detected=False)


def _fallback_nvidia() -> Optional[GPUInfo]:
    """Fallback NVIDIA detection via nvidia-smi CLI."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None
        parts = result.stdout.strip().split(',')
        name = parts[0].strip()
        vram = int(float(parts[1].strip()))
        return GPUInfo(
            vendor='nvidia', name=name, vram_mb=vram,
            l2_cache_mb=0, l1_cache_kb=0,
            hbm_bandwidth_gb_s=0, sm_count=0, clock_mhz=0,
            compute_capability='unknown', driver_version='',
        )
    except Exception:
        return None


def _estimate_nvidia_bandwidth(name: str, vram_mb: int) -> float:
    """Estimate HBM bandwidth from GPU name."""
    name_lower = name.lower()
    # Known bandwidths in GB/s
    if 'h100' in name_lower:
        return 3350.0
    if 'a100' in name_lower:
        return 1555.0 if vram_mb >= 80000 else 2039.0
    if 'a10' in name_lower:
        return 600.0
    if 'l40s' in name_lower:
        return 864.0
    if '4090' in name_lower:
        return 1008.0
    if '4080' in name_lower:
        return 717.0
    if '4070' in name_lower:
        return 504.0 if 'ti' in name_lower or 'super' in name_lower else 256.0
    if '4060' in name_lower:
        return 288.0
    if '3050' in name_lower:
        return 224.0
    return 256.0  # Default


# ---------------------------------------------------------------------------
# AMD ROCm detection
# ---------------------------------------------------------------------------

def detect_amd() -> Optional[GPUInfo]:
    """Detect AMD GPU via rocm-smi CLI."""
    try:
        result = subprocess.run(
            ['rocm-smi', '--showproductname', '--showmeminfo', 'vram',
             '--csv'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return None

        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            return None

        # Parse CSV output
        data = lines[1].split(',')
        name = data[0].strip() if len(data) > 0 else 'AMD GPU'

        # VRAM
        vram_match = re.search(r'(\d+)', data[1] if len(data) > 1 else '0')
        vram_mb = int(vram_match.group(1)) if vram_match else 0

        # MI300 has Infinity Cache; approximate L2 from known values
        l2_fallback = {'mi300': 256, 'mi250': 128, 'mi210': 64, 'mi100': 64}
        l2_mb = 0
        for pattern, size in l2_fallback.items():
            if pattern in name.lower():
                l2_mb = size
                break

        # Bandwidth estimates
        bw_fallback = {'mi300': 5300, 'mi250': 1600, 'mi210': 1200, 'mi100': 1200}
        bw = 0.0
        for pattern, b in bw_fallback.items():
            if pattern in name.lower():
                bw = float(b)
                break

        return GPUInfo(
            vendor='amd', name=name, vram_mb=vram_mb,
            l2_cache_mb=l2_mb, l1_cache_kb=0,
            hbm_bandwidth_gb_s=bw, sm_count=0, clock_mhz=0,
            compute_capability='unknown', driver_version='ROCm',
        )
    except FileNotFoundError:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Apple Silicon detection
# ---------------------------------------------------------------------------

def detect_apple() -> Optional[GPUInfo]:
    """Detect Apple Silicon GPU via system_profiler."""
    if sys.platform != 'darwin':
        return None

    try:
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType', '-json'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        displays = data.get('SPDisplaysDataType', [])
        if not displays:
            return None

        # Apple Silicon: unified memory architecture
        # GPU info is in the chip description
        chip_info = displays[0]
        name = chip_info.get('sppci_model', 'Apple Silicon')
        vram_match = re.search(r'(\d+)\s*GB', str(chip_info))
        vram_mb = int(vram_match.group(1)) * 1024 if vram_match else 0

        # Apple GPUs use system-level cache, not discrete L2
        # Approximate from known values
        l2_fallback = {'m3 max': 48, 'm3 pro': 36, 'm3': 24,
                       'm2 max': 48, 'm2 pro': 36, 'm2': 24,
                       'm1 max': 48, 'm1 pro': 36, 'm1': 24}
        l2_mb = 0
        for pattern, size in l2_fallback.items():
            if pattern in name.lower():
                l2_mb = size
                break

        # Bandwidth estimates (GB/s)
        bw_fallback = {'m3 max': 400, 'm3 pro': 200, 'm3': 100,
                       'm2 max': 400, 'm2 pro': 200, 'm2': 100,
                       'm1 max': 400, 'm1 pro': 200, 'm1': 68}
        bw = 0.0
        for pattern, b in bw_fallback.items():
            if pattern in name.lower():
                bw = float(b)
                break

        return GPUInfo(
            vendor='apple', name=name, vram_mb=vram_mb,
            l2_cache_mb=l2_mb, l1_cache_kb=0,
            hbm_bandwidth_gb_s=bw, sm_count=0, clock_mhz=0,
            compute_capability='Metal', driver_version='macOS',
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# TPU detection
# ---------------------------------------------------------------------------

def detect_tpu() -> Optional[GPUInfo]:
    """Detect Google TPU via environment variables."""
    tpu_name = os.environ.get('TPU_NAME', '')
    if not tpu_name:
        return None

    # TPU v5p: 95 GB HBM per chip, ~2.8 TB/s BW
    tpu_type = os.environ.get('TPU_TYPE', 'v5p')
    tpu_chips = int(os.environ.get('TPU_CHIPS', '1'))

    bw_map = {'v5p': 2800, 'v5e': 1600, 'v4': 1200, 'v3': 900}
    hbm_map = {'v5p': 95, 'v5e': 48, 'v4': 32, 'v3': 16}

    bw = bw_map.get(tpu_type, 0)
    hbm_mb = hbm_map.get(tpu_type, 0) * 1024

    return GPUInfo(
        vendor='tpu', name=f'TPU {tpu_type} ({tpu_chips} chips)',
        vram_mb=hbm_mb * tpu_chips,
        l2_cache_mb=0,  # TPU doesn't have traditional L2
        l1_cache_kb=0,
        hbm_bandwidth_gb_s=bw, sm_count=tpu_chips, clock_mhz=0,
        compute_capability=tpu_type, driver_version='libtpu',
    )


# ---------------------------------------------------------------------------
# Unified detection
# ---------------------------------------------------------------------------

def detect_gpu() -> GPUInfo:
    """
    Auto-detect the primary GPU, trying all vendors.

    Returns:
        GPUInfo for the first detected GPU, or CPU fallback.
    """
    detectors = [
        detect_nvidia,
        detect_amd,
        detect_apple,
        detect_tpu,
    ]

    for detector in detectors:
        gpu = detector()
        if gpu is not None and gpu.detected and gpu.l2_cache_mb > 0:
            return gpu

    # Try again with relaxed requirements
    for detector in detectors:
        gpu = detector()
        if gpu is not None:
            return gpu

    return GPUInfo(
        vendor='cpu', name='CPU-only (no GPU detected)',
        vram_mb=0, l2_cache_mb=0, l1_cache_kb=0,
        hbm_bandwidth_gb_s=0, sm_count=0, clock_mhz=0,
        compute_capability='none', driver_version='',
        detected=False,
    )


def detect_all_gpus() -> List[GPUInfo]:
    """Try to detect all available GPUs across vendors."""
    gpus = []
    for detector in [detect_nvidia, detect_amd, detect_apple, detect_tpu]:
        gpu = detector()
        if gpu is not None and gpu.detected:
            gpus.append(gpu)
    return gpus


# ---------------------------------------------------------------------------
# k* recommendation
# ---------------------------------------------------------------------------

def recommend_kstar(
    gpu: GPUInfo,
    model_dim: int = 4096,
    safety_margin: float = 0.75,
) -> int:
    """
    Recommend optimal GRC compression rank k* for the detected GPU.

    The formula k* = L2_MB × 48.0 (derived as 1536/32 on AD106) gives the
    rank at which the attention working set fits in L2 with safety_margin.

    Args:
        gpu: Detected GPU info.
        model_dim: Model hidden dimension (default 4096 for Llama-8B).
        safety_margin: Fraction of L2 to use (0.75 = 25% headroom).

    Returns:
        Recommended k*, rounded to nearest power of 2.
    """
    if gpu.l2_cache_mb <= 0:
        # No L2 info: guess from VRAM
        if gpu.vram_mb >= 40000:  # 40GB+
            return 1536
        elif gpu.vram_mb >= 20000:  # 24GB
            return 1024
        elif gpu.vram_mb >= 8000:   # 8GB
            return 768
        else:
            return 512

    # Vendor-specific scaling factor
    if gpu.vendor == 'apple':
        # Apple Silicon unified memory: L2 is system-level, use conservative
        constant = 24.0
    elif gpu.vendor == 'tpu':
        # TPU: no traditional L2, use HBM bandwidth proxy
        constant = 16.0
    elif gpu.vendor == 'amd':
        # AMD: similar GPU architecture, use same constant
        constant = 48.0
    else:
        # NVIDIA: empirically calibrated at 48.0 = 1536/32 on AD106
        constant = 48.0

    # Scale by model dimension relative to Llama-8B's d=4096
    dim_scale = 4096.0 / max(model_dim, 1)

    k_float = gpu.l2_cache_mb * constant * dim_scale * safety_margin

    # Round to nearest power of 2
    kstar = 1
    while kstar * 2 <= k_float:
        kstar *= 2

    return max(min(kstar, 4096), 64)  # Clamp to reasonable range


def kstar_table() -> List[Dict]:
    """Generate cross-vendor k* prediction table."""
    # Known GPU configurations
    configs = [
        ('RTX 4070 Laptop', 32, 256, 'nvidia'),
        ('RTX 4090', 72, 1008, 'nvidia'),
        ('RTX 4080', 64, 717, 'nvidia'),
        ('A100', 40, 1555, 'nvidia'),
        ('A10G', 24, 600, 'nvidia'),
        ('L40S', 48, 864, 'nvidia'),
        ('H100', 50, 3350, 'nvidia'),
        ('L4', 36, 300, 'nvidia'),
        ('MI300X', 256, 5300, 'amd'),
        ('MI250X', 128, 1600, 'amd'),
        ('M3 Max', 48, 400, 'apple'),
        ('M2 Ultra', 72, 800, 'apple'),
        ('TPU v5p', 0, 2800, 'tpu'),
    ]

    rows = []
    for name, l2, bw, vendor in configs:
        gpu = GPUInfo(
            vendor=vendor, name=name,
            vram_mb=0, l2_cache_mb=l2, l1_cache_kb=0,
            hbm_bandwidth_gb_s=bw, sm_count=0, clock_mhz=0,
            compute_capability='', driver_version='',
        )
        kstar = recommend_kstar(gpu)
        rows.append({
            'gpu': name,
            'vendor': vendor,
            'l2_mb': l2,
            'hbm_gb_s': bw,
            'kstar_predicted': kstar,
            'kstar_fallback': kstar,
            'status': 'predicted' if vendor != 'nvidia' else (
                'measured' if l2 == 32 else 'empirical'
            ),
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Cross-Vendor GPU Detection')
    parser.add_argument('--json', action='store_true', help='JSON output')
    parser.add_argument('--all', action='store_true', help='Show all GPUs')
    parser.add_argument('--table', action='store_true', help='Show k* prediction table')
    parser.add_argument('--model-dim', type=int, default=4096,
                        help='Model hidden dimension')
    args = parser.parse_args()

    if args.table:
        table = kstar_table()
        if args.json:
            print(json.dumps(table, indent=2))
        else:
            print(f"{'GPU':<20} {'Vendor':<8} {'L2 MB':>6} {'BW GB/s':>8} {'k*':>5} {'Status':<12}")
            print("-" * 65)
            for row in table:
                print(f"{row['gpu']:<20} {row['vendor']:<8} {row['l2_mb']:>6} "
                      f"{row['hbm_gb_s']:>8.0f} {row['kstar_predicted']:>5} "
                      f"{row['status']:<12}")
    elif args.all:
        gpus = detect_all_gpus()
        if args.json:
            print(json.dumps([g.__dict__ for g in gpus], indent=2, default=str))
        else:
            for gpu in gpus:
                print(f"\n  {gpu.name} ({gpu.vendor})")
                print(f"    VRAM: {gpu.vram_mb}MB  L2: {gpu.l2_cache_mb}MB  BW: {gpu.hbm_bandwidth_gb_s:.0f} GB/s")
                print(f"    k* ({args.model_dim}d): {recommend_kstar(gpu, args.model_dim)}")
    else:
        gpu = detect_gpu()
        kstar = recommend_kstar(gpu, args.model_dim)
        if args.json:
            out = gpu.__dict__.copy()
            out['kstar_recommendation'] = kstar
            print(json.dumps(out, indent=2, default=str))
        else:
            print(f"  GPU: {gpu.name} ({gpu.vendor})")
            print(f"  VRAM: {gpu.vram_mb} MB")
            print(f"  L2 Cache: {gpu.l2_cache_mb} MB")
            print(f"  HBM Bandwidth: {gpu.hbm_bandwidth_gb_s:.0f} GB/s")
            print(f"  Compute: {gpu.compute_capability}")
            print(f"  Recommended k* (d={args.model_dim}): {kstar}")
            if not gpu.detected:
                print(f"  WARNING: GPU detection incomplete; k* is a best guess.")
