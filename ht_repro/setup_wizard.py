"""
Setup wizard — auto-detects environment and installs dependencies.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# ── Dependency Map ─────────────────────────────────────────────────
DEPS_T1 = ["numpy", "scipy", "mpmath", "sympy"]
DEPS_T2 = DEPS_T1 + ["torch", "transformers", "tqdm"]
DEPS_T3 = DEPS_T2 + ["bitsandbytes", "psutil", "pynvml"]

def detect_python() -> str:
    """Detect Python version and executable."""
    return f"Python {sys.version_info.major}.{sys.version_info.minor} ({sys.executable})"

def detect_gpu() -> dict:
    """Detect GPU availability and specs."""
    info = {"available": False, "name": "None", "vram_gb": 0, "cuda": False}
    try:
        import torch
        if torch.cuda.is_available():
            info["available"] = True
            info["cuda"] = True
            info["name"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            info["vram_gb"] = round(vram, 1)
    except ImportError:
        pass
    return info

def detect_os() -> str:
    """Detect OS."""
    if sys.platform == "win32":
        return "Windows"
    elif sys.platform == "darwin":
        return "macOS"
    else:
        return "Linux"

def detect_disk_space() -> dict:
    """Check available disk space."""
    try:
        total, used, free = shutil.disk_usage(str(ROOT))
        return {"total_gb": round(total/(1024**3), 1), "free_gb": round(free/(1024**3), 1)}
    except Exception:
        return {"total_gb": 0, "free_gb": 0}

def check_deps(tier: str = "T1") -> dict:
    """Check which dependencies are installed."""
    deps = {"T1": DEPS_T1, "T2": DEPS_T2, "T3": DEPS_T3}.get(tier, DEPS_T1)
    status = {}
    for dep in deps:
        try:
            __import__(dep)
            status[dep] = True
        except ImportError:
            status[dep] = False
    return status

def install_deps(tier: str = "T1") -> bool:
    """Install missing dependencies for a tier."""
    deps = {"T1": DEPS_T1, "T2": DEPS_T2, "T3": DEPS_T3}.get(tier, DEPS_T1)
    missing = [d for d in deps if not _is_installed(d)]
    if not missing:
        return True

    print(f"Installing {len(missing)} missing packages: {', '.join(missing)}")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
            check=True, capture_output=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False

def _is_installed(package: str) -> bool:
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def run_setup(interactive: bool = True) -> dict:
    """Run the full setup wizard. Returns environment report."""
    report = {
        "os": detect_os(),
        "python": detect_python(),
        "gpu": detect_gpu(),
        "disk": detect_disk_space(),
        "deps": {},
        "tier_available": "T1",
    }

    report["deps"] = {
        "T1": check_deps("T1"),
        "T2": check_deps("T2"),
        "T3": check_deps("T3"),
    }

    # Determine available tier
    t1_ok = all(report["deps"]["T1"].values())
    t2_ok = all(report["deps"]["T2"].values()) and report["gpu"]["available"]
    t3_ok = all(report["deps"]["T3"].values()) and report["gpu"]["available"] and report["gpu"]["vram_gb"] >= 24

    if t3_ok: report["tier_available"] = "T3"
    elif t2_ok: report["tier_available"] = "T2"
    elif t1_ok: report["tier_available"] = "T1"

    if interactive:
        print_setup_report(report)

    return report

def print_setup_report(report: dict):
    """Pretty-print the setup report."""
    B = "\033[1m"; G = "\033[92m"; Y = "\033[93m"; R = "\033[91m"; C = "\033[0m"

    print(f"\n{B}═══ HyperTensor Environment Report ═══{C}\n")
    print(f"  OS:      {report['os']}")
    print(f"  Python:  {report['python']}")

    gpu = report['gpu']
    if gpu['available']:
        print(f"  GPU:     {G}{gpu['name']} ({gpu['vram_gb']} GB VRAM){C}")
    else:
        print(f"  GPU:     {Y}None — CPU-only mode{C}")

    disk = report['disk']
    print(f"  Disk:    {disk['free_gb']} GB free of {disk['total_gb']} GB")

    print(f"\n  {B}Dependencies:{C}")
    for tier in ("T1", "T2", "T3"):
        deps = report['deps'][tier]
        installed = sum(1 for v in deps.values() if v)
        total = len(deps)
        color = G if installed == total else (Y if installed > 0 else R)
        print(f"    {tier}: {color}{installed}/{total} installed{C}  [{', '.join(f'{'✓' if v else '✗'} {k}' for k, v in deps.items())}]")

    tier_color = {"T1": G, "T2": Y, "T3": R}[report['tier_available']]
    print(f"\n  → Available tier: {tier_color}{B}{report['tier_available']}{C}")
    print(f"    (T1 = CPU-only, T2 = consumer GPU, T3 = datacenter GPU)\n")
