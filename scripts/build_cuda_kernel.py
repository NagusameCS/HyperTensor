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
#  ::::::::::::::::::::::.......................+@@@-......................:::::::::::::::::::::
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
#  :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............:::::::::::::::
#  ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............:::::::::::::::
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

"""Build script for the HyperRetro fused dual-Q8 GEMV CUDA kernel.

Usage (Linux with CUDA toolkit)::

    python scripts/build_cuda_kernel.py

This compiles ``hyperretro/kernels/csrc/cuda/gemv_dual_q8_0.cu`` into a
shared library and installs it as ``hyperretro_kernels_cuda``.

On Windows without MSVC, use the pure-torch GPU backend
(``hyperretro.kernels.gpu``) which requires no compilation.
"""
from __future__ import annotations

import os
import sys
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CUDA_SRC = ROOT / "hyperretro" / "kernels" / "csrc" / "cuda" / "gemv_dual_q8_0.cu"

def _find_cuda_home() -> str | None:
    for path in [
        os.environ.get("CUDA_HOME", ""),
        os.environ.get("CUDA_PATH", ""),
        "/usr/local/cuda",
        "/opt/cuda",
    ]:
        if path and Path(path).is_dir():
            return path
    # Try nvcc on PATH
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).parent.parent)
    return None


def build_with_setuptools():
    """Build via torch.utils.cpp_extension.CUDAExtension (Linux only)."""
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension
        from setuptools import setup
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install torch setuptools")
        sys.exit(1)

    cuda_home = _find_cuda_home()
    if cuda_home is None:
        print("CUDA toolkit not found. Set CUDA_HOME or install CUDA.")
        print("Falling back to pure-torch GPU backend — no compilation needed.")
        return False

    os.environ.setdefault("CUDA_HOME", cuda_home)
    print(f"CUDA_HOME={cuda_home}")

    # Check nvcc
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        print("nvcc not found on PATH")
        return False

    # Detect SM architectures
    try:
        result = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, timeout=10
        )
        print(f"nvcc: {result.stdout.split(chr(10))[0]}")
    except Exception:
        pass

    # Get torch include paths
    torch_include = Path(torch.__path__[0]) / "include"
    csrc_dir = CUDA_SRC.parent

    ext = CUDAExtension(
        name="hyperretro_kernels_cuda",
        sources=[str(CUDA_SRC)],
        include_dirs=[
            str(torch_include),
            str(torch_include / "torch" / "csrc" / "api" / "include"),
            str(csrc_dir),
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-gencode", "arch=compute_80,code=sm_80",
                "-gencode", "arch=compute_89,code=sm_89",
                "-gencode", "arch=compute_90,code=sm_90",
            ],
        },
    )

    # Build in-place
    setup(
        name="hyperretro_kernels_cuda",
        ext_modules=[ext],
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
        script_args=["build_ext", "--inplace"],
    )
    return True


def build_with_nvcc():
    """Build directly with nvcc (fallback for non-setuptools envs)."""
    cuda_home = _find_cuda_home()
    if cuda_home is None:
        print("CUDA not found, skipping direct nvcc build")
        return False

    try:
        import torch
    except ImportError:
        print("torch not installed, skipping direct nvcc build")
        return False

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        print("nvcc not found")
        return False

    torch_include = Path(torch.__path__[0]) / "include"
    out_dir = ROOT / "build" / "cuda"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_so = out_dir / "gemv_dual_q8_0.cuda.so"

    cmd = [
        nvcc,
        "-O3", "-shared",
        "-o", str(out_so),
        str(CUDA_SRC),
        f"-I{torch_include}",
        f"-I{torch_include}/torch/csrc/api/include",
        "--gpu-architecture=sm_89",
        "--use_fast_math",
    ]

    print(f"Building: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"nvcc build FAILED:\n{result.stderr}")
        return False
    print(f"Built: {out_so} ({out_so.stat().st_size / 1024:.0f} KB)")
    return True


def main():
    print(f"CUDA source: {CUDA_SRC}")
    if not CUDA_SRC.exists():
        print(f"ERROR: CUDA source not found at {CUDA_SRC}")
        sys.exit(1)

    if sys.platform == "win32":
        print("Windows detected — CUDA wheels require MSVC host compiler.")
        print("The pure-torch GPU backend (hyperretro.kernels.gpu) works without compilation.")
        print("For production CUDA kernels, build on Linux CI with this script.")
        print("Skipping CUDA compilation.")
        return

    print("Attempting setuptools CUDAExtension build...")
    ok = build_with_setuptools()
    if not ok:
        print("setuptools build failed, trying direct nvcc...")
        ok = build_with_nvcc()
    if ok:
        print("CUDA kernel built successfully.")
    else:
        print("CUDA kernel build skipped — using pure-torch GPU backend.")


if __name__ == "__main__":
    main()
