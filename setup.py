"""
setup.py — HyperTensor build & install

Builds the `libhypercore` shared library via CMake and exposes it as a
Python package.  The Python layer (hypercore/) loads the compiled .so/.dylib
at runtime via ctypes; no CPython extension ABI is needed.

Usage
-----
# Development install (builds in-place; CMake output goes to build/):
    pip install -e .

# Wheel build:
    pip install build && python -m build

# Manual CMake (if you prefer to drive the build yourself):
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel
    # Then set HT_LIB_DIR=build/ before importing

Requirements
------------
- CMake >= 3.18
- A C11 compiler  (gcc, clang, or MSVC)
- LAPACKE  (liblapacke-dev on Ubuntu, openblas on macOS)
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).parent.resolve()


class CMakeBuild(build_ext):
    """Drive CMake from inside setuptools so `pip install .` just works.

    If CMake is not available or the build fails, the C extension is
    skipped and only Python packages are installed.
    """

    def build_extension(self, ext):
        # Check for cmake
        if not shutil.which("cmake"):
            print("[hypercore] CMake not found — skipping C extension build")
            return

        build_dir = Path(self.build_temp) / ext.name
        build_dir.mkdir(parents=True, exist_ok=True)

        install_dir = Path(self.build_lib) / "hypercore"
        install_dir.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            "-DHT_BUILD_RUNTIME=OFF",
            "-DHT_BUILD_TESTS=OFF",
        ]

        if platform.system() == "Windows":
            if shutil.which("ninja"):
                cmake_args += ["-GNinja"]
        else:
            if shutil.which("ninja"):
                cmake_args += ["-GNinja"]

        build_args = ["--build", str(build_dir), "--config", "Release",
                      "--parallel", str(os.cpu_count() or 4)]

        try:
            subprocess.check_call(
                ["cmake", str(ROOT)] + cmake_args,
                cwd=build_dir,
            )
            subprocess.check_call(["cmake"] + build_args)
            subprocess.check_call(
                ["cmake", "--install", str(build_dir), "--config", "Release"],
            )
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"[hypercore] CMake build failed: {e}")
            print("[hypercore] Skipping C extension — Python packages only")
            return

        # Copy shared library
        lib_suffix = {
            "Linux":   ".so",
            "Darwin":  ".dylib",
            "Windows": ".dll",
        }.get(platform.system(), ".so")

        lib_name = f"libhypercore{lib_suffix}"
        src = install_dir / "lib" / lib_name
        dst = Path(self.build_lib) / "hypercore" / lib_name

        if src.exists():
            shutil.copy2(src, dst)
        else:
            alt = install_dir / "bin" / f"hypercore{lib_suffix}"
            if alt.exists():
                shutil.copy2(alt, dst)
            else:
                print(f"[hypercore] Built library not found — skipping")


# Only register the C extension if cmake is available
_cmake_ext = Extension("hypercore._hypercore", sources=[])
_ext_modules = [_cmake_ext] if shutil.which("cmake") else []
_cmdclass = {"build_ext": CMakeBuild} if shutil.which("cmake") else {}

setup(
    name="hypertensor",
    version="1.1.0",
    description=(
        "Riemannian geometry for transformer compression, speculative decoding, "
        "safety, and the Riemann Hypothesis"
    ),
    author="William 'Nagusame' Ken Ohara Stewart",
    license="MIT",
    python_requires=">=3.10",
    packages=[
        "hypercore",
        "hyperretro",
        "hyperretro.bench",
        "hyperretro.hf",
        "hyperretro.kernels",
        "hyperretro.models",
        "hyperretro.vllm",
    ],
    package_dir={
        "hypercore": "hypercore",
        "hyperretro": "hyperretro",
        "hyperretro.bench": "hyperretro/bench",
        "hyperretro.hf": "hyperretro/hf",
        "hyperretro.kernels": "hyperretro/kernels",
        "hyperretro.models": "hyperretro/models",
        "hyperretro.vllm": "hyperretro/vllm",
    },
    ext_modules=_ext_modules,
    cmdclass=_cmdclass,
    entry_points={
        "console_scripts": [
            "hyperretro=hyperretro.cli:main",
        ],
    },
    install_requires=[
        "numpy>=1.24",
    ],
    extras_require={
        "torch": ["torch>=2.0"],
        "full": [
            "torch>=2.0",
            "transformers>=4.40",
            "safetensors>=0.4",
            "bitsandbytes>=0.43",
        ],
        "repro": [
            "numpy>=1.24",
            "scipy>=1.11",
            "mpmath>=1.3",
            "sympy>=1.12",
        ],
        "om": ["open-mythos"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
    ],
)
