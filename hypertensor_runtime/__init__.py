"""
hypertensor-runtime — installable package wrapping the geodessical C binary and
libhypercore native library for the current platform.

This is a thin packaging shim. The actual binaries live under
`hypertensor_runtime/bin/<platform>/` and are produced by:
  - CMake (libhypercore.so / .dylib / .dll)
  - The host build script (geodessical / geodessical.exe)

Build wheels per platform with cibuildwheel; see deploy/cibuildwheel.toml.
"""
from __future__ import annotations
import os, platform, subprocess, sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent


def _platform_dir() -> str:
    s = platform.system().lower()
    m = platform.machine().lower()
    if s == "windows": return f"win-{m}"
    if s == "darwin":  return f"mac-{m}"
    return f"linux-{m}"


def bin_dir() -> Path:
    return PKG_DIR / "bin" / _platform_dir()


def geodessical_path() -> Path | None:
    exe = "geodessical.exe" if platform.system() == "Windows" else "geodessical"
    p = bin_dir() / exe
    return p if p.exists() else None


def libhypercore_path() -> Path | None:
    sysname = platform.system()
    name = ("hypercore.dll" if sysname == "Windows"
            else "libhypercore.dylib" if sysname == "Darwin"
            else "libhypercore.so")
    p = bin_dir() / name
    return p if p.exists() else None


def run_geodessical(*args: str) -> int:
    g = geodessical_path()
    if not g:
        print(f"geodessical binary not bundled for {_platform_dir()}", file=sys.stderr)
        print("Build from source: see CMakeLists.txt and build_host.ps1", file=sys.stderr)
        return 127
    return subprocess.call([str(g), *args])


def cli_main() -> int:
    return run_geodessical(*sys.argv[1:])


__version__ = "0.1.0"
__all__ = ["geodessical_path", "libhypercore_path", "run_geodessical", "bin_dir"]
