"""
ht_repro.runtime_loader — locate and load the geodessical / libhypercore native runtime.

Search order:
  1. $HT_RUNTIME_PATH (explicit override)
  2. package-bundled binary at ht_repro/bin/<platform>/
  3. system installs (/usr/local/lib, /usr/lib, $HT_LIB_DIR)
  4. project build dirs (../build_host, ../build)
"""
from __future__ import annotations
import ctypes, os, platform, sys
from pathlib import Path

_PKG = Path(__file__).resolve().parent


def _platform_dir() -> str:
    s = platform.system().lower()
    m = platform.machine().lower()
    if s == "windows": return f"win-{m}"
    if s == "darwin":  return f"mac-{m}"
    return f"linux-{m}"


def _bin_name(stem: str) -> str:
    if platform.system() == "Windows":
        return f"{stem}.exe" if stem in {"geodessical"} else f"{stem}.dll"
    if platform.system() == "Darwin":
        return f"{stem}" if stem in {"geodessical"} else f"lib{stem}.dylib"
    return f"{stem}" if stem in {"geodessical"} else f"lib{stem}.so"


def _search(name: str) -> Path | None:
    candidates: list[Path] = []
    if os.environ.get("HT_RUNTIME_PATH"):
        candidates.append(Path(os.environ["HT_RUNTIME_PATH"]) / name)
    candidates.append(_PKG / "bin" / _platform_dir() / name)
    # Sibling hypertensor-runtime package (if installed)
    try:
        import hypertensor_runtime as _hr  # type: ignore
        candidates.append(Path(_hr.__file__).parent / "bin" / _platform_dir() / name)
    except Exception:
        pass
    for d in (os.environ.get("HT_LIB_DIR"), "/usr/local/lib", "/usr/lib", "/opt/hypertensor/lib"):
        if d: candidates.append(Path(d) / name)
    root = _PKG.parent
    for sub in ("build_host", "build", "build_release", "hypertensor_runtime/bin/" + _platform_dir()):
        candidates.append(root / sub / name)
    for c in candidates:
        if c.exists(): return c
    return None


def find_geodessical() -> Path | None:
    return _search(_bin_name("geodessical"))


def load_libhypercore() -> ctypes.CDLL | None:
    p = _search(_bin_name("hypercore"))
    if not p: return None
    try: return ctypes.CDLL(str(p))
    except OSError: return None


def diagnostic() -> dict:
    g = find_geodessical()
    return {
        "platform": _platform_dir(),
        "geodessical": str(g) if g else None,
        "libhypercore_loadable": load_libhypercore() is not None,
        "HT_RUNTIME_PATH": os.environ.get("HT_RUNTIME_PATH"),
        "HT_LIB_DIR": os.environ.get("HT_LIB_DIR"),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(diagnostic(), indent=2))
