"""
ht_repro.models — Automatic model download + cache management.

Wraps huggingface_hub when available. Falls back to direct HTTP for raw files.
Caches metadata in ht_repro.storage so grafting / benchmarking tools can
resolve a HF repo_id to a local path automatically.
"""
from __future__ import annotations
import os, shutil, sys, time
from pathlib import Path
from typing import Optional

from . import storage

# Default cache: ~/.ht-repro/models  (override with HT_REPRO_MODELS env var)
CACHE_ROOT = Path(os.environ.get("HT_REPRO_MODELS", Path.home() / ".ht-repro" / "models"))


def _dir_size_gb(p: Path) -> float:
    if not p.exists(): return 0.0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1024**3, 3)


def ensure(repo_id: str, revision: str = "main", allow_patterns: list[str] | None = None,
           force: bool = False, quiet: bool = False) -> str:
    """Make sure `repo_id` is downloaded. Returns local path.
    Uses HF cache if huggingface_hub is installed; otherwise asks the user
    to install it. If already cached (per our registry), returns immediately
    unless force=True."""
    if not force:
        m = storage.get_model(repo_id)
        if m and m["local_path"] and Path(m["local_path"]).exists():
            storage.register_model(repo_id, m["local_path"], m["size_gb"])  # bump last_used
            if not quiet: print(f"[models] cached: {repo_id} -> {m['local_path']}")
            return m["local_path"]

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed. Run: pip install huggingface_hub")

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not quiet: print(f"[models] downloading {repo_id} (revision={revision}) ...")
    t0 = time.time()
    local = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(CACHE_ROOT),
        allow_patterns=allow_patterns,
        local_files_only=False,
    )
    size = _dir_size_gb(Path(local))
    storage.register_model(repo_id, local, size, meta={"revision": revision,
                                                        "downloaded_in_s": round(time.time()-t0,1)})
    if not quiet: print(f"[models] done: {local}  ({size} GB)")
    return local


def resolve(repo_id_or_path: str) -> str:
    """Accept either a HF repo_id or a local path. If it looks like a path
    that exists, return as-is. Otherwise download via ensure()."""
    p = Path(repo_id_or_path)
    if p.exists():
        return str(p)
    return ensure(repo_id_or_path)


def prefetch(*repo_ids: str, quiet: bool = False) -> dict[str, str]:
    """Download multiple models. Returns repo_id -> local_path."""
    out = {}
    for r in repo_ids:
        try:
            out[r] = ensure(r, quiet=quiet)
        except Exception as e:
            out[r] = f"ERROR: {e}"
    return out


def evict(repo_id: str) -> bool:
    """Delete a cached model. Returns True if removed."""
    m = storage.get_model(repo_id)
    if not m or not m["local_path"]: return False
    p = Path(m["local_path"])
    if p.exists():
        # snapshot_download stores symlinks; resolve to actual snapshots dir parent
        snap = p
        # walk up to a `models--org--name` folder if present
        for parent in p.parents:
            if parent.name.startswith("models--"):
                snap = parent; break
        if snap.exists(): shutil.rmtree(snap, ignore_errors=True)
    return True


def list_cached() -> list[dict]:
    return storage.list_models()


def cache_root() -> str:
    return str(CACHE_ROOT)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(prog="ht-repro-models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("get"); p.add_argument("repo_id"); p.add_argument("--revision", default="main")
    p = sub.add_parser("evict"); p.add_argument("repo_id")
    sub.add_parser("path")
    args = ap.parse_args()
    if args.cmd == "list":
        for m in list_cached():
            print(f"  {m['repo_id']:<40} {m['size_gb']:>6} GB  {m['local_path']}")
    elif args.cmd == "get":
        print(ensure(args.repo_id, revision=args.revision))
    elif args.cmd == "evict":
        print("removed" if evict(args.repo_id) else "not found")
    elif args.cmd == "path":
        print(cache_root())
