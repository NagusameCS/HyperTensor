"""
inventory_models.py
====================

Walk C:\\Users\\legom\\models, gather all GGUF files, parse the GGUF header for
architecture, parameter count (approx via tensor sizes), embedding dim, layer
count, vocab size, and quantisation. Emit:

  * ``docs/figures/model_inventory.json``
  * ``docs/figures/model_inventory.md`` (Markdown table)

The data is what backs the "Model memory inventory" line in Paper 0.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path(r"C:\Users\legom\models")

# GGUF tag types
GGUF_TYPES = {
    0: ("u8", 1, "<B"), 1: ("i8", 1, "<b"),
    2: ("u16", 2, "<H"), 3: ("i16", 2, "<h"),
    4: ("u32", 4, "<I"), 5: ("i32", 4, "<i"),
    6: ("f32", 4, "<f"),
    7: ("bool", 1, "<?"),
    8: ("string", None, None),
    9: ("array", None, None),
    10: ("u64", 8, "<Q"), 11: ("i64", 8, "<q"),
    12: ("f64", 8, "<d"),
}


def read_gguf_string(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def read_gguf_value(f, t: int):
    if t == 8:
        return read_gguf_string(f)
    if t == 9:
        sub_t = struct.unpack("<I", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        return [read_gguf_value(f, sub_t) for _ in range(n)]
    name, size, fmt = GGUF_TYPES[t]
    return struct.unpack(fmt, f.read(size))[0]


def parse_gguf_header(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return {"error": f"not GGUF (magic={magic!r})"}
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        kv: Dict[str, Any] = {}
        for _ in range(n_kv):
            key = read_gguf_string(f)
            t = struct.unpack("<I", f.read(4))[0]
            try:
                kv[key] = read_gguf_value(f, t)
            except Exception as e:
                kv[key] = f"<read-fail:{e}>"
                break
    return {"version": version, "n_tensors": n_tensors, "n_kv": n_kv, "kv": kv}


def summarise(path: Path) -> Dict[str, Any]:
    size_mb = round(path.stat().st_size / (1024 * 1024), 1)
    if size_mb < 1:
        return {"path": str(path), "size_MB": size_mb, "stub": True}
    try:
        h = parse_gguf_header(path)
    except Exception as e:
        return {"path": str(path), "size_MB": size_mb, "error": str(e)}
    kv = h.get("kv", {})
    arch = kv.get("general.architecture", "?")
    rec = {
        "path": str(path).replace(str(MODELS_ROOT), "<models>"),
        "size_MB": size_mb,
        "arch": arch,
        "name": kv.get("general.name", path.stem),
        "quant": kv.get("general.file_type", "?"),
        "n_tensors": h.get("n_tensors"),
    }
    # Architecture-specific keys
    for key, full_keys in (
        ("embedding_length",   [f"{arch}.embedding_length"]),
        ("block_count",        [f"{arch}.block_count"]),
        ("context_length",     [f"{arch}.context_length"]),
        ("feed_forward_length", [f"{arch}.feed_forward_length"]),
        ("head_count",         [f"{arch}.attention.head_count", f"{arch}.head_count"]),
        ("head_count_kv",      [f"{arch}.attention.head_count_kv", f"{arch}.head_count_kv"]),
        ("vocab_size",         [f"{arch}.vocab_size", "tokenizer.ggml.tokens"]),
    ):
        for full in full_keys:
            if full in kv:
                v = kv[full]
                if isinstance(v, list):
                    rec[key] = len(v)
                else:
                    rec[key] = v
                break

    # Decode quant code → readable name (llama.cpp file_type enum)
    quant_map = {
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0",
        8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
        13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
        17: "Q5_K_M", 18: "Q6_K", 30: "BF16",
    }
    if isinstance(rec["quant"], int):
        rec["quant"] = quant_map.get(rec["quant"], f"ftype{rec['quant']}")
    return rec


def main():
    if not MODELS_ROOT.exists():
        print(f"models root not found: {MODELS_ROOT}")
        return
    rows: List[Dict[str, Any]] = []
    for path in sorted(MODELS_ROOT.rglob("*.gguf")):
        rows.append(summarise(path))

    out_dir = REPO / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_inventory.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")

    md = ["# Local Model Inventory", "",
          f"Generated from `{MODELS_ROOT}` on disk.", "",
          "| Model | Arch | Quant | Size (MB) | dim | layers | heads | vocab | ctx |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r.get("stub"):
            md.append(f"| {Path(r['path']).name} | _stub (0 B)_ | — | 0 | — | — | — | — | — |")
            continue
        md.append("| {n} | {a} | {q} | {sz} | {d} | {l} | {h} | {v} | {c} |".format(
            n=r.get("name", Path(r["path"]).name),
            a=r.get("arch", "?"),
            q=r.get("quant", "?"),
            sz=r.get("size_MB", "?"),
            d=r.get("embedding_length", "—"),
            l=r.get("block_count", "—"),
            h=r.get("head_count", "—"),
            v=r.get("vocab_size", "—"),
            c=r.get("context_length", "—"),
        ))
    (out_dir / "model_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for line in md:
        print(line)


if __name__ == "__main__":
    main()
