"""
compute_kint.py  ---  Paper A cross-model generalisation data

For each GGUF model, sample a subset of layers, compute the joint
Gram eigenspectrum of {W_Q, W_K, W_V}, and find k_int (the rank needed
to capture 95% of the joint Frobenius energy).  Outputs a JSON summary
and a row for the LaTeX generalisation table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _field_int(reader, suffix: str, default: int = 0) -> int:
    for f in reader.fields.values():
        if f.name.endswith(suffix):
            return int(f.parts[-1][0])
    return default


def load_qkv(reader, layer_idx: int):
    """Return (Wq, Wk, Wv) as float32 arrays for one layer.

    Supports split tensors (attn_q/attn_k/attn_v) and fused attn_qkv.
    """
    from gguf import dequantize

    tm = {t.name: t for t in reader.tensors}
    prefix = f"blk.{layer_idx}"

    def _get(key: str) -> np.ndarray:
        t = tm[key]
        return dequantize(t.data, t.tensor_type).astype(np.float32)

    q_key = f"{prefix}.attn_q.weight"
    k_key = f"{prefix}.attn_k.weight"
    v_key = f"{prefix}.attn_v.weight"

    if q_key in tm and k_key in tm and v_key in tm:
        return _get(q_key), _get(k_key), _get(v_key)

    qkv_key = f"{prefix}.attn_qkv.weight"
    if qkv_key in tm:
        Wqkv = _get(qkv_key)
        q_rows = _field_int(reader, ".attention.head_count") * _field_int(reader, ".attention.key_length")
        kv_rows = _field_int(reader, ".attention.head_count_kv") * _field_int(reader, ".attention.key_length")

        if q_rows > 0 and kv_rows > 0 and (q_rows + 2 * kv_rows) <= Wqkv.shape[0]:
            Wq = Wqkv[:q_rows, :]
            Wk = Wqkv[q_rows:q_rows + kv_rows, :]
            Wv = Wqkv[q_rows + kv_rows:q_rows + 2 * kv_rows, :]
            return Wq, Wk, Wv

        if Wqkv.shape[0] % 3 == 0:
            s = Wqkv.shape[0] // 3
            return Wqkv[:s, :], Wqkv[s:2 * s, :], Wqkv[2 * s:, :]

        raise KeyError(f"{prefix}: unsupported fused attn_qkv layout {Wqkv.shape}")

    raise KeyError(f"{prefix}: missing required Q/K/V tensors")


def joint_gram_eigvals(Wq, Wk, Wv):
    d = Wq.shape[1]
    K = Wq.T @ Wq + Wk.T @ Wk + Wv.T @ Wv
    K = K / (np.linalg.norm(K, "fro") + 1e-12)
    eigvals = np.linalg.eigvalsh(K)
    eigvals = np.sort(eigvals)[::-1]
    return eigvals.astype(np.float64)


def kint_at_var(eigvals: np.ndarray, threshold: float = 0.95) -> int:
    total = float(np.sum(np.maximum(eigvals, 0.0)))
    if total == 0:
        return len(eigvals)
    cumsum = 0.0
    for i, v in enumerate(eigvals):
        cumsum += max(float(v), 0.0)
        if cumsum / total >= threshold:
            return i + 1
    return len(eigvals)


def n_layers_gguf(gguf_path: str) -> int:
    from gguf import GGUFReader

    r = GGUFReader(gguf_path)
    for f in r.fields.values():
        if f.name.endswith(".block_count"):
            return int(f.parts[-1][0])
    indices = set()
    for t in r.tensors:
        parts = t.name.split(".")
        if parts[0] == "blk" and len(parts) > 1 and parts[1].isdigit():
            indices.add(int(parts[1]))
    return max(indices) + 1 if indices else 0


def model_dim_gguf(gguf_path: str) -> int:
    from gguf import GGUFReader

    r = GGUFReader(gguf_path)
    for f in r.fields.values():
        if f.name.endswith(".embedding_length"):
            return int(f.parts[-1][0])
    for t in r.tensors:
        if ".attn_q.weight" in t.name or ".attn_qkv.weight" in t.name:
            return int(t.shape[-1])
    return 0


def main():
    from gguf import GGUFReader

    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", required=True, help="Path to GGUF blob")
    ap.add_argument("--name", required=True, help="Model display name")
    ap.add_argument("--layers", nargs="+", type=int, default=None,
                    help="Layer indices to sample (default: 5 evenly spaced)")
    ap.add_argument("--out", default="docs/figures/paper-a/kint_30b",
                    help="Output directory")
    ap.add_argument("--threshold", type=float, default=0.95,
                    help="Variance threshold for k_int (default 0.95)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = GGUFReader(args.blob)
    n_layers = n_layers_gguf(args.blob)
    d = model_dim_gguf(args.blob)
    print(f"[kint] model={args.name}  n_layers={n_layers}  d={d}", flush=True)

    if args.layers is None:
        idxs = [int(round(i * (n_layers - 1) / 4)) for i in range(5)]
    else:
        idxs = args.layers

    print(f"[kint] sampling layers: {idxs}", flush=True)

    kints = []
    used_layers = []
    skipped = []

    for layer in idxs:
        print(f"  layer {layer:3d}  loading Q/K/V ...", end="", flush=True)
        try:
            Wq, Wk, Wv = load_qkv(reader, layer)
        except KeyError as e:
            msg = str(e)
            skipped.append({"layer": layer, "reason": msg})
            print(f" skipped ({msg})", flush=True)
            continue

        print(f" shapes Q={Wq.shape} K={Wk.shape} V={Wv.shape} ...", end="", flush=True)
        eigvals = joint_gram_eigvals(Wq, Wk, Wv)
        k = kint_at_var(eigvals, args.threshold)
        used_layers.append(layer)
        kints.append(k)
        print(f" k_int={k}  (k/d={k/d:.4f})", flush=True)

    if not kints:
        print("[kint] ERROR: no valid layers were processed", file=sys.stderr)
        sys.exit(1)

    mean_kint = float(np.mean(kints))
    mean_ratio = mean_kint / d

    print(f"\n[kint] {args.name}: mean k_int={mean_kint:.1f}  mean k/d={mean_ratio:.4f}", flush=True)
    print(f"[kint] range: [{min(kints)}, {max(kints)}]", flush=True)
    if skipped:
        print(f"[kint] skipped layers: {len(skipped)}", flush=True)

    result = {
        "model": args.name,
        "blob": args.blob,
        "n_layers": n_layers,
        "d": d,
        "threshold": args.threshold,
        "requested_layers": idxs,
        "used_layers": used_layers,
        "skipped_layers": skipped,
        "per_layer_kint": kints,
        "mean_kint": mean_kint,
        "mean_kint_over_d": mean_ratio,
        "min_kint": min(kints),
        "max_kint": max(kints),
    }

    tag = args.name.replace(" ", "_").replace("/", "_")
    out_file = out_dir / f"kint_{tag}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[kint] saved -> {out_file}", flush=True)

    print("\nLaTeX row:")
    print(f"  {args.name} & {d} & {int(round(mean_kint))} & ${mean_ratio:.4f}$ " + r"\\")


if __name__ == "__main__":
    main()

