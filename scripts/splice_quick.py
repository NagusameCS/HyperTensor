#!/usr/bin/env python3
"""Fast chimeric splice measurement — Phase 2+4 only."""
import sys, json, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from grc_distill import build_shared_basis, sink_indices, _load_attn_weights_gguf, _n_layers_gguf

model = "models/smollm2-135m-instruct-q8_0.gguf"
k = 32; T = 32
n_layers = _n_layers_gguf(model)
print(f"Model: {model}")
print(f"Layers: {n_layers}, k={k}, sink_T={T}\n")

results = []
for layer in [0, 10, 25]:
    Wq, Wk, Wv = _load_attn_weights_gguf(model, layer)
    sinks = sink_indices(Wq, Wk, Wv, T)
    Wq_p, Wk_p, Wv_p = Wq.copy(), Wk.copy(), Wv.copy()
    Wq_p[:, sinks] = 0.0; Wk_p[:, sinks] = 0.0; Wv_p[:, sinks] = 0.0

    P = build_shared_basis(Wq_p, Wk_p, Wv_p)
    P_k = P[:, :k]

    # Splice residual: project attention into intrinsic subspace
    for sname, W in [("Q", Wq), ("K", Wk), ("V", Wv)]:
        Wp = W @ P_k @ P_k.T
        err = np.linalg.norm(W - Wp, "fro") / max(np.linalg.norm(W, "fro"), 1e-10)
        en = np.linalg.norm(Wp, "fro")**2 / max(np.linalg.norm(W, "fro")**2, 1e-10)
        results.append({"layer": layer, "slot": sname, "rel_err": round(float(err), 4),
                        "energy": round(float(en), 4)})

    # LoRA rho estimate
    P_perp = P[:, k:]
    tres, rec = 0.0, 0.0
    for W in [Wq, Wk, Wv]:
        V = W @ P_perp
        eta = np.linalg.norm(V, "fro")
        tres += eta**2
        r = 8; ke = min(r, min(V.shape))
        if ke > 0:
            U, S, Vt = np.linalg.svd(V, full_matrices=False)
            rec += np.sum(S[:ke]**2)
    rho = rec / max(tres, 1e-10)
    results.append({"layer": layer, "rho": round(float(rho), 4),
                    "residual_f2": round(float(tres), 1), "recoverable_f2": round(float(rec), 1)})

    print(f"Layer {layer}: {len(sinks)} sinks, rho={rho:.4f}")
    sr = [r for r in results if r.get("slot")]
    for r in sr[-3:]:
        print(f"  {r['slot']}: err={r['rel_err']:.4f}, energy={r['energy']:.4f}")

out = Path("benchmarks/chimeric_splice")
out.mkdir(parents=True, exist_ok=True)
with open(out / "splice_residual.json", "w") as f:
    json.dump({"config": {"model": model, "k": k, "sink_T": T}, "results": results}, f, indent=2)
print(f"\nDone — {out / 'splice_residual.json'}")
