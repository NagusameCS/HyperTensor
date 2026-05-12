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

"""Tier-item #7: behavioral-residue invariant.

The volume mentions a "behavioral-residue invariant" as narrative only; no
formal statement appears in any .tex file (grep confirmed: 0 hits).  The
most natural operational reading is:

    Let h_l(x) be the residual-stream activation at layer l on input x.
    Let p(.|x) be the next-token distribution.
    Decompose h_l = h_pred + h_residue, where h_pred is the projection
    of h_l onto the row-space of the unembedding W_U (or a learned
    linear next-token probe), and h_residue is the orthogonal complement.
    Claim: ablating h_residue (replacing it with zero) leaves the output
    distribution KL-invariant up to a small epsilon, while ablating
    h_pred breaks it.

This script tests that operational reading on SmolLM2-135M.  CPU-only
(but uses GPU if available); n=64 short sequences, no training.

Output: benchmarks/behavioral_residue_invariant.json
"""
from __future__ import annotations
import json, sys, os
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "benchmarks" / "behavioral_residue_invariant.json"

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
N_SEQ    = 32
SEQ_LEN  = 24
LAYERS   = [0, 7, 15, 22, 29]


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(device).eval()
    W_U = model.lm_head.weight.detach()  # [V, d]

    # Build the "predictive subspace" P_pred = top-r left singular vectors of W_U
    # (rows of W_U span the predictive subspace; SVD gives an orthonormal basis).
    U, S, _ = torch.linalg.svd(W_U.T.float(), full_matrices=False)  # W_U.T : [d, V]
    # cumulative variance threshold
    cum = (S**2).cumsum(0) / (S**2).sum()
    r = int((cum < 0.95).sum().item()) + 1
    Q_pred = U[:, :r].to(device)  # [d, r]
    Pp = Q_pred @ Q_pred.T        # [d, d] projector onto predictive subspace

    # Build inputs
    rng = np.random.default_rng(0)
    vocab = tok.vocab_size
    ids = torch.tensor(rng.integers(low=10, high=vocab-10, size=(N_SEQ, SEQ_LEN)),
                       dtype=torch.long, device=device)

    # Reference logits
    with torch.no_grad():
        out_ref = model(ids).logits  # [B, T, V]
        log_ref = torch.log_softmax(out_ref, dim=-1)

    results = {"layers": [], "r_pred": r, "d": int(W_U.shape[1])}

    def hooked_forward(layer_idx: int, mode: str):
        """mode: 'ablate_residue' or 'ablate_pred'."""
        layer = model.model.layers[layer_idx]
        def hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h_dtype = h.dtype
            h32 = h.float()
            if mode == "ablate_residue":
                # keep h_pred only:  h <- Pp h
                h_new = h32 @ Pp.T
            elif mode == "ablate_pred":
                # remove h_pred:    h <- (I - Pp) h
                h_new = h32 - h32 @ Pp.T
            else:
                h_new = h32
            h_new = h_new.to(h_dtype)
            if isinstance(out, tuple):
                return (h_new,) + out[1:]
            return h_new
        return layer.register_forward_hook(hook)

    for L in LAYERS:
        kls = {}
        for mode in ("ablate_residue", "ablate_pred"):
            handle = hooked_forward(L, mode)
            try:
                with torch.no_grad():
                    out_a = model(ids).logits
                    log_a = torch.log_softmax(out_a, dim=-1)
                # KL(p_ref || p_a) per token, then mean
                p_ref = log_ref.exp()
                kl = (p_ref * (log_ref - log_a)).sum(-1)  # [B,T]
                kls[mode] = float(kl.mean().item())
            finally:
                handle.remove()
        ratio = kls["ablate_pred"] / max(kls["ablate_residue"], 1e-12)
        results["layers"].append({
            "layer": L,
            "kl_ablate_residue": kls["ablate_residue"],
            "kl_ablate_pred":    kls["ablate_pred"],
            "ratio_pred_over_residue": ratio,
        })
        print(f"L={L:2d}  KL(ablate_residue)={kls['ablate_residue']:.4e}  "
              f"KL(ablate_pred)={kls['ablate_pred']:.4e}  ratio={ratio:.2f}")

    # Verdict
    ratios = [r["ratio_pred_over_residue"] for r in results["layers"]]
    results["verdict"] = {
        "mean_ratio": float(np.mean(ratios)),
        "min_ratio":  float(np.min(ratios)),
        "invariant_holds_if_ratio_gg_1": all(r > 5.0 for r in ratios),
        "interpretation": (
            "If ratio >> 1 across all probed layers, ablating the predictive "
            "subspace breaks output much more than ablating the residue, "
            "supporting the behavioral-residue-invariant reading."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    print(f"verdict: invariant holds = {results['verdict']['invariant_holds_if_ratio_gg_1']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
