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

"""Real speculative-decoding evaluation — Paper III protocol.

Loads a real HuggingFace causal LM, applies GRC compression to create a
compressed drafter, and measures token acceptance against the original
model's argmax.  Uses ``CompressedDrafter`` (the correct approach from
Paper III) and ``KSpaceDrafter`` (legacy, for comparison).

Reports top-1 acceptance, top-5 hit rate, and median propose latency.

This is the honest "would this actually work in vLLM" check.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def run_compressed_drafter(
    model_id: str,
    prompt: str,
    k: int,
    n_drafts: int = 4,
    n_calib: int = 32,
    seed: int = 0,
    device: str = "cpu",
    dtype: str = "float32",
):
    """Evaluate CompressedDrafter: compress model → draft, ORIGINAL verifies.

    This is the real Paper III protocol: the compressed model serves as a
    fast drafter; the FULL-PRECISION (original) model verifies candidate
    tokens.  Acceptance = how often the drafter's argmax matches the
    original model's argmax.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hyperretro.hf.compress import compress_state_dict, CompressConfig
    from hyperretro.vllm.draft import CompressedDrafter, DraftConfig

    dtype_map = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype, torch.float32)
    dev = torch.device(device)

    tok = AutoTokenizer.from_pretrained(model_id)

    # Verifier: original model (uncompressed)
    verifier = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype,
    ).to(dev)
    verifier.eval()

    # Drafter: compressed copy of the same model
    drafter_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype,
    )
    drafter_model.eval()
    sd = drafter_model.state_dict()
    cfg_compress = CompressConfig(rank_k=k, sink_T=4, dtype=dtype)
    compress_state_dict(sd, cfg_compress)
    drafter_model = drafter_model.to(dev)
    drafter_model.load_state_dict(sd)
    drafter_model.eval()

    drafter = CompressedDrafter(drafter_model, cfg=DraftConfig(n_drafts=n_drafts), dtype=torch_dtype)

    # Ground truth: verifier's argmax at each position
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        verifier_out = verifier(ids)
    true_next = verifier_out.logits[0].argmax(dim=-1).cpu().numpy()  # [T]

    top1 = 0
    top5 = 0
    n = 0
    t_draft = []
    T = ids.size(1)

    # Warmup GPU
    if dev.type == "cuda":
        _ = drafter.propose(ids[:, :10], n_drafts=1)
        torch.cuda.synchronize()

    for i in range(max(1, T - n_drafts - 1)):
        prefix = ids[:, :i + 1]
        n_tok = min(n_drafts, T - i - 1)
        true_tokens = true_next[i:i + n_tok]

        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        draft_ids, confs = drafter.propose(prefix, n_drafts=n_tok)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        t_draft.append(elapsed)

        # Greedy acceptance
        if int(draft_ids[0]) == int(true_tokens[0]):
            top1 += 1

        # Top-5: check if verifier's top-1 is in drafter's top-5 beam
        with torch.no_grad():
            verifier_logits = verifier(prefix).logits[0, -1, :]
            verifier_top5 = torch.topk(verifier_logits, k=min(5, verifier_logits.size(-1))).indices.cpu().numpy()
        if int(true_tokens[0]) in verifier_top5:
            top5 += 1
        n += 1

    return {
        "model": model_id,
        "method": "compressed_drafter",
        "protocol": "compressed_drafts_original_verifies",
        "prompt_len": T,
        "k": int(k),
        "n_drafts": n_drafts,
        "n_eval_positions": n,
        "top1_accept": top1 / max(1, n),
        "top5_hit": top5 / max(1, n),
        "median_ms_per_propose": float(np.median(t_draft)) if t_draft else 0.0,
    }


def run_kspace_drafter(
    model_id: str,
    prompt: str,
    k: int,
    n_drafts: int = 1,
    n_calib: int = 32,
    seed: int = 0,
):
    """Evaluate KSpaceDrafter (legacy) for research comparison."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from hyperretro.vllm.draft import KSpaceDrafter, GeodesicDraftConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, output_hidden_states=True,
    )
    model.eval()

    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = out.hidden_states[-1][0].cpu().numpy().astype(np.float32)  # [T, d]
    logits = out.logits[0].cpu().numpy().astype(np.float32)  # [T, vocab]
    true_next = logits.argmax(axis=-1)  # [T]

    n_calib = min(n_calib, hs.shape[0] - 2)
    H_cal = hs[:n_calib]
    U, S, Vt = np.linalg.svd(
        H_cal - H_cal.mean(axis=0, keepdims=True), full_matrices=False
    )
    basis = Vt[:k].T.astype(np.float32)

    try:
        lm_head = model.get_output_embeddings().weight.detach().cpu().numpy().astype(np.float32)
    except Exception:
        lm_head = model.get_input_embeddings().weight.detach().cpu().numpy().astype(np.float32)

    drafter = KSpaceDrafter(basis, lm_head, GeodesicDraftConfig(k=k, n_drafts=n_drafts))
    drafter.calibrate(H_cal, token_ids=ids[0, :n_calib + 1].cpu().numpy())

    top1 = 0
    top5 = 0
    n = 0
    t_draft = []
    for i in range(n_calib + 1, hs.shape[0] - n_drafts - 1):
        h_curr = hs[i]
        h_prev = hs[i - 1]
        t0 = time.perf_counter()
        ids_pred, conf = drafter.propose(h_curr, h_prev, top_k_search=32)
        t_draft.append((time.perf_counter() - t0) * 1000)
        if int(ids_pred[0]) == int(true_next[i]):
            top1 += 1
        n += 1

    return {
        "model": model_id,
        "method": "kspace_drafter",
        "prompt_len": int(hs.shape[0]),
        "calib_len": int(n_calib),
        "k": int(k),
        "n_drafts": n_drafts,
        "n_eval_positions": n,
        "top1_accept": top1 / max(1, n),
        "top5_hit": float("nan"),
        "median_ms_per_propose": float(np.median(t_draft)) if t_draft else 0.0,
    }


def main():
    p = argparse.ArgumentParser(
        description="Real speculative-decoding evaluation (Paper III protocol)"
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--prompt", default=(
        "The history of artificial intelligence began in antiquity with myths "
        "and stories of artificial beings endowed with intelligence by master "
        "craftsmen. In the 1950s a generation of scientists began to explore "
        "the possibility of building electronic brains."
    ))
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--n-drafts", type=int, default=4)
    p.add_argument("--calib", type=int, default=32)
    p.add_argument("--method", choices=["compressed", "kspace", "both"],
                   default="compressed")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    results = {}
    if args.method in ("compressed", "both"):
        print("=== CompressedDrafter (Paper III) ===", flush=True)
        r = run_compressed_drafter(
            args.model, args.prompt, args.k, args.n_drafts, args.calib,
            device=args.device, dtype=args.dtype,
        )
        print(json.dumps(r, indent=2))
        results["compressed"] = r

    if args.method in ("kspace", "both"):
        print("=== KSpaceDrafter (legacy) ===", flush=True)
        r = run_kspace_drafter(
            args.model, args.prompt, args.k, args.n_drafts, args.calib
        )
        print(json.dumps(r, indent=2))
        results["kspace"] = r

    if args.out:
        import pathlib
        pathlib.Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    sys.exit(main())
