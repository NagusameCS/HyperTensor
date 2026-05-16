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

"""Proper speculative decoding simulation — multi-token acceptance + speedup.

Measures the REAL performance of HyperRetro speculative decoding:
  1. Drafter proposes γ tokens autoregressively (fast, incremental)
  2. Verifier scores ALL γ tokens in ONE forward pass
  3. Greedy acceptance: accept matching prefix
  4. Report: mean acceptance length, effective speedup, draft latency

This is the honest "would this actually work in production" benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def simulate_speculative_decode(
    model_id: str,
    prompt: str,
    k: int,
    gamma: int = 4,
    device: str = "cuda",
    dtype: str = "float32",
    max_cycles: int = 50,
    drafter_path: str | None = None,
):
    """Simulate speculative decoding with HyperRetro drafter.

    Protocol (greedy, matches Leviathan et al. 2023 §2):
      1. Drafter generates γ tokens from prefix
      2. Verifier runs ONE forward pass on [prefix + all γ draft tokens]
      3. For i in 0..γ-1: accept draft[i] if verifier_argmax[i] == draft[i]
      4. Accept matching prefix, append to generated tokens
      5. Repeat

    Args:
        model_id: HuggingFace model ID for the verifier.
        drafter_path: If provided, load a pre-compressed/distilled model
            as drafter (skips GRC compression). Otherwise compress from
            model_id at rank k.

    Returns metrics dict.
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
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- Load models ---
    print(f"Loading verifier: {model_id}")
    verifier = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype,
    ).to(dev)
    verifier.eval()

    # --- Load drafter ---
    if drafter_path:
        print(f"Loading pre-compressed drafter: {drafter_path}")
        drafter_model = AutoModelForCausalLM.from_pretrained(
            drafter_path, torch_dtype=torch_dtype,
        ).to(dev)
        drafter_model.eval()
    else:
        print(f"Loading drafter (compressing at k={k})...")
        drafter_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype,
        )
        sd = drafter_model.state_dict()
        cfg = CompressConfig(rank_k=k, sink_T=4, dtype=dtype)
        compress_state_dict(sd, cfg)
        drafter_model = drafter_model.to(dev)
        drafter_model.load_state_dict(sd)
        drafter_model.eval()
    drafter = CompressedDrafter(drafter_model, cfg=DraftConfig(n_drafts=gamma),
                                dtype=torch_dtype)

    # --- Run simulation ---
    ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
    generated = ids.clone()

    total_accepted = 0
    total_cycles = 0
    draft_times = []
    verify_times = []
    accept_lengths = []

    if dev.type == "cuda":
        # Warmup
        _ = drafter.propose(generated[:, :10], n_drafts=1)
        _ = verifier(generated[:, :10])
        torch.cuda.synchronize()

    for cycle in range(max_cycles):
        if generated.size(1) > 500:  # safety limit
            break

        # 1. DRAFT: generate γ tokens
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        draft_ids, confs = drafter.propose(generated, n_drafts=gamma)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        draft_time = (time.perf_counter() - t0) * 1000
        draft_times.append(draft_time)

        if len(draft_ids) == 0:
            break

        # 2. VERIFY: score all draft tokens in one forward pass
        draft_tensor = torch.as_tensor(draft_ids, dtype=torch.long,
                                       device=dev).unsqueeze(0)  # [1, γ]
        verify_input = torch.cat([generated, draft_tensor], dim=1)  # [1, T+γ]

        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            verify_out = verifier(verify_input)
            # Logits at position P predict token at P+1.
            # We need predictions for draft positions T through T+γ-1.
            # verify_logits[T-1] → token T (1st draft)
            # verify_logits[T]   → token T+1 (2nd draft)
            # ...
            # verify_logits[T+γ-2] → token T+γ-1 (γ-th draft)
            # verify_logits[T+γ-1] → token T+γ (correction token)
            T = generated.size(1)
            verify_logits = verify_out.logits[0, T-1:T+gamma, :]  # [γ+1, vocab]
            verify_argmax = verify_logits.argmax(dim=-1).cpu().numpy()  # [γ+1]

        if dev.type == "cuda":
            torch.cuda.synchronize()
        verify_time = (time.perf_counter() - t1) * 1000
        verify_times.append(verify_time)

        # 3. ACCEPT: greedy matching prefix
        # verify_argmax[i] = verifier prediction at draft position i
        # draft_ids[i] = drafter's token at position i
        n_accept = 0
        for i in range(gamma):
            if int(draft_ids[i]) == int(verify_argmax[i]):
                n_accept += 1
            else:
                break

        accept_lengths.append(n_accept)
        total_accepted += n_accept
        total_cycles += 1

        # 4. ADVANCE: append accepted drafts + one correction token
        # The correction token is verify_argmax[n_accept] (verifier's
        # token at the first non-matching position, or next token after
        # all accepted drafts).
        if n_accept > 0:
            accepted_tokens = draft_ids[:n_accept]
            new_tokens = torch.as_tensor(accepted_tokens, dtype=torch.long,
                                         device=dev).unsqueeze(0)
            generated = torch.cat([generated, new_tokens], dim=1)
        
        correction = int(verify_argmax[n_accept])
        corr_tensor = torch.as_tensor([[correction]], dtype=torch.long, device=dev)
        generated = torch.cat([generated, corr_tensor], dim=1)

        if (cycle + 1) % 10 == 0:
            mean_accept = total_accepted / total_cycles if total_cycles > 0 else 0
            print(f"  cycle {cycle+1:3d}: accepted={n_accept}, "
                  f"mean_accept={mean_accept:.2f}, "
                  f"draft={draft_time:.1f}ms, verify={verify_time:.1f}ms",
                  flush=True)

    # --- Metrics ---
    mean_accept = total_accepted / total_cycles if total_cycles > 0 else 0
    median_draft = float(np.median(draft_times)) if draft_times else 0
    median_verify = float(np.median(verify_times)) if verify_times else 0

    # Each cycle produces: accepted drafts + 1 correction token
    tokens_per_cycle = mean_accept + 1.0
    # Speedup vs autoregressive (1 token per verify pass):
    speedup = tokens_per_cycle / 1.0

    # Wall-clock speedup: tokens_per_cycle * baseline_ms_per_token / (draft_ms + verify_ms)
    baseline_ms_per_token = median_verify  # one autoregressive step ≈ one verify pass
    wallclock_speedup = (tokens_per_cycle * baseline_ms_per_token) / (median_draft + median_verify) if (median_draft + median_verify) > 0 else 1.0

    result = {
        "model": model_id,
        "k": k,
        "gamma": gamma,
        "dtype": dtype,
        "device": device,
        "drafter_source": drafter_path if drafter_path else f"compressed_from_{model_id}_k{k}",
        "total_cycles": total_cycles,
        "total_accepted_tokens": total_accepted,
        "mean_acceptance_length": float(mean_accept),
        "acceptance_rate": float(mean_accept / gamma) if gamma > 0 else 0,
        "acceptance_lengths": accept_lengths[:20],
        "median_draft_ms": median_draft,
        "median_verify_ms": median_verify,
        "tokens_per_cycle": float(tokens_per_cycle),
        "speedup_vs_autoregressive": float(speedup),
        "wallclock_speedup": float(wallclock_speedup),
    }

    del verifier, drafter_model
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser(
        description="Proper speculative decoding simulation"
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--prompt", default=(
        "The history of artificial intelligence began in antiquity with "
        "myths and stories of artificial beings endowed with intelligence "
        "by master craftsmen. In the 1950s a generation of scientists"
    ))
    p.add_argument("--k", type=int, default=640)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--cycles", type=int, default=50)
    p.add_argument("--drafter-path", default=None,
                   help="Path to pre-compressed/distilled drafter model (skip GRC)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print(f"HyperRetro Speculative Decoding Simulation")
    print(f"  Model: {args.model}  k={args.k}  γ={args.gamma}  dtype={args.dtype}")
    print(f"  Max cycles: {args.cycles}")
    print()

    result = simulate_speculative_decode(
        args.model, args.prompt, args.k,
        gamma=args.gamma, device=args.device, dtype=args.dtype,
        max_cycles=args.cycles, drafter_path=args.drafter_path,
    )

    print(f"\n{'='*55}")
    print(f"RESULTS")
    print(f"{'='*55}")
    print(f"  Cycles completed:           {result['total_cycles']}")
    print(f"  Total drafts accepted:      {result['total_accepted_tokens']}")
    print(f"  Mean acceptance length:     {result['mean_acceptance_length']:.2f} / {args.gamma}")
    print(f"  Acceptance rate:            {result['acceptance_rate']:.1%}")
    print(f"  Tokens per cycle:           {result['tokens_per_cycle']:.2f}")
    print(f"  Speedup vs autoregressive:  {result['speedup_vs_autoregressive']:.2f}×")
    print(f"  Wall-clock speedup:         {result['wallclock_speedup']:.2f}×")
    print(f"  Median draft time:          {result['median_draft_ms']:.1f} ms")
    print(f"  Median verify time:         {result['median_verify_ms']:.1f} ms")
    print(f"  First 10 accept lengths:    {result['acceptance_lengths'][:10]}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
