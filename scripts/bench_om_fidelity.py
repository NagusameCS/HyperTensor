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

"""OpenMythos vs HyperRetro-compressed OpenMythos: forward pass fidelity.

Measures how faithfully the compressed model reproduces the original's
outputs. This is the "no training" benchmark — a clean comparison of
architecture-level compression quality.

Metrics:
- Weight reconstruction error (per-matrix Frobenius)
- Layer output cosine similarity
- Final logit KL divergence
- Per-token top-5 overlap

Run: python scripts/bench_om_fidelity.py
"""

import json, sys, time
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
sys.path.insert(0, str(_ROOT))

import torch
import torch.nn.functional as F
import numpy as np

from open_mythos import mythos_1b, OpenMythos, MythosConfig
from hyperretro.hf.openmythos import compress_openmythos

# Use a small config for speed
SMALL_CFG = MythosConfig(
    vocab_size=1000,
    dim=256,
    n_heads=8,
    n_kv_heads=2,
    max_seq_len=128,
    max_loop_iters=4,
    prelude_layers=1,
    coda_layers=1,
    n_experts=4,
    n_shared_experts=1,
    n_experts_per_tok=2,
    expert_dim=128,
    attn_type="gqa",
    kv_lora_rank=64,
    q_lora_rank=128,
    qk_rope_head_dim=16,
    qk_nope_head_dim=16,
    v_head_dim=16,
    lora_rank=4,
    act_threshold=0.99,
)

BATCH, SEQ = 2, 16
SEED = 42
OUT = _ROOT / "benchmarks" / "om_fidelity.json"


def build_and_run(model, input_ids, n_loops=4):
    """Run forward pass, return (logits, hidden_states_dict)."""
    model.eval()
    hidden_states = {}

    # Register hooks to capture intermediate outputs
    hooks = []
    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states[name] = output[0].detach().clone()
            else:
                hidden_states[name] = output.detach().clone()
        return hook

    for name, module in model.named_modules():
        if "recurrent.block" in name or "prelude" in name or "coda" in name:
            if "TransformerBlock" in str(type(module)):
                h = module.register_forward_hook(make_hook(name))
                hooks.append(h)

    with torch.no_grad():
        logits = model(input_ids, n_loops=n_loops)

    for h in hooks:
        h.remove()

    return logits, hidden_states


def reconstruct_weights(compressed_sd, orig_sd_for_shapes=None):
    """Reconstruct dense weights from compressed state_dict."""
    from hyperretro.hf.factor_int4 import unpack_int4_rows
    from hyperretro.hf.factor_quantize import dequantize_blockwise_int4

    dense_sd = {}
    factored_map = {}  # prefix -> (A, B) — where prefix is the dense weight key prefix
    int4_map = {}      # base -> (packed, scales, awq)

    # --- Pass 1: collect all .q entries and non-quantized entries ---
    for key, tensor in compressed_sd.items():
        if key.endswith(".q"):
            base = key[:-2]
            scales_key = base + ".scales"
            awq_key = base + ".awq_scales"
            if scales_key in compressed_sd:
                packed = compressed_sd[key].cpu().numpy()
                scales = compressed_sd[scales_key].cpu().numpy()
                awq = compressed_sd.get(awq_key)
                if awq is not None:
                    awq = awq.cpu().numpy()
                int4_map[base] = (packed, scales, awq)
        elif key.endswith(".scales") or key.endswith(".awq_scales"):
            continue

    # --- Pass 2: identify factored pairs ---
    # Non-int4: .factored_A and .factored_B exist directly
    # Int4: .factored_A.q and .factored_B.q exist in int4_map
    for key in list(compressed_sd.keys()):
        if key.endswith(".factored_A"):
            base = key[:-len(".factored_A")]
            b_key = base + ".factored_B"
            if b_key in compressed_sd:
                factored_map[base] = [compressed_sd[key], compressed_sd[b_key]]
        elif key.endswith(".factored_B"):
            base = key[:-len(".factored_B")]
            # Already handled above if .factored_A exists
            if base not in factored_map:
                a_key = base + ".factored_A"
                if a_key in compressed_sd:
                    factored_map[base] = [compressed_sd[a_key], compressed_sd[key]]

    # Detect int4-quantized factored pairs via int4_map
    for base in list(int4_map.keys()):
        if base.endswith(".factored_A"):
            prefix = base[:-len(".factored_A")]
            b_base = prefix + ".factored_B"
            if b_base in int4_map:
                packed_a, scales_a, awq_a = int4_map[base]
                packed_b, scales_b, awq_b = int4_map[b_base]
                # Unpack: n_cols = packed.shape[1] * 2 (packed stores pairs)
                a_ncols = packed_a.shape[1] * 2
                b_ncols = packed_b.shape[1] * 2
                W_qa = unpack_int4_rows(packed_a, a_ncols)
                W_qb = unpack_int4_rows(packed_b, b_ncols)
                A = dequantize_blockwise_int4(W_qa, scales_a)
                B = dequantize_blockwise_int4(W_qb, scales_b)
                if awq_a is not None:
                    A = A / awq_a[None, :]
                if awq_b is not None:
                    B = B / awq_b[None, :]
                factored_map[prefix] = [torch.from_numpy(A), torch.from_numpy(B)]

    # --- Pass 3: materialize factored weights, trimming odd-column overestimate ---
    for base, (A_t, B_t) in factored_map.items():
        A = A_t.float().cpu().numpy() if hasattr(A_t, "cpu") else np.asarray(A_t)
        B = B_t.float().cpu().numpy() if hasattr(B_t, "cpu") else np.asarray(B_t)
        weight_key = base + ".weight"
        W = (B.astype(np.float64) @ A.astype(np.float64)).astype(np.float32)
        # Trim if overestimated (odd column count in original)
        if orig_sd_for_shapes and weight_key in orig_sd_for_shapes:
            orig = orig_sd_for_shapes[weight_key]
            orig_shape = tuple(orig.shape) if hasattr(orig, "shape") else None
            if orig_shape and W.shape != orig_shape:
                W = W[:orig_shape[0], :orig_shape[1]]
        dense_sd[weight_key] = torch.from_numpy(W)

    # --- Pass 4: dequantize non-factored int4 weights ---
    factored_bases = set(factored_map.keys())
    for base, (packed, scales, awq) in int4_map.items():
        # Skip if part of a factored pair
        if any(base.startswith(fb + ".factored_") for fb in factored_bases):
            continue
        if any(base == fb + ".factored_A" or base == fb + ".factored_B" for fb in factored_bases):
            continue
        n_cols = packed.shape[1] * 2
        W_q = unpack_int4_rows(packed, n_cols)
        W = dequantize_blockwise_int4(W_q, scales)
        if awq is not None:
            W = W / awq[None, :]
        # Map back: the original key might be {base}.weight or just {base}
        dense_key = base
        if not dense_key.endswith(".weight") and ".factored_" not in dense_key:
            # This was a non-factored weight that got quantized
            dense_sd[dense_key + ".weight"] = torch.from_numpy(W)
        else:
            dense_sd[dense_key] = torch.from_numpy(W)

    # --- Pass 5: copy remaining non-quantized, non-factored tensors ---
    for key, tensor in compressed_sd.items():
        if key.endswith((".q", ".scales", ".awq_scales", ".factored_A", ".factored_B")):
            continue
        if key not in dense_sd:
            dense_sd[key] = tensor

    return dense_sd


def _infer_factored_cols(prefix, compressed_sd, which):
    """Infer number of columns for factored A or B matrix."""
    # Try to find the original weight to get shape
    weight_key = prefix + ".weight"
    # Not available in compressed form; use heuristics
    # For OpenMythos MLA: A is (rank, in_dim), B is (out_dim, rank)
    # We can infer from the .scales shape or from B's row count
    if which == "A":
        # A shape is (rank, in_features)
        # Try to get in_features from the B matrix
        b_qkey = prefix + ".factored_B.q"
        if b_qkey in compressed_sd:
            return compressed_sd[b_qkey].shape[0]  # out_features? No, that's B's rows
        # Fallback: from A's .scales
        a_scales_key = prefix + ".factored_A.scales"
        if a_scales_key in compressed_sd:
            scales = compressed_sd[a_scales_key]
            if scales.ndim == 1:
                return compressed_sd[prefix + ".factored_A.q"].shape[1] * 2
            elif scales.ndim == 2:
                blk = compressed_sd[prefix + ".factored_A.q"].shape[1] * 2
                # n_blocks * block_size gives an upper bound
                n_blocks = scales.shape[1]
                # Block size is typically 128
                return n_blocks * 128
    elif which == "B":
        # B shape is (out_features, rank)
        # rank comes from A
        a_qkey = prefix + ".factored_A.q"
        if a_qkey in compressed_sd:
            return compressed_sd[a_qkey].shape[0]  # A's rows = rank
        return 64
    return 256

    return dense_sd


def _infer_cols_from_A(base, compressed_sd):
    """Infer columns for factored_A from manifest or heuristics."""
    # factored_A is (rank, in_features)
    # Look at matching B to infer rank
    b_key = base + ".factored_B"
    if b_key in compressed_sd:
        B = compressed_sd[b_key]
        rank = B.shape[1] if hasattr(B, "shape") else B.shape[-1]
        # in_features = A_cols, but we don't know without more info
        # For MLA: q_down is (q_lora_rank, dim), kv_down is (kv_lora_rank, dim)
    # Fallback: use shape from A.q
    a_qkey = base + ".factored_A.q"
    if a_qkey in compressed_sd:
        packed = compressed_sd[a_qkey]
        n_rows = packed.shape[0]
        # For OpenMythos MLA: A is either (q_lora_rank, dim) or (kv_lora_rank, dim)
        # We'll try to infer from the module path
        return packed.shape[1] * 2
    return 256  # default


def _infer_cols_from_B(base, compressed_sd):
    """Infer columns for factored_B."""
    a_key = base + ".factored_A"
    if a_key in compressed_sd:
        A = compressed_sd[a_key]
        return A.shape[0] if hasattr(A, "shape") else 64
    return 64


def compare_outputs(orig_logits, recon_logits):
    """Compare two sets of logits."""
    # KL divergence (logits → probs)
    orig_probs = F.softmax(orig_logits.float(), dim=-1)
    recon_probs = F.softmax(recon_logits.float(), dim=-1)

    kl = F.kl_div(
        recon_probs.log(), orig_probs, reduction="batchmean"
    ).item()

    # Cosine similarity (flattened)
    o_flat = orig_logits.float().reshape(-1)
    r_flat = recon_logits.float().reshape(-1)
    cos_sim = F.cosine_similarity(o_flat.unsqueeze(0), r_flat.unsqueeze(0)).item()

    # Top-5 overlap
    _, orig_top5 = orig_logits.float().topk(5, dim=-1)
    _, recon_top5 = recon_logits.float().topk(5, dim=-1)
    overlap = 0
    total = 0
    for b in range(orig_top5.shape[0]):
        for t in range(orig_top5.shape[1]):
            overlap += len(set(orig_top5[b, t].tolist()) & set(recon_top5[b, t].tolist()))
            total += 5
    top5_acc = overlap / total if total > 0 else 0

    # Max probability correlation
    orig_max = orig_probs.max(dim=-1).values
    recon_max = recon_probs.max(dim=-1).values
    max_corr = float(torch.corrcoef(
        torch.stack([orig_max.flatten(), recon_max.flatten()])
    )[0, 1].item())

    return {
        "kl_divergence": round(kl, 6),
        "cosine_similarity": round(cos_sim, 6),
        "top5_overlap": round(top5_acc, 4),
        "max_prob_correlation": round(max_corr, 4),
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()

    # 1. Build original model
    print("=" * 60)
    print("Building OpenMythos (small config for speed) ...")
    cfg = SMALL_CFG
    model_orig = OpenMythos(cfg)
    total_params = sum(p.numel() for p in model_orig.parameters())
    total_mb = total_params * 2 / 1e6
    print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  fp16 size: {total_mb:.1f} MB")

    # 2. Run original forward pass
    input_ids = torch.randint(0, cfg.vocab_size, (BATCH, SEQ))
    print(f"\nRunning original forward pass (B={BATCH}, T={SEQ}) ...")
    t_fwd = time.time()
    logits_orig, hidden_orig = build_and_run(model_orig, input_ids, n_loops=4)
    fwd_time_orig = time.time() - t_fwd
    print(f"  Logits shape: {tuple(logits_orig.shape)}")
    print(f"  Hidden states captured: {len(hidden_orig)}")
    print(f"  Forward time: {fwd_time_orig:.2f}s")

    # 3. Compress
    print(f"\nCompressing with HyperRetro (ffn_rank=64, attn_rank=32, int4=True) ...")
    sd_orig = {k: v.clone() for k, v in model_orig.state_dict().items()}
    sd_comp, manifest, report = compress_openmythos(
        model_orig, ffn_rank=64, attn_rank=32, int4=True,
    )
    compressed_keys = len(sd_comp)
    print(f"  Compressed tensors: {compressed_keys} (was {len(sd_orig)})")
    print(f"  Attn factored: {report['attn_factored']}")
    print(f"  FFN factored: {report['ffn_factored']}")

    # 4. Reconstruct weights
    print(f"\nReconstructing dense weights from compressed form ...")
    t_recon = time.time()
    dense_sd = reconstruct_weights(sd_comp, sd_orig)
    recon_time = time.time() - t_recon
    print(f"  Reconstructed keys: {len(dense_sd)}")
    print(f"  Reconstruction time: {recon_time:.2f}s")

    # 5. Measure per-matrix reconstruction error
    print(f"\nPer-matrix reconstruction errors:")
    frob_errors = {}
    for key in sorted(sd_orig.keys()):
        if key not in dense_sd:
            continue
        W_orig = sd_orig[key]
        W_recon = dense_sd[key]
        if hasattr(W_orig, "dim") and W_orig.dim() == 2:
            frob = float(torch.norm(W_orig.float() - W_recon.float(), "fro"))
            frob_rel = frob / max(float(torch.norm(W_orig.float(), "fro")), 1e-12)
            frob_errors[key] = frob_rel

    if frob_errors:
        errors_sorted = sorted(frob_errors.items(), key=lambda x: -x[1])
        for key, err in errors_sorted[:8]:
            print(f"  {key:50s}  rel_err={err:.6f}")
        if len(errors_sorted) > 8:
            print(f"  ... ({len(errors_sorted) - 8} more)")
        mean_err = float(np.mean(list(frob_errors.values())))
        max_err = float(np.max(list(frob_errors.values())))
        print(f"  Mean rel error: {mean_err:.6f}")
        print(f"  Max rel error:  {max_err:.6f}")

    # 6. Build reconstructed model and run forward pass
    print(f"\nBuilding reconstructed model ...")
    model_recon = OpenMythos(cfg)
    # Load reconstructed state dict (may have missing keys due to factoring)
    missing, unexpected = model_recon.load_state_dict(dense_sd, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    # Show first few missing to diagnose
    for mk in missing[:5]:
        print(f"    MISSING: {mk}")
    for uk in unexpected[:3]:
        print(f"    UNEXPECTED: {uk}")

    print(f"\nRunning reconstructed forward pass ...")
    t_fwd2 = time.time()
    logits_recon, hidden_recon = build_and_run(model_recon, input_ids, n_loops=4)
    fwd_time_recon = time.time() - t_fwd2

    # 7. Compare outputs
    print(f"\n{'=' * 60}")
    print(f"OUTPUT FIDELITY COMPARISON")
    print(f"{'=' * 60}")
    metrics = compare_outputs(logits_orig, logits_recon)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 8. Per-layer hidden state comparison
    print(f"\nPer-layer hidden state cosine similarity:")
    for name in sorted(hidden_orig.keys()):
        if name in hidden_recon:
            h_orig = hidden_orig[name].float().reshape(-1)
            h_recon = hidden_recon[name].float().reshape(-1)
            cos = F.cosine_similarity(h_orig.unsqueeze(0), h_recon.unsqueeze(0)).item()
            indicator = "" if cos > 0.99 else ("" if cos > 0.90 else "")
            print(f"  {indicator} {name:45s}  cos={cos:.6f}")

    # 9. Size comparison
    orig_mb = total_mb
    # Estimate compressed size
    comp_bytes = 0
    for k, v in sd_comp.items():
        if hasattr(v, "numel"):
            if v.dtype == torch.uint8:
                comp_bytes += v.numel()
            elif v.dtype == torch.float16:
                comp_bytes += v.numel() * 2
            else:
                comp_bytes += v.numel() * 4
    comp_mb = comp_bytes / 1e6

    print(f"\n{'=' * 60}")
    print(f"SIZE COMPARISON")
    print(f"{'=' * 60}")
    print(f"  Original (fp16):     {orig_mb:.1f} MB")
    print(f"  Compressed (int4):   {comp_mb:.1f} MB")
    print(f"  Shrink:              {orig_mb/comp_mb:.1f}x")
    print(f"  Forward time orig:   {fwd_time_orig:.2f}s")
    print(f"  Forward time recon:  {fwd_time_recon:.2f}s")

    # 10. Save results
    results = {
        "model": "OpenMythos (small)",
        "params": total_params,
        "params_m": round(total_params / 1e6, 1),
        "ffn_rank": 64,
        "attn_rank": 32,
        "int4": True,
        "forward_time_orig_s": round(fwd_time_orig, 3),
        "forward_time_recon_s": round(fwd_time_recon, 3),
        "orig_mb": round(orig_mb, 1),
        "compressed_mb": round(comp_mb, 1),
        "shrink": round(orig_mb / comp_mb, 2),
        "output_metrics": metrics,
        "mean_weight_relerr": round(mean_err, 6) if frob_errors else None,
        "max_weight_relerr": round(max_err, 6) if frob_errors else None,
        "n_missing_keys": len(missing),
        "missing_key_examples": missing[:10],
        "wall_s": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {OUT}")

    # Verdict
    print(f"\n{'=' * 60}")
    print(f"VERDICT")
    print(f"{'=' * 60}")
    kl = metrics["kl_divergence"]
    cos = metrics["cosine_similarity"]
    top5 = metrics["top5_overlap"]
    shrink = results["shrink"]

    if kl < 0.1 and cos > 0.99:
        print(f" EXCELLENT fidelity: KL={kl:.4f}, cos={cos:.4f}, top5={top5:.2%}")
        print(f"   The compressed model faithfully reproduces the original.")
    elif kl < 1.0 and cos > 0.95:
        print(f"  GOOD fidelity: KL={kl:.4f}, cos={cos:.4f}, top5={top5:.2%}")
        print(f"   Minor degradation; distill would close the gap.")
    else:
        print(f" POOR fidelity: KL={kl:.4f}, cos={cos:.4f}, top5={top5:.2%}")
        print(f"   Compression is too aggressive for this config.")
        print(f"   Missing keys: {len(missing)} — factored weights not restored.")

    print(f"   Shrink: {shrink:.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
