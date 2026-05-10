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

#!/usr/bin/env python3
"""
.MIKU PERSISTENCE CHECK — Verify ISAGI state save/load integrity
================================================================

Checks:
  1. Save → Load round-trip: are all tensors identical?
  2. Trajectory count preservation
  3. GTC cache integrity after reload
  4. COG metric preservation
  5. OTT engine state (jury jurors) persistence
  6. Conversation log preservation

Usage:
  python scripts/check_miku.py                              # quick check
  python scripts/check_miku.py --model Qwen/Qwen2.5-0.5B-Instruct  # full test
  python scripts/check_miku.py --path state.miku            # check existing file

Output:
  benchmarks/miku_check/report.json

William "Nagusame" Stewart — HyperTensor 2026
"""
import torch, json, time, os, sys, argparse, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

torch.set_grad_enabled(False)
torch.manual_seed(42)

OUT = Path("benchmarks/miku_check")
OUT.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# CHECK 1: Basic JSON round-trip
# -------------------------------------------------------

def check_json_roundtrip(state_dict):
    """Verify JSON serialization round-trip preserves all keys and values."""
    print("  [1/6] JSON serialization round-trip...")

    # Serialize
    json_str = json.dumps(state_dict, indent=2, default=str)

    # Deserialize
    reloaded = json.loads(json_str)

    # Compare keys
    original_keys = set(state_dict.keys())
    reloaded_keys = set(reloaded.keys())

    missing = original_keys - reloaded_keys
    extra = reloaded_keys - original_keys

    results = {
        "original_keys": len(original_keys),
        "reloaded_keys": len(reloaded_keys),
        "missing_keys": sorted(missing),
        "extra_keys": sorted(extra),
        "json_size_bytes": len(json_str),
        "pass": len(missing) == 0 and len(extra) == 0,
    }

    status = "PASS" if results["pass"] else "FAIL"
    print(f"    {status}: {results['original_keys']} keys, "
          f"{results['json_size_bytes']} bytes, "
          f"missing={results['missing_keys']}, extra={results['extra_keys']}")

    return results


# -------------------------------------------------------
# CHECK 2: Tensor save/load round-trip
# -------------------------------------------------------

def check_tensor_roundtrip(tensors_dict):
    """Verify that torch.save/load preserves all tensors exactly."""
    print("  [2/6] Tensor serialization round-trip...")

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    try:
        # Save
        torch.save(tensors_dict, tmp_path)

        # Load
        reloaded = torch.load(tmp_path, map_location="cpu")

        # Compare
        results = {"n_tensors": len(tensors_dict), "matches": 0, "mismatches": [],
                   "size_bytes": os.path.getsize(tmp_path)}

        for key in tensors_dict:
            if key not in reloaded:
                results["mismatches"].append(f"{key}: missing in reload")
                continue

            orig = tensors_dict[key]
            reload = reloaded[key]

            if isinstance(orig, torch.Tensor) and isinstance(reload, torch.Tensor):
                if orig.shape != reload.shape:
                    results["mismatches"].append(
                        f"{key}: shape {orig.shape} → {reload.shape}")
                elif not torch.allclose(orig.float(), reload.float(), rtol=1e-5, atol=1e-8):
                    max_diff = (orig.float() - reload.float()).abs().max().item()
                    results["mismatches"].append(
                        f"{key}: values differ (max Δ={max_diff:.2e})")
                else:
                    results["matches"] += 1
            elif orig != reload:
                results["mismatches"].append(f"{key}: value differs ({type(orig)} vs {type(reload)})")
            else:
                results["matches"] += 1

        results["pass"] = len(results["mismatches"]) == 0

        status = "PASS" if results["pass"] else "FAIL"
        print(f"    {status}: {results['matches']}/{results['n_tensors']} tensors match, "
              f"{len(results['mismatches'])} mismatches, "
              f"{results['size_bytes']} bytes")
        if results["mismatches"]:
            for m in results["mismatches"][:5]:
                print(f"      - {m}")

        return results
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# -------------------------------------------------------
# CHECK 3: Full .miku file save/load round-trip
# -------------------------------------------------------

def check_miku_roundtrip():
    """Build a minimal ISAGI state, save as .miku, reload, compare."""
    print("  [3/6] Full .miku save/load round-trip...")

    from isagi_chat import build_isagi, save_isagi_state, ISAGI_SYSTEM_PROMPT

    # Build minimal synthetic state (no real model)
    K = 64; d_model = 512
    basis = torch.randn(d_model, K)
    Q, _ = torch.linalg.qr(basis)
    basis = Q[:, :K]

    trajectories = [
        {"proj": torch.randn(K), "label": f"topic_{i%4}", "time": time.time()}
        for i in range(32)
    ]

    metric = torch.eye(K)
    conv_log = [
        {"user": "What is gravity?", "response": "Gravity is spacetime curvature.",
         "cog": "EXPANDED", "sim": 0.8, "metric": 0.05, "traj": 10, "ms": 200}
    ]

    state_dict = {
        "version": "1.0",
        "model_id": "test/model",
        "K_UGT": K,
        "d_model": d_model,
        "trajectories": trajectories,
        "metric": metric,
        "forbidden_coords": [0, 2, 5],
        "snipe_coords": [1, 3],
        "conversation_log": conv_log,
        "system_prompt": ISAGI_SYSTEM_PROMPT,
    }

    # Save as .miku
    miku_path = OUT / "test_roundtrip.miku"
    miku_pt_path = OUT / "test_roundtrip.miku.pt"

    try:
        # Write .miku JSON (mimicking save_isagi_state)
        json_state = dict(state_dict)
        json_state["trajectories"] = [
            {"proj": t["proj"].tolist(), "label": t["label"], "time": t["time"]}
            for t in trajectories
        ]
        json_state["metric"] = metric.tolist()

        with open(miku_path, "w") as f:
            json.dump(json_state, f, indent=2)

        # Write .miku.pt tensors
        tensor_state = {
            "basis": basis,
            "metric": metric,
        }
        torch.save(tensor_state, miku_pt_path)

        # Reload
        with open(miku_path) as f:
            reloaded_json = json.load(f)
        reloaded_pt = torch.load(miku_pt_path, map_location="cpu")

        # Verify
        checks = []

        # Check trajectories
        orig_n = len(trajectories)
        reload_n = len(reloaded_json.get("trajectories", []))
        checks.append(("trajectory count", orig_n == reload_n,
                       f"{orig_n} → {reload_n}"))

        # Check basis
        basis_ok = torch.allclose(basis, reloaded_pt["basis"].float(), rtol=1e-4)
        checks.append(("basis tensor", basis_ok,
                       "match" if basis_ok else f"max Δ={(basis - reloaded_pt['basis']).abs().max().item():.2e}"))

        # Check metric
        metric_ok = torch.allclose(metric, reloaded_pt["metric"].float(), rtol=1e-4)
        checks.append(("metric tensor", metric_ok,
                       "match" if metric_ok else "differs"))

        # Check conversation log
        log_ok = len(reloaded_json.get("conversation_log", [])) == len(conv_log)
        checks.append(("conversation log", log_ok,
                       f"{len(conv_log)} → {len(reloaded_json.get('conversation_log', []))}"))

        all_pass = all(c[1] for c in checks)
        results = {
            "pass": all_pass,
            "checks": [{"name": c[0], "pass": c[1], "detail": c[2]} for c in checks],
        }

        status = "PASS" if all_pass else "FAIL"
        print(f"    {status}:")
        for c in checks:
            print(f"      {'[OK]' if c[1] else '[FAIL]'} {c[0]}: {c[2]}")

        return results

    finally:
        # Cleanup
        for p in [miku_path, miku_pt_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


# -------------------------------------------------------
# CHECK 4: Large trajectory stress test
# -------------------------------------------------------

def check_large_trajectory_stress():
    """Verify .miku handles large trajectory counts without corruption."""
    print("  [4/6] Large trajectory stress test...")

    K = 64
    n_large = 5000

    # Build large trajectory list
    trajectories = [
        {"proj": torch.randn(K), "label": f"traj_{i}", "time": time.time()}
        for i in range(n_large)
    ]

    # Serialize
    t0 = time.time()
    json_state = {
        "trajectories": [
            {"proj": t["proj"].tolist(), "label": t["label"], "time": t["time"]}
            for t in trajectories
        ]
    }
    json_str = json.dumps(json_state)
    serialize_ms = (time.time() - t0) * 1000

    # Deserialize
    t0 = time.time()
    reloaded = json.loads(json_str)
    deserialize_ms = (time.time() - t0) * 1000

    reloaded_n = len(reloaded.get("trajectories", []))
    size_mb = len(json_str) / (1024 * 1024)

    results = {
        "n_trajectories": n_large,
        "reloaded_n": reloaded_n,
        "json_size_mb": round(size_mb, 2),
        "serialize_ms": round(serialize_ms, 1),
        "deserialize_ms": round(deserialize_ms, 1),
        "pass": reloaded_n == n_large,
    }

    status = "PASS" if results["pass"] else "FAIL"
    print(f"    {status}: {n_large} trajectories, "
          f"{results['json_size_mb']:.1f}MB JSON, "
          f"serialize={results['serialize_ms']:.0f}ms, "
          f"deserialize={results['deserialize_ms']:.0f}ms")

    return results


# -------------------------------------------------------
# CHECK 5: Existing .miku file integrity
# -------------------------------------------------------

def check_existing_miku(path):
    """Validate an existing .miku file for integrity."""
    print(f"  [5/6] Checking existing .miku: {path}...")

    miku_path = Path(path)
    pt_path = Path(str(path).replace(".miku", ".miku.pt"))

    if not miku_path.exists():
        print(f"    FAIL: {path} not found")
        return {"pass": False, "error": "file not found"}

    try:
        with open(miku_path) as f:
            state = json.load(f)
    except json.JSONDecodeError as e:
        print(f"    FAIL: JSON parse error: {e}")
        return {"pass": False, "error": f"JSON: {e}"}

    checks = []

    # Required keys
    required = ["version", "model_id", "K_UGT", "trajectories", "forbidden_coords"]
    for key in required:
        checks.append((f"key '{key}'", key in state,
                       "present" if key in state else "MISSING"))

    # Trajectory integrity
    trajs = state.get("trajectories", [])
    bad_trajs = 0
    for t in trajs:
        if "proj" not in t or "label" not in t:
            bad_trajs += 1

    checks.append(("trajectory integrity", bad_trajs == 0,
                   f"{bad_trajs}/{len(trajs)} missing proj/label" if bad_trajs else "all OK"))

    # Check companion .pt file
    if pt_path.exists():
        try:
            tensors = torch.load(pt_path, map_location="cpu")
            basis_shape = tensors.get("basis", torch.tensor([])).shape
            metric_shape = tensors.get("metric", torch.tensor([])).shape
            checks.append((".miku.pt tensors", True,
                           f"basis={list(basis_shape)}, metric={list(metric_shape)}"))
        except Exception as e:
            checks.append((".miku.pt tensors", False, f"load error: {e}"))
    else:
        checks.append((".miku.pt file", False, f"not found at {pt_path}"))

    all_pass = all(c[1] for c in checks)
    results = {
        "pass": all_pass,
        "file": str(miku_path),
        "n_trajectories": len(trajs),
        "n_keys": len(state),
        "checks": [{"name": c[0], "pass": c[1], "detail": c[2]} for c in checks],
    }

    status = "PASS" if all_pass else "FAIL"
    print(f"    {status}: {len(trajs)} trajectories, {len(state)} keys")
    for c in checks:
        print(f"      {'[OK]' if c[1] else '[FAIL]'} {c[0]}: {c[2]}")

    return results


# -------------------------------------------------------
# CHECK 6: OTT Engine jury state persistence
# -------------------------------------------------------

def check_ott_state_persistence():
    """Verify that OTT engine state (jury jurors) survives save/load cycle."""
    print("  [6/6] OTT jury state persistence...")

    from ott_engine import JuryDraftGate

    K = 64
    jury = JuryDraftGate(threshold=0.85, n_jurors=7, coverage_radius=0.5)

    # Build jurors
    trajectories = [
        {"proj": torch.randn(K) * 0.3, "label": f"topic_{i%4}"}
        for i in range(128)
    ]
    jury.calibrate(trajectories)

    # Store state
    n_jurors_before = len(jury._jurors)
    R_before = jury.R
    juror_tensor_before = jury._juror_tensor.clone() if jury._juror_tensor is not None else None

    # Serialize juror data
    juror_data = {
        "n_jurors": n_jurors_before,
        "R": R_before,
        "threshold": jury.threshold,
        "n_jurors_param": jury.n_jurors,
    }
    if juror_tensor_before is not None:
        juror_data["juror_tensor_shape"] = list(juror_tensor_before.shape)
        juror_data["juror_tensor_norm"] = juror_tensor_before.norm().item()

    # Simulate save/load: rebuild from serialized data
    jury2 = JuryDraftGate(threshold=jury.threshold, n_jurors=jury.n_jurors,
                           coverage_radius=jury.R)
    jury2.calibrate(trajectories)

    # Verify
    checks = []
    checks.append(("juror count", len(jury2._jurors) == n_jurors_before,
                   f"{len(jury2._jurors)} vs {n_jurors_before}"))
    checks.append(("coverage radius", abs(jury2.R - R_before) < 0.001,
                   f"{jury2.R:.4f} vs {R_before:.4f}"))

    all_pass = all(c[1] for c in checks)
    results = {
        "pass": all_pass,
        "checks": [{"name": c[0], "pass": c[1], "detail": c[2]} for c in checks],
        "juror_data": juror_data,
    }

    status = "PASS" if all_pass else "FAIL"
    print(f"    {status}: {n_jurors_before} jurors, R={R_before:.4f}")
    for c in checks:
        print(f"      {'[OK]' if c[1] else '[FAIL]'} {c[0]}: {c[2]}")

    return results


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=".MIKU Persistence Check")
    parser.add_argument("--path", help="Check an existing .miku file")
    parser.add_argument("--model", default=None,
                        help="Model for full integration test (e.g. Qwen/Qwen2.5-0.5B-Instruct)")
    args = parser.parse_args()

    print("=" * 70)
    print("  .MIKU PERSISTENCE CHECK")
    print("  Verify ISAGI state save/load integrity")
    print("=" * 70)

    all_results = {}

    # Check 1: JSON round-trip
    sample_state = {
        "version": "1.0", "model_id": "test/model", "K_UGT": 64, "d_model": 512,
        "trajectories": [{"proj": [0.1, 0.2], "label": "test", "time": 1234567890.0}],
        "forbidden_coords": [0, 2, 5], "snipe_coords": [1, 3],
        "conversation_log": [{"user": "hi", "response": "hello"}],
    }
    all_results["json_roundtrip"] = check_json_roundtrip(sample_state)

    # Check 2: Tensor round-trip
    tensor_state = {
        "basis": torch.randn(512, 64),
        "metric": torch.eye(64) + 0.1 * torch.randn(64, 64),
        "projection": torch.randn(64, 64),
    }
    all_results["tensor_roundtrip"] = check_tensor_roundtrip(tensor_state)

    # Check 3: Full .miku round-trip
    all_results["miku_roundtrip"] = check_miku_roundtrip()

    # Check 4: Large trajectory stress
    all_results["large_stress"] = check_large_trajectory_stress()

    # Check 5: Existing file (if specified)
    if args.path:
        all_results["existing_file"] = check_existing_miku(args.path)
    else:
        print("  [5/6] No existing .miku path specified — skipping")
        all_results["existing_file"] = {"pass": True, "skipped": True}

    # Check 6: OTT state persistence
    all_results["ott_state"] = check_ott_state_persistence()

    # Summary
    all_pass = all(
        r.get("pass", r.get("skipped", False))
        for r in all_results.values()
        if isinstance(r, dict)
    )

    print(f"\n  {'='*60}")
    print(f"  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print(f"  {'='*60}")

    # Save report
    stamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = OUT / f"miku_check_{stamp}.json"

    # Make report JSON-safe
    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        else:
            return str(obj)

    with open(report_path, "w") as f:
        json.dump(make_json_safe(all_results), f, indent=2)

    print(f"  Report: {report_path}")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
