"""
Test runner — executes verification scripts with timeout, progress, and result capture.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple, Optional

from .catalog import find_test, EXPECTED_OUTPUTS

ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS = ROOT / "benchmarks"
RESULTS_FILE = BENCHMARKS / "ht_repro_results.json"

# ── Spinner ────────────────────────────────────────────────────────
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_spin_idx = 0

def spin():
    global _spin_idx
    c = SPINNER[_spin_idx % len(SPINNER)]
    _spin_idx += 1
    return c

# ── GPU Detection ──────────────────────────────────────────────────
_gpu_cache = None

def has_gpu() -> bool:
    global _gpu_cache
    if _gpu_cache is not None:
        return _gpu_cache
    try:
        import torch
        _gpu_cache = torch.cuda.is_available()
    except ImportError:
        _gpu_cache = False
    return _gpu_cache

def can_run_tier(tier: str) -> bool:
    if tier == "T1":
        return True
    if tier == "T2":
        return has_gpu()
    if tier == "T3":
        return has_gpu()  # We still check but default to skip
    return False

# ── Core Runner ────────────────────────────────────────────────────

def run_test(test_id: str, verbose: bool = False, timeout_override: int = 0
             ) -> Tuple[bool, str, float, dict]:
    """
    Run a single test. Returns (passed, summary, elapsed_seconds, details).
    """
    test = find_test(test_id)
    if not test:
        return False, f"Unknown test: {test_id}", 0.0, {}

    script_path = ROOT / test["script"]
    if not script_path.exists():
        return False, f"Script not found: {script_path}", 0.0, {}

    timeout = timeout_override or test.get("timeout", 120)
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT), env=env,
            capture_output=True, text=True,
            timeout=timeout,
        )
        elapsed = time.time() - t0
        output = (result.stdout + result.stderr)

        # Check expected patterns
        expected = test.get("desc", "")
        passed = expected.lower() in output.lower() if expected else result.returncode == 0

        # Also check against expected outputs database
        exp = EXPECTED_OUTPUTS.get(test_id, {})
        exp_matches = {}
        for key, expected_val in exp.items():
            if isinstance(expected_val, bool):
                indicator = "true" if expected_val else "false"
                exp_matches[key] = indicator in output.lower()
            elif isinstance(expected_val, (int, float)):
                exp_matches[key] = str(expected_val) in output
            elif isinstance(expected_val, str):
                exp_matches[key] = expected_val.lower() in output.lower()

        # Summary: last non-empty lines
        lines = [l.strip() for l in output.split("\n") if l.strip()]
        summary = "\n".join(lines[-3:]) if lines else "(no output)"

        details = {
            "test_id": test_id,
            "passed": passed,
            "time": round(elapsed, 2),
            "output_last_lines": summary,
            "expected_matches": exp_matches,
            "all_expected_match": all(exp_matches.values()) if exp_matches else None,
        }

        return passed, summary, elapsed, details

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        return False, f"TIMEOUT after {timeout}s", elapsed, {"test_id": test_id, "passed": False, "time": elapsed}
    except Exception as e:
        elapsed = time.time() - t0
        return False, str(e), elapsed, {"test_id": test_id, "passed": False, "time": elapsed, "error": str(e)}

# ── Batch Runner ───────────────────────────────────────────────────

def run_batch(test_ids: list, progress_callback=None, verbose: bool = False
              ) -> dict:
    """Run multiple tests with progress tracking."""
    results = load_results()
    run_record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tests": {},
        "passed": 0, "failed": 0, "skipped": 0,
        "total_time": 0.0,
    }

    for i, tid in enumerate(test_ids):
        test = find_test(tid)
        if not test:
            run_record["tests"][tid] = {"status": "error", "reason": "Not found"}
            run_record["failed"] += 1
            continue

        if progress_callback:
            progress_callback(i, len(test_ids), test["name"], test["tier"])

        # Check tier
        if not can_run_tier(test["tier"]):
            reason = "Needs GPU" if test["tier"] in ("T2", "T3") else f"Tier {test['tier']} not available"
            run_record["tests"][tid] = {"status": "skipped", "reason": reason}
            run_record["skipped"] += 1
            continue

        if test["tier"] == "T3" and not os.environ.get("HT_REPRO_FORCE_T3"):
            run_record["tests"][tid] = {"status": "skipped", "reason": "T3 hardware required (set HT_REPRO_FORCE_T3=1 to force)"}
            run_record["skipped"] += 1
            continue

        passed, summary, elapsed, details = run_test(tid, verbose=verbose)
        run_record["total_time"] += elapsed

        if passed:
            run_record["tests"][tid] = {"status": "pass", "time": elapsed, "details": details}
            run_record["passed"] += 1
        else:
            run_record["tests"][tid] = {"status": "fail", "time": elapsed, "summary": summary[:500], "details": details}
            run_record["failed"] += 1

    results["runs"].append(run_record)
    save_results(results)
    return run_record

# ── Results Persistence ────────────────────────────────────────────

def load_results() -> dict:
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {"runs": []}

def save_results(results: dict):
    BENCHMARKS.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

def last_run() -> Optional[dict]:
    results = load_results()
    if results["runs"]:
        return results["runs"][-1]
    return None
