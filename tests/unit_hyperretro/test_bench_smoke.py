"""Smoke-test the benchmark harness end-to-end (synthetic only)."""
from hyperretro.bench.run import (
    run_kernel_bench,
    run_speculative_bench,
)


def test_kernel_bench_smoke():
    r = run_kernel_bench(rows=64, in_dim=64, iters=3)
    assert r["rows"] == 64 and r["in_dim"] == 64
    res = r["results"]
    assert "hyperretro_fused_q8" in res
    assert "baseline_two_q8_gemvs" in res
    assert "baseline_torch_matmul_fp32" in res
    assert res["hyperretro_fused_q8"]["median_ms"] >= 0
    assert r["accuracy"]["q8_vs_fp32_max_abs_err"] >= 0


def test_speculative_bench_smoke():
    r = run_speculative_bench(d_model=64, k=8, vocab=128, n_steps=16, seed=0)
    assert 0.0 <= r["geodesic"]["acceptance"] <= 1.0
    assert 0.0 <= r["random_baseline"]["acceptance"] <= 1.0
    # Geodesic should at least tie the random baseline on a smooth synthetic
    # trajectory (low bar; this is a sanity check, not a paper-grade claim).
    assert r["geodesic"]["acceptance"] >= r["random_baseline"]["acceptance"] - 0.1
