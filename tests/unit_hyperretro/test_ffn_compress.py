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

"""Tests for FFN compression."""
import numpy as np
import pytest

from hyperretro.hf.ffn_compress import (
    FFNCompressConfig,
    compress_ffn_state_dict,
    _group_ffn_by_layer,
    build_ffn_shared_basis,
    truncated_svd_project,
)


def _fake_swiglu_sd(n_layers=2, hidden=64, intermediate=128, rng=None):
    rng = rng or np.random.default_rng(0)
    sd = {}
    for li in range(n_layers):
        sd[f"model.layers.{li}.mlp.gate_proj.weight"] = rng.standard_normal((intermediate, hidden)).astype(np.float32)
        sd[f"model.layers.{li}.mlp.up_proj.weight"] = rng.standard_normal((intermediate, hidden)).astype(np.float32)
        sd[f"model.layers.{li}.mlp.down_proj.weight"] = rng.standard_normal((hidden, intermediate)).astype(np.float32)
    return sd


def _fake_gpt2_sd(n_layers=2, hidden=64, intermediate=256, rng=None):
    rng = rng or np.random.default_rng(0)
    sd = {}
    for li in range(n_layers):
        sd[f"transformer.h.{li}.mlp.c_fc.weight"] = rng.standard_normal((hidden, intermediate)).astype(np.float32)
        sd[f"transformer.h.{li}.mlp.c_proj.weight"] = rng.standard_normal((intermediate, hidden)).astype(np.float32)
    return sd


def test_group_ffn_swiglu():
    sd = _fake_swiglu_sd(n_layers=3)
    g = _group_ffn_by_layer(sd)
    assert set(g.keys()) == {0, 1, 2}
    for layer in g.values():
        assert set(layer.keys()) == {"gate", "up", "down"}


def test_group_ffn_gpt2():
    sd = _fake_gpt2_sd(n_layers=2)
    g = _group_ffn_by_layer(sd)
    assert set(g.keys()) == {0, 1}
    for layer in g.values():
        assert set(layer.keys()) == {"fc", "proj"}


def test_swiglu_compression_reduces_rank():
    sd = _fake_swiglu_sd(n_layers=1, hidden=64, intermediate=128)
    orig_gate = sd["model.layers.0.mlp.gate_proj.weight"].copy()
    orig_down = sd["model.layers.0.mlp.down_proj.weight"].copy()

    cfg = FFNCompressConfig(rank_in=8, rank_out=8)
    stats = compress_ffn_state_dict(sd, cfg)

    new_gate = sd["model.layers.0.mlp.gate_proj.weight"]
    new_down = sd["model.layers.0.mlp.down_proj.weight"]

    # Shapes preserved (full-shape storage).
    assert new_gate.shape == orig_gate.shape
    assert new_down.shape == orig_down.shape
    # gate full-matrix rank bounded by k=8 (small tolerance) under SVD mode.
    assert np.linalg.matrix_rank(new_gate, tol=1e-4) <= 8 + 1
    # down full-matrix rank bounded by 8.
    assert np.linalg.matrix_rank(new_down, tol=1e-4) <= 8 + 1
    # Stats recorded.
    s = stats["layer_0"]
    assert s["rank_in"] == 8
    assert s["rank_out"] == 8
    assert s["arch"] == "swiglu"
    assert s["mode"] == "svd"
    assert 0.0 < s["frob_relerr_gate"] < 1.0
    assert 0.0 < s["frob_relerr_down"] < 1.0


def test_swiglu_shared_mode_still_works():
    sd = _fake_swiglu_sd(n_layers=1, hidden=64, intermediate=128)
    orig_gate = sd["model.layers.0.mlp.gate_proj.weight"].copy()
    cfg = FFNCompressConfig(rank_in=8, rank_out=0, mode="shared")
    stats = compress_ffn_state_dict(sd, cfg)
    new_gate = sd["model.layers.0.mlp.gate_proj.weight"]
    assert new_gate.shape == orig_gate.shape
    # column-rank bounded by k=8 (shared basis projects columns).
    assert np.linalg.matrix_rank(new_gate, tol=1e-4) <= 8 + 1
    assert stats["layer_0"]["mode"] == "shared"


def test_swiglu_full_rank_is_lossless():
    sd = _fake_swiglu_sd(n_layers=1, hidden=32, intermediate=64)
    orig = {k: v.copy() for k, v in sd.items()}
    # rank_in == min(W.shape)=32, so SVD adds no info loss; arrays match orig.
    cfg = FFNCompressConfig(rank_in=32, rank_out=32)
    stats = compress_ffn_state_dict(sd, cfg)
    for k in orig:
        np.testing.assert_allclose(sd[k], orig[k], atol=1e-4)


def test_gpt2_compression():
    sd = _fake_gpt2_sd(n_layers=1, hidden=64, intermediate=256)
    orig_fc = sd["transformer.h.0.mlp.c_fc.weight"].copy()
    cfg = FFNCompressConfig(rank_out=16)
    stats = compress_ffn_state_dict(sd, cfg)
    new_fc = sd["transformer.h.0.mlp.c_fc.weight"]
    assert new_fc.shape == orig_fc.shape
    assert np.linalg.matrix_rank(new_fc, tol=1e-4) <= 16 + 1
    s = stats["layer_0"]
    assert s["arch"] == "gpt2_mlp"
    assert s["rank_fc"] == 16
    assert "sigma_fc_next" in s


def test_truncated_svd_exact_low_rank():
    rng = np.random.default_rng(42)
    # Build an exact rank-4 matrix.
    U = rng.standard_normal((20, 4))
    V = rng.standard_normal((10, 4))
    W = U @ V.T  # rank 4
    W_r, sigma_next = truncated_svd_project(W, r=4)
    np.testing.assert_allclose(W_r, W, atol=1e-10)
    assert sigma_next < 1e-10


def test_shared_basis_orthonormal():
    rng = np.random.default_rng(1)
    Wg = rng.standard_normal((48, 32)).astype(np.float32)
    Wu = rng.standard_normal((48, 32)).astype(np.float32)
    P = build_ffn_shared_basis(Wg, Wu)
    # Should be 32x32 and orthonormal.
    assert P.shape == (32, 32)
    np.testing.assert_allclose(P.T @ P, np.eye(32), atol=1e-4)


def test_only_in_compresses_gate_up_only():
    sd = _fake_swiglu_sd(n_layers=1, hidden=64, intermediate=128)
    orig_down = sd["model.layers.0.mlp.down_proj.weight"].copy()
    cfg = FFNCompressConfig(rank_in=8, rank_out=0)
    stats = compress_ffn_state_dict(sd, cfg)
    # down unchanged.
    np.testing.assert_allclose(sd["model.layers.0.mlp.down_proj.weight"], orig_down)
    # gate changed.
    assert stats["layer_0"]["rank_in"] == 8
    assert "rank_out" not in stats["layer_0"]
