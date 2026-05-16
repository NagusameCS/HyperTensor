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

"""Unit tests for hyperretro.hf.factored on-disk factored storage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file, save_file

from hyperretro.hf.factored import (
    FactoredEntry,
    _adaptive_rank,
    _shared_basis_factor,
    _svd_factor,
    _svd_factor_adaptive,
    _torch_module,
    build_manifest,
    factor_attn_state_dict,
    factor_ffn_state_dict,
)


def test_svd_factor_round_trip():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((16, 24)).astype(np.float32)
    A, B = _svd_factor(W, k=16)
    rec = B @ A
    assert np.allclose(rec, W, atol=1e-5)


def test_svd_factor_low_rank_exact():
    rng = np.random.default_rng(1)
    U = rng.standard_normal((20, 5)).astype(np.float32)
    V = rng.standard_normal((5, 30)).astype(np.float32)
    W = U @ V
    A, B = _svd_factor(W, k=5)
    rec = B @ A
    assert np.allclose(rec, W, atol=1e-4)


def test_adaptive_rank_truncation():
    """Spectrum with sharp cliff should be truncated at the cliff."""
    rng = np.random.default_rng(2)
    s_top = np.array([1.0, 0.9, 0.8])
    s_tail = np.full(20, 1e-6)
    s = np.concatenate([s_top, s_tail])
    U = np.linalg.qr(rng.standard_normal((40, len(s))))[0]
    Vt = np.linalg.qr(rng.standard_normal((len(s), 30)))[0].T[: len(s)]
    W = (U * s) @ Vt
    k = _adaptive_rank(W, max_k=10, rel_tol=1e-3)
    assert k == 3, f"expected truncation to top-3, got {k}"


def test_shared_basis_factor_shapes():
    rng = np.random.default_rng(3)
    d_in = 24
    d_out_q = 24
    d_out_kv = 8
    Wq = rng.standard_normal((d_out_q, d_in)).astype(np.float32)
    Wk = rng.standard_normal((d_out_kv, d_in)).astype(np.float32)
    Wv = rng.standard_normal((d_out_kv, d_in)).astype(np.float32)
    A, Bs = _shared_basis_factor(Wq, Wk, Wv, k=12)
    assert A.shape == (12, d_in)
    assert Bs["q"].shape == (d_out_q, 12)
    assert Bs["k"].shape == (d_out_kv, 12)
    assert Bs["v"].shape == (d_out_kv, 12)


def test_factored_linear_matches_dense():
    FactoredLinear = _torch_module()
    rng = np.random.default_rng(4)
    d_in, d_out, k = 8, 16, 4
    U = rng.standard_normal((d_out, k)).astype(np.float32)
    V = rng.standard_normal((k, d_in)).astype(np.float32)
    W = U @ V
    bias = rng.standard_normal(d_out).astype(np.float32)

    A_t = torch.from_numpy(V)
    B_t = torch.from_numpy(U)
    bias_t = torch.from_numpy(bias)

    flin = FactoredLinear(A_t, B_t, bias_t)
    dense = torch.nn.Linear(d_in, d_out)
    with torch.no_grad():
        dense.weight.copy_(torch.from_numpy(W))
        dense.bias.copy_(bias_t)

    x = torch.from_numpy(rng.standard_normal((2, 5, d_in)).astype(np.float32))
    y_f = flin(x)
    y_d = dense(x)
    assert torch.allclose(y_f, y_d, atol=1e-5), \
        f"FactoredLinear mismatch: max diff {(y_f - y_d).abs().max()}"


def test_factor_attn_state_dict_round_trip(tmp_path):
    """Round-trip: build fake state_dict, factor, ensure manifest is well-formed
    and dense keys are removed."""
    rng = np.random.default_rng(5)
    d_in = 32
    d_out_q = 32
    d_out_kv = 8

    sd: dict[str, torch.Tensor] = {}
    layer_keys: dict[int, dict[str, str]] = {}
    for li in range(2):
        Wq = torch.from_numpy(rng.standard_normal((d_out_q, d_in)).astype(np.float32))
        Wk = torch.from_numpy(rng.standard_normal((d_out_kv, d_in)).astype(np.float32))
        Wv = torch.from_numpy(rng.standard_normal((d_out_kv, d_in)).astype(np.float32))
        q_key = f"model.layers.{li}.self_attn.q_proj.weight"
        k_key = f"model.layers.{li}.self_attn.k_proj.weight"
        v_key = f"model.layers.{li}.self_attn.v_proj.weight"
        sd[q_key] = Wq
        sd[k_key] = Wk
        sd[v_key] = Wv
        layer_keys[li] = {"q": q_key, "k": k_key, "v": v_key}

    sd, entries = factor_attn_state_dict(
        sd, rank=20, layer_keys=layer_keys, rel_tol=1e-6,
    )
    assert len(entries) == 2
    for e in entries:
        assert e.rank <= 20
        assert e.in_features == d_in
        # original keys gone
        assert e.keys["q"] not in sd
        assert e.keys["k"] not in sd
        assert e.keys["v"] not in sd
        # factored keys present
        prefix = e.keys["q"].rsplit(".q_proj.weight", 1)[0]
        assert f"{prefix}.factored_A" in sd
        for slot in ("q", "k", "v"):
            assert f"{prefix}.factored_B{slot}" in sd

    manifest = build_manifest(entries, shared=True)
    assert manifest["shared_basis"] is True
    assert len(manifest["layers"]) == 2


def test_factor_ffn_skips_when_no_savings():
    """FFN factoring should only commit when k*(m+n) < m*n."""
    rng = np.random.default_rng(6)
    d_in = 16
    d_out = 24
    W = torch.from_numpy(rng.standard_normal((d_out, d_in)).astype(np.float32))
    sd = {"model.layers.0.mlp.gate_proj.weight": W}
    # max_rank big enough that adaptive picks full rank → no byte savings
    entries = factor_ffn_state_dict(sd, max_rank=d_in, rel_tol=1e-12)
    # Full-rank random matrix at extreme tol -> rank d_in -> storage
    # d_in*(d_out+d_in) = 16*40 = 640 vs d_out*d_in = 384. No savings.
    assert len(entries) == 0
    assert "model.layers.0.mlp.gate_proj.weight" in sd


def test_factor_ffn_commits_when_low_rank():
    """Truly low-rank FFN matrix should be factored."""
    rng = np.random.default_rng(7)
    d_in = 64
    d_out = 64
    r = 4
    U = rng.standard_normal((d_out, r)).astype(np.float32)
    V = rng.standard_normal((r, d_in)).astype(np.float32)
    W = torch.from_numpy(U @ V)
    sd = {"model.layers.0.mlp.gate_proj.weight": W}
    entries = factor_ffn_state_dict(sd, max_rank=8, rel_tol=1e-4)
    assert len(entries) == 1
    e = entries[0]
    assert e["rank"] <= 8
    assert "model.layers.0.mlp.gate_proj.weight" not in sd
    assert "model.layers.0.mlp.gate_proj.factored_A" in sd
    assert "model.layers.0.mlp.gate_proj.factored_B" in sd
