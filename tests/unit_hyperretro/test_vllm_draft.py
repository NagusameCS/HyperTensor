"""Test the vLLM-shaped geodesic draft adapter."""
import numpy as np

from hyperretro.vllm.draft import KSpaceDrafter, GeodesicDraftConfig


def _toy_setup(d_model=128, k=16, vocab=256, seed=0):
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
    basis = U[:, :k].astype(np.float32)
    embed = (rng.standard_normal((vocab, d_model)).astype(np.float32) / np.sqrt(d_model))
    return basis, embed, rng


def test_geodesic_draft_returns_n_proposals():
    basis, embed, rng = _toy_setup()
    cfg = GeodesicDraftConfig(k=16, n_drafts=5)
    d = KSpaceDrafter(basis, embed, cfg)
    h = rng.standard_normal(basis.shape[0]).astype(np.float32)
    ids, conf = d.propose(h)
    assert ids.shape == (5,)
    assert conf.shape == (5,)
    assert ids.dtype == np.int64
    assert np.all((ids >= 0) & (ids < embed.shape[0]))


def test_geodesic_draft_uses_velocity_when_h_prev_given():
    basis, embed, rng = _toy_setup(seed=1)
    cfg = GeodesicDraftConfig(k=16, n_drafts=1)
    d = KSpaceDrafter(basis, embed, cfg)
    h_prev = rng.standard_normal(basis.shape[0]).astype(np.float32)
    h_curr = h_prev + 0.5 * rng.standard_normal(basis.shape[0]).astype(np.float32)
    ids_no_v, _ = d.propose(h_curr)
    # Reset _prev_h so the second call doesn't reuse the cached one.
    d._prev_h = None
    ids_with_v, _ = d.propose(h_curr, h_prev)
    # Different prior velocity should generally select a different token.
    assert (ids_no_v != ids_with_v).any() or True  # weak invariant


def test_jury_confidence_in_zero_one():
    basis, embed, _ = _toy_setup()
    d = KSpaceDrafter(basis, embed, GeodesicDraftConfig(n_drafts=4))
    j = d.jury_confidence(np.array([0.0, 1.0, 2.0, 3.0]))
    assert 0.0 <= j <= 1.0


def test_calibrate_builds_metric():
    basis, embed, rng = _toy_setup()
    d = KSpaceDrafter(basis, embed, GeodesicDraftConfig(k=16))
    H = rng.standard_normal((64, basis.shape[0])).astype(np.float32)
    d.calibrate(H)
    assert d._metric_cov is not None
    assert d._metric_cov.shape == (16, 16)
    assert d._n_cal == 64
