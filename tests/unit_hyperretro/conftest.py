"""Test fixtures for HyperRetro.

We import-protect torch since the kernel/vllm tests must run on CI hosts
that don't necessarily have torch wheels available.
"""
import os
import sys
import pytest

# Force the numpy fallback by default in unit tests so we exercise the
# Python reference path deterministically. Individual tests that want
# the C extension or torch path opt in by unsetting this fixture.
@pytest.fixture(autouse=True)
def _force_numpy_fallback(monkeypatch):
    monkeypatch.setenv("HYPERRETRO_FORCE_FALLBACK", "1")
    # Reset cached backend selection.
    import hyperretro.kernels as k
    k._BACKEND = None
    k._CEXT = None
    yield
    k._BACKEND = None
    k._CEXT = None
