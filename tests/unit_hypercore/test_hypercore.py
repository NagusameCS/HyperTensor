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

"""Smoke tests for hypertensor-core (hypercore) geometric modules."""
import numpy as np
import pytest
import torch


class TestGeodesicMetric:
    """GeodesicMetric -- core geometric reasoning."""

    def test_import(self):
        from hypercore import GeodesicMetric
        m = GeodesicMetric()
        assert m is not None

    def test_measure_collapse_returns_tuple(self):
        from hypercore import GeodesicMetric
        m = GeodesicMetric(dim=64)
        logits = torch.randn(1, 16, 128)
        collapse, n_collapsed = m.measure_collapse(logits)
        assert isinstance(collapse, float)
        assert isinstance(n_collapsed, int)


class TestHallucinationGuard:
    """HallucinationGuard -- hallucination detection."""

    def test_class_exists(self):
        from hypercore import HallucinationGuard
        assert HallucinationGuard is not None


class TestGenerationMetrics:
    """GenerationMetrics -- per-token metrics."""

    def test_import(self):
        from hypercore import GenerationMetrics
        gm = GenerationMetrics()
        assert gm is not None


class TestLazyImports:
    """All lazy imports resolve without error."""

    LAZY_NAMES = [
        "AxiomGauge",
        "ThermalRankController",
        "OnlineOjaBasis",
        "TreeDrafter",
        "EagleFeatureDrafter",
        "GCGAttack",
        "AutoPromptAttack",
        "PAIRAttack",
        "NativeLinear",
        "RiemannianAdamW",
        "KExpansionScheduler",
    ]

    @pytest.mark.parametrize("name", LAZY_NAMES)
    def test_lazy_import_resolves(self, name):
        import hypercore
        obj = getattr(hypercore, name)
        assert obj is not None, f"{name} resolved to None"


class TestRegimeDetector:
    """RegimeDetector — geometric regime change detection (Principle 6)."""

    def test_import(self):
        from hypercore import RegimeDetector
        rd = RegimeDetector()
        assert rd is not None
        assert rd.k == 12
        assert rd.window_size == 252

    def test_regime_assessment_import(self):
        from hypercore import RegimeAssessment, RegimeSignal
        ra = RegimeAssessment()
        assert ra.rci == 0.0
        assert ra.confidence == 0.0
        assert not ra.regime_change

        rs = RegimeSignal(name="test", raw=0.5, normalized=0.6, fired=True)
        assert rs.name == "test"
        assert rs.fired

    def test_fit_and_check_no_regime_change(self):
        """Detector should NOT fire on data from the same regime."""
        from hypercore.regime_detector import (
            RegimeDetector, generate_coupled_dynamics,
        )
        np.random.seed(42)
        traj = generate_coupled_dynamics(N=30, T=200, D=32, regime_change_at=200)
        assert traj.shape == (30, 200, 32)

        rd = RegimeDetector(intrinsic_dim=8, window_size=50, threshold=0.45)
        rd.fit(traj)

        # Prime window then check on training-regime data
        for t in range(50, 150):
            rd.check(traj[:, t, :])
        result = rd.check(traj[:, -1, :])
        assert result.rci < 0.65, f"RCI={result.rci:.3f} — false positive on training regime"
        assert not result.regime_change

    def test_regime_change_detected(self):
        """Detector SHOULD fire after a structural coupling change."""
        from hypercore.regime_detector import (
            RegimeDetector, generate_coupled_dynamics,
        )
        np.random.seed(123)
        # Generate with regime change at step 350 — use many entities
        # and a calibrated threshold for stable VAR(1) dynamics
        N = 50
        traj = generate_coupled_dynamics(
            N=N, T=500, D=32, regime_change_at=350,
            coupling_strength=0.7, noise=0.02,
        )

        rd = RegimeDetector(intrinsic_dim=8, window_size=100, threshold=0.40,
                            knn_k=min(5, N - 1))
        rd.fit(traj[:300])  # fit on regime 1 only

        # Prime window and check post-change
        for t in range(200, 350):
            rd.check(traj[:, t, :])

        fired_count = 0
        total_checks = 0
        for t in range(360, 500, 5):
            result = rd.check(traj[:, t, :])
            total_checks += 1
            if result.regime_change:
                fired_count += 1

        assert fired_count > 0, (
            f"Regime change not detected after coupling shift "
            f"(fired {fired_count}/{total_checks})"
        )

    def test_volatility_regime_detected(self):
        """Detector SHOULD fire on a volatility regime shift (curvature spike)."""
        from hypercore.regime_detector import (
            RegimeDetector, generate_volatility_regime,
        )
        np.random.seed(456)
        traj = generate_volatility_regime(
            N=20, T=500, D=32, regime_change_at=350,
            base_vol=0.02, high_vol=0.15,
        )

        rd = RegimeDetector(intrinsic_dim=8, window_size=100, threshold=0.50)
        rd.fit(traj[:300])

        fired_count = 0
        total_checks = 0
        for t in range(360, 500, 5):
            result = rd.check(traj[:, t, :])
            total_checks += 1
            if result.regime_change:
                fired_count += 1

        assert fired_count > 0, (
            f"Volatility regime change not detected (fired {fired_count}/{total_checks})"
        )

    def test_five_signals_present(self):
        """Every assessment must contain exactly five named signals."""
        from hypercore.regime_detector import (
            RegimeDetector, generate_coupled_dynamics,
        )
        np.random.seed(789)
        traj = generate_coupled_dynamics(N=20, T=200, D=32, regime_change_at=200)

        rd = RegimeDetector(intrinsic_dim=8, window_size=50)
        rd.fit(traj)
        result = rd.check(traj[:, -1, :])

        assert len(result.signals) == 5
        expected_names = {
            "manifold_deviation", "curvature_anomaly",
            "neighbor_instability", "spectral_drift", "geodesic_misalignment",
        }
        actual_names = {s.name for s in result.signals}
        assert actual_names == expected_names, f"Got {actual_names}"

    def test_jury_confidence_bounds(self):
        """Jury confidence J must be in [0, 1]."""
        from hypercore.regime_detector import RegimeDetector
        rd = RegimeDetector(intrinsic_dim=4, window_size=20)

        from hypercore.regime_detector import RegimeSignal
        # All signals low
        sigs_low = [RegimeSignal(name=f"s{i}", normalized=0.0, weight=0.2)
                     for i in range(5)]
        rci, J = rd._aggregate(sigs_low)
        assert 0.0 <= J <= 1.0
        assert J < 0.1  # should be near 0

        # All signals high — J = 1 - (1-0.2)^5 = 1 - 0.8^5 ≈ 0.672
        sigs_high = [RegimeSignal(name=f"s{i}", normalized=1.0, weight=0.2)
                      for i in range(5)]
        rci, J = rd._aggregate(sigs_high)
        assert 0.65 < J <= 1.0  # J = 1 - 0.8^5 ≈ 0.672

    def test_custom_weights(self):
        """Custom signal weights should be accepted and normalised."""
        from hypercore.regime_detector import RegimeDetector
        rd = RegimeDetector(intrinsic_dim=4, weights=[0.4, 0.3, 0.1, 0.1, 0.1])
        assert np.allclose(rd.weights.sum(), 1.0)

    def test_reset_clears_history(self):
        """reset() should clear rolling windows but keep fitted state."""
        from hypercore.regime_detector import (
            RegimeDetector, generate_coupled_dynamics,
        )
        np.random.seed(999)
        traj = generate_coupled_dynamics(N=20, T=200, D=32, regime_change_at=200)

        rd = RegimeDetector(intrinsic_dim=8, window_size=50)
        rd.fit(traj)
        rd.check(traj[:, -1, :])
        assert len(rd._proj_history) > 0
        rd.reset()
        assert len(rd._proj_history) == 0
        assert rd._fitted  # still fitted

    def test_not_fitted_raises(self):
        """Calling check() before fit() must raise RuntimeError."""
        import pytest
        from hypercore.regime_detector import RegimeDetector
        rd = RegimeDetector()
        with pytest.raises(RuntimeError, match="fit"):
            rd.check(np.random.randn(20, 32))


class TestAxiomGauge:
    """AxiomGauge -- GL(d) diagonal gauge optimization."""

    def test_import_only(self):
        from hypercore import AxiomGauge
        assert AxiomGauge is not None


class TestThermalRankController:
    """ThermalRankController -- temperature-driven rank."""

    def test_import_and_init(self):
        from hypercore import ThermalRankController
        ctrl = ThermalRankController()
        assert ctrl is not None


class TestOnlineOjaBasis:
    """OnlineOjaBasis -- rejection-driven PCA."""

    def test_import_only(self):
        from hypercore import OnlineOjaBasis
        assert OnlineOjaBasis is not None


class TestTreeDrafter:
    """TreeDrafter -- tree speculative decoding."""

    def test_import_and_config(self):
        from hypercore import TreeDrafter
        from scripts.tree_spec import TreeSpecConfig
        cfg = TreeSpecConfig(num_heads=2, max_branch=2, max_depth=3)
        assert cfg.num_heads == 2


class TestNativeLinear:
    """NativeLinear -- train on compressed manifold."""

    def test_import_and_forward(self):
        from hypercore import NativeLinear
        layer = NativeLinear(d=64, k=8)
        x = torch.randn(4, 64)
        y = layer(x)
        assert y.shape == (4, 64)


class TestRiemannianAdamW:
    """RiemannianAdamW -- manifold-respecting optimizer."""

    def test_import_and_init(self):
        from hypercore import NativeLinear, RiemannianAdamW
        layer = NativeLinear(d=64, k=8)
        opt = RiemannianAdamW(layer.parameters(), lr=1e-4)
        assert opt is not None
