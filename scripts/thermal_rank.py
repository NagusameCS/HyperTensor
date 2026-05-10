#!/usr/bin/env python3
"""
Thermal Rank Controller: Temperature-Driven Compression Rank Scheduler (Paper II §Thermal)

Implements a closed-loop rank controller that reduces compression rank when
GPU temperature exceeds a threshold, preventing thermal throttling. Also
computes tokens-per-joule efficiency with an optional power-budget constraint.

Reference: Stewart, "Geodesic Projection Pipeline," HyperTensor Paper II, 2026.

Usage:
    from thermal_rank import ThermalRankController
    ctrl = ThermalRankController(r_min=256, r_max=1536)
    current_rank = ctrl.update(temperature_c=72.0)
    tpj = ctrl.tokens_per_joule(tok_per_sec=35.6, power_watts=103.0)
"""

import time
import threading
from typing import Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ThermalConfig:
    """Configuration for the thermal rank controller."""
    r_min: int = 256           # Minimum allowable rank
    r_max: int = 1536          # Maximum (baseline) rank
    t_low_c: float = 65.0      # Temperature below which full rank is used
    t_high_c: float = 85.0     # Temperature at which r_min is reached
    poll_interval_s: float = 0.5  # Seconds between temperature reads
    power_budget_w: Optional[float] = None  # Optional power cap in watts


@dataclass
class ThermalState:
    """Current thermal controller state."""
    rank: int
    temperature_c: float
    power_w: float
    sm_clock_mhz: int
    timestamp: float
    throttled: bool
    tokens_per_joule: float = 0.0


class ThermalRankController:
    """
    Closed-loop temperature-driven compression rank scheduler.

    When GPU temperature rises, compression rank is linearly reduced to
    decrease memory traffic and heat generation, keeping the GPU below
    the throttling threshold. The controller is closed-loop stable as
    long as the heat-generation rate at r_min is below the cooling
    capacity at t_high.

    Tokens-per-joule efficiency is also tracked, enabling energy-aware
    rank planning (the TPJ diffplan term from Paper II).
    """

    def __init__(self, config: Optional[ThermalConfig] = None):
        """
        Args:
            config: ThermalConfig with bounds, thresholds, and optional power cap.
        """
        self.config = config or ThermalConfig()
        self._current_rank = self.config.r_max
        self._lock = threading.Lock()
        self._history: list = []  # List of ThermalState records

    @property
    def current_rank(self) -> int:
        """Current compression rank."""
        with self._lock:
            return self._current_rank

    def update(
        self,
        temperature_c: float,
        power_w: float = 0.0,
        sm_clock_mhz: int = 0,
        tok_per_sec: Optional[float] = None,
    ) -> ThermalState:
        """
        Update the controller with a new temperature reading.

        Args:
            temperature_c: GPU temperature in Celsius.
            power_w: GPU power draw in watts (optional).
            sm_clock_mhz: SM clock frequency in MHz (optional).
            tok_per_sec: Current decode throughput (optional, for TPJ).

        Returns:
            ThermalState with the new rank and telemetry.
        """
        cfg = self.config

        # Linear interpolation between t_low (r_max) and t_high (r_min)
        if temperature_c <= cfg.t_low_c:
            frac = 0.0  # No reduction
        elif temperature_c >= cfg.t_high_c:
            frac = 1.0  # Full reduction
        else:
            frac = (temperature_c - cfg.t_low_c) / (cfg.t_high_c - cfg.t_low_c)

        # Compute rank
        rank_float = cfg.r_max - (cfg.r_max - cfg.r_min) * frac

        # Apply optional power cap
        if cfg.power_budget_w is not None and power_w > cfg.power_budget_w:
            power_frac = (power_w - cfg.power_budget_w) / max(power_w, 1.0)
            rank_float = max(rank_float - power_frac * (cfg.r_max - cfg.r_min), cfg.r_min)

        rank = int(round(rank_float))
        rank = max(cfg.r_min, min(cfg.r_max, rank))

        with self._lock:
            self._current_rank = rank

        throttled = temperature_c >= cfg.t_high_c

        tpj = 0.0
        if tok_per_sec is not None and power_w > 0:
            tpj = tok_per_sec / power_w

        state = ThermalState(
            rank=rank,
            temperature_c=temperature_c,
            power_w=power_w,
            sm_clock_mhz=sm_clock_mhz,
            timestamp=time.time(),
            throttled=throttled,
            tokens_per_joule=tpj,
        )
        self._history.append(state)

        # Keep only last 1000 entries
        if len(self._history) > 1000:
            self._history = self._history[-1000:]

        return state

    def tokens_per_joule(self, tok_per_sec: float, power_w: float) -> float:
        """Compute instantaneous tokens-per-joule efficiency."""
        if power_w <= 0:
            return 0.0
        return tok_per_sec / power_w

    def tpj_diffplan_gradient(
        self,
        current_rank: int,
        baseline_tpj: float,
        lambda_tpj: float = 0.1,
    ) -> float:
        """
        Compute the tokens-per-joule gradient term for the differentiable
        rank plan (Paper II §Thermal, TPJ diffplan term).

        Args:
            current_rank: Current soft rank (float).
            baseline_tpj: Baseline tokens-per-joule at r_max.
            lambda_tpj: TPJ regularisation weight.

        Returns:
            Gradient contribution from TPJ term.
        """
        # The TPJ term encourages reducing rank when TPJ drops
        # Simple linear model: d(TPJ)/dr ≈ baseline_tpj / r_max
        # ∂L/∂r ≈ λ * baseline_tpj / r_max
        return lambda_tpj * baseline_tpj / max(self.config.r_max, 1)

    def estimate_energy_saving(
        self, rank_before: int, rank_after: int, tok_per_sec: float, power_w: float
    ) -> float:
        """
        Estimate energy saving (joules per hour) from rank reduction.

        Args:
            rank_before: Rank before reduction.
            rank_after: Rank after reduction.
            tok_per_sec: Current throughput.
            power_w: Current power draw.

        Returns:
            Estimated joules saved per hour of sustained operation.
        """
        if rank_before <= rank_after:
            return 0.0
        # Rough model: power ∝ memory bandwidth ∝ rank^(1/2)
        power_ratio = (rank_after / rank_before) ** 0.5
        power_after = power_w * power_ratio
        joules_per_hour = (power_w - power_after) * 3600.0
        return max(0.0, joules_per_hour)

    def get_history(self, last_n: int = 100) -> list:
        """Return recent telemetry history."""
        return self._history[-last_n:]

    def summary(self) -> dict:
        """Return a summary dict suitable for JSON export."""
        if not self._history:
            return {"status": "no_data"}
        temps = [s.temperature_c for s in self._history]
        ranks = [s.rank for s in self._history]
        tpjs = [s.tokens_per_joule for s in self._history if s.tokens_per_joule > 0]
        return {
            "config": {
                "r_min": self.config.r_min,
                "r_max": self.config.r_max,
                "t_low_c": self.config.t_low_c,
                "t_high_c": self.config.t_high_c,
            },
            "current_rank": self.current_rank,
            "n_samples": len(self._history),
            "temperature": {
                "mean": sum(temps) / len(temps),
                "min": min(temps),
                "max": max(temps),
                "p90": sorted(temps)[int(0.9 * len(temps))],
            },
            "rank": {
                "mean": sum(ranks) / len(ranks),
                "min": min(ranks),
                "max": max(ranks),
            },
            "tokens_per_joule": {
                "mean": sum(tpjs) / len(tpjs) if tpjs else 0,
            } if tpjs else None,
        }


# ---------------------------------------------------------------------------
# GPU telemetry helper (NVML-based)
# ---------------------------------------------------------------------------
def get_nvml_telemetry() -> Optional[Tuple[float, float, int]]:
    """
    Read GPU temperature, power, and SM clock via NVML.
    Returns (temperature_c, power_w, sm_clock_mhz) or None if NVML unavailable.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
        clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        pynvml.nvmlShutdown()
        return (float(temp), power, int(clock))
    except (ImportError, Exception):
        return None


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Thermal Rank Controller — Self-Test")
    print("=" * 60)

    cfg = ThermalConfig(r_min=256, r_max=1536, t_low_c=65.0, t_high_c=85.0)
    ctrl = ThermalRankController(cfg)

    # Simulate a heating scenario
    scenario = [
        (40.0, 15.0, 2235),   # Cold start
        (55.0, 45.0, 2235),   # Warming up
        (70.0, 80.0, 2235),   # Mid-range
        (78.0, 103.0, 2235),  # Approaching throttle
        (84.0, 109.0, 2100),  # Near throttle, clock dropping
        (87.0, 95.0, 1800),   # Throttled
        (72.0, 75.0, 2235),   # Cooling down
        (50.0, 50.0, 2235),   # Recovered
    ]

    for temp, power, clock in scenario:
        state = ctrl.update(temp, power, clock, tok_per_sec=35.0)
        flag = "[THROTTLED]" if state.throttled else "[OK]"
        print(f"  T={temp:5.1f}C  P={power:6.1f}W  "
              f"clock={clock:4d}MHz  rank={state.rank:4d}  "
              f"TPJ={state.tokens_per_joule:.3f}  {flag}")

    print(f"\n  Final rank: {ctrl.current_rank}")
    print(f"  History samples: {len(ctrl._history)}")

    # NVML check
    telemetry = get_nvml_telemetry()
    if telemetry:
        t, p, c = telemetry
        print(f"\n  NVML live: T={t:.1f}C  P={p:.1f}W  clock={c}MHz")
    else:
        print("\n  NVML: not available (expected on non-GPU host)")

    print(f"\n  Summary: {ctrl.summary()}")
    print("\n  Thermal Rank module: OK")
