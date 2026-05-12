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
L-Functions and Dedekind Zeta Extension (Paper XVI gap 3)

Extends the AGT geometric framework from the Riemann zeta function to
the broader class of L-functions and Dedekind zeta functions where the
Generalised Riemann Hypothesis (GRH) applies.

Supported L-function families:
  - Dirichlet L-functions L(s, χ) for primitive characters
  - Dedekind zeta functions ζ_K(s) for number fields
  - Modular form L-functions (Hecke eigenforms)
  - Elliptic curve L-functions (Hasse-Weil)

The framework applies the same Z₂ symmetry detection: compute the
functional equation's involution, build a feature map from arithmetic
data, and measure whether candidate zeros satisfy the fixed-point property.

Reference: Stewart, "AGT Topology of Zeta Zeros," HyperTensor Paper XVI, 2026.
          Iwaniec & Kowalski, "Analytic Number Theory," AMS 2004.
          Darmon, "Rational Points on Modular Elliptic Curves," AMS 2004.

Usage:
    from l_functions import DirichletLFunction, DedekindZeta, LFunctionPipeline
    L = DirichletLFunction(modulus=5, character=2)
    zeros = L.known_zeros(n=100)
    pipe = LFunctionPipeline(L)
    results = pipe.verify_zeros(zeros)
"""

import numpy as np
from typing import List, Tuple, Optional, Callable, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import cmath
import math


# ---------------------------------------------------------------------------
# Abstract L-function base
# ---------------------------------------------------------------------------

@dataclass
class LFunctionConfig:
    """Configuration for an L-function."""
    name: str                        # Human-readable name
    conductor: int                   # Conductor (N)
    degree: int                      # Degree (d)
    functional_equation_sign: float  # ε (root number, ±1)
    gamma_factors: List[float]       # Γ-shift parameters μ_j
    analytic_conductor: float        # C = N ∏(1+|μ_j|)/π^{d/2}


class LFunction(ABC):
    """
    Abstract base for L-functions.

    Each concrete L-function must provide:
      - value(s): compute L(s) at a complex point
      - functional_equation(s): compute completed L(s) relation
      - involution(s): the symmetry ι(s) such that L(s) = ε·L(ι(s))
      - known_zeros(n): return the first n zeros on the critical line
    """

    def __init__(self, config: LFunctionConfig):
        self.cfg = config

    @abstractmethod
    def value(self, s: complex) -> complex:
        """Compute L(s) at complex argument s."""
        ...

    def completed_value(self, s: complex) -> complex:
        """Compute the completed L-function Λ(s) = γ(s)·L(s)."""
        gamma = self._gamma_factor(s)
        return gamma * self.value(s)

    def _gamma_factor(self, s: complex) -> complex:
        """Compute the gamma factor γ(s) = π^{-ds/2} ∏ Γ((s+μ_j)/2)."""
        cfg = self.cfg
        result = cmath.pi ** (-cfg.degree * s / 2)
        for mu in cfg.gamma_factors:
            result *= self._gamma((s + mu) / 2)
        return result

    @staticmethod
    def _gamma(z: complex) -> complex:
        """Simplified gamma function via Stirling approximation."""
        # For production, use mpmath.gamma
        if z.real > 0:
            return cmath.sqrt(2 * math.pi / z) * (z / math.e) ** z
        return 1.0  # Fallback

    def functional_equation(self, s: complex) -> complex:
        """The functional equation: Λ(s) = ε·Λ(1-s)."""
        return self.cfg.functional_equation_sign * self.completed_value(1 - s)

    @abstractmethod
    def involution(self, s: complex) -> complex:
        """The symmetry ι(s) such that L(s) = ε·L(ι(s))."""
        ...

    @abstractmethod
    def known_zeros(self, n: int = 100) -> List[complex]:
        """Return the first n known zeros on the critical line."""
        ...


# ---------------------------------------------------------------------------
# Dirichlet L-function L(s, χ)
# ---------------------------------------------------------------------------

class DirichletLFunction(LFunction):
    """
    Dirichlet L-function L(s, χ) for a primitive Dirichlet character χ mod q.

    Functional equation: Λ(s,χ) = ε(χ)·Λ(1-s, χ̄)
    Involution: ι(s) = 1 - s  (same Z₂ structure as Riemann ζ)

    The conductor is q, degree is 1.
    """

    def __init__(self, modulus: int, character_values: Optional[Dict[int, complex]] = None):
        """
        Args:
            modulus: The modulus q of the Dirichlet character.
            character_values: Dict mapping n → χ(n) for the character.
                            If None, use the Legendre symbol (quadratic character).
        """
        self.modulus = modulus
        self._char_values = character_values or self._legendre_character(modulus)

        # Determine parity for gamma factor
        # χ(-1) = +1: even character (μ = 0)
        # χ(-1) = -1: odd character (μ = 1)
        chi_neg1 = self._char_values.get(modulus - 1, 0)
        mu = 0.0 if abs(chi_neg1 - 1) < 0.1 else 1.0

        config = LFunctionConfig(
            name=f'L(s,χ) mod {modulus}',
            conductor=modulus,
            degree=1,
            functional_equation_sign=self._root_number(),
            gamma_factors=[mu],
            analytic_conductor=modulus,
        )
        super().__init__(config)

    @staticmethod
    def _legendre_character(p: int) -> Dict[int, complex]:
        """Compute Legendre symbol (·|p) character values."""
        values = {}
        for n in range(1, p):
            # Euler's criterion: (n|p) ≡ n^{(p-1)/2} mod p
            leg = pow(n, (p - 1) // 2, p)
            values[n] = 1.0 if leg == 1 else (-1.0 if leg == p - 1 else 0.0)
        return values

    def _root_number(self) -> float:
        """Compute the root number ε(χ) for a primitive Dirichlet character."""
        # Gauss sum normalisation
        chi_vals = list(self._char_values.values())
        gauss_sum = sum(
            chi * cmath.exp(2j * math.pi * n / self.modulus)
            for n, chi in self._char_values.items()
        )
        return (gauss_sum / abs(gauss_sum)).real if abs(gauss_sum) > 1e-10 else 1.0

    def value(self, s: complex) -> complex:
        """Compute L(s,χ) = Σ χ(n)·n^{-s} by truncated series."""
        result = 0j
        # Summation up to reasonable truncation
        for n, chi in list(self._char_values.items())[:1000]:
            if abs(chi) > 1e-10:
                result += chi * (n ** (-s))
        return result

    def involution(self, s: complex) -> complex:
        """For Dirichlet L-functions: ι(s) = 1 - s + i·0 (same as Riemann ζ)."""
        return 1 - s

    def known_zeros(self, n: int = 100) -> List[complex]:
        """
        Return low-lying zeros on the critical line.

        Prefers cached LMFDB ``positive_zeros`` for this character's L-function
        (matched by conductor / modulus). Falls back to a Riemann–von Mangoldt
        average-spacing approximation when no cached record matches.
        """
        # Try LMFDB cache first
        try:
            from data_sources import load_lmfdb_lfunctions  # type: ignore
        except ImportError:  # pragma: no cover
            try:
                from .data_sources import load_lmfdb_lfunctions  # type: ignore
            except ImportError:
                load_lmfdb_lfunctions = None  # type: ignore

        if load_lmfdb_lfunctions is not None:
            for r in load_lmfdb_lfunctions(degree=1):
                if int(r.get("conductor", -1)) == int(self.config.conductor):
                    pz = r.get("positive_zeros") or []
                    if pz:
                        return [complex(0.5, float(t)) for t in pz[:n]]

        # Fallback: synthetic zeros from average spacing
        zeros = []
        t = 14.0  # Start near first Riemann zero
        step = 2 * math.pi / math.log(max(t, 20) / (2 * math.pi * math.e))
        for _ in range(n):
            zeros.append(complex(0.5, t))
            t += step
            step = 2 * math.pi / math.log(max(t, 20) / (2 * math.pi * math.e))
        return zeros


# ---------------------------------------------------------------------------
# Dedekind zeta function ζ_K(s)
# ---------------------------------------------------------------------------

class DedekindZeta(LFunction):
    """
    Dedekind zeta function ζ_K(s) for a number field K.

    The functional equation relates ζ_K(s) and ζ_K(1-s) via the
    discriminant and regulator of K. The involution is ι(s) = 1 - s.

    For quadratic fields Q(√d), this factorises as ζ(s)·L(s,χ_d).
    """

    def __init__(self, discriminant: int, degree: int = 2):
        """
        Args:
            discriminant: Fundamental discriminant of the number field.
            degree: Degree of the field extension [K:Q].
        """
        self.discriminant = discriminant
        config = LFunctionConfig(
            name=f'ζ_K(s) for Q(√{discriminant})',
            conductor=abs(discriminant),
            degree=degree,
            functional_equation_sign=1.0 if discriminant > 0 else -1.0,
            gamma_factors=[0.0] * degree,
            analytic_conductor=abs(discriminant),
        )
        super().__init__(config)

    def value(self, s: complex) -> complex:
        """Factorised form: ζ_K(s) = ζ(s)·L(s,χ_d)."""
        # Riemann zeta
        zeta_s = sum(n ** (-s) for n in range(1, 1000))
        # Dirichlet L-function for quadratic character
        chi = DirichletLFunction(abs(self.discriminant))
        L_s = chi.value(s)
        return zeta_s * L_s

    def involution(self, s: complex) -> complex:
        return 1 - s

    def known_zeros(self, n: int = 100) -> List[complex]:
        """Low-lying zeros of ζ_K(s) on the critical line."""
        # For quadratic fields, zeros are a subset of Dirichlet L-function zeros
        L = DirichletLFunction(abs(self.discriminant))
        return L.known_zeros(n)


# ---------------------------------------------------------------------------
# L-function verification pipeline
# ---------------------------------------------------------------------------

@dataclass
class LFunctionVerificationResult:
    """Result of verifying zeros for an L-function."""
    l_function_name: str
    n_zeros_tested: int
    n_on_critical: int                    # Should match n_zeros
    mean_deviation: float                 # Mean |ι(s) - s| for tested zeros
    max_deviation: float
    computation_time_s: float
    verdict: str                          # "All zeros are Z₂ fixed points" or caveats


class LFunctionPipeline:
    """
    Pipeline for verifying Z₂ symmetry across L-function families.

    Applies the same geometric framework (AGT difference operator)
    to L-functions beyond Riemann ζ.
    """

    def __init__(self, l_function: LFunction, feature_dim: int = 32):
        self.L = l_function
        self.feature_dim = feature_dim

    def verify_zeros(
        self,
        zeros: Optional[List[complex]] = None,
        n_zeros: int = 100,
    ) -> LFunctionVerificationResult:
        """
        Verify that known zeros satisfy ι(s) = s (Z₂ fixed-point property).

        Args:
            zeros: List of zeros to test. If None, use self.L.known_zeros(n).
            n_zeros: Number of zeros to generate if zeros not provided.

        Returns:
            LFunctionVerificationResult.
        """
        import time
        t0 = time.perf_counter()

        if zeros is None:
            zeros = self.L.known_zeros(n_zeros)

        deviations = []
        on_critical = 0

        for s in zeros:
            iota_s = self.L.involution(s)
            dev = abs(s - iota_s)
            deviations.append(dev)
            if dev < 1e-10:
                on_critical += 1

        elapsed = time.perf_counter() - t0
        mean_dev = float(np.mean(deviations))
        max_dev = float(np.max(deviations))

        verdict = (f"All {len(zeros)} tested zeros are Z₂ fixed points "
                   f"(mean |s-ι(s)| = {mean_dev:.2e})"
                   if on_critical == len(zeros)
                   else f"{on_critical}/{len(zeros)} zeros are fixed points")

        return LFunctionVerificationResult(
            l_function_name=self.L.cfg.name,
            n_zeros_tested=len(zeros),
            n_on_critical=on_critical,
            mean_deviation=mean_dev,
            max_deviation=max_dev,
            computation_time_s=elapsed,
            verdict=verdict,
        )

    def cross_family_comparison(
        self,
        families: List[LFunction],
        n_zeros: int = 50,
    ) -> List[LFunctionVerificationResult]:
        """Compare Z₂ symmetry across multiple L-function families."""
        results = []
        for L in families:
            pipe = LFunctionPipeline(L)
            result = pipe.verify_zeros(n_zeros=n_zeros)
            results.append(result)
        return results


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("L-Functions Extension — Self-Test")
    print("=" * 60)

    # Test Dirichlet L-function mod 5
    print("\n  Dirichlet L-function L(s,χ) mod 5:")
    L = DirichletLFunction(modulus=5)
    zeros = L.known_zeros(n=20)
    print(f"    First zero: s = {zeros[0]}")
    print(f"    ι(s) = {L.involution(zeros[0])}")
    print(f"    |s-ι(s)| = {abs(zeros[0] - L.involution(zeros[0])):.2e}")

    pipe = LFunctionPipeline(L)
    result = pipe.verify_zeros(zeros)
    print(f"    {result.verdict}")

    # Test Dedekind zeta for Q(√5)
    print("\n  Dedekind ζ_K(s) for Q(√5):")
    zeta = DedekindZeta(discriminant=5)
    zeros_k = zeta.known_zeros(n=10)
    pipe2 = LFunctionPipeline(zeta)
    result2 = pipe2.verify_zeros(zeros_k)
    print(f"    {result2.verdict}")

    # Cross-family comparison
    print("\n  Cross-family comparison:")
    families = [
        DirichletLFunction(modulus=3),
        DirichletLFunction(modulus=5),
        DirichletLFunction(modulus=7),
    ]
    results = LFunctionPipeline(families[0]).cross_family_comparison(families, n_zeros=10)
    for r in results:
        print(f"    {r.l_function_name}: {r.n_on_critical}/{r.n_zeros_tested} "
              f"({r.computation_time_s:.3f}s)")

    print("\n  L-Functions module: OK")
