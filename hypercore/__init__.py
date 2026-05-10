"""
hypercore — HyperTensor Core Geometric Modules
================================================
v1.0 | May 8, 2026

Submodules:
  geodesic_metric  — GeodesicMetric, HallucinationGuard, GenerationMetrics
  axiom_gauge      — AxiomGauge: GL(d) diagonal gauge optimization
  thermal_rank     — ThermalRankController: temperature-driven rank scheduler
  online_oja       — OnlineOjaBasis: rejection-driven Oja PCA update
  tree_spec        — TreeDrafter, EagleFeatureDrafter: Medusa/EAGLE drafting
  red_team         — GCGAttack, AutoPromptAttack, PAIRAttack: safety red-team

Usage:
  from hypercore import GeodesicMetric, HallucinationGuard
  from hypercore.axiom_gauge import AxiomGauge
  from hypercore.thermal_rank import ThermalRankController
"""

# Core geometric metric (always available)
from hypercore.geodesic_metric import (
    GeodesicMetric,
    HallucinationGuard,
    GenerationMetrics,
)

# Optional modules — import on demand, fail gracefully if dependencies missing
__all__ = [
    'GeodesicMetric',
    'HallucinationGuard',
    'GenerationMetrics',
    'AxiomGauge',
    'ThermalRankController',
    'OnlineOjaBasis',
    'TreeDrafter',
    'EagleFeatureDrafter',
    'GCGAttack',
    'AutoPromptAttack',
    'PAIRAttack',
]

def __getattr__(name):
    """Lazy import for optional modules."""
    if name == 'AxiomGauge':
        from scripts.axiom_gauge import AxiomGauge
        return AxiomGauge
    if name == 'ThermalRankController':
        from scripts.thermal_rank import ThermalRankController
        return ThermalRankController
    if name == 'OnlineOjaBasis':
        from scripts.online_oja import OnlineOjaBasis
        return OnlineOjaBasis
    if name in ('TreeDrafter', 'EagleFeatureDrafter'):
        from scripts.tree_spec import TreeDrafter, EagleFeatureDrafter
        return TreeDrafter if name == 'TreeDrafter' else EagleFeatureDrafter
    if name in ('GCGAttack', 'AutoPromptAttack', 'PAIRAttack'):
        from scripts.red_team import GCGAttack, AutoPromptAttack, PAIRAttack
        return {'GCGAttack': GCGAttack, 'AutoPromptAttack': AutoPromptAttack,
                'PAIRAttack': PAIRAttack}[name]
    raise AttributeError(f"module 'hypercore' has no attribute '{name}'")
