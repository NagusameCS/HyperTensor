"""
Test catalog — all 18 verification scripts with metadata.
Tests are auto-discovered from the HyperTensor scripts/ directory.
New tests can be added by creating a script and registering it here,
or by placing a `# ht-repro: <id> <tier> <paper> <group>` comment.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS = ROOT / "scripts"
BENCHMARKS = ROOT / "benchmarks"

# ── Manual Catalog (auto-discovery supplements this) ───────────────
MANUAL_CATALOG: List[dict] = [
    # ── T1 (CPU-only) ──
    {"id":"smoke","name":"60-Second Smoke Test (Riemann core math)","script":"scripts/faithfulness_rigorous.py","tier":"T1","paper":"XVI-XVIII","group":"core","timeout":60,"desc":"SV1=8.94, Z₂ exact, rank-1 proven"},
    {"id":"jury-proof","name":"Jury Proof (8 Theorems)","script":"scripts/jury_final.py","tier":"T1","paper":"Foundation","group":"jury","timeout":30,"desc":"J = 1−∏(1−cᵢ), 8 theorems verified"},
    {"id":"jury-horizon","name":"Jury Horizon + J-Decay","script":"scripts/jury_horizon.py","tier":"T1","paper":"Foundation","group":"jury","timeout":30,"desc":"d_h = R·(−ln(1−0.5^(1/N))), knowledge boundary"},
    {"id":"jury-scaling","name":"Jury Scaling (174× speedup)","script":"scripts/jury_scaling.py","tier":"T1","paper":"Foundation","group":"jury","timeout":30,"desc":"174× at 128 jurors, 153× at 512"},
    {"id":"jury-ensemble","name":"Jury Ensemble (Reg vs Class)","script":"scripts/jury_ensemble.py","tier":"T1","paper":"Foundation","group":"jury","timeout":30,"desc":"Regression beats classification, 3-temperature optimal"},
    {"id":"riemann-lmfdb","name":"LMFDB Meta-Jury (54,949 zeros)","script":"scripts/validate_riemann_lmfdb.py","tier":"T1","paper":"XVIII","group":"riemann","timeout":120,"desc":"100% on critical, TPR=1.0, FPR=0.0, Pearson r=1.0"},
    {"id":"riemann-agt","name":"AGT v3 — Zeta Zero Topology","script":"scripts/agt_v3.py","tier":"T1","paper":"XVI","group":"riemann","timeout":120,"desc":"98% detection, 1392× separation, k90=k95=1"},
    {"id":"riemann-comprehensive","name":"Comprehensive Riemann Verify","script":"scripts/riemann_comprehensive_verify.py","tier":"T1","paper":"XVI-XVIII","group":"riemann","timeout":120,"desc":"ALL 9 TESTS PASSED, D(s) rank-1 exact"},
    {"id":"safe-ogd","name":"Safe OGD — Zero Leakage","script":"scripts/verify_safe_loss_aczel.py","tier":"T1","paper":"XIII","group":"safety","timeout":30,"desc":"0% forbidden leakage, Q_fᵀ·P_safe=0"},
    {"id":"gtc-vs-rag","name":"GTC vs RAG Benchmark","script":"scripts/gtc_vs_rag.py","tier":"T1","paper":"VIII","group":"runtime","timeout":60,"desc":"30.9 µs/q, 5.96 KB/record, ~16× faster"},
    {"id":"bp-ns-bound","name":"BP/NS Bound Verification","script":"scripts/verify_bp_ns_bound.py","tier":"T1","paper":"Audit","group":"audit","timeout":30,"desc":"160/160 trials pass, tightness ratio=0.425"},
    {"id":"behavioral-residue","name":"Behavioral Residue Invariant","script":"scripts/verify_behavioral_residue_invariant.py","tier":"T1","paper":"Audit","group":"audit","timeout":120,"desc":"Layers 0-22 hold, layer 29 under investigation"},
    # ── T2 (Consumer GPU) ──
    {"id":"bilateral-ugt","name":"Bilateral UGT (Subspace Overlap)","script":"scripts/bilateral_ugt.py","tier":"T2","paper":"XI","group":"living","timeout":600,"desc":"overlap > 0.99, Wielandt-Hoffman"},
    {"id":"acm-prototype","name":"ACM — Learned Involution","script":"scripts/acm_prototype.py","tier":"T2","paper":"XVII","group":"riemann","timeout":120,"desc":"ι²≈id (ε=0.009), fixed-point detection"},
    {"id":"grc-distill","name":"GRC Light Distillation (Phase 1)","script":"scripts/grc_distill.py","tier":"T2","paper":"V","group":"compression","timeout":120,"desc":"ρ ratio, recoverable-energy bound"},
    # ── T3 (Datacenter GPU) ──
    {"id":"agt-scale","name":"AGT at 50K Primes (Full Scale)","script":"scripts/agt_scale_ec2.py","tier":"T3","paper":"XVI","group":"riemann","timeout":1800,"desc":"100% detection, 800× separation"},
    {"id":"cog-10k","name":"COG 10K Interaction Test","script":"scripts/cog_10k.py","tier":"T3","paper":"XV","group":"living","timeout":3600,"desc":"COG converged, zero novel in final 7000"},
]

# ── Expected Outputs Database ──────────────────────────────────────
EXPECTED_OUTPUTS = {
    "smoke": {"sv1": 8.944272, "z2_exact": True, "error_at_k12": 0.0},
    "jury-proof": {"theorems_verified": 8},
    "jury-horizon": {"j_decay_table": True, "d_h_derived": True},
    "jury-scaling": {"speedup_at_128": "~174x", "speedup_at_512": "~153x"},
    "jury-ensemble": {"regression_better": True},
    "riemann-lmfdb": {"on_critical_pct": 100.0, "tpr": 1.0, "fpr": 0.0},
    "riemann-agt": {"detection_rate": "≥98%", "k90": 1, "k95": 1},
    "riemann-comprehensive": {"all_tests_pass": True},
    "safe-ogd": {"forbidden_leakage": 0.0},
    "gtc-vs-rag": {"lookup_us": "~30.9", "record_kb": "~5.96"},
    "bp-ns-bound": {"trials_pass": 160, "total": 160},
    "behavioral-residue": {"layers_0_22_hold": True},
}

def load_catalog() -> List[dict]:
    """Load full catalog, including auto-discovered tests."""
    catalog = list(MANUAL_CATALOG)
    # Auto-discover: scan scripts/ for files with # ht-repro: markers
    if SCRIPTS.exists():
        for f in sorted(SCRIPTS.glob("*.py")):
            try:
                content = f.read_text(encoding="utf-8", errors="replace")[:2000]
                m = re.search(r'#\s*ht-repro:\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)', content)
                if m and not any(t["script"] == f"scripts/{f.name}" for t in catalog):
                    catalog.append({
                        "id": m.group(1), "name": f.stem.replace("_", " ").title(),
                        "script": f"scripts/{f.name}",
                        "tier": m.group(2), "paper": m.group(3), "group": m.group(4),
                        "timeout": 120, "desc": "Auto-discovered",
                    })
            except Exception:
                pass
    return catalog

def find_test(test_id: str) -> Optional[dict]:
    """Find a test by ID."""
    catalog = load_catalog()
    for t in catalog:
        if t["id"] == test_id:
            return t
    return None

def tests_by_tier(tier: str) -> List[dict]:
    return [t for t in load_catalog() if t["tier"] == tier]

def tests_by_group(group: str) -> List[dict]:
    return [t for t in load_catalog() if t["group"].lower() == group.lower()]

def tests_by_paper(paper_label: str) -> List[dict]:
    return [t for t in load_catalog() if paper_label in t["paper"]]
