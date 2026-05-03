#!/usr/bin/env python3
"""
MASTER EXPERIMENT PLAN — Prove every unproven prediction across HyperTensor Papers I-X.

STATUS LEGEND: [PENDING] = unproven, [RUNNING] = in progress, [DONE] = measured
"""

EXPERIMENTS = {
    # =========================================================================
    # PAPER F: Task-Level Asymmetric Degradation
    # =========================================================================
    "F1": {
        "paper": "Paper VI (F)",
        "claim": "MMLU (knowledge retrieval) survives compression <2% drop at k=1024",
        "status": "PENDING",
        "experiment": "Run MMLU benchmark on GRC-compressed SmolLM2-135M at k={256,512,1024,1536,full}",
        "metric": "MMLU accuracy vs compression rank",
        "pass": "≤2% drop at k=1024",
        "requires": ["./geodessical2.exe", "MMLU dataset", "task_bench.py"],
        "runtime": "~2 hrs on RTX 4070"
    },
    "F2": {
        "paper": "Paper VI (F)",
        "claim": "GSM8K (math reasoning) collapses >5% drop at k=1024",
        "status": "PENDING",
        "experiment": "Run GSM8K benchmark on GRC-compressed model at k={256,512,1024,1536,full}",
        "metric": "GSM8K exact-match vs compression rank",
        "pass": ">5% drop at k=1024",
        "requires": ["./geodessical2.exe", "GSM8K dataset", "task_bench.py"],
        "runtime": "~3 hrs on RTX 4070"
    },
    "F3": {
        "paper": "Paper VI (F)",
        "claim": "HumanEval (code) shows bimodal degradation",
        "status": "PENDING",
        "experiment": "Run HumanEval on GRC-compressed model at k={256,512,1024,1536,full}",
        "metric": "pass@1 vs compression rank",
        "pass": "Bimodal: easy survive, hard collapse",
        "requires": ["./geodessical2.exe", "HumanEval dataset"],
        "runtime": "~1 hr on RTX 4070"
    },
    "F4": {
        "paper": "Paper VI (F)",
        "claim": "Safe compression frontier at k=1024 for all tasks",
        "status": "PENDING",
        "experiment": "Cross-task analysis: find min k where ALL tasks stay within 95% of baseline",
        "metric": "k value at 95% preservation threshold per task",
        "pass": "k=1024 preserves ≥95% on all tasks",
        "requires": ["Results from F1, F2, F3"],
        "runtime": "0 (analysis only)"
    },

    # =========================================================================
    # PAPER I: Super-Baseline Universality
    # =========================================================================
    "I1": {
        "paper": "Paper IX (I)",
        "claim": "RTX 4090 (72MB L2) peaks at k*=1536 with 1.04-1.06 throughput",
        "status": "PENDING",
        "experiment": "Benchmark GRC kernel on RTX 4090 at k={256,512,768,1024,1280,1536,1792,full}",
        "metric": "Throughput (tok/s) vs k, find k*",
        "pass": "k*=1536, throughput ratio 1.04-1.06",
        "requires": ["RTX 4090 GPU access"],
        "runtime": "Need GPU access"
    },
    "I2": {
        "paper": "Paper IX (I)",
        "claim": "A100 (40MB L2) peaks at k*=1024 with 1.04-1.06 throughput",
        "status": "PENDING",
        "experiment": "Benchmark on A100 via EC2 or cloud",
        "metric": "Throughput vs k, find k*",
        "pass": "k*=1024, throughput ratio 1.04-1.06",
        "requires": ["AWS p4d instance or Lambda A100"],
        "runtime": "~2 hrs on A100"
    },
    "I3": {
        "paper": "Paper IX (I)",
        "claim": "H100 (50MB L2) peaks at k*=1280 with 1.02-1.04 throughput",
        "status": "PENDING",
        "experiment": "Benchmark on H100 via cloud",
        "metric": "Throughput vs k, find k*",
        "pass": "k*=1280, throughput ratio 1.02-1.04",
        "requires": ["H100 access (rare)"],
        "runtime": "Need GPU access"
    },
    "I4": {
        "paper": "Paper IX (I)",
        "claim": "Super-baseline effect transfers to KV-cache projection",
        "status": "PENDING",
        "experiment": "Implement GRC kernel for KV-cache projection, benchmark throughput",
        "metric": "Throughput with KV-cache GRC vs uncompressed",
        "pass": "Throughput increase at optimal k",
        "requires": ["Modified geodessical binary or simulation"],
        "runtime": "~1 hr (simulation)"
    },
    "I5": {
        "paper": "Paper IX (I)",
        "claim": "Super-baseline effect transfers to LoRA-augmented FFN",
        "status": "PENDING",
        "experiment": "Simulate fused GRC+LoRA FFN kernel throughput",
        "metric": "FLOPS/byte ratio analysis confirms fusion benefit",
        "pass": "Fusion benefit > FLOP overhead",
        "requires": ["Analytic simulation in Python"],
        "runtime": "~30 min (simulation)"
    },
    "I6": {
        "paper": "Paper IX (I)",
        "claim": "L40S (EC2, 48MB L2) follows same prediction curve",
        "status": "PENDING",
        "experiment": "Benchmark on EC2 g6e.xlarge (L40S) at varying k",
        "metric": "Throughput vs k, find k*",
        "pass": "L40S shows super-baseline at predicted k*",
        "requires": ["EC2 g6e.xlarge", "geodessical binary for Linux"],
        "runtime": "~1.5 hrs on L40S"
    },

    # =========================================================================
    # PAPER H: GTC as Vector Database
    # =========================================================================
    "H1": {
        "paper": "Paper VIII (H)",
        "claim": "GTC predicts tokens 15.5 faster than vector-DB RAG (analytic)",
        "status": "PENDING",
        "experiment": "Build Python simulation: GTC token prediction vs FAISS retrieval + LLM decode",
        "metric": "End-to-end latency for 10K Wikipedia queries",
        "pass": "GTC latency < RAG latency / 10 (conservative)",
        "requires": ["FAISS", "HuggingFace model", "Wikipedia dataset"],
        "runtime": "~2 hrs (simulation + benchmark)"
    },
    "H2": {
        "paper": "Paper VIII (H)",
        "claim": "Hybrid GTC+RAG combines scalability with precision",
        "status": "PENDING",
        "experiment": "Implement and benchmark hybrid: FAISS narrows → GTC refines",
        "metric": "Precision@k vs latency trade-off",
        "pass": "Hybrid Pareto-dominates both pure approaches",
        "requires": ["Results from H1"],
        "runtime": "~1 hr (analysis + simulation)"
    },

    # =========================================================================
    # PAPER G: FFN Clustering
    # =========================================================================
    "G1": {
        "paper": "Paper VII (G)",
        "claim": "FFN cluster compression recovers 21-25% of global compression error",
        "status": "DONE (local reconstruction)",
        "experiment": "ALREADY MEASURED: local reconstruction error. Need PPL.",
        "metric": "Already have reconstruction data",
        "pass": "N/A (already done)",
        "requires": [],
        "runtime": "0"
    },
    "G2": {
        "paper": "Paper VII (G)",
        "claim": "End-to-end PPL on WikiText-2 validates cluster compression",
        "status": "PENDING",
        "experiment": "Run perplexity measurement on FFN-cluster-compressed model",
        "metric": "PPL vs compression: global SVD vs per-cluster SVD",
        "pass": "Per-cluster PPL < global PPL by ≥15% of gap to uncompressed",
        "requires": ["HuggingFace evaluate", "WikiText-2", "FFN weight modification"],
        "runtime": "~2 hrs on RTX 4070"
    },
    "G3": {
        "paper": "Paper VII (G)",
        "claim": "Combined attention+FFN compression at C=4,k_frac=0.50 gives ~2.5 byte savings",
        "status": "PENDING",
        "experiment": "Compute total model bytes for attention GRC + FFN cluster compression",
        "metric": "Total model size vs uncompressed",
        "pass": "≥2.0 byte savings",
        "requires": ["Results from G2 + Paper A data"],
        "runtime": "~30 min (computation)"
    },

    # =========================================================================
    # PAPER E: GRC Light Distillation
    # =========================================================================
    "E1": {
        "paper": "Paper V (E)",
        "claim": "Per-matrix SVD beats shared-basis GRC for individual projections",
        "status": "DONE (per-matrix data)",
        "experiment": "ALREADY MEASURED: +79.4% SmolLM2, +18.9% Llama-8B",
        "metric": "N/A (already done)",
        "pass": "N/A",
        "requires": [],
        "runtime": "0"
    },
    "E2": {
        "paper": "Paper V (E)",
        "claim": "LoRA distillation recovers PPL after GRC compression",
        "status": "PENDING",
        "experiment": "Run end-to-end PPL on GRC-compressed+distilled model",
        "metric": "PPL: uncompressed vs compressed vs compressed+distilled",
        "pass": "Distillation recovers ≥70% of PPL gap",
        "requires": ["HF evaluate", "WikiText-2", "GRC projection + LoRA adaptation"],
        "runtime": "~3 hrs on RTX 4070"
    },

    # =========================================================================
    # PAPER J: CECI Chimeric Splicing
    # =========================================================================
    "J1": {
        "paper": "Paper X (J)",
        "claim": "Within-model within-band splicing is viable (GD<0.92, ρ>0.15)",
        "status": "DONE (systematic sweep)",
        "experiment": "ALREADY MEASURED: 120 pairs, 0 viable at k=32. FALSIFIED within-model.",
        "metric": "N/A",
        "pass": "N/A (negative result)",
        "requires": [],
        "runtime": "0"
    },
    "J2": {
        "paper": "Paper X (J)",
        "claim": "Cross-model splicing with shared-init dedicated models is viable",
        "status": "RUNNING",
        "experiment": "Train pure math + language models → CECI splice at k=128",
        "metric": "GD, overlap, ρ_CECI per layer",
        "pass": "≥30% layers viable (GD<0.90, ρ>0.30)",
        "requires": ["Model M (training)", "Model L (queued)", "ceci_cross_model.py"],
        "runtime": "~2.5 hrs total"
    },

    # =========================================================================
    # PAPER A-D: Core GRC Theoretical Framework
    # =========================================================================
    "A1": {
        "paper": "Paper I (A)",
        "claim": "GRC projection preserves ≥95% of attention signal at k≥256 (135M model)",
        "status": "PENDING (end-to-end)",
        "experiment": "Measure end-to-end PPL at each k: full, 1536, 1024, 512, 256, 128",
        "metric": "PPL vs k",
        "pass": "≤5% PPL increase at k≥256",
        "requires": ["HF evaluate", "WikiText-2", "GRC projection code"],
        "runtime": "~2 hrs on RTX 4070"
    },
}

def print_plan():
    total = len(EXPERIMENTS)
    done = sum(1 for e in EXPERIMENTS.values() if e['status'] == 'DONE')
    running = sum(1 for e in EXPERIMENTS.values() if e['status'] == 'RUNNING')
    pending = sum(1 for e in EXPERIMENTS.values() if e['status'] == 'PENDING')
    
    print("=" * 80)
    print(f"HYPER TENSOR MASTER EXPERIMENT PLAN")
    print(f"  Total experiments: {total}")
    print(f"  Done: {done} | Running: {running} | Pending: {pending}")
    print("=" * 80)
    
    for eid, exp in sorted(EXPERIMENTS.items()):
        status_icon = {"PENDING": "", "RUNNING": "", "DONE": ""}[exp['status']]
        print(f"\n{status_icon} {eid}: {exp['paper']}")
        print(f"   Claim: {exp['claim'][:100]}...")
        print(f"   Experiment: {exp['experiment'][:120]}...")
        print(f"   Pass criterion: {exp['pass']}")
        print(f"   Est. runtime: {exp['runtime']}")
    
    print("\n" + "=" * 80)
    print("EXECUTABLE NOW (local RTX 4070):")
    executable = [eid for eid, e in EXPERIMENTS.items() 
                  if e['status'] == 'PENDING' and 'EC2' not in e['requires'] 
                  and 'H100' not in str(e['requires']) and 'A100' not in str(e['requires'])
                  and 'RTX 4090' not in str(e['requires'])]
    for eid in executable:
        print(f"  - {eid}: {EXPERIMENTS[eid]['experiment'][:80]}...")
    
    print(f"\nNEEDS CLOUD GPU:")
    cloud = [eid for eid, e in EXPERIMENTS.items() 
             if any(g in str(e['requires']) for g in ['EC2', 'A100', 'H100', 'RTX 4090', 'Lambda'])]
    for eid in cloud:
        print(f"  - {eid}: needs {[r for r in EXPERIMENTS[eid]['requires'] if any(g in r for g in ['EC2','A100','H100','4090','Lambda'])][:2]}")
    
    return executable


if __name__ == '__main__':
    executable = print_plan()
    print(f"\n{len(executable)} experiments executable NOW.")
