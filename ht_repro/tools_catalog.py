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

"""
ht-repro tools catalog — all HyperTensor utility scripts organized by category.
Each entry maps a tool ID to its script, description, and usage example.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

TOOLS = {
    # ── Grafting ──────────────────────────────────────────────────
    "graft": {
        "ceci-sweep": {
            "script": "scripts/ceci_systematic_sweep.py",
            "desc": "Systematic CECI k-vs-layer sweep: find compatible model pairs",
            "usage": "ht-repro tools graft ceci-sweep",
            "tier": "T2",
        },
        "ceci-cross": {
            "script": "scripts/ceci_cross_model.py",
            "desc": "Cross-model CECI: Grassmann distance between math & language models",
            "usage": "ht-repro tools graft ceci-cross",
            "tier": "T2",
        },
        "chimeric-splice": {
            "script": "scripts/chimeric_splice.py",
            "desc": "Horimiya chimeric model splicing pipeline (math+language fusion)",
            "usage": "ht-repro tools graft chimeric-splice",
            "tier": "T2",
        },
        "hyper-graft": {
            "script": "scripts/hyper_graft.py",
            "desc": "Full HYPER GRAFT pipeline: CECI protocol + Danish-named model grafting",
            "usage": "ht-repro tools graft hyper-graft",
            "tier": "T2",
        },
        "hyper-graft-cpu": {
            "script": "scripts/hyper_graft_cpu.py",
            "desc": "CPU-only HYPER GRAFT for resource-constrained environments",
            "usage": "ht-repro tools graft hyper-graft-cpu",
            "tier": "T1",
        },
        "publish-grafts": {
            "script": "scripts/publish_grafts.py",
            "desc": "Publish grafted models to HuggingFace and Ollama with verification",
            "usage": "ht-repro tools graft publish-grafts",
            "tier": "T2",
        },
        "contrastive-fusion": {
            "script": "scripts/contrastive_fusion.py",
            "desc": "Contrastive fusion of model pairs via UGT-aligned weight blending",
            "usage": "ht-repro tools graft contrastive-fusion",
            "tier": "T2",
        },
        "zone-fusion": {
            "script": "scripts/zone_weighted_fusion.py",
            "desc": "Zone-weighted model fusion using UGT domain coordinates",
            "usage": "ht-repro tools graft zone-fusion",
            "tier": "T2",
        },
        "quick-splice": {
            "script": "scripts/splice_quick.py",
            "desc": "Quick single-layer model splice for rapid prototyping",
            "usage": "ht-repro tools graft quick-splice",
            "tier": "T2",
        },
    },

    # ── Benchmarking ──────────────────────────────────────────────
    "bench": {
        "quick": {
            "script": "scripts/benchmarks_quick.py",
            "desc": "Quick 7-test benchmark suite (bulletproof audit)",
            "usage": "ht-repro tools bench quick",
            "tier": "T1",
        },
        "super-baseline": {
            "script": "scripts/benchmark_super_baseline.py",
            "desc": "Super-baseline benchmark across all paper claims",
            "usage": "ht-repro tools bench super-baseline",
            "tier": "T2",
        },
        "jury-decay": {
            "script": "scripts/benchmark_jury_decay.py",
            "desc": "Jury accuracy vs. Euclidean decay benchmark",
            "usage": "ht-repro tools bench jury-decay",
            "tier": "T1",
        },
        "multi-ppl": {
            "script": "scripts/multi_dataset_ppl.py",
            "desc": "Multi-dataset perplexity (WikiText2, C4, PTB)",
            "usage": "ht-repro tools bench multi-ppl",
            "tier": "T2",
        },
        "task-bench": {
            "script": "scripts/task_bench.py",
            "desc": "Task benchmark harness (GSM8K, MMLU, HumanEval)",
            "usage": "ht-repro tools bench task-bench",
            "tier": "T2",
        },
        "task-bench-cpu": {
            "script": "scripts/task_bench_python.py",
            "desc": "Pure-Python task benchmark (no compiled binary needed)",
            "usage": "ht-repro tools bench task-bench-cpu",
            "tier": "T1",
        },
        "creativity": {
            "script": "scripts/creativity_benchmark.py",
            "desc": "MIKU Creativity Benchmark (5-dimension creativity scoring)",
            "usage": "ht-repro tools bench creativity",
            "tier": "T2",
        },
        "native-ppl": {
            "script": "scripts/native_ppl_parity.py",
            "desc": "NativeLinear PPL parity benchmark vs full-rank baseline",
            "usage": "ht-repro tools bench native-ppl",
            "tier": "T2",
        },
        "spec-decode": {
            "script": "scripts/bench_llama8b_specdecode.py",
            "desc": "Speculative decode throughput benchmark for Llama-8B",
            "usage": "ht-repro tools bench spec-decode",
            "tier": "T2",
        },
        "mcr-ablation": {
            "script": "scripts/bench_mcr_ablation.py",
            "desc": "MCR (model compression ratio) ablation benchmark",
            "usage": "ht-repro tools bench mcr-ablation",
            "tier": "T1",
        },
    },

    # ── Training ──────────────────────────────────────────────────
    "train": {
        "distill": {
            "script": "scripts/distill_runner.py",
            "desc": "GRC light-distillation runner (teacher -> student with LoRA r=8)",
            "usage": "ht-repro tools train distill",
            "tier": "T2",
        },
        "native": {
            "script": "scripts/native_train.py",
            "desc": "NativeLinear training: B*C*B^T decomposition on Stiefel manifold",
            "usage": "ht-repro tools train native",
            "tier": "T2",
        },
        "native-7b": {
            "script": "scripts/native_7b_final.py",
            "desc": "Final native training run for 7B models",
            "usage": "ht-repro tools train native-7b",
            "tier": "T3",
        },
        "shf-build": {
            "script": "scripts/shf_full_build.py",
            "desc": "Full SHF (Spectral Hamiltonian Flow) model build with curvature loss",
            "usage": "ht-repro tools train shf-build",
            "tier": "T2",
        },
        "shf-loss": {
            "script": "scripts/shf_loss.py",
            "desc": "SHF curvature-regularized loss for any training loop",
            "usage": "ht-repro tools train shf-loss",
            "tier": "T1",
        },
        "pure-lora": {
            "script": "scripts/train_pure_lora.py",
            "desc": "Train pure LoRA adapters for single-skill specialization",
            "usage": "ht-repro tools train pure-lora",
            "tier": "T2",
        },
        "pure-model": {
            "script": "scripts/train_pure_model.py",
            "desc": "Train pure full-model single-skill specialization",
            "usage": "ht-repro tools train pure-model",
            "tier": "T3",
        },
        "dedicated": {
            "script": "scripts/train_dedicated_models.py",
            "desc": "Train dedicated single-skill models (math, code, language)",
            "usage": "ht-repro tools train dedicated",
            "tier": "T3",
        },
    },

    # ── Compression ───────────────────────────────────────────────
    "compress": {
        "ffn-cluster": {
            "script": "scripts/ffn_cluster_compress.py",
            "desc": "FFN weight clustering compression (k-means on gate/up/down)",
            "usage": "ht-repro tools compress ffn-cluster",
            "tier": "T2",
        },
        "ffn-activation": {
            "script": "scripts/ffn_real_activations.py",
            "desc": "FFN compression with real activation-weighted importance",
            "usage": "ht-repro tools compress ffn-activation",
            "tier": "T2",
        },
        "grc-distill": {
            "script": "scripts/grc_distill.py",
            "desc": "GRC distillation: shared basis + LoRA fit for PPL recovery",
            "usage": "ht-repro tools compress grc-distill",
            "tier": "T2",
        },
        "sink-aware": {
            "script": "scripts/sink_aware_grc.py",
            "desc": "Sink-aware GRC: preserve top-T high-magnitude attention columns",
            "usage": "ht-repro tools compress sink-aware",
            "tier": "T2",
        },
        "attnres-sweep": {
            "script": "scripts/attnres_sweep.py",
            "desc": "AttnRes x GRC interaction sweep (find optimal k from L2 cache)",
            "usage": "ht-repro tools compress attnres-sweep",
            "tier": "T2",
        },
        "spectra": {
            "script": "scripts/analysis/compute_spectra.py",
            "desc": "Compute SVD spectra of attention vs FFN weights",
            "usage": "ht-repro tools compress spectra",
            "tier": "T1",
        },
    },

    # ── GTC / Manifold ────────────────────────────────────────────
    "gtc": {
        "build": {
            "script": "scripts/build_gtc_library.py",
            "desc": "Build GTC trajectory library from model activations",
            "usage": "ht-repro tools gtc build",
            "tier": "T2",
        },
        "benchmark": {
            "script": "scripts/gtc_vs_rag.py",
            "desc": "Head-to-head GTC vs RAG (coverage, latency, scaling)",
            "usage": "ht-repro tools gtc benchmark",
            "tier": "T1",
        },
        "jury-accelerated": {
            "script": "scripts/jury_gtc.py",
            "desc": "Jury-accelerated GTC trajectory cache routing",
            "usage": "ht-repro tools gtc jury-accelerated",
            "tier": "T1",
        },
        "geodesic-compile": {
            "script": "scripts/geodesic_compile.py",
            "desc": "Compile geodesic trajectory data into compact cache records",
            "usage": "ht-repro tools gtc geodesic-compile",
            "tier": "T1",
        },
        "geodesic-compiler": {
            "script": "scripts/geodesic_compiler.py",
            "desc": "Native k-space model compiler via geodesic projection",
            "usage": "ht-repro tools gtc geodesic-compiler",
            "tier": "T2",
        },
        "manifold-drift": {
            "script": "scripts/eval_manifold_drift.py",
            "desc": "Evaluate manifold drift over training/adaptation epochs",
            "usage": "ht-repro tools gtc manifold-drift",
            "tier": "T1",
        },
        "ship-of-theseus": {
            "script": "scripts/eval_ship_of_theseus.py",
            "desc": "Ship of Theseus: knowledge persistence under weight replacement",
            "usage": "ht-repro tools gtc ship-of-theseus",
            "tier": "T2",
        },
        "batch-jacobi": {
            "script": "scripts/gtc/batch_jacobi.py",
            "desc": "Batched Jacobi propagation for GTC lookup acceleration",
            "usage": "ht-repro tools gtc batch-jacobi",
            "tier": "T1",
        },
        "validity-radius": {
            "script": "scripts/gtc/validity_radius.py",
            "desc": "Measure GTC validity radius for trajectory cache",
            "usage": "ht-repro tools gtc validity-radius",
            "tier": "T1",
        },
    },

    # ── Safety ────────────────────────────────────────────────────
    "safety": {
        "safe-ogd": {
            "script": "scripts/safe_ogd.py",
            "desc": "Safe OGD: constrained creative exploration with forbidden-subspace projection",
            "usage": "ht-repro tools safety safe-ogd",
            "tier": "T1",
        },
        "safe-ogd-cog": {
            "script": "scripts/safe_ogd_cog.py",
            "desc": "Safe OGD integrated with COG living manifold for creative safety",
            "usage": "ht-repro tools safety safe-ogd-cog",
            "tier": "T2",
        },
        "teh-roc": {
            "script": "scripts/teh_roc.py",
            "desc": "TEH ROC curve: detection rate vs false positive sweep",
            "usage": "ht-repro tools safety teh-roc",
            "tier": "T1",
        },
        "teh-multicat": {
            "script": "scripts/teh_multicat.py",
            "desc": "Multi-category TEH: per-category forbidden coordinate detection (8 categories)",
            "usage": "ht-repro tools safety teh-multicat",
            "tier": "T2",
        },
        "teh-scaling": {
            "script": "scripts/teh_scaling.py",
            "desc": "TEH scaling analysis: harm detection vs model scale",
            "usage": "ht-repro tools safety teh-scaling",
            "tier": "T2",
        },
        "snipe": {
            "script": "scripts/snipe_specificity.py",
            "desc": "SNIPE specificity: incremental coordinate ablation per harm category",
            "usage": "ht-repro tools safety snipe",
            "tier": "T1",
        },
        "multi-snipe": {
            "script": "scripts/multi_snipe.py",
            "desc": "Multi-category behavioral sniping across 8 harm categories",
            "usage": "ht-repro tools safety multi-snipe",
            "tier": "T2",
        },
        "red-team": {
            "script": "scripts/red_team.py",
            "desc": "Red-team attack library (GCG, AutoPrompt, PAIR) with TEH measurement",
            "usage": "ht-repro tools safety red-team",
            "tier": "T2",
        },
        "hallucination-guard": {
            "script": "scripts/test_hallucination_guard.py",
            "desc": "Test hallucination guard using TEH-based detection",
            "usage": "ht-repro tools safety hallucination-guard",
            "tier": "T1",
        },
    },

    # ── UGT ───────────────────────────────────────────────────────
    "ugt": {
        "bilateral": {
            "script": "scripts/bilateral_ugt.py",
            "desc": "Bilateral UGT: hot-swap taxonomy between independently trained models",
            "usage": "ht-repro tools ugt bilateral",
            "tier": "T2",
        },
        "bilateral-7b": {
            "script": "scripts/bilateral_7b.py",
            "desc": "Bilateral UGT hot-swap on 7B-scale models",
            "usage": "ht-repro tools ugt bilateral-7b",
            "tier": "T3",
        },
        "probe": {
            "script": "scripts/probe_ugt_zones.py",
            "desc": "Probe UGT zone integrity under perturbation",
            "usage": "ht-repro tools ugt probe",
            "tier": "T1",
        },
        "ablation": {
            "script": "scripts/ugt_ablation.py",
            "desc": "UGT coordinate ablation: measure impact of removing taxonomy axes",
            "usage": "ht-repro tools ugt ablation",
            "tier": "T1",
        },
        "domain-map": {
            "script": "scripts/ugt_domain_mapper.py",
            "desc": "UGT domain mapper: allocate k-space dimensions to knowledge zones",
            "usage": "ht-repro tools ugt domain-map",
            "tier": "T1",
        },
        "random-basis": {
            "script": "scripts/ugt_random_basis_ablation.py",
            "desc": "Ablation study: random basis vs UGT basis comparison",
            "usage": "ht-repro tools ugt random-basis",
            "tier": "T1",
        },
        "zone-recovery": {
            "script": "scripts/ugt_zone_recovery.py",
            "desc": "UGT zone recovery: verify taxonomy zones survive noise injection",
            "usage": "ht-repro tools ugt zone-recovery",
            "tier": "T1",
        },
    },

    # ── Models / Infrastructure ───────────────────────────────────
    "models": {
        "download-hf": {
            "script": "scripts/download_model.py",
            "desc": "Download model from HuggingFace with 4-bit quantization",
            "usage": "ht-repro tools models download-hf --model meta-llama/Llama-3.1-8B-Instruct",
            "tier": "any",
        },
        "batch-download": {
            "script": "scripts/download_models.py",
            "desc": "Batch download multiple models for local use",
            "usage": "ht-repro tools models batch-download",
            "tier": "any",
        },
        "download-datasets": {
            "script": "scripts/download_datasets.py",
            "desc": "Download benchmark datasets (WikiText2, C4, PTB)",
            "usage": "ht-repro tools models download-datasets",
            "tier": "any",
        },
        "export-ollama": {
            "script": "scripts/export_to_ollama.py",
            "desc": "Merge LoRA adapter + export full model to Ollama/GGUF format",
            "usage": "ht-repro tools models export-ollama --model-path ./my-model",
            "tier": "any",
        },
        "token-setup": {
            "script": None,  # Built-in command
            "desc": "Configure HuggingFace token for model downloads (interactive setup)",
            "usage": "ht-repro tools models token-setup",
            "tier": "any",
        },
        "token-status": {
            "script": None,
            "desc": "Check HuggingFace token status and validity",
            "usage": "ht-repro tools models token-status",
            "tier": "any",
        },
        "gpu-check": {
            "script": "scripts/gpu_check.py",
            "desc": "GPU capability check (VRAM, CUDA, compute capability)",
            "usage": "ht-repro tools models gpu-check",
            "tier": "any",
        },
        "inventory": {
            "script": "scripts/inventory_models.py",
            "desc": "Inventory all local models with size, format, and status",
            "usage": "ht-repro tools models inventory",
            "tier": "any",
        },
        "ollama-clone": {
            "script": None,
            "desc": "Clone a model from Ollama to local HF format",
            "usage": "ht-repro tools models ollama-clone --name llama3",
            "tier": "any",
        },
    },

    # ── ISAGI / Interactive ───────────────────────────────────────
    "isagi": {
        "chat": {
            "script": "scripts/isagi_chat.py",
            "desc": "ISAGI interactive chat: adaptive living model with GTC cache + jury",
            "usage": "ht-repro tools isagi chat",
            "tier": "T2",
        },
        "benchmark": {
            "script": "scripts/isagi_benchmark.py",
            "desc": "ISAGI reasoning benchmark (multi-turn problem solving)",
            "usage": "ht-repro tools isagi benchmark",
            "tier": "T2",
        },
    },
}
