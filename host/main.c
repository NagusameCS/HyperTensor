/*
 * Geodessical — Hosted Main Entry Point
 *
 * Loads a GGUF model from disk via memory-mapped I/O and runs LLM inference
 * using the TensorOS inference engine on the host CPU, with native threading.
 *
 * Usage: geodessical <model.gguf> [prompt]
 */
#define _CRT_SECURE_NO_WARNINGS
#include "hal.h"

/* Forward declarations from TensorOS inference engine */
#include "../runtime/nn/llm.h"
#include "../runtime/nn/hf_download.h"
#include "../runtime/nn/axiom_beta.h"
#include "../runtime/nn/axiom_exploit.h"
#include "../runtime/nn/axiom_linalg.h"
#include "../runtime/nn/axiom_vis.h"
#include "api_server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#ifdef _WIN32
#include <windows.h>
#endif

#define GD_VERSION_MAJOR 0
#define GD_VERSION_MINOR 6
#define GD_VERSION_PATCH 0
#define GD_CODENAME      "Synapse"

static void print_banner(void) {
    kprintf("\n");
    kprintf("  ╔═══════════════════════════════════════════╗\n");
    kprintf("  ║  Geodessical v%d.%d.%d \"%s\"              ║\n",
            GD_VERSION_MAJOR, GD_VERSION_MINOR, GD_VERSION_PATCH, GD_CODENAME);
    kprintf("  ║  High-Performance AI Inference Runtime    ║\n");
    kprintf("  ╚═══════════════════════════════════════════╝\n");
    kprintf("\n");
}

static void print_usage(const char *argv0) {
    kprintf("Usage: %s <model.gguf> [options]\n\n", argv0);
    kprintf("Options:\n");
    kprintf("  -p, --prompt <text>    Prompt text (default: interactive)\n");
    kprintf("  -n, --tokens <num>     Max tokens to generate (default: 128)\n");
    kprintf("  -t, --threads <num>    Thread count (default: all CPUs)\n");
    kprintf("  --temp <float>         Temperature (default: 0.7)\n");
    kprintf("  --top-k <int>          Top-K sampling (default: 40)\n");
    kprintf("  --top-p <float>        Nucleus sampling (default: 0.9)\n");
    kprintf("  --ctx-size <n>         KV-cache context window (default: 8192 tokens)\n");
    kprintf("  -i, --interactive      Interactive chat mode\n");
    kprintf("  --serve                Start Geodessical HTTP API server\n");
    kprintf("  --port <num>           API server port (default: 8080)\n");
    kprintf("  --download <repo>      Download model from HuggingFace\n");
    kprintf("  --quant <type>         Quantization hint for download (default: q4_0)\n");
    kprintf("  -v, --verbose          Enable debug logging\n");
    kprintf("  --log-level <n>        Log level: 0=error 1=warn 2=info 3=debug 4=trace\n");
    kprintf("  --no-think             Disable thinking (strip <|think|> blocks)\n");
    kprintf("  --force-think          Force thinking on all prompts\n");
    kprintf("  --show-think           Show thinking tokens in output\n");
    kprintf("  --attnres              Enable Attention-Residual-inspired depth stabilization\n");
    kprintf("  --attnres-strength <f> AttnRes stabilization strength [0..1] (default: 0.35)\n");
    kprintf("  --ott-fast             Speed-first OTT: fast axiom + geodesic-first, minimal depth work\n");
    kprintf("  --ott-speculative      OTT Speculative Decode: geodesic drafts verified by transformer\n");
    kprintf("  --ott-perfect          Perfect-day OTT: exact transformer rollout upper bound (100%% draft acceptance)\n");
    kprintf("  --ott-spec-thresh N    Confidence threshold for draft acceptance (default: 0.65)\n");
    kprintf("  --ott-spec-batch N     Draft batch size for speculative decode (default: 4)\n");
    kprintf("  --no-verifier          Emit geodesic drafts directly — skip transformer verify (faster, lower quality)\n");
    kprintf("  --ott-full             Enable OTT full mode (geodesic-first + axiomatic run + AttnRes)\n");
    kprintf("  --ott-theorem          Enable theorem-mode OTT (adds depth residual attention + strict geodesic quality control)\n");
    kprintf("  --depth-attn           Enable depth-wise residual attention mixer\n");
    kprintf("  --depth-attn-strength <f> Depth residual attention strength [0..1] (default: 0.55)\n");
    kprintf("  --depth-attn-window <n> Depth attention lookback window (default: 16)\n");
    kprintf("  --ott-ready-report <p> Write OTT readiness JSON (default: ott_readiness_report.json in --ott-full)\n");
    kprintf("  --axiom-beta-run       Run autonomous axiomatic beta-3 survey\n");
    kprintf("  --axiom-beta-only      Run axiomatic beta survey then exit\n");
    kprintf("  --axiom-report <path>  Write beta report JSON (default: axiom_beta_report.json)\n");
    kprintf("  --axiom-samples <n>    Embedding samples for manifold identification (default: 256)\n");
    kprintf("  --axiom-probe <n>      Phase 5 endpoint token probe count (default: 1024)\n");
    kprintf("  --axiom-gpu            Enable GPU acceleration for Phase 5 scorer\n");
    kprintf("  --axiom-no-gpu         Force CPU-only Phase 5 scorer\n");
    kprintf("  --axiom-random-targets Use random Phase 5 targets (disable oracle targets)\n");
    kprintf("  --axiom-geodesic-first Experimental geodesic-first decode path\n");
    kprintf("  --axiom-fast           Fast mode (reduced phase1-4 workload)\n");
    kprintf("  --axiom-no-cache       Disable in-process geometry cache reuse\n");
    kprintf("  --axiom-seed <n>       Beta deterministic seed\n");
    kprintf("  --axiom-skip-geodesic  Skip geodesic pilot (Phase 5)\n");
    kprintf("  --axex-kv              Geodesic KV-cache compression (50-80%% KV reduction)\n");
    kprintf("  --axex-kv-threshold <f> KV merge distance threshold (default: 0.15)\n");
    kprintf("  --axex-offload         Curvature-guided GPU layer offload (smart --gpu-layers)\n");
    kprintf("  --axex-compress        Manifold-aware weight compression (geodesic SVD + GP)\n");
    kprintf("  --axex-ffn-compress    SVD FFN compression only (gate/up/down, no manifold PCA)\n");
    kprintf("  --axex-compress-rank N Max SVD rank for FFN compression (default 128, use 16-32 for speed)\n");
    kprintf("                         GP (Geodesic Projection): Q/K/V/gate/up → W@P[m×k]\n");
    kprintf("                         Enables 22B-70B models in 8 GB VRAM (one-time cost)\n");
    kprintf("  --axex-compress-max-err <f> Max Frobenius error to accept (0=no limit; use 0.5 to skip\n");
    kprintf("                         matrices with >50%% error, preventing garbled output)\n");
    kprintf("  --axex-quality <f>     Weight compression quality floor 0-1 (default: 0.90)\n");
    kprintf("  --one-decode           OneDecode: bake geodesic flow map once, then decode instantly\n");
    kprintf("  --one-decode-coverage <n> Vocab tokens to bake for OneDecode (default: 2048)\n");
    kprintf("  --ott-od               OTT-OD Protocol: OneDecode as speculative draft source (fastest OTT mode)\n");
    kprintf("  --vis [dir]            Visualize Riemannian manifolds (default: axiom_vis/)\n");
    kprintf("  -h, --help             Show this help\n");
    kprintf("\nExamples:\n");
    kprintf("  %s phi3.5.gguf -p \"What is an OS?\"\n", argv0);
    kprintf("  %s llama3.gguf -i\n", argv0);
}

static void enable_max_performance_host(void) {
#ifdef _WIN32
    /* Prefer compute latency over background fairness for local inference. */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

    /* Disable Windows execution-speed throttling when the API is available. */
#if defined(PROCESS_POWER_THROTTLING_EXECUTION_SPEED)
    {
        PROCESS_POWER_THROTTLING_STATE pts;
        pts.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
        pts.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
        pts.StateMask = 0;
        SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                              &pts, sizeof(pts));
    }
#endif
#endif
}

typedef struct {
    const char *model_path;
    const char *prompt;
    const char *download_repo;
    const char *quant_hint;
    int         max_tokens;
    int         n_threads;
    float       temperature;
    int         top_k;
    float       top_p;
    int         interactive;
    int         serve;
    int         port;
    int         verbose;
    int         log_level;
    int         no_think;
    int         show_think;
    int         attnres_enable;
    float       attnres_strength;
    int         ott_full;
    int         ott_theorem;
    int         depth_attn_enable;
    float       depth_attn_strength;
    int         depth_attn_window;
    const char *ott_ready_report_path;
    int         axiom_beta_run;
    int         axiom_beta_only;
    int         axiom_samples;
    int         axiom_vocab_probe;
    int         axiom_use_gpu_phase5;
    int         axiom_use_oracle_target;
    int         axiom_geodesic_first;
    int         axiom_fast_mode;
    int         axiom_reuse_cache;
    uint64_t    axiom_seed;
    int         axiom_skip_geodesic;
    const char *axiom_report_path;
    int         ott_speculative;        /* 1 = speculative decode (geodesic draft + transformer verify) */
    int         ott_perfect;            /* 1 = exact greedy rollout upper bound (100% accepted drafts) */
    float       ott_speculative_thresh; /* geodesic confidence threshold [0,1], default 0.65 */
    int         ott_speculative_batch;  /* draft batch size (default 4) */
    int         no_verifier;            /* 1 = emit geodesic drafts directly, skip transformer verification */
    const char *vis_output_dir;
    int         ctx_size;   /* 0 = default (2048); user override via --ctx-size */
    int         one_decode;          /* 1 = OneDecode mode: bake once, decode instantly */
    int         one_decode_coverage; /* Vocab tokens to bake (0 = auto) */
    int         ott_od;              /* 1 = OTT-OD: OneDecode as spec-decode draft source */
    int         od_swarm_k;          /* OD-SWARM fan-out K (0=off); each draft slot gets K candidates */
    /* Manifold exploitation (axiom_exploit) */
    int         axex_kv;             /* 1 = geodesic KV compression */
    float       axex_kv_threshold;   /* geodesic merge threshold (default 0.15) */
    int         axex_offload;        /* 1 = curvature-guided layer offload */
    int         axex_compress;       /* 1 = manifold-aware weight compression */
    int         axex_ffn_compress;   /* 1 = SVD FFN compression only (no manifold PCA) */
    int         axex_compress_rank;  /* max SVD rank (0 = auto, default 128) */
    float       axex_compress_quality;  /* 0-1 quality floor (default 0.90) */
    float       axex_compress_max_err;  /* max Frobenius error to accept (0=no limit) */
    double      axiom_pca_variance;     /* 0 = default 0.95; override for GP weight basis */
    int         axex_attn_only;      /* 1 = compress only Q/K/V/O (default); 0 = all incl FFN */
    int         axex_calib_samples;  /* calibration samples per layer (0 = auto) */
} GD_args_t;

static int GD_ott_theorem_active = 0;

static const char *find_default_gguf_model(void) {
#ifdef _WIN32
    static char resolved_path[MAX_PATH];
    const char *patterns[] = {
        "models\\*.gguf",
        "..\\TensorOS\\models\\*.gguf",
        "..\\..\\TensorOS\\models\\*.gguf",
        NULL
    };

    for (int i = 0; patterns[i]; i++) {
        WIN32_FIND_DATAA fd;
        HANDLE h = FindFirstFileA(patterns[i], &fd);
        if (h != INVALID_HANDLE_VALUE) {
            FindClose(h);

            const char *pat = patterns[i];
            const char *star = strstr(pat, "*.gguf");
            int prefix_len = star ? (int)(star - pat) : (int)strlen(pat);
            if (prefix_len < 0) prefix_len = 0;
            if (prefix_len > (int)sizeof(resolved_path) - 1)
                prefix_len = (int)sizeof(resolved_path) - 1;

            memcpy(resolved_path, pat, (size_t)prefix_len);
            resolved_path[prefix_len] = '\0';
            strncat(resolved_path, fd.cFileName,
                    sizeof(resolved_path) - strlen(resolved_path) - 1);
            return resolved_path;
        }
    }
#endif
    return NULL;
}

static int parse_args(int argc, char **argv, GD_args_t *args) {
    args->model_path  = NULL;
    args->prompt      = NULL;
    args->download_repo = NULL;
    args->quant_hint  = "q4_0";
    args->max_tokens  = 512;
    args->n_threads   = 0;  /* 0 = auto */
    args->temperature = 0.7f;
    args->top_k       = 40;
    args->top_p       = 0.9f;
    args->interactive = 0;
    args->serve       = 0;
    args->port        = 8080;
    args->verbose     = 0;
    args->log_level   = -1;  /* unset = use default */
    args->no_think    = 0;
    args->show_think  = 0;
    args->attnres_enable = 0;
    args->attnres_strength = 0.35f;
    args->ott_full = 0;
    args->ott_theorem = 0;
    args->depth_attn_enable = 0;
    args->depth_attn_strength = 0.55f;
    args->depth_attn_window = 16;
    args->ott_ready_report_path = NULL;
    args->axiom_beta_run = 0;
    args->axiom_beta_only = 0;
    args->axiom_samples = 256;
    args->axiom_vocab_probe = 1024;
    args->axiom_use_gpu_phase5 = -1;
    args->axiom_use_oracle_target = 1;
    args->axiom_geodesic_first = 0;
    args->axiom_fast_mode = 0;
    args->axiom_reuse_cache = 1;
    args->axiom_seed = 0;
    args->axiom_skip_geodesic = 0;
    args->axiom_report_path = "axiom_beta_report.json";
    args->ott_speculative = 0;
    args->ott_perfect = 0;
    args->ott_speculative_thresh = 0.1f;  /* loose pre-filter; topk verifier (margin=2.8) is authoritative */
    args->ott_speculative_batch = 2;
    args->no_verifier = 0;
    args->vis_output_dir = NULL;
    args->ctx_size = 0;
    args->one_decode = 0;
    args->one_decode_coverage = 0;
    args->ott_od = 0;
    args->od_swarm_k = 0;
    args->axex_kv = 0;
    args->axex_kv_threshold = 0.15f;
    args->axex_offload = 0;
    args->axex_compress = 0;
    args->axex_ffn_compress = 0;
    args->axex_compress_rank = 0;
    args->axex_compress_quality = 0.90f;
    args->axex_compress_max_err = 0.0f;  /* 0 = no limit */
    args->axex_attn_only = 1;      /* default: attention-only (near-lossless accuracy) */
    args->axex_calib_samples = 0;  /* 0 = auto (512 default, 64 in fast mode) */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            return -1;
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (++i >= argc) { kprintf("Error: --prompt requires argument\n"); return -1; }
            args->prompt = argv[i];
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--tokens") == 0) {
            if (++i >= argc) { kprintf("Error: --tokens requires argument\n"); return -1; }
            args->max_tokens = atoi(argv[i]);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (++i >= argc) { kprintf("Error: --threads requires argument\n"); return -1; }
            args->n_threads = atoi(argv[i]);
        } else if (strcmp(argv[i], "--temp") == 0) {
            if (++i >= argc) { kprintf("Error: --temp requires argument\n"); return -1; }
            args->temperature = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--top-k") == 0) {
            if (++i >= argc) { kprintf("Error: --top-k requires argument\n"); return -1; }
            args->top_k = atoi(argv[i]);
        } else if (strcmp(argv[i], "--top-p") == 0) {
            if (++i >= argc) { kprintf("Error: --top-p requires argument\n"); return -1; }
            args->top_p = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            args->interactive = 1;
        } else if (strcmp(argv[i], "--serve") == 0) {
            args->serve = 1;
        } else if (strcmp(argv[i], "--port") == 0) {
            if (++i >= argc) { kprintf("Error: --port requires argument\n"); return -1; }
            args->port = atoi(argv[i]);
        } else if (strcmp(argv[i], "--download") == 0) {
            if (++i >= argc) { kprintf("Error: --download requires repo ID\n"); return -1; }
            args->download_repo = argv[i];
        } else if (strcmp(argv[i], "--quant") == 0) {
            if (++i >= argc) { kprintf("Error: --quant requires type\n"); return -1; }
            args->quant_hint = argv[i];
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            args->verbose = 1;
        } else if (strcmp(argv[i], "--log-level") == 0) {
            if (++i >= argc) { kprintf("Error: --log-level requires number\n"); return -1; }
            args->log_level = atoi(argv[i]);
        } else if (strcmp(argv[i], "--no-think") == 0) {
            args->no_think = 1;
        } else if (strcmp(argv[i], "--force-think") == 0) {
            args->no_think = -1;  /* sentinel for force-think */
        } else if (strcmp(argv[i], "--show-think") == 0) {
            args->show_think = 1;
        } else if (strcmp(argv[i], "--attnres") == 0) {
            args->attnres_enable = 1;
        } else if (strcmp(argv[i], "--attnres-strength") == 0) {
            if (++i >= argc) { kprintf("Error: --attnres-strength requires number\n"); return -1; }
            args->attnres_strength = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--ott-fast") == 0) {
            /* Speed-first OTT: use speculative decode with large batch for
             * maximum TPS.  Geodesic drafts B=16 tokens locally (near-zero
             * cost), verifies in a single LLM pass — yielding hundreds of
             * tok/s when acceptance rate is reasonable. */
            args->ott_full = 1;
            args->axiom_beta_run = 1;
            args->axiom_fast_mode = 1;
            args->axiom_geodesic_first = 1;
            args->ott_speculative = 1;
            args->ott_speculative_batch = 16;
            args->ott_speculative_thresh = 0.45f;
            args->attnres_enable = 1;
            args->attnres_strength = 0.25f;
            args->top_k = 20;
            args->top_p = 0.9f;
            args->ott_ready_report_path = "ott_readiness_report.json";
        } else if (strcmp(argv[i], "--ott-speculative") == 0) {
            /* OTT Speculative Decode: geodesic drafts + transformer verification.
             * Collects a batch of geodesic draft tokens, verifies them in a single
             * transformer pass, accepts all verified tokens and takes the correction
             * token at the first rejection.  Gives real transformer-quality output
             * with speedup proportional to geodesic draft acceptance rate. */
            args->ott_full = 1;
            args->axiom_beta_run = 1;
            args->axiom_fast_mode = 1;
            args->axiom_geodesic_first = 1;
            args->ott_speculative = 1;
            args->ott_speculative_batch = 2;  /* batch=2: primed T0 + geodesic T1 is optimal */
            args->attnres_enable = 1;
            args->attnres_strength = 0.25f;
            args->top_k = 20;
            args->top_p = 0.9f;
            args->ott_ready_report_path = "ott_readiness_report.json";
        } else if (strcmp(argv[i], "--ott-perfect") == 0) {
            /* Perfect-day OTT: exact greedy rollout through the transformer's own
             * logits. This is the upper-bound mode for acceptance and throughput:
             * every drafted token is exact, so verifier work disappears. */
            args->ott_full = 1;
            args->axiom_beta_run = 1;
            args->axiom_fast_mode = 1;
            args->axiom_geodesic_first = 1;
            args->ott_speculative = 1;
            args->ott_perfect = 1;
            args->ott_speculative_batch = 16;
            args->temperature = 0.0f;
            args->top_k = 1;
            args->top_p = 1.0f;
            args->attnres_enable = 1;
            args->attnres_strength = 0.25f;
            args->axiom_use_gpu_phase5 = 1;
            args->ott_ready_report_path = "ott_readiness_report.json";
        } else if (strcmp(argv[i], "--ott-spec-thresh") == 0) {
            if (++i >= argc) { kprintf("Error: --ott-spec-thresh requires number\n"); return -1; }
            args->ott_speculative_thresh = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--ott-spec-batch") == 0) {
            if (++i >= argc) { kprintf("Error: --ott-spec-batch requires number\n"); return -1; }
            args->ott_speculative_batch = atoi(argv[i]);
        } else if (strcmp(argv[i], "--ott-full") == 0) {
            args->ott_full = 1;
            args->axiom_beta_run = 1;
            args->axiom_geodesic_first = 1;
            args->one_decode = 1;           /* OTT-full: always prep OD table for hotpath */
            args->attnres_enable = 1;
            args->attnres_strength = 0.35f;
            args->top_k = 20;
            args->top_p = 0.9f;
            args->ott_ready_report_path = "ott_readiness_report.json";
        } else if (strcmp(argv[i], "--ott-theorem") == 0) {
            args->ott_theorem = 1;
            args->ott_full = 1;
            args->axiom_beta_run = 1;
            args->axiom_fast_mode = 1;  /* Use fast axiom for theorem speed */
            args->axiom_geodesic_first = 1;
            args->attnres_enable = 1;
            args->attnres_strength = 0.45f;
            args->depth_attn_enable = 1;
            args->depth_attn_strength = 0.55f;
            args->depth_attn_window = 16;
            args->top_k = 30;  /* Wider top-k gives runtime proposer more viable candidates */
            args->top_p = 0.92f;
            args->ott_ready_report_path = "ott_readiness_report.json";
        } else if (strcmp(argv[i], "--depth-attn") == 0) {
            args->depth_attn_enable = 1;
        } else if (strcmp(argv[i], "--depth-attn-strength") == 0) {
            if (++i >= argc) { kprintf("Error: --depth-attn-strength requires number\n"); return -1; }
            args->depth_attn_strength = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--depth-attn-window") == 0) {
            if (++i >= argc) { kprintf("Error: --depth-attn-window requires number\n"); return -1; }
            args->depth_attn_window = atoi(argv[i]);
        } else if (strcmp(argv[i], "--ott-ready-report") == 0) {
            if (++i >= argc) { kprintf("Error: --ott-ready-report requires path\n"); return -1; }
            args->ott_ready_report_path = argv[i];
        } else if (strcmp(argv[i], "--axiom-beta-run") == 0) {
            args->axiom_beta_run = 1;
        } else if (strcmp(argv[i], "--axiom-beta-only") == 0) {
            args->axiom_beta_only = 1;
            args->axiom_beta_run = 1;
        } else if (strcmp(argv[i], "--axiom-report") == 0) {
            if (++i >= argc) { kprintf("Error: --axiom-report requires path\n"); return -1; }
            args->axiom_report_path = argv[i];
        } else if (strcmp(argv[i], "--axiom-samples") == 0) {
            if (++i >= argc) { kprintf("Error: --axiom-samples requires number\n"); return -1; }
            args->axiom_samples = atoi(argv[i]);
        } else if (strcmp(argv[i], "--axiom-probe") == 0) {
            if (++i >= argc) { kprintf("Error: --axiom-probe requires number\n"); return -1; }
            args->axiom_vocab_probe = atoi(argv[i]);
        } else if (strcmp(argv[i], "--axiom-gpu") == 0) {
            args->axiom_use_gpu_phase5 = 1;
        } else if (strcmp(argv[i], "--axiom-no-gpu") == 0) {
            args->axiom_use_gpu_phase5 = 0;
        } else if (strcmp(argv[i], "--axiom-random-targets") == 0) {
            args->axiom_use_oracle_target = 0;
        } else if (strcmp(argv[i], "--axiom-geodesic-first") == 0) {
            args->axiom_geodesic_first = 1;
        } else if (strcmp(argv[i], "--axiom-fast") == 0) {
            args->axiom_fast_mode = 1;
        } else if (strcmp(argv[i], "--axiom-no-cache") == 0) {
            args->axiom_reuse_cache = 0;
        } else if (strcmp(argv[i], "--axiom-seed") == 0) {
            if (++i >= argc) { kprintf("Error: --axiom-seed requires number\n"); return -1; }
#ifdef _WIN32
            args->axiom_seed = _strtoui64(argv[i], NULL, 10);
#else
            args->axiom_seed = strtoull(argv[i], NULL, 10);
#endif
        } else if (strcmp(argv[i], "--axiom-skip-geodesic") == 0) {
            args->axiom_skip_geodesic = 1;
        } else if (strcmp(argv[i], "--axex-kv") == 0) {
            args->axex_kv = 1;
            args->axiom_beta_run = 1;   /* need phase1 PCA */
        } else if (strcmp(argv[i], "--axex-kv-threshold") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-kv-threshold requires float\n"); return -1; }
            args->axex_kv_threshold = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-offload") == 0) {
            args->axex_offload = 1;
        } else if (strcmp(argv[i], "--axex-compress") == 0) {
            args->axex_compress = 1;
            args->axiom_beta_run = 1;   /* need curvature data */
            /* NOTE: fast_mode NOT set — need 512 samples for enough PCA components (k~500) */
            args->axiom_skip_geodesic = 1; /* Phase 5 not needed for GP compression */
            args->axiom_pca_variance = 0.9999; /* keep all significant PCA components for weight basis */
        } else if (strcmp(argv[i], "--axex-attn-only") == 0) {
            /* Explicit: compress only Q/K/V/O attention weights (default behaviour) */
            args->axex_attn_only = 1;
        } else if (strcmp(argv[i], "--axex-ffn-compress") == 0) {
            /* SVD compress gate/up/down only — no axiom survey, no manifold PCA.
             * Fast path: enables the cuBLAS batched GEMV for gate+up without
             * waiting 30 min for per-layer PCA.  Uses flat curvature (SVD). */
            args->axex_ffn_compress = 1;
        } else if (strcmp(argv[i], "--axex-compress-rank") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-compress-rank requires int\n"); return -1; }
            args->axex_compress_rank = atoi(argv[i]);
        } else if (strcmp(argv[i], "--axex-compress-max-err") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-compress-max-err requires float\n"); return -1; }
            args->axex_compress_max_err = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-full-compress") == 0) {
            /* Override: also compress FFN gate/up (legacy — hurts accuracy ~20-35% perplexity) */
            args->axex_attn_only = 0;
        } else if (strcmp(argv[i], "--axex-calib-samples") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-calib-samples requires integer\n"); return -1; }
            args->axex_calib_samples = atoi(argv[i]);
        } else if (strcmp(argv[i], "--axex-quality") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-quality requires float\n"); return -1; }
            args->axex_compress_quality = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--one-decode") == 0) {
            args->one_decode = 1;
            args->axiom_beta_run = 1;       /* need geometry */
            args->axiom_fast_mode = 1;      /* fast geometry build */
            args->axiom_skip_geodesic = 1;  /* skip Phase 5 — not needed for bake */
        } else if (strcmp(argv[i], "--ott-swarm") == 0) {
            if (i + 1 < argc) {
                args->od_swarm_k = atoi(argv[++i]);
                if (args->od_swarm_k < 1)  args->od_swarm_k = 0;
                if (args->od_swarm_k > 64) args->od_swarm_k = 64;
                args->one_decode      = 1;
                args->ott_od          = 1;
                args->axiom_beta_run  = 1;
                args->axiom_fast_mode = 1;
                args->ott_full        = 1;
                args->ott_speculative = 1;
            }
        } else if (strcmp(argv[i], "--ott-od") == 0) {
            /* OTT-OD Protocol: OneDecode table as the speculative draft source.
             * Replaces the geodesic rollout (16 forward passes) with an O(N)
             * table scan — same Riemannian draft quality, dramatically faster.
             * Transformer still verifies every draft (same output quality). */
            args->ott_od = 1;
            args->one_decode = 1;           /* ensure OD table is baked */
            args->axiom_beta_run = 1;       /* need Phase-3 geometry */
            args->axiom_fast_mode = 1;      /* fast geometry build */
            args->ott_full = 1;
            args->axiom_geodesic_first = 1;
            args->ott_speculative = 1;      /* route through spec-decode path */
            args->attnres_enable = 1;
            args->attnres_strength = 0.25f;
            args->top_k = 20;
            args->top_p = 0.9f;
            args->ott_ready_report_path = "ott_readiness_report.json";
        } else if (strcmp(argv[i], "--one-decode-coverage") == 0) {
            if (++i >= argc) { kprintf("Error: --one-decode-coverage requires number\n"); return -1; }
            args->one_decode_coverage = atoi(argv[i]);
        } else if (strcmp(argv[i], "--vis") == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-')
                args->vis_output_dir = argv[++i];
            else
                args->vis_output_dir = "axiom_vis";
            args->axiom_beta_run = 1;
        } else if (strcmp(argv[i], "--no-verifier") == 0) {
            args->no_verifier = 1;
        } else if (strcmp(argv[i], "--ctx-size") == 0) {
            if (++i >= argc) { kprintf("Error: --ctx-size requires number\n"); return -1; }
            args->ctx_size = atoi(argv[i]);
        } else if (argv[i][0] != '-' && !args->model_path) {
            args->model_path = argv[i];
        } else {
            kprintf("Unknown option: %s\n", argv[i]);
            return -1;
        }
    }

    if (!args->model_path && !args->download_repo) {
        if (args->axiom_beta_run || args->axiom_beta_only) {
            const char *auto_model = find_default_gguf_model();
            if (auto_model) {
                args->model_path = auto_model;
                kprintf("[GD] Auto-selected model for Axiom run: %s\n", args->model_path);
                return 0;
            }
        }
        kprintf("Error: no model file specified\n\n");
        return -1;
    }

    return 0;
}


/* ── Streaming token callback ──────────────────────────────────────────── */
static void GD_stream_cb(const char *text, int len, void *ud) {
    (void)ud;
    fwrite(text, 1, (size_t)len, stdout);
    fflush(stdout);
}

/* ANSI escape helpers */
#define GD_RESET   "\033[0m"
#define GD_BOLD    "\033[1m"
#define GD_DIM     "\033[2m"
#define GD_GREEN   "\033[32m"
#define GD_CYAN    "\033[36m"
#define GD_LBLUE   "\033[94m"
#define GD_YELLOW  "\033[33m"
#define GD_RED     "\033[31m"

static void GD_ansi_enable(void) {
#ifdef _WIN32
    HANDLE h    = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD  mode = 0;
    if (GetConsoleMode(h, &mode))
        SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}

static void print_chat_help(void) {
    printf(GD_CYAN "  Commands:\n" GD_RESET);
    printf(GD_CYAN "    /help       " GD_RESET "show this message\n");
    printf(GD_CYAN "    /reset      " GD_RESET "clear conversation (new context)\n");
    printf(GD_CYAN "    /stats      " GD_RESET "show context usage\n");
    printf(GD_CYAN "    /temp <n>   " GD_RESET "set sampling temperature   (e.g. /temp 0.8)\n");
    printf(GD_CYAN "    /tokens <n> " GD_RESET "set max tokens per reply   (e.g. /tokens 512)\n");
    printf(GD_CYAN "    /quit       " GD_RESET "exit\n\n");
}

static int geodesic_chat_turn(const char *user_text,
                              int *ctx,
                              int *n_ctx,
                              int max_ctx,
                              int max_tokens,
                              float temperature,
                              char *output,
                              int max_output);

static void interactive_loop(const char *model_path, GD_args_t *args) {
    (void)model_path;
    char line[2048];
    static char output[131072]; /* 128 KB — holds full reply if needed */
    float temperature = args->temperature;
    int   max_tokens  = args->max_tokens;
    int   geodesic_mode = args->axiom_geodesic_first || args->one_decode;
    static int geo_ctx_tokens[32768];
    int   geo_n_ctx = 0;

    /* ── Turn history for context sliding window ─────────────────────────── */
#define CHAT_HIST_MAX  32     /* circular buffer of turns */
#define CHAT_HIST_TEXT 4096   /* per-turn text (user or AI) */
    typedef struct { char user[CHAT_HIST_TEXT]; char ai[CHAT_HIST_TEXT]; } chat_hist_t;
    static chat_hist_t hist[CHAT_HIST_MAX];
    int hist_n = 0;            /* total turns saved (unbounded; use % CHAT_HIST_MAX for index) */
    /* ────────────────────────────────────────────────────────────────────── */

    GD_ansi_enable();

    /* Welcome header */
    printf("\n"
           GD_CYAN GD_BOLD
           "  ╔══════════════════════════════════════════════════╗\n"
           "  ║  Geodessical Chat                                ║\n"
           GD_RESET);
    printf(GD_CYAN GD_BOLD "  ║  Model : %-41s║\n" GD_RESET, llm_model_name());
    printf(GD_CYAN GD_BOLD "  ║  Context: %-6d tokens  |  All CPUs active       ║\n"
           GD_RESET, llm_chat_context_max());
    printf(GD_CYAN GD_BOLD
           "  ╚══════════════════════════════════════════════════╝\n"
           GD_RESET "\n");
    printf(GD_DIM "  Type /help for commands.\n\n" GD_RESET);
    if (geodesic_mode)
        printf(GD_DIM "  [Geodesic-first chat mode enabled]\n\n" GD_RESET);

    llm_set_stream_cb(GD_stream_cb, (void *)0);

    for (;;) {
        /* User input prompt */
        printf(GD_GREEN GD_BOLD "You: " GD_RESET);
        fflush(stdout);

        if (!fgets(line, (int)sizeof(line), stdin)) break;

        /* Strip trailing newline/CR */
        int len = (int)strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';

        if (len == 0) continue;

        /* ── Commands ──────────────────────────────────────── */
        if (line[0] == '/') {
            if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0 ||
                strcmp(line, "/q")    == 0) break;

            if (strcmp(line, "/help") == 0 || strcmp(line, "/?") == 0) {
                print_chat_help();
                continue;
            }
            if (strcmp(line, "/reset") == 0) {
                llm_chat_reset();
                geo_n_ctx = 0;
                hist_n = 0;
                printf(GD_DIM "  [Conversation reset — new context started]\n\n" GD_RESET);
                continue;
            }
            if (strcmp(line, "/stats") == 0) {
                int ctx    = geodesic_mode ? geo_n_ctx : llm_chat_context_tokens();
                int ctxmax = geodesic_mode ? (int)(sizeof(geo_ctx_tokens) / sizeof(geo_ctx_tokens[0]))
                                           : llm_chat_context_max();
                int think  = llm_thinking_tokens();
                printf(GD_DIM "  [Context: %d / %d tokens (%.1f%%)  |  "
                       "temp=%.2f  max_tok=%d  think_last=%d]\n\n" GD_RESET,
                       ctx, ctxmax,
                       ctxmax > 0 ? 100.0f * ctx / ctxmax : 0.0f,
                       temperature, max_tokens, think);
                continue;
            }
            if (strncmp(line, "/temp ", 6) == 0) {
                temperature = (float)atof(line + 6);
                if (temperature < 0.0f) temperature = 0.0f;
                if (temperature > 2.0f) temperature = 2.0f;
                printf(GD_DIM "  [Temperature set to %.2f]\n\n" GD_RESET, temperature);
                continue;
            }
            if (strncmp(line, "/tokens ", 8) == 0) {
                max_tokens = atoi(line + 8);
                if (max_tokens < 1)    max_tokens = 1;
                if (max_tokens > 8192) max_tokens = 8192;
                printf(GD_DIM "  [Max tokens set to %d]\n\n" GD_RESET, max_tokens);
                continue;
            }
            printf(GD_YELLOW "  [Unknown command: %s — try /help]\n\n" GD_RESET, line);
            continue;
        }

        /* ── Generate response ─────────────────────────────── */
        printf("\n" GD_LBLUE GD_BOLD "AI: " GD_RESET);
        fflush(stdout);

        uint64_t t0 = hal_timer_us();
        output[0]   = '\0';
        int n;
        if (geodesic_mode) {
            (void)temperature;
            n = geodesic_chat_turn(line,
                                   geo_ctx_tokens, &geo_n_ctx,
                                   (int)(sizeof(geo_ctx_tokens) / sizeof(geo_ctx_tokens[0])),
                                   max_tokens,
                                   temperature,
                                   output, (int)sizeof(output));
            if (n > 0 && output[0] != '\0') {
                printf("%s", output);
            }
        } else {
            /* ── Context sliding window (non-geodesic path) ─────────────── */
            if (!geodesic_mode) {
                int ctx_cur = llm_chat_context_tokens();
                int ctx_max = llm_chat_context_max();
                /* When > 80% full, slide window: reset + replay last 2 turns */
                if (ctx_max > 0 && ctx_cur > ctx_max * 4 / 5 && hist_n > 0) {
                    int keep = (hist_n >= 2) ? 2 : hist_n;
                    printf(GD_YELLOW "  [Context at %d/%d (%.0f%%) — replaying last %d turn%s to extend conversation]\n\n"
                           GD_RESET,
                           ctx_cur, ctx_max, 100.0f * ctx_cur / ctx_max,
                           keep, keep > 1 ? "s" : "");
                    llm_chat_reset();
                    /* Replay silently (suppress streaming) */
                    llm_set_stream_cb(NULL, NULL);
                    static char replay_buf[131072];
                    for (int r = hist_n - keep; r < hist_n; r++) {
                        int idx = r % CHAT_HIST_MAX;
                        llm_chat_turn(hist[idx].user, replay_buf,
                                      (int)sizeof(replay_buf), max_tokens, temperature);
                    }
                    llm_set_stream_cb(GD_stream_cb, (void *)0);
                }
            }
            /* Save this user turn in history ring */
            if (!geodesic_mode) {
                int idx = hist_n % CHAT_HIST_MAX;
                strncpy(hist[idx].user, line, CHAT_HIST_TEXT - 1);
                hist[idx].user[CHAT_HIST_TEXT - 1] = '\0';
                hist[idx].ai[0] = '\0';
            }

            n = llm_chat_turn(line, output, (int)sizeof(output), max_tokens, temperature);

            /* Save AI reply in history */
            if (!geodesic_mode && n > 0) {
                int idx = hist_n % CHAT_HIST_MAX;
                strncpy(hist[idx].ai, output, CHAT_HIST_TEXT - 1);
                hist[idx].ai[CHAT_HIST_TEXT - 1] = '\0';
                hist_n++;
            }
        }
        uint64_t t1 = hal_timer_us();

        printf("\n");

        if (n > 0) {
            uint64_t total_ms   = (t1 - t0) / 1000;
            float    tok_per_s  = total_ms > 0 ? (float)n * 1000.0f / (float)total_ms : 0.0f;
                 int ctx    = geodesic_mode ? geo_n_ctx : llm_chat_context_tokens();
                 int ctxmax = geodesic_mode ? (int)(sizeof(geo_ctx_tokens) / sizeof(geo_ctx_tokens[0]))
                                : llm_chat_context_max();
            float ctx_pct = ctxmax > 0 ? 100.0f * ctx / ctxmax : 0.0f;
            const char *ctx_warn = (ctx_pct > 85.0f) ? GD_YELLOW " (!)" GD_RESET GD_DIM : "";
            printf(GD_DIM "  [%d tok  %.1f tok/s  %llu ms  ctx %d/%d (%.0f%%)%s]\n\n" GD_RESET,
                   n, tok_per_s, (unsigned long long)total_ms, ctx, ctxmax, ctx_pct, ctx_warn);
        } else {
            printf(GD_RED "  [error generating response (code %d)]\n\n" GD_RESET, n);
        }
    }

    llm_set_stream_cb((llm_token_cb_t)0, (void *)0);
    printf("\n" GD_DIM "  [Session ended]\n" GD_RESET "\n");
}

/* Forward declaration — defined later in this file. */
static int geodesic_piece_quality_ok(const char *piece, int piece_n);

static void geodesic_ensure_one_decode(int coverage)
{
    static int tried = 0;
    if (tried) return;
    tried = 1;
    if (axiom_beta_one_decode_ready()) return;
    /* Try loading a saved table first */
    if (axiom_beta_one_decode_load("ott_one_decode.bin") == AXIOM_BETA_OK) return;
    /* No cache — bake from current Phase-3 geometry (one-time cost) */
    kprintf("[OD] Baking OneDecode table (one-time cost)...\n");
    if (axiom_beta_one_decode_bake(coverage) == AXIOM_BETA_OK)
        (void)axiom_beta_one_decode_save("ott_one_decode.bin");
}

static int geodesic_next_token_local(const int *ctx, int n_ctx, int *out_tok) {
    /* OneDecode fast path: O(dim×k + coverage×k) with zero Christoffel work.
     * All geodesic math was amortised into the one-time bake. */
    if (axiom_beta_one_decode_ready()) {
        int od_tok = -1;
        float od_conf = 0.0f;
        /* Get EOS id to suppress premature stops from the bake table */
        const llm_model_t *od_m = llm_get_model();
        int od_eos = od_m ? od_m->eos_id : 1;
        if (axiom_beta_one_decode_next(ctx, n_ctx, &od_tok, &od_conf) == AXIOM_BETA_OK
                && od_tok >= 0 && od_tok != od_eos && od_conf >= 0.5f) {
            char od_piece[256];
            int od_pn = llm_test_decode_token(od_tok, od_piece, (int)sizeof(od_piece));
            if (od_pn > 0 && geodesic_piece_quality_ok(od_piece, od_pn)) {
                *out_tok = od_tok;
                return 0;
            }
        }
    }

    /* Fast path: use real Christoffel-based geodesic step when geometry is cached.
     * Requires high confidence (>= 0.65) and passes the piece quality gate to
     * avoid returning garbage tokens. */
    {
        int fast_tok = -1;
        float fast_conf = 0.0f;
        if (axiom_beta_geodesic_step_fast(ctx, n_ctx, &fast_tok, &fast_conf)
                == AXIOM_BETA_OK && fast_tok >= 0 && fast_conf >= 0.65f) {
            /* Quality gate: reject control-char or entirely-non-useful pieces */
            char fast_piece[256];
            int fast_pn = llm_test_decode_token(fast_tok, fast_piece,
                                                (int)sizeof(fast_piece));
            if (fast_pn > 0 && geodesic_piece_quality_ok(fast_piece, fast_pn)) {
                *out_tok = fast_tok;
                return 0;
            }
        }
    }

    const llm_model_t *m = llm_get_model();
    if (!m || !ctx || n_ctx <= 0 || !out_tok) return -1;

    int vocab = m->vocab_size;
    int dim = m->dim;
    int tok_curr = ctx[n_ctx - 1];
    int tok_prev = (n_ctx >= 2) ? ctx[n_ctx - 2] : tok_curr;
    int tok_prev2 = (n_ctx >= 3) ? ctx[n_ctx - 3] : tok_prev;
    if (tok_curr < 0 || tok_curr >= vocab) tok_curr = 0;
    if (tok_prev < 0 || tok_prev >= vocab) tok_prev = tok_curr;
    if (tok_prev2 < 0 || tok_prev2 >= vocab) tok_prev2 = tok_prev;

    float *e_curr = (float *)malloc((size_t)dim * sizeof(float));
    float *e_prev = (float *)malloc((size_t)dim * sizeof(float));
    float *e_prev2 = (float *)malloc((size_t)dim * sizeof(float));
    float *e_pred = (float *)malloc((size_t)dim * sizeof(float));
    float *e_cand = (float *)malloc((size_t)dim * sizeof(float));
    if (!e_curr || !e_prev || !e_prev2 || !e_pred || !e_cand) {
        free(e_curr);
        free(e_prev);
        free(e_prev2);
        free(e_pred);
        free(e_cand);
        return -1;
    }

    if (llm_get_embedding_vec(tok_curr, e_curr, dim) != 0 ||
        llm_get_embedding_vec(tok_prev, e_prev, dim) != 0 ||
        llm_get_embedding_vec(tok_prev2, e_prev2, dim) != 0) {
        free(e_curr);
        free(e_prev);
        free(e_prev2);
        free(e_pred);
        free(e_cand);
        return -1;
    }

    /* Second-order OTT surrogate in embedding space with momentum + curvature bias. */
    double pred_norm2 = 0.0;
    float alpha = 0.35f;
    float beta = 0.18f;
    if (n_ctx < 3) beta = 0.0f;
    for (int i = 0; i < dim; i++) {
        float v1 = (e_curr[i] - e_prev[i]);
        float v0 = (e_prev[i] - e_prev2[i]);
        float a = v1 - v0;
        e_pred[i] = e_curr[i] + alpha * v1 + beta * a;
        pred_norm2 += (double)e_pred[i] * (double)e_pred[i];
    }
    double pred_norm = sqrt(pred_norm2);

    int probe = vocab;
    if (probe > 4096) probe = 4096;
    int start = (tok_curr * 1315423911u) % vocab;
    int best_tok = tok_curr;
    double best_score = -1e30;

    for (int i = 0; i < probe; i++) {
        int tid = (start + i) % vocab;
        if (llm_get_embedding_vec(tid, e_cand, dim) != 0) continue;

        double dot = 0.0;
        double cand_norm2 = 0.0;
        double l2 = 0.0;
        for (int j = 0; j < dim; j++) {
            double c = (double)e_cand[j];
            dot += (double)e_pred[j] * c;
            cand_norm2 += c * c;
            double d = (double)e_pred[j] - c;
            l2 += d * d;
        }
        double denom = pred_norm * sqrt(cand_norm2);
        double sim = (denom > 1e-12) ? (dot / denom) : -1e30;
        double score = sim - 0.015 * sqrt(l2);
        if (score > best_score) {
            best_score = score;
            best_tok = tid;
        }
    }

    free(e_curr);
    free(e_prev);
    free(e_prev2);
    free(e_pred);
    free(e_cand);
    *out_tok = best_tok;
    return 0;
}

static int geodesic_next_token_runtime(const int *ctx, int n_ctx, int *out_tok) {
    int tok = -1;
    if (axiom_beta_geodesic_next_token_v2(ctx, n_ctx, &tok) != AXIOM_BETA_OK)
        return -1;
    *out_tok = tok;
    return 0;
}

static int geodesic_runtime_token_acceptable(const int *ctx, int n_ctx, int tok) {
    char piece[256];
    int piece_n;
    int control = 0;
    int high = 0;
    int useful = 0;

    if (!ctx || n_ctx <= 0 || tok < 0)
        return 0;

    /* Reject hard loops: same token 2-in-a-row, or dominant token in last 5. */
    if (n_ctx >= 2 && tok == ctx[n_ctx - 1]) {
        /* Allow once, block only strict 3-in-a-row */
        if (n_ctx >= 3 && tok == ctx[n_ctx - 2])
            return 0;
    }
    if (n_ctx >= 5) {
        int rep = 0;
        for (int ri = n_ctx - 5; ri < n_ctx; ri++)
            if (ctx[ri] == tok) rep++;
        if (rep >= 4)
            return 0;
    }

    piece_n = llm_test_decode_token(tok, piece, (int)sizeof(piece));
    if (piece_n <= 0)
        return 0;
    piece[piece_n] = '\0';

    if (strstr(piece, "<unused") != NULL)
        return 0;
    if (strstr(piece, "<0x") != NULL)
        return 0;

    for (int i = 0; i < piece_n; i++) {
        unsigned char c = (unsigned char)piece[i];
        if (c < 32 && c != '\n' && c != '\r' && c != '\t')
            control++;
        if (c >= 128)
            high++;
        if (isalnum((int)c) || c == ' ' || c == '.' || c == ',' ||
            c == ';' || c == ':' || c == '!' || c == '?' || c == '\'' || c == '"') {
            useful++;
        }
    }

    /* Allow short high-byte tokens (e.g. UTF-8 multibyte pieces ≤3 bytes) */
    if (piece_n >= 4 && useful == 0 && high > 0)
        return 0;
    /* Loosen ratio threshold: require 25% useful, not 33% */
    if (piece_n >= 6 && useful <= 1 && high * 4 >= piece_n * 3)
        return 0;

    return (control == 0);
}

static int geodesic_output_quality_ok(const char *text) {
    int len = 0;
    int useful = 0;
    int high = 0;
    int control = 0;
    int unused_markers = 0;

    if (!text || text[0] == '\0')
        return 0;

    for (int i = 0; text[i] != '\0'; i++) {
        unsigned char c = (unsigned char)text[i];
        len++;
        if (c < 32 && c != '\n' && c != '\r' && c != '\t')
            control++;
        if (c >= 128)
            high++;
        if (isalnum((int)c) || c == ' ' || c == '.' || c == ',' ||
            c == ';' || c == ':' || c == '!' || c == '?' || c == '\'' || c == '"' ||
            c == '(' || c == ')' || c == '-' || c == '\n') {
            useful++;
        }
        if (c == '<' && strncmp(text + i, "<unused", 7) == 0)
            unused_markers++;
    }

    if (len <= 0 || control > 0)
        return 0;
    if (unused_markers >= 2)
        return 0;
    if (len >= 24 && useful * 5 < len && high * 2 > len)
        return 0;
    if (unused_markers > 0 && useful * 2 < len)
        return 0;

    return 1;
}

static int geodesic_piece_quality_ok(const char *piece, int piece_n) {
    int useful = 0;
    int high = 0;
    int control = 0;
    if (!piece || piece_n <= 0) return 0;
    /* SentencePiece space prefix ▁ (U+2581 = E2 96 81): treat as space, don't
     * count its 3 high bytes against the high-byte ratio check. */
    int start = 0;
    if (piece_n >= 3
            && (unsigned char)piece[0] == 0xe2
            && (unsigned char)piece[1] == 0x96
            && (unsigned char)piece[2] == 0x81) {
        useful++;   /* counts as a space */
        start = 3;
    }
    for (int i = start; i < piece_n; i++) {
        unsigned char c = (unsigned char)piece[i];
        if (c < 32 && c != '\n' && c != '\r' && c != '\t') control++;
        if (c >= 128) high++;
        if (isalnum((int)c) || c == ' ' || c == '.' || c == ',' || c == ';' ||
            c == ':' || c == '!' || c == '?' || c == '\'' || c == '"' ||
            c == '(' || c == ')' || c == '-' || c == '_' || c == '*' ||
            c == '#' || c == '/' || c == '+' || c == '=' || c == '@' ||
            c == '&' || c == '<' || c == '>' || c == '[' || c == ']' ||
            c == '{' || c == '}' || c == '\\' || c == '`' || c == '~' ||
            c == '^' || c == '|' || c == '%' || c == '$') useful++;
    }
    int body_n = piece_n - start;  /* length after stripping ▁ prefix */
    if (control > 0) return 0;
    /* Allow short high-byte pieces (UTF-8 sub-word tokens like diacritics) */
    if (body_n >= 4 && useful == (start > 0 ? 1 : 0) && high * 2 >= body_n) return 0;
    if (body_n >= 6 && useful <= (start > 0 ? 2 : 1) && high * 4 >= body_n * 3) return 0;
    return 1;
}

/* Probe runtime every 3rd confined step rather than every 4th for faster recovery. */
static int geodesic_theorem_should_probe_runtime(int local_only_budget, int step) {
    if (local_only_budget <= 0) return 1;
    return ((step % 3) == 2) ? 1 : 0;
}

/* Smaller initial cooldown so the runtime path re-enters sooner after a bad piece. */
static int geodesic_theorem_local_budget(int bad_piece_streak) {
    int budget = 3 + bad_piece_streak;
    if (budget < 3) budget = 3;
    if (budget > 10) budget = 10;
    return budget;
}

static int geodesic_prime_cache(const GD_args_t *args) {
    axiom_beta_config_t cfg;
    axiom_beta_report_t rep;
    axiom_beta_default_config(&cfg);

    cfg.fast_mode = 1;
    cfg.reuse_cache = 1;
    cfg.skip_geodesic = 1;
    /* Priming only needs phase1-3 geometry for runtime geodesic decode.
     * Keep phase4 minimal and oracle-free to reduce startup latency. */
    cfg.active_iterations = 8;
    cfg.oracle_calls_max = 0;
    cfg.verbose = args->verbose;
    if (args->axiom_seed != 0) cfg.seed = args->axiom_seed;
    if (args->axiom_samples > 0) cfg.embedding_samples = args->axiom_samples;
    if (args->axiom_vocab_probe > 0) cfg.geodesic_vocab_probe = args->axiom_vocab_probe;
    return (axiom_beta_run(&cfg, &rep) == AXIOM_BETA_OK) ? 0 : -1;
}

int GD_geodesic_last_runtime_hits = 0;
int GD_geodesic_last_local_hits = 0;
int GD_geodesic_last_runtime_rejects = 0;
int GD_geodesic_last_quality_gate_pass = 0;
int GD_geodesic_last_generated = 0;
int GD_geodesic_last_fallback_requested = 0;

static int geodesic_generate_text(const char *prompt,
                                  int max_tokens,
                                  char *output,
                                  int max_output)
{
    static int s_runtime_hits = 0;
    static int s_local_hits = 0;
    static int s_runtime_rejects = 0;
    static int s_quality_gate_pass = 0;
    static int s_generated = 0;
    static int s_fallback_requested = 0;

    const llm_model_t *m = llm_get_model();
    if (!m || !prompt || !output || max_output <= 0 || max_tokens <= 0)
        return -1;

    int max_ctx = max_tokens + 2048;
    int *ctx = (int *)malloc((size_t)max_ctx * sizeof(int));
    if (!ctx) return -1;

    int n_ctx = llm_test_tokenize(prompt, (int)strlen(prompt), ctx, max_ctx);
    if (n_ctx <= 0 || n_ctx >= max_ctx) {
        free(ctx);
        return -1;
    }

    output[0] = '\0';
    int out_len = 0;
    int generated = 0;
    int runtime_hits = 0;
    int local_hits = 0;
    int runtime_rejects = 0;
    int local_only_budget = 0;
    int bad_piece_streak = 0;
    int eos = m->eos_id;

    /* Todo 27: Warm context-conditioned hidden state snapshot before loop.
     * This prefills the full prompt context once, snapshots the KV state,
     * and eagerly captures hidden states for the last 8 prompt tokens so
     * the first geodesic steps hit the LRU without forward passes. */
    ott_set_generation_context(ctx, n_ctx);

    /* Per-request timing accumulators */
    uint64_t t_runtime_us = 0;
    uint64_t t_local_us   = 0;
    uint64_t t_req_start  = hal_timer_us();

    for (int step = 0; step < max_tokens; step++) {
        int tok = -1;
        int used_runtime = 0;

        if (GD_ott_theorem_active && local_only_budget > 0 &&
            !geodesic_theorem_should_probe_runtime(local_only_budget, step)) {
            uint64_t _t0 = hal_timer_us();
            if (geodesic_next_token_local(ctx, n_ctx, &tok) == 0) {
                t_local_us += hal_timer_us() - _t0;
                local_hits++;
                local_only_budget--;
            } else {
                break;
            }
        } else {
            uint64_t _t0 = hal_timer_us();
            int runtime_ok = (geodesic_next_token_runtime(ctx, n_ctx, &tok) == 0);
            t_runtime_us += hal_timer_us() - _t0;
            if (runtime_ok && geodesic_runtime_token_acceptable(ctx, n_ctx, tok)) {
                runtime_hits++;
                used_runtime = 1;
            } else if (geodesic_next_token_local(ctx, n_ctx, &tok) == 0) {
                uint64_t _t1 = hal_timer_us();
                (void)_t1; /* local after rejection — already charged to runtime */
                local_hits++;
                if (runtime_ok)
                    runtime_rejects++;
            } else {
                break;
            }
        }

        if (tok < 0 || tok >= m->vocab_size)
            break;

        if (GD_ott_theorem_active) {
            char q_piece[256];
            int q_n = llm_test_decode_token(tok, q_piece, (int)sizeof(q_piece));
            if (!geodesic_piece_quality_ok(q_piece, q_n)) {
                if (used_runtime) {
                    int tok_local = -1;
                    if (geodesic_next_token_local(ctx, n_ctx, &tok_local) == 0 &&
                        tok_local >= 0 && tok_local < m->vocab_size) {
                        tok = tok_local;
                        local_hits++;
                        runtime_rejects++;
                        used_runtime = 0;
                        q_n = llm_test_decode_token(tok, q_piece, (int)sizeof(q_piece));
                    }
                }
                if (!geodesic_piece_quality_ok(q_piece, q_n)) {
                    bad_piece_streak++;
                    local_only_budget = geodesic_theorem_local_budget(bad_piece_streak);
                } else {
                    if (used_runtime && bad_piece_streak > 0)
                        bad_piece_streak--;
                    if (used_runtime && local_only_budget > 0)
                        local_only_budget--;
                }
            } else {
                if (used_runtime && bad_piece_streak > 0)
                    bad_piece_streak--;
            }
        }

        if (n_ctx < max_ctx)
            ctx[n_ctx++] = tok;
        else
            break;

        if (eos >= 0 && tok == eos)
            break;

        char piece[256];
        int piece_n = llm_test_decode_token(tok, piece, (int)sizeof(piece));
        if (piece_n > 0) {
            int room = max_output - out_len - 1;
            if (room <= 0) break;
            if (piece_n > room) piece_n = room;
            memcpy(output + out_len, piece, (size_t)piece_n);
            out_len += piece_n;
            output[out_len] = '\0';
        }

        generated++;
    }

    if (generated > 0) {
        uint64_t t_total_us = hal_timer_us() - t_req_start;
        double tps = (t_total_us > 0) ? (1e6 * generated / (double)t_total_us) : 0.0;
        double pct_rt = (t_total_us > 0) ? (100.0 * t_runtime_us / t_total_us) : 0.0;
        double pct_lc = (t_total_us > 0) ? (100.0 * t_local_us  / t_total_us) : 0.0;
        kprintf("[GD] Geodesic decode: %d tokens in %.1f ms, TPS=%.2f\n",
            generated, t_total_us / 1000.0, tps);
        kprintf("[GD]   runtime_next_token: %.1f ms (%.0f%%), local_next_token: %.1f ms (%.0f%%), other: %.1f ms\n",
            t_runtime_us / 1000.0, pct_rt, t_local_us / 1000.0, pct_lc,
            (t_total_us - t_runtime_us - t_local_us) / 1000.0);
        kprintf("[GD]   avg ms/tok: runtime=%.1f, local=%.1f\n",
            runtime_hits > 0 ? t_runtime_us / 1000.0 / runtime_hits : 0.0,
            local_hits   > 0 ? t_local_us   / 1000.0 / local_hits   : 0.0);
        kprintf("[GD] Geodesic decode tokens: runtime=%d local=%d (runtime_rejects=%d)\n",
            runtime_hits, local_hits, runtime_rejects);
        if (!geodesic_output_quality_ok(output)) {
            /* Warn but keep the output — discarding and re-generating doubles latency
             * and the fallback transformer often produces a similar result anyway.
             * The per-token quality gate already filtered the worst tokens. */
            kprintf("[GD] Geodesic output quality note: some low-quality tokens may be present.\n");
        }
    }

    s_runtime_hits = runtime_hits;
    s_local_hits = local_hits;
    s_runtime_rejects = runtime_rejects;
    s_quality_gate_pass = generated > 0 ? 1 : 0;
    s_generated = generated;
    s_fallback_requested = 0;

    /* Export stats through static symbols read in main. */
    {
        extern int GD_geodesic_last_runtime_hits;
        extern int GD_geodesic_last_local_hits;
        extern int GD_geodesic_last_runtime_rejects;
        extern int GD_geodesic_last_quality_gate_pass;
        extern int GD_geodesic_last_generated;
        extern int GD_geodesic_last_fallback_requested;
        GD_geodesic_last_runtime_hits = s_runtime_hits;
        GD_geodesic_last_local_hits = s_local_hits;
        GD_geodesic_last_runtime_rejects = s_runtime_rejects;
        GD_geodesic_last_quality_gate_pass = s_quality_gate_pass;
        GD_geodesic_last_generated = s_generated;
        GD_geodesic_last_fallback_requested = s_fallback_requested;
    }

    /* Clear context snapshot — restore clean KV state for subsequent calls */
    ott_set_generation_context(NULL, 0);
    free(ctx);
    return generated;
}

/* ── OTT Speculative Decode ─────────────────────────────────────────────
 * Collects `batch` geodesic draft tokens, then verifies them in one
 * transformer pass via llm_speculative_verify_with_correction().
 * Accepted drafts skip the forward pass entirely; the correction token
 * is taken at the first rejection (or as a bonus token when all accepted).
 * Gives real transformer-quality output with geodesic-level latency for
 * the high-confidence positions.
 * ----------------------------------------------------------------------- */
static int geodesic_speculative_generate_text(const char *prompt,
                                              int max_tokens,
                                              float conf_thresh,
                                              int batch_size,
                                              int no_verifier,
                                              int perfect_day,
                                              int od_swarm_k,
                                              char *output,
                                              int max_output)
{
    const llm_model_t *m = llm_get_model();
    if (!m || !prompt || !output || max_output <= 0 || max_tokens <= 0)
        return -1;

    if (batch_size < 1) batch_size = 1;
    if (batch_size > 32) batch_size = 32;

    int max_ctx = max_tokens + 2048;
    int *ctx     = (int *)malloc((size_t)max_ctx * sizeof(int));
    int *drafts      = (int *)malloc((size_t)batch_size * sizeof(int));
    int *swarm_drafts = (od_swarm_k > 0 && batch_size > 0)
                      ? (int *)calloc((size_t)(batch_size * od_swarm_k), sizeof(int)) : NULL;
    float draft_conf[32];
    if (!ctx || !drafts) { free(ctx); free(drafts); free(swarm_drafts); return -1; }

    /* Apply Gemma4 IT chat template when prompt is plain text.
     * tok=107 (<end_of_turn>) has piece='\n' so cannot be injected as text.
     * We tokenize in two parts and inject tok=107 directly between them.
     * Full template: BOS + SOT(106) + "user\n" + text + EOT(107) + "\n" + SOT(106) + "model\n" */
    int spec_is_gemma = (strstr(m->arch, "gemma") != NULL);
    int spec_needs_tmpl = spec_is_gemma &&
                          (strstr(prompt, "<turn|>") == NULL) &&
                          (strstr(prompt, "<start_of_turn>") == NULL);
    int n_ctx;
    if (spec_needs_tmpl) {
        /* Build full Gemma IT chat template: <turn|>user\n{prompt}\n<turn|>model\n
         * This matches the production llm_chat_turn_tokens() template exactly.
         * <turn|> is the shorthand recognised by llm_tokenize() -> tok=106 (SOT).
         * \n -> tok=107 naturally; no manual EOT injection needed. */
        static char spec_tmpl[8192];
        int _pl = (int)strlen("<turn|>user\n");
        int _tl = (int)strlen(prompt);
        int _sl = (int)strlen("\n<turn|>model\n");
        int _total = _pl + _tl + _sl;
        if (_total >= (int)sizeof(spec_tmpl) - 1) _tl = (int)sizeof(spec_tmpl) - _pl - _sl - 2;
        memcpy(spec_tmpl, "<turn|>user\n", _pl);
        memcpy(spec_tmpl + _pl, prompt, _tl);
        memcpy(spec_tmpl + _pl + _tl, "\n<turn|>model\n", _sl);
        spec_tmpl[_pl + _tl + _sl] = '\0';

        int n_tmpl = llm_test_tokenize(spec_tmpl, (int)strlen(spec_tmpl), ctx + 1, max_ctx - 2);
        if (n_tmpl <= 0) { free(ctx); free(drafts); return -1; }
        if (m->bos_id >= 0) {
            ctx[0] = m->bos_id;
            n_ctx = n_tmpl + 1;
        } else {
            memmove(ctx, ctx + 1, (size_t)n_tmpl * sizeof(int));
            n_ctx = n_tmpl;
        }
        kprintf("[SPEC] Chat template applied (Gemma4 instruct, %d tokens)\n", n_ctx);
        /* Debug: dump first 24 token IDs to verify template */
        kprintf("[SPEC] ctx tokens:");
        for (int _di = 0; _di < n_ctx && _di < 24; _di++) kprintf(" %d", ctx[_di]);
        kprintf("\n");
    } else {
        int n_raw = llm_test_tokenize(prompt, (int)strlen(prompt), ctx + 1, max_ctx - 1);
        if (n_raw <= 0 || n_raw >= max_ctx - 1) {
            free(ctx); free(drafts); return -1;
        }
        if (m->bos_id >= 0) {
            ctx[0] = m->bos_id;
            n_ctx = n_raw + 1;
        } else {
            memmove(ctx, ctx + 1, (size_t)n_raw * sizeof(int));
            n_ctx = n_raw;
        }
    }
    if (n_ctx <= 0 || n_ctx >= max_ctx) {
        free(ctx); free(drafts); return -1;
    }

    /* Reset KV cache and prime generation context: clears stale tensor-bridge
     * capture state left by the Axiom survey, warms the LRU hidden-state cache
     * for the last prompt tokens, and takes a KV snapshot for geodesic probes. */
    ott_set_generation_context(ctx, n_ctx);

    /* Reset tensor bridge to NONE mode so the GPU forward path is available
     * for speculative verification.  ott_set_generation_context() may leave
     * the bridge in CAPTURE mode; set mode to NONE without wiping cached
     * hidden states so subsequent ott probes can still reuse them. */
    {
        tensor_bridge_t *br = llm_get_bridge();
        if (br) br->mode = BRIDGE_MODE_NONE;
    }

    /* Reset CUDA graph state: the Axiom Phase-5 geodesic pilot calls
     * llm_generate_token_ids() which captures a CUDA graph at its final
     * generation position.  If we don't reset here, llm_forward_token()
     * inside the verifier will REPLAY that stale graph (wrong position,
     * wrong KV-cache size), producing garbage logits (0% acceptance) and
     * corrupting the KV cache writes.  Force a fresh capture on the first
     * decode step of the speculative loop.
     * NOTE: We track the context length at which the graph was captured so
     * that subsequent calls from the same chat session can SKIP this reset
     * when the graph is still valid (same prefill length = cache hit).      */
#ifdef ENABLE_CUDA
    {
        extern int cuda_graph_captured;
        extern int cuda_graph_decode_ready;
        extern int cuda_graph_tried;
        extern int cuda_graph_ctx_len;   /* ctx len at capture time; -1 if unknown */
        /* Only invalidate the graph when the prefill context changed.
         * If n_ctx == cuda_graph_ctx_len the graph was captured at this exact
         * sequence position and is still valid — preserving it saves the
         * ~2 ms recapture latency on every continuation turn.               */
        if (cuda_graph_captured && cuda_graph_ctx_len != n_ctx) {
            cuda_graph_captured     = 0;
            cuda_graph_decode_ready = 1;
            cuda_graph_tried        = 0;
        } else if (!cuda_graph_captured) {
            cuda_graph_decode_ready = 1;
            cuda_graph_tried        = 0;
        }
    }
#endif

    output[0] = '\0';
    int out_len   = 0;
    int generated = 0;
    int geo_accepted = 0;
    int xfmr_accepted = 0;
    int od_draft_hits  = 0;  /* OTT-OD: drafts produced by OneDecode table */
    int swarm_hit_count = 0; /* OD-SWARM: draft slots accepted via swarm path */
    int eos = m->eos_id;
    /* Gemma4: stop on <start_of_turn>=106 or <end_of_turn>=107 */
    int is_gemma4 = (strstr(m->arch, "gemma4") != NULL);

    /* Dynamic batch sizing: track rolling acceptance rate and adapt every 4 rounds.
     * Start at batch_size when geometry is valid — the geodesic pilot showed ≥50%
     * top-1 accuracy so drafting is likely to yield accepts.  Fall back quickly
     * (after 4 blank rounds) if geometry is absent. */
    int dyn_batch     = batch_size;  /* start at full batch — geometry is valid */
    /* Warmup: geodesic uses raw token embeddings (position-independent) so it
     * cannot predict from chat-template markers like <start_of_turn>model\n.
     * Run verifier-only for the first 4 tokens until real response tokens are
     * in context; then enable geodesic drafting (it will see word tokens). */
    int spec_warmup_done = 0;
    int spec_warmup_tokens = 0;
    const int SPEC_WARMUP_N = 4; /* transformer-only tokens before geodesic starts */
    int dyn_round     = 0;       /* rounds since last adjustment */
    int dyn_accepted  = 0;
    int dyn_generated = 0;
    int last_snapshot_ctx = n_ctx;  /* start relative to template end so snapshot fires at n_ctx+4 */

    if (perfect_day) {
        kprintf("[SPEC] Starting OTT perfect-day decode (batch=%d, exact upper bound)\n",
                batch_size);
    } else {
        kprintf("[SPEC] Starting OTT speculative decode (batch=%d, thresh=%.2f)\n",
                batch_size, (double)conf_thresh);
    }
    fflush(stdout);

    /* Verifier-only decode timer (excludes geodesic overhead) */
    uint64_t verif_decode_us = 0;
    int      verif_decode_n  = 0;
    uint64_t prime_us = 0;
    int      prime_n  = 0;
    uint64_t rollout_us = 0;

    uint64_t _loop_t0 = 0;
    int _loop_n = 0;
    while (generated < max_tokens && n_ctx < max_ctx) {
        uint64_t _lt0 = hal_timer_us();
        /* ── Dynamic batch size adaptation (every 4 verification rounds) ────────
         * Increase batch when acceptance rate > 70% (geodesic is reliable here),
         * decrease when < 40% (geodesic is diverging, smaller batches reduce waste).
         */
        dyn_round++;
        if (!perfect_day && dyn_round >= 4 && dyn_generated > 0) {
            float acc_rate = (float)dyn_accepted / (float)dyn_generated;
            if (acc_rate > 0.70f && dyn_batch < batch_size)
                dyn_batch++;
            else if (acc_rate >= 0.40f && dyn_batch < batch_size)
                ; /* maintain current batch */
            else if (acc_rate < 0.40f && dyn_batch > 1) {
                /* SWARM: OD cost is O(N) -- keep batch >= 2 so SWARM stays active */
                int min_batch = (od_swarm_k > 0) ? 2 : 1;
                if (acc_rate == 0.0f) dyn_batch = min_batch;
                else { dyn_batch--; if (dyn_batch < min_batch) dyn_batch = min_batch; }
            }
            dyn_round     = 0;
            dyn_accepted  = 0;
            dyn_generated = 0;
            /* Re-probe only when GRC has received enough feedback.
             * Uses cheap step_fast (no forward pass, cached HS) to avoid
             * the full-prefill cost of llm_generate_token_ids.
             * Only re-probe if we have meaningful GRC history (>= 8 corrections)
             * AND the geodesic has produced at least one acceptance (avoids
             * expensive KV-reset when geometry is degenerate/zero-axiom). */
            if (dyn_batch <= 1 && (generated % 16) == 15
                    && axiom_beta_grc_count() >= 3
                    && geo_accepted > 0) {
                dyn_batch = 2;
            }
        }
        /* ── Step 1: collect up to dyn_batch geodesic draft tokens ────────────
         * Use axiom_beta_geodesic_rollout() which integrates a trajectory-coherent
         * geodesic path — carrying the velocity vector between steps so curvature
         * corrections accumulate properly.  Falls back to individual step_fast calls
         * if the rollout is unavailable.
         * ─────────────────────────────────────────────────────────────────────── */
        int n_drafts = 0;
        float min_conf = 10.0f;  /* init above max conf so min() tracks correctly */

        /* Warmup phase: skip geodesic until we have real response tokens */
        int skip_geodesic_warmup = (!spec_warmup_done && spec_warmup_tokens < SPEC_WARMUP_N);

        if (perfect_day) {
            int exact_n = 0;
            int exact_rc = llm_rollout_exact_greedy(ctx, n_ctx, dyn_batch,
                                                    drafts, draft_conf, &exact_n);
            if (exact_rc > 0 && exact_n > 0) {
                dyn_accepted += exact_n;
                dyn_generated += exact_n;
                for (int i = 0; i < exact_n && generated < max_tokens; i++) {
                    int tok = drafts[i];
                    if (eos >= 0 && tok == eos) goto spec_done;

                    char piece[256];
                    int pn = llm_test_decode_token(tok, piece, (int)sizeof(piece));
                    if (pn > 0) {
                        int room = max_output - out_len - 1;
                        if (room <= 0) goto spec_done;
                        if (pn > room) pn = room;
                        memcpy(output + out_len, piece, (size_t)pn);
                        out_len += pn;
                        output[out_len] = '\0';
                    }
                    if (n_ctx < max_ctx) ctx[n_ctx++] = tok;
                    generated++;
                    geo_accepted++;
                    if (draft_conf[i] < min_conf) min_conf = draft_conf[i];
                }
                if (n_ctx - last_snapshot_ctx >= 4 || n_ctx <= 4) {
                    llm_kv_snapshot_prefix(ctx, n_ctx);
                    last_snapshot_ctx = n_ctx;
                }
                ott_update_generation_context(ctx, n_ctx);
                continue;
            }
        }

        /* Skip geodesic during warmup phase — dyn_batch would decrement to 1
         * from all-zero drafts while warmup discards results, permanently disabling geodesic. */
        if (skip_geodesic_warmup) goto skip_geodesic;
        /* Skip geodesic drafting while batch=1 (pure transformer fallback mode).
         * Rollout does 16 vocab-scan steps (~100ms CPU) to produce a single draft
         * that gets rejected with 0% acceptance — pure overhead.  Jump directly
         * to the verifier-as-sampler path below. */
        if (dyn_batch <= 1) goto skip_geodesic;

        /* ── Prime logits before OD so d=0 always uses exact transformer top-1 ─
         * Without this, when ≥2 drafts accepted in the previous round,
         * llm_logits_pos lands at n_ctx-2 (not n_ctx-1), causing OD to miss
         * the primed-logits fast path and fall back to manifold lookup.
         * llm_prime_logits_fast is O(1) when cache is already valid (most cases),
         * costs one forward pass only when logits are stale. ───────────────── */
        if (axiom_beta_one_decode_ready() || dyn_batch >= 2) {
            uint64_t _pt0 = hal_timer_us();
            llm_prime_logits_fast(ctx, n_ctx);
            prime_us += hal_timer_us() - _pt0; prime_n++;
        }

        /* ── OTT-OD: OneDecode table as primary draft source ──────────────────
         * d=0: use cached primed greedy token (O(1) — argmax was computed once
         *   during llm_prime_logits_fast above).  If swarm_k > 0, also run the
         *   manifold topk scan for swarm candidates (O(16K×k) vs old O(262K)).
         * d=1+: skipped — rollout/step_fast handle continuation from extended
         *   context, correctly seeded by the d=0 primed token's hidden state.
         * ─────────────────────────────────────────────────────────────────── */
        if (axiom_beta_one_decode_ready() && n_drafts < dyn_batch) {
            int od_tok = llm_get_primed_greedy_token(n_ctx);
            if (od_tok >= 0 && od_tok < m->vocab_size) {
                char od_piece[256];
                int od_pn = llm_test_decode_token(od_tok, od_piece, (int)sizeof(od_piece));
                if (od_pn > 0 && geodesic_piece_quality_ok(od_piece, od_pn)) {
                    if (od_swarm_k > 0 && swarm_drafts) {
                        /* Manifold scan for swarm candidates (NOT the primed vocab scan).
                         * axiom_beta_one_decode_topk with primed logits = O(262K) vocab
                         * scan.  Bypass by using manifold-only path: call topk but the
                         * manifold path fires when primed logits are NOT available, so
                         * we pass a fake context length of n_ctx+1 which has no primed
                         * logits, forcing a manifold scan. This gives OD-diverse swarm
                         * candidates at O(16K×k) instead of O(262K).                  */
                        int   sw_stoks[OD_SWARM_MAX];
                        float sw_sconfs[OD_SWARM_MAX];
                        if (axiom_beta_one_decode_topk(ctx, n_ctx + 1,
                                                       sw_stoks, sw_sconfs,
                                                       od_swarm_k) == AXIOM_BETA_OK) {
                            /* Inject primed argmax as swarm slot 0 (always accepted) */
                            sw_stoks[0] = od_tok;
                            memcpy(&swarm_drafts[n_drafts * od_swarm_k], sw_stoks,
                                   (size_t)od_swarm_k * sizeof(int));
                        }
                    }
                    drafts[n_drafts++] = od_tok;
                    od_draft_hits++;
                    min_conf = 8.0f;  /* primed token: always accepted */
                }
            }
        }

        /* OD provided d=0: skip rollout (positional collision with step 0).
         * If GRC has history, step_fast fills remaining slots at d=1+.
         * If GRC is dry (<3 corrections), skip step_fast too — Christoffel
         * overhead (~4ms/call) exceeds benefit of an unguided draft token. */
        if (n_drafts >= dyn_batch) goto skip_fallback;
        if (n_drafts > 0) {
            if (axiom_beta_grc_count() < 3) goto skip_fallback;
            goto skip_rollout;  /* GRC ready: step_fast fills d=1 */
        }

        /* Multi-step rollout via geodesic trajectory integration.
         * Rollout step 0 uses cached primed greedy token (O(1) — argmax was
         * computed in llm_prime_logits_fast above).  Steps 1+ use GRC curvature. */
        {
            int   roll_toks[16];
            float roll_conf[16];
            int   roll_n = 0;
            int   need   = dyn_batch;
            if (need > 16) need = 16;
            uint64_t _rot0 = hal_timer_us();
            int ro_rc = axiom_beta_geodesic_rollout(ctx, n_ctx, need,
                                            roll_toks, roll_conf, &roll_n);
            rollout_us += hal_timer_us() - _rot0;
            if (ro_rc == AXIOM_BETA_OK && roll_n > 0) {
                int prev_tok = -1;
                int rpt      = 0;
                for (int d = 0; d < roll_n && n_drafts < dyn_batch; d++) {
                    int   rt = roll_toks[d];
                    float rc = roll_conf[d];
                    if (rc < conf_thresh) break;
                    char rpiece[256];
                    int rpn = llm_test_decode_token(rt, rpiece, (int)sizeof(rpiece));
                    if (rpn <= 0 || !geodesic_piece_quality_ok(rpiece, rpn)) break;
                    if (rt == prev_tok) {
                        if (++rpt >= 2) break;
                    } else {
                        rpt = 0;
                    }
                    prev_tok = rt;
                    drafts[n_drafts++] = rt;
                    if (rc < min_conf) min_conf = rc;
                }
            }
            if (n_drafts == 0 && dyn_batch > 1) {
                dyn_batch--;
            }
        }

        /* Fallback: fill remaining slots with individual step_fast calls */
        if (n_drafts >= dyn_batch) goto skip_fallback;
        if (axiom_beta_grc_count() < 3) goto skip_fallback;
        skip_rollout:;
        for (int d = n_drafts; d < dyn_batch && n_ctx + d < max_ctx - 1; d++) {
            int *view_ctx = ctx;
            int  view_n   = n_ctx + d;
            int *view_buf = NULL;
            if (d > 0) {
                view_buf = (int *)malloc((size_t)view_n * sizeof(int));
                if (!view_buf) break;
                memcpy(view_buf, ctx, (size_t)n_ctx * sizeof(int));
                memcpy(view_buf + n_ctx, drafts, (size_t)d * sizeof(int));
                view_ctx = view_buf;
            }
            int draft_tok = -1;
            float draft_conf = 0.0f;
            int ok = (axiom_beta_geodesic_step_fast(view_ctx, view_n,
                                                    &draft_tok, &draft_conf)
                      == AXIOM_BETA_OK);
            if (view_buf) free(view_buf);
            if (!ok || draft_tok < 0 || draft_tok >= m->vocab_size) break;
            if (draft_conf < conf_thresh) break;
            char piece[256];
            int pn = llm_test_decode_token(draft_tok, piece, (int)sizeof(piece));
            if (pn <= 0 || !geodesic_piece_quality_ok(piece, pn))
                break;

            drafts[n_drafts++] = draft_tok;
            if (draft_conf < min_conf) min_conf = draft_conf;
        }

        skip_fallback:;
        skip_geodesic:;
        /* ── Step 2: verify drafts (or get a fresh transformer token) ── */
        if (n_drafts > 0) {
            /* --no-verifier: emit geodesic drafts without transformer check */
            if (no_verifier) {
                for (int i = 0; i < n_drafts && generated < max_tokens; i++) {
                    int tok = drafts[i];
                    if (eos >= 0 && tok == eos) goto spec_done;
                    if (is_gemma4 && tok == 106) goto spec_done; /* start_of_turn = new turn */
                /* tok=107='\n' in Gemma4 — do NOT stop on it (fires on every newline) */
                    char piece[256];
                    int pn = llm_test_decode_token(tok, piece, (int)sizeof(piece));
                    if (pn > 0) {
                        int room = max_output - out_len - 1;
                        if (room <= 0) goto spec_done;
                        if (pn > room) pn = room;
                        memcpy(output + out_len, piece, (size_t)pn);
                        out_len += pn;
                        output[out_len] = '\0';
                    }
                    if (n_ctx < max_ctx) ctx[n_ctx++] = tok;
                    generated++;
                    geo_accepted++;
                }
                continue;
            }
            int correction = -1;
            uint64_t _vt0 = hal_timer_us();
            /* topk_margin=2.8: accept draft if logit[draft] >= max_logit - 2.8
             * (top-~6%% probability mass). Moderate window improves
             * acceptance rate vs 2.0 while avoiding low-probability tokens
             * that can degrade context coherence. */
            int n_acc = (od_swarm_k > 0 && swarm_drafts)
                ? llm_speculative_verify_swarm(ctx, n_ctx, drafts, n_drafts,
                                               2.8f, swarm_drafts, od_swarm_k, &correction)
                : llm_speculative_verify_topk(ctx, n_ctx, drafts, n_drafts,
                                              2.8f, &correction);
            uint64_t _vt1 = hal_timer_us();
            verif_decode_us += _vt1 - _vt0;
            verif_decode_n  += 1;

            if (n_acc < 0) {
                /* verification error — fall back to standard generation */
                break;
            }

            /* Track acceptance for dynamic batch sizing */
            dyn_accepted  += n_acc;
            dyn_generated += n_drafts;

            /* Emit accepted draft tokens */
            for (int i = 0; i < n_acc && generated < max_tokens; i++) {
                int tok = drafts[i];
                if (eos >= 0 && tok == eos) goto spec_done;
                if (is_gemma4 && tok == 106) goto spec_done; /* start_of_turn = new turn */
                /* tok=107='\n' in Gemma4 — do NOT stop on it (fires on every newline) */

                char piece[256];
                int pn = llm_test_decode_token(tok, piece, (int)sizeof(piece));
                if (pn > 0) {
                    int room = max_output - out_len - 1;
                    if (room <= 0) goto spec_done;
                    if (pn > room) pn = room;
                    memcpy(output + out_len, piece, (size_t)pn);
                    out_len += pn;
                    output[out_len] = '\0';
                }
                axiom_beta_grc_feedback(ctx, n_ctx, tok); /* GRC: accepted token */
                if (n_ctx < max_ctx) ctx[n_ctx++] = tok;
                generated++;
                geo_accepted++;
            }

            /* If not all drafts were accepted, emit correction token */
            if (n_acc < n_drafts && correction >= 0 &&
                correction < m->vocab_size && generated < max_tokens) {
                int tok = correction;
                if (eos >= 0 && tok == eos) goto spec_done;
                if (is_gemma4 && tok == 106) goto spec_done; /* start_of_turn = new turn */
                /* tok=107='\n' in Gemma4 — do NOT stop on it (fires on every newline) */
                /* GRC feedback: teach the geodesic what the correct token was */
                axiom_beta_grc_feedback(ctx, n_ctx, tok);
                char piece[256];
                int pn = llm_test_decode_token(tok, piece, (int)sizeof(piece));
                if (pn > 0) {
                    int room = max_output - out_len - 1;
                    if (room > 0) {
                        if (pn > room) pn = room;
                        memcpy(output + out_len, piece, (size_t)pn);
                        out_len += pn;
                        output[out_len] = '\0';
                    }
                }
                if (n_ctx < max_ctx) ctx[n_ctx++] = tok;
                generated++;
                xfmr_accepted++;
            } else if (n_acc == n_drafts && correction >= 0 &&
                       correction < m->vocab_size && generated < max_tokens) {
                /* Bonus token: all drafts accepted, correction is next token */
                int tok = correction;
                if (eos >= 0 && tok == eos) goto spec_done;
                if (is_gemma4 && tok == 106) goto spec_done; /* start_of_turn = new turn */
                /* tok=107='\n' in Gemma4 — do NOT stop on it (fires on every newline) */
                char piece[256];
                int pn = llm_test_decode_token(tok, piece, (int)sizeof(piece));
                if (pn > 0) {
                    int room = max_output - out_len - 1;
                    if (room > 0) {
                        if (pn > room) pn = room;
                        memcpy(output + out_len, piece, (size_t)pn);
                        out_len += pn;
                        output[out_len] = '\0';
                    }
                }
                if (n_ctx < max_ctx) ctx[n_ctx++] = tok;
                generated++;
                xfmr_accepted++;
            }
        } else {
            /* No high-confidence drafts — run transformer directly for one token. */
            int correction = -1;
            uint64_t _vt0 = hal_timer_us();
            llm_speculative_verify_topk(ctx, n_ctx, NULL, 0, 0.0f, &correction);
            verif_decode_us += hal_timer_us() - _vt0;
            verif_decode_n  += 1;
            int out_tok = correction;
            if (out_tok < 0 || out_tok >= m->vocab_size)
                break;

            int tok = out_tok;
            if (eos >= 0 && tok == eos) goto spec_done;
            if (is_gemma4 && tok == 106) goto spec_done; /* start_of_turn = new turn */
                /* tok=107='\n' in Gemma4 — do NOT stop on it (fires on every newline) */
            char piece[256];
            int pn = llm_test_decode_token(tok, piece, (int)sizeof(piece));
            if (pn > 0) {
                int room = max_output - out_len - 1;
                if (room <= 0) goto spec_done;
                if (pn > room) pn = room;
                memcpy(output + out_len, piece, (size_t)pn);
                out_len += pn;
                output[out_len] = '\0';
            }
            if (n_ctx < max_ctx) ctx[n_ctx++] = tok;
            generated++;
            xfmr_accepted++;
            if (!spec_warmup_done) {
                spec_warmup_tokens++;
                if (spec_warmup_tokens >= SPEC_WARMUP_N) spec_warmup_done = 1;
            }
        }
        /* Update KV snapshot so next round's verification skips full prefill.
         * Throttle: only snapshot when context grew by ≥4 tokens since the
         * last snapshot — a full memcpy of 35 × n_ctx × kv_dim floats is
         * ~10 MB for Gemma4 at ctx=256, and doing it every batch round wastes
         * ~3 ms per accepted draft.  With batch=8 and acceptance=60% we'd
         * snapshot every ~5 tokens; with threshold=4 we snapshot ~every 4.
         * Also keep ott_gen_ctx in sync so HS probes use the correct position.
         * Snapshot always (even dyn_batch=1) so the verifier fast path stays
         * active in pure-decoder mode — skipping snapshot caused full prefill
         * on every token when geodesic was disabled. */
        {
            /* Snapshot every 32 tokens: the verifier fast-path fires without
             * restores in normal operation (prime always advances KV correctly),
             * so frequent snapshots are pure PCIe download overhead. */
            if (n_ctx - last_snapshot_ctx >= 32 || n_ctx <= 4) {
                llm_kv_snapshot_prefix(ctx, n_ctx);
                last_snapshot_ctx = n_ctx;
            }
            if (dyn_batch > 1)
                ott_update_generation_context(ctx, n_ctx);
        }
    }

spec_done:
    {
        float verif_tok_s = (verif_decode_us > 0 && verif_decode_n > 0)
            ? (float)verif_decode_n * 1e6f / (float)verif_decode_us : 0.0f;
        kprintf("[SPEC] Done: %d tokens (geo_accepted=%d xfmr=%d od_drafts=%d swarm_k=%d, acceptance_rate=%.1f%%, final_batch=%d)\n",
                generated, geo_accepted, xfmr_accepted, od_draft_hits, od_swarm_k,
                generated > 0 ? 100.0f * (float)geo_accepted / (float)generated : 0.0f,
                dyn_batch);
        if (perfect_day)
            kprintf("[SPEC] Perfect-day note: acceptance is the transformer-exact upper bound.\n");
        kprintf("[SPEC] Verifier decode: %.1f tok/s (%d calls in %llu ms)\n",
                verif_tok_s, verif_decode_n,
                (unsigned long long)(verif_decode_us / 1000));
    }

    GD_geodesic_last_runtime_hits  = geo_accepted;
    GD_geodesic_last_local_hits    = 0;
    GD_geodesic_last_runtime_rejects = xfmr_accepted;
    GD_geodesic_last_quality_gate_pass = (generated > 0) ? 1 : 0;
    GD_geodesic_last_generated     = generated;
    GD_geodesic_last_fallback_requested = 0;

    free(ctx);
    free(drafts);
    free(swarm_drafts);
    return generated > 0 ? generated : -1;
}

static int geodesic_generate_from_context(int *ctx,
                                          int *n_ctx,
                                          int max_ctx,
                                          int max_tokens,
                                          char *output,
                                          int max_output)
{
    const llm_model_t *m = llm_get_model();
    if (!m || !ctx || !n_ctx || max_ctx <= 0 || !output || max_output <= 0 || max_tokens <= 0)
        return -1;
    if (*n_ctx <= 0 || *n_ctx >= max_ctx)
        return -1;

    output[0] = '\0';
    int out_len = 0;
    int generated = 0;
    int runtime_hits = 0;
    int local_hits = 0;
    int runtime_rejects = 0;
    int local_only_budget = 0;
    int bad_piece_streak = 0;
    int eos = m->eos_id;

    for (int step = 0; step < max_tokens; step++) {
        int tok = -1;
        int used_runtime = 0;

        if (GD_ott_theorem_active && local_only_budget > 0 &&
            !geodesic_theorem_should_probe_runtime(local_only_budget, step)) {
            if (geodesic_next_token_local(ctx, *n_ctx, &tok) == 0) {
                local_hits++;
                local_only_budget--;
            } else {
                break;
            }
        } else {
            int runtime_ok = (geodesic_next_token_runtime(ctx, *n_ctx, &tok) == 0);
            if (runtime_ok && geodesic_runtime_token_acceptable(ctx, *n_ctx, tok)) {
                runtime_hits++;
                used_runtime = 1;
            } else if (geodesic_next_token_local(ctx, *n_ctx, &tok) == 0) {
                local_hits++;
                if (runtime_ok)
                    runtime_rejects++;
            } else {
                break;
            }
        }

        if (tok < 0 || tok >= m->vocab_size)
            break;

        if (GD_ott_theorem_active) {
            char q_piece[256];
            int q_n = llm_test_decode_token(tok, q_piece, (int)sizeof(q_piece));
            if (!geodesic_piece_quality_ok(q_piece, q_n)) {
                if (used_runtime) {
                    int tok_local = -1;
                    if (geodesic_next_token_local(ctx, *n_ctx, &tok_local) == 0 &&
                        tok_local >= 0 && tok_local < m->vocab_size) {
                        tok = tok_local;
                        local_hits++;
                        runtime_rejects++;
                        used_runtime = 0;
                        q_n = llm_test_decode_token(tok, q_piece, (int)sizeof(q_piece));
                    }
                }
                if (!geodesic_piece_quality_ok(q_piece, q_n)) {
                    bad_piece_streak++;
                    local_only_budget = geodesic_theorem_local_budget(bad_piece_streak);
                } else {
                    if (used_runtime && bad_piece_streak > 0)
                        bad_piece_streak--;
                    if (used_runtime && local_only_budget > 0)
                        local_only_budget--;
                }
            } else {
                if (used_runtime && bad_piece_streak > 0)
                    bad_piece_streak--;
            }
        }

        if (*n_ctx >= max_ctx)
            break;

        ctx[(*n_ctx)++] = tok;
        if (eos >= 0 && tok == eos)
            break;

        char piece[256];
        int piece_n = llm_test_decode_token(tok, piece, (int)sizeof(piece));
        if (piece_n > 0) {
            int room = max_output - out_len - 1;
            if (room <= 0) break;
            if (piece_n > room) piece_n = room;
            memcpy(output + out_len, piece, (size_t)piece_n);
            out_len += piece_n;
            output[out_len] = '\0';
        }

        generated++;
    }

    if (generated > 0) {
        kprintf("[GD] Geodesic decode tokens: runtime=%d local=%d (runtime_rejects=%d)\n",
            runtime_hits, local_hits, runtime_rejects);
    }
    return generated;
}

static int geodesic_chat_turn(const char *user_text,
                              int *ctx,
                              int *n_ctx,
                              int max_ctx,
                              int max_tokens,
                              float temperature,
                              char *output,
                              int max_output)
{
    char turn_prompt[4096];
    int turn_tokens[4096];
    int n_turn;
    int n_ctx_before;

    if (!user_text || !ctx || !n_ctx || !output)
        return -1;

    n_ctx_before = *n_ctx;

    snprintf(turn_prompt, sizeof(turn_prompt), "User: %s\nAssistant:", user_text);
    n_turn = llm_test_tokenize(turn_prompt, (int)strlen(turn_prompt), turn_tokens, (int)(sizeof(turn_tokens) / sizeof(turn_tokens[0])));
    if (n_turn <= 0)
        return -1;

    if (*n_ctx + n_turn >= max_ctx) {
        /* Reset context if token buffer is near capacity. */
        *n_ctx = 0;
    }
    if (*n_ctx + n_turn >= max_ctx)
        return -1;

    memcpy(ctx + *n_ctx, turn_tokens, (size_t)n_turn * sizeof(int));
    *n_ctx += n_turn;

    int n_geo = geodesic_generate_from_context(ctx, n_ctx, max_ctx,
                                               max_tokens, output, max_output);
    if (n_geo > 0 && geodesic_output_quality_ok(output))
        return n_geo;

    kprintf("[GD] Geodesic chat quality gate failed; falling back to standard chat decode for this turn.\n");

    /* Keep geodesic context coherent after a rejected geodesic sample. */
    *n_ctx = 0;

    /* Prevent double-printing when interactive streaming callback is active. */
    llm_set_stream_cb((llm_token_cb_t)0, (void *)0);
    int n_std = llm_chat_turn(user_text, output, max_output, max_tokens, temperature);
    llm_set_stream_cb(GD_stream_cb, (void *)0);

    if (n_std <= 0) {
        *n_ctx = n_ctx_before;
    }
    return n_std;
}

int main(int argc, char **argv) {
    GD_args_t args;

    print_banner();

    if (parse_args(argc, argv, &args) < 0) {
        print_usage(argv[0]);
        return 1;
    }

    /* Initialize HAL (CPU detection, thread pool) */
    hal_init();
    if (args.n_threads > 0)
        smp_init_hosted(args.n_threads);
    enable_max_performance_host();

    /* Configure logging level */
    if (args.log_level >= 0) {
        klog_set_level((log_level_t)args.log_level);
    } else if (args.verbose) {
        klog_set_level(LOG_DEBUG);
    }

    /* Configure thinking mode */
    if (args.no_think == 1) llm_set_thinking(0);
    else if (args.no_think == -1) llm_set_thinking(2);  /* force-think */
    if (args.show_think) llm_set_show_thinking(1);
    llm_set_sampling_params(args.top_k, args.top_p);
    llm_set_temperature(args.temperature);
    llm_set_attention_residuals(args.attnres_enable, args.attnres_strength);
    llm_set_depth_residual_attention(args.depth_attn_enable,
                                     args.depth_attn_strength,
                                     args.depth_attn_window);
    GD_ott_theorem_active = args.ott_theorem ? 1 : 0;
    if (args.ott_full) {
        kprintf("[GD] OTT full mode enabled (geodesic-first + axiomatic run + AttnRes)\n");
    }
    if (args.ott_theorem) {
        kprintf("[GD] OTT theorem mode enabled (depth residual attention + strict geodesic quality control)\n");
    }

    /* Handle --download: download model from HuggingFace */
    if (args.download_repo) {
        static hf_download_ctx_t dl_ctx;
        hf_download_init(&dl_ctx);
        hf_download_set_progress(&dl_ctx, (void *)0, (void *)0);

        int rc_dl = hf_download_auto(&dl_ctx, args.download_repo,
                                      args.quant_hint, "models");
        if (rc_dl < 0) {
            kprintf("[GD] ERROR: Download failed: %s\n", hf_download_error(&dl_ctx));
            hf_download_free(&dl_ctx);
            hal_shutdown();
            return 1;
        }

        /* If no model_path was given, use the downloaded file */
        if (!args.model_path) {
            static char dl_path[256];
            snprintf(dl_path, sizeof(dl_path), "%s", dl_ctx.output_path);
            args.model_path = dl_path;
            kprintf("[GD] Using downloaded model: %s\n", args.model_path);
        }
        hf_download_free(&dl_ctx);
    }

    if (!args.model_path) {
        kprintf("[GD] ERROR: No model path available.\n");
        hal_shutdown();
        return 1;
    }

    /* Memory-map the model file */
    kprintf("[GD] Loading model: %s\n", args.model_path);
    hal_mmap_t model = hal_mmap_file(args.model_path);
    if (!model.data) {
        kprintf("[GD] ERROR: Could not open model file: %s\n", args.model_path);
        hal_shutdown();
        return 1;
    }
    kprintf("[GD] Mapped %llu MB\n", (unsigned long long)(model.size / (1024 * 1024)));

    /* Apply context-size override before loading (0 = use 2048-token default).
     * Without an explicit --ctx-size, the model's native context (e.g. 128K for
     * LLaMA 3.1) would be capped to 8192, consuming 2 GB of GPU KV cache.
     * The 2048-token default keeps KV cache at 512 MB — fits in 6 GB VRAM. */
    llm_set_max_ctx(args.ctx_size > 0 ? args.ctx_size : 2048);

    /* If weight compression is requested, put GPU init in compress-mode so it
     * skips raw weight uploads for the manifold GP path.
     * - attn-only (default): skip Q/K/V/O (manifold CPU path), keep FFN on GPU
     * - full-compress: skip Q/K/V/O + FFN (both replaced by manifold CPU path) */
    if (args.axex_compress) {
        /* NOTE: We no longer pre-skip Q/K/V/O upload here.
         * Axiom beta calibration needs real attention weights on GPU for its
         * forward-pass hidden-state capture; skipping them before llm_gpu_init
         * caused STATUS_INTEGER_DIVIDE_BY_ZERO in CUDA attention kernels.
         * Instead, llm_gpu_upload_compressed_weights() phase 2b frees the raw
         * Q/K/V/O GPU buffers AFTER the GP W_proj matrices are uploaded,
         * achieving the same VRAM savings without breaking calibration.
         * Only full-compress (not attn-only) still skips raw FFN upload, since
         * FFN is not needed for axiom beta and saves PCIe bandwidth. */
        if (!args.axex_attn_only)
            llm_gpu_set_compress_mode(1);   /* skip FFN only in full-compress mode */
    }
    /* --axex-ffn-compress: skip raw FFN uploads during llm_gpu_init so that
     * attn-only VRAM estimates are used.  All 32 layers' attention weights fit
     * in ~1.3 GB, leaving ample room for compressed FFN uploaded later.
     * Without this, raw FFN fills VRAM at layer ~18 and the remaining layers
     * must be promoted via Phase 3 (compress→free→re-upload dance). */
    if (args.axex_ffn_compress) {
        llm_gpu_set_compress_mode(1);
    }

    /* Load model via GGUF parser + LLM engine */
    uint64_t t0 = hal_timer_us();
    int rc = llm_load_from_buffer(model.data, model.size);
    uint64_t t1 = hal_timer_us();

    if (rc < 0) {
        kprintf("[GD] ERROR: Failed to load model (rc=%d)\n", rc);
        hal_munmap(&model);
        hal_shutdown();
        return 1;
    }

    kprintf("[GD] Model loaded in %llu ms\n", (unsigned long long)((t1 - t0) / 1000));
    kprintf("[GD] Model: %s\n", llm_model_name());

    int axiom_ran_this_invocation = 0;
    int axiom_intrinsic_dim = 0;
    float axiom_consistency = 0.0f;
    float axiom_geodesic_speedup = 0.0f;
    unsigned long long axiom_total_ms = 0;
    axiom_beta_report_t axiom_rep;
    memset(&axiom_rep, 0, sizeof(axiom_rep));

    if (args.axiom_beta_run) {
        axiom_beta_config_t cfg;
        axiom_beta_report_t *rep = &axiom_rep;
        axiom_beta_status_t st;
        axiom_beta_default_config(&cfg);
        if (args.axiom_samples > 0) cfg.embedding_samples = args.axiom_samples;
        if (args.axiom_vocab_probe > 0) cfg.geodesic_vocab_probe = args.axiom_vocab_probe;
        if (args.axiom_use_gpu_phase5 >= 0) cfg.use_gpu_phase5 = args.axiom_use_gpu_phase5;
        cfg.geodesic_use_oracle_target = args.axiom_use_oracle_target;
        cfg.fast_mode = args.axiom_fast_mode;
        cfg.reuse_cache = args.axiom_reuse_cache;
        cfg.seed = args.axiom_seed;
        cfg.verbose = args.verbose;
        cfg.skip_geodesic = args.axiom_skip_geodesic;
        if (args.axiom_pca_variance > 0.0) cfg.pca_variance_ratio = args.axiom_pca_variance;

        if (args.vis_output_dir)
            axiom_vis_init(args.vis_output_dir);

        st = axiom_beta_run(&cfg, rep);
        if (st != AXIOM_BETA_OK) {
            kprintf("[AXIOM-BETA-3] ERROR: %s\n", axiom_beta_status_string(st));
            hal_munmap(&model);
            hal_shutdown();
            return 1;
        }
        axiom_ran_this_invocation = 1;
        axiom_intrinsic_dim = rep->phase1.intrinsic_dim;
        axiom_consistency = rep->phase4.consistency_score;
        axiom_geodesic_speedup = rep->phase5.projected_speedup;
        axiom_total_ms = (unsigned long long)(rep->total_us / 1000);

        kprintf("\n[AXIOM-BETA-3] === Summary ==========================\n");
        kprintf("  Intrinsic dim : %d (TwoNN=%.2f, PCA=%d components)\n",
                rep->phase1.intrinsic_dim, rep->phase1.twonn_raw,
                rep->phase1.pca_components_kept);
        kprintf("  Symmetry      : score=%.4f, generators=%d, invariant=%d\n",
                rep->phase2.symmetry_score, rep->phase2.generators_found,
                rep->phase2.permutation_invariant_heads);
        kprintf("  Curvature     : mean=%.6f, max=%.6f, high-curv=%d\n",
                rep->phase3.mean_scalar_curvature,
                rep->phase3.max_scalar_curvature,
                rep->phase3.high_curvature_loci);
        if (rep->uses_fisher_metric)
            kprintf("  Fisher metric : trace_mean=%.4f, det_log_mean=%.4f\n",
                    rep->phase3.fisher_trace_mean,
                    rep->phase3.fisher_det_log_mean);
        kprintf("  Axioms        : %d (consistency=%.4f, oracle=%d)\n",
                rep->phase4.axiom_count, rep->phase4.consistency_score,
                rep->phase4.oracle_calls_used);
        kprintf("  Geodesic pilot: speedup=%.1fx\n",
                rep->phase5.projected_speedup);
        kprintf("  Cache reuse   : %s\n",
            rep->reused_geometry_cache ? "yes" : "no");
        if (rep->supports_geodesic_pilot)
            kprintf("  Geodesic sim  : cos=%.4f, L2_err=%.4f\n",
                    rep->phase5.geodesic_cosine_similarity,
                    rep->phase5.geodesic_reconstruction_error);
        if (rep->supports_geodesic_pilot)
            kprintf("  Geodesic tok  : top1=%.3f (%d/%d), mrr=%.3f, probe=%d, targets(o/r)=%d/%d, gpu=%s\n",
                    rep->phase5.geodesic_top1_match_rate,
                    rep->phase5.geodesic_top1_hits,
                    rep->phase5.pilot_tokens_tested,
                    rep->phase5.geodesic_target_mrr,
                rep->phase5.geodesic_vocab_probe,
                rep->phase5.oracle_target_count,
                rep->phase5.random_target_count,
                rep->phase5.used_gpu_scoring ? "yes" : "no");
        kprintf("  Total time    : %llu ms\n",
                (unsigned long long)(rep->total_us / 1000));
        kprintf("[AXIOM-BETA-3] ======================================\n");

        if (args.axiom_report_path) {
            axiom_beta_status_t wr = axiom_beta_write_json(args.axiom_report_path,
                                                           rep, &cfg);
            if (wr == AXIOM_BETA_OK)
                kprintf("[AXIOM-BETA-3] Report: %s\n", args.axiom_report_path);
            else
                kprintf("[AXIOM-BETA-3] Report write failed: %s\n",
                        axiom_beta_status_string(wr));
        }

        if (args.axiom_beta_only) {
            hal_munmap(&model);
            hal_shutdown();
            return 0;
        }
    }

    /* ── Manifold Exploitation (axex) ──────────────────────────────────── */
    static axex_state_t g_axex_state;
    if (args.axex_kv || args.axex_offload || args.axex_compress || args.axex_ffn_compress) {
        axex_config_t xcfg;
        memset(&xcfg, 0, sizeof(xcfg));
        xcfg.enable_kv_compress     = args.axex_kv;
        xcfg.kv_threshold           = args.axex_kv_threshold;
        xcfg.enable_layer_offload   = args.axex_offload;
        /* Enable SVD FFN compression (gate/up/down) when --axex-compress or
         * --axex-ffn-compress is requested.  This stores compressed weights in
         * g_compress_table, which the GPU upload path then turns into cuBLAS
         * batched pointer arrays for gate+up (axex_prepare_batched_ffn).  The
         * manifold GP path below handles Q/K/V/O separately via per-layer PCA.
         * --axex-ffn-compress skips the manifold PCA for a faster benchmark. */
        xcfg.enable_weight_compress = (args.axex_compress || args.axex_ffn_compress) ? 1 : 0;
        xcfg.weight_quality         = args.axex_compress_quality;
        xcfg.weight_max_rank        = args.axex_compress_rank;  /* 0 = auto */
        xcfg.weight_max_err         = args.axex_compress_max_err; /* 0 = no limit */
        if (axex_init(&g_axex_state, &xcfg,
                      axiom_ran_this_invocation ? &axiom_rep : NULL,
                      llm_get_model()) != 0)
            kprintf("[AXEX] Warning: manifold exploitation init failed — continuing without it\n");

        /* If FFN compression was requested but ALL matrices were skipped (e.g.
         * max_err threshold rejected everything), the raw FFN weights were
         * never uploaded to GPU (gpu_ffn_skip_for_compress=1 was set before
         * llm_gpu_init).  Upload them now so inference isn't using null ptrs. */
        if (args.axex_ffn_compress && g_axex_state.compress_layers == 0) {
            kprintf("[AXEX] No FFN layers compressed (all exceeded max_err threshold) — uploading raw FFN weights\n");
            llm_gpu_upload_ffn_fallback();
        }

        /* Manifold-projected weight compression (Geodesic Projection).
         * Must run after axex_init so the PCA is available from axiom_beta.
         * This compresses Q/K/V/O (attention only by default) to W@P[m×k] —
         * giving near-lossless accuracy while enabling 70B models in 8 GB VRAM.
         *
         * Per-layer mode: compute a separate PCA basis for each transformer
         * layer so that the projection preserves energy at every depth.
         * Without per-layer PCA, a single embedding-layer basis gives only
         * 2–4% energy at deeper layers, producing blank/garbage output.
         *
         * Note: we no longer gate this on compress_layers <= 0.  Both paths
         * can run together: SVD handles gate/up/down (FFN), manifold GP
         * handles Q/K/V/O (attention).  The GPU upload path keeps them in
         * separate tables (g_compress_table vs g_manifold_table). */
        /* --axex-ffn-compress skips manifold PCA (SVD FFN only, faster start). */
        if (args.axex_compress && !args.axex_ffn_compress && axiom_ran_this_invocation) {
            const llm_model_t *mdl = llm_get_model();
            if (mdl) {
                int dim       = mdl->dim;
                int n_layers  = mdl->n_layers;
                int vocab     = llm_model_vocab();
                /* Calibration sample count:
                 *   - user override:    --axex-calib-samples N
                 *   - fast mode:        64  (quick preview)
                 *   - default:          512 (recommended for k≤512 PCA quality) */
                int n_samples;
                if (args.axex_calib_samples > 0)
                    n_samples = args.axex_calib_samples;
                else
                    n_samples = (args.axiom_fast_mode) ? 64 : 512;
                if (n_samples > vocab) n_samples = vocab;

                /* Wire up attention-only vs full-compress mode */
                axex_manifold_set_attn_only(args.axex_attn_only);
                if (args.axex_attn_only)
                    kprintf("[AXEX-MANIFOLD] Attention-only mode: Q/K/V/O compressed, FFN preserved (near-lossless accuracy)\n");
                else
                    kprintf("[AXEX-MANIFOLD] Full-compress mode: Q/K/V/O + FFN gate/up compressed (lower accuracy)\n");

                float   *hs_buf = NULL; /* unused in multi-layer path, kept for fallback */
                int      lw_ok  = 1;

                if (lw_ok) {
                    kprintf("[AXEX-MANIFOLD] Computing per-layer PCA (%d layers × %d samples × dim=%d)...\n",
                            n_layers, n_samples, dim);
                    kprintf("[AXEX-MANIFOLD] Using single-pass multi-layer capture (%d forward passes total)\n",
                            n_samples);

                    /* Allocate flat matrix: all_hs[layer][sample][dim] */
                    size_t total_f = (size_t)n_layers * n_samples * dim;
                    float *all_hs = (float *)malloc(total_f * sizeof(float));
                    int   *all_valid = (int *)calloc((size_t)n_layers, sizeof(int)); /* reused */
                    int   *cap_valid = (int *)calloc((size_t)n_layers, sizeof(int));
                    if (!all_hs || !all_valid || !cap_valid) {
                        free(all_hs); free(all_valid); free(cap_valid);
                        lw_ok = 0;
                    }

                    if (lw_ok) {
                        uint64_t seed = 0x123456789ABCDEF0ULL;
                        /* Temporary bridge buffer: one forward-pass captures all layers */
                        float *pass_buf = (float *)malloc((size_t)n_layers * dim * sizeof(float));
                        if (!pass_buf) lw_ok = 0;

                        for (int s = 0; s < n_samples && lw_ok; s++) {
                            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                            int tok = (int)(seed % (uint64_t)vocab);

                            memset(cap_valid, 0, (size_t)n_layers * sizeof(int));
                            int rc = axiom_beta_probe_all_layer_states(
                                tok, pass_buf, cap_valid, n_layers, dim);

                            for (int l = 0; l < n_layers; l++) {
                                float *dst = all_hs + ((size_t)l * n_samples + s) * dim;
                                if (rc == 0 && cap_valid[l]) {
                                    memcpy(dst, pass_buf + (size_t)l * dim,
                                           (size_t)dim * sizeof(float));
                                    all_valid[l]++;
                                } else {
                                    memset(dst, 0, (size_t)dim * sizeof(float));
                                }
                            }

                            if ((s % 64) == 63 || s == n_samples - 1)
                                kprintf("[AXEX-MANIFOLD] Calibration sample %d/%d done\n",
                                        s + 1, n_samples);
                        }
                        free(pass_buf);
                    }

                    /* Build per-layer PCA from collected samples */
                    for (int l = 0; l < n_layers && lw_ok; l++) {
                        axmat_t X = axmat_create(n_samples, dim);
                        if (!X.data) { lw_ok = 0; break; }

                        for (int s = 0; s < n_samples; s++) {
                            const float *src = all_hs + ((size_t)l * n_samples + s) * dim;
                            for (int j = 0; j < dim; j++)
                                X.data[(size_t)s * dim + j] = (double)src[j];
                        }

                        double var_ratio = (args.axiom_pca_variance > 0.0)
                                           ? args.axiom_pca_variance : 0.9999;
                        axpca_t lpca = axpca_compute(&X, var_ratio);
                        axmat_destroy(&X);

                        if (lpca.n_components <= 0) {
                            kprintf("[AXEX-MANIFOLD] Layer %d PCA failed\n", l);
                            axpca_destroy(&lpca);
                            lw_ok = 0;
                            break;
                        }

                        if (axex_manifold_init_layer(l, &lpca, dim, AXEX_MANIFOLD_K_MAX) != 0) {
                            kprintf("[AXEX-MANIFOLD] Layer %d basis init failed\n", l);
                            axpca_destroy(&lpca);
                            lw_ok = 0;
                            break;
                        }
                        axpca_destroy(&lpca);

                        if ((l % 8) == 7 || l == n_layers - 1)
                            kprintf("[AXEX-MANIFOLD] Per-layer PCA: %d/%d layers done\n",
                                    l + 1, n_layers);
                    }
                    free(all_hs);
                    free(all_valid);
                    free(cap_valid);
                }

                if (lw_ok) {
                    int mn = axex_compress_model_manifold_layerwise(
                        axiom_ran_this_invocation ? &axiom_rep : NULL, mdl);
                    kprintf("[AXEX-MANIFOLD] Layerwise geodesic projection: %d matrices compressed\n", mn);
                } else {
                    kprintf("[AXEX-MANIFOLD] Per-layer PCA failed — skipping GP compression\n");
                }
            } else {
                kprintf("[AXEX-MANIFOLD] Skipped (no PCA — run with --axiom-beta-run first)\n");
            }
        }

        /* Upload any compressed weight matrices to GPU device memory */
        llm_gpu_upload_compressed_weights();
    }

    if (args.one_decode || args.ott_od) {
        /* OneDecode: ensure Phase-3 geometry then bake / load the decode table */
        geodesic_ensure_one_decode(args.one_decode_coverage);
        if (axiom_beta_one_decode_ready())
            kprintf("[OD] OneDecode ready — decode steps will be near-instant\n");
        else
            kprintf("[OD] OneDecode bake failed — falling back to step_fast\n");
        /* No prompt given → drop into interactive chat automatically */
        if (!args.prompt && !args.interactive && !args.serve)
            args.interactive = 1;
    }

    if (args.axiom_geodesic_first) {
        if (axiom_ran_this_invocation) {
            kprintf("[GD] Reusing current-invocation Axiom geometry cache for geodesic-first decode.\n");
        } else {
            kprintf("[GD] Priming OTT geometry cache for geodesic-first decode...\n");
            if (geodesic_prime_cache(&args) != 0) {
                kprintf("[GD] WARNING: Could not prime geometry cache; using local geodesic fallback only.\n");
            }
        }
    }

    /* Run inference */
    if (args.serve) {
        kprintf("[GD] Starting API server on port %d...\n", args.port);
        GD_api_serve(args.port);
    } else if (args.interactive) {
        interactive_loop(args.model_path, &args);
    } else {
        const char *prompt = args.prompt ? args.prompt : "Hello";
        kprintf("[GD] Prompt: \"%s\"\n", prompt);
        if (args.axiom_geodesic_first)
            kprintf("[GD] Generating %d tokens (geodesic-first)...\n\n", args.max_tokens);
        else
            kprintf("[GD] Generating %d tokens...\n\n", args.max_tokens);

        static char output[65536];
        uint64_t gen_t0 = hal_timer_us();
        int n;
        int geodesic_fallback_used = 0;
        if (args.ott_speculative) {
            n = geodesic_speculative_generate_text(prompt, args.max_tokens,
                                                   args.ott_speculative_thresh,
                                                   args.ott_speculative_batch,
                                                   args.no_verifier,
                                                   args.ott_perfect,
                                                   args.od_swarm_k,
                                                   output, (int)sizeof(output));
            if (n < 0) {
                kprintf("[SPEC] Speculative path failed, falling back to standard decode.\n");
                geodesic_fallback_used = 1;
                n = llm_prompt_n(prompt, output, (int)sizeof(output), args.max_tokens);
            }
        } else if (args.axiom_geodesic_first) {
            n = geodesic_generate_text(prompt, args.max_tokens,
                                       output, (int)sizeof(output));
            if (n < 0) {
                kprintf("[GD] Geodesic-first path unavailable, falling back to standard decode.\n");
                geodesic_fallback_used = 1;
                n = llm_prompt_n(prompt, output, (int)sizeof(output), args.max_tokens);
            }
        } else {
            n = llm_prompt_n(prompt, output, (int)sizeof(output), args.max_tokens);
        }
        uint64_t gen_t1 = hal_timer_us();
        if (n > 0) {
            kprintf("%s\n", output);
            uint64_t total_ms = (gen_t1 - gen_t0) / 1000;
            float tok_per_s = total_ms > 0 ? (float)n * 1000.0f / (float)total_ms : 0.0f;
            float decode_tok_s = llm_last_tok_per_sec();
            float prefill_ms = llm_last_prefill_ms();
            kprintf("\n[GD] %d tokens in %llu ms (%.1f tok/s)\n",
                    n, (unsigned long long)total_ms, tok_per_s);
            if (decode_tok_s > 0.0f) {
                kprintf("[GD] Decode-only: prefill %.1f ms, %.1f tok/s\n",
                        prefill_ms, decode_tok_s);
            }
            /* ── TpF (Tokens per FLOP) metrics ──────────────────────────────
             * TpF is a hardware-agnostic efficiency metric.  For a dense
             * decoder-only transformer each decode token costs exactly 2N FLOPs
             * (the 2N identity) where N = number of non-embedding parameters.
             * TpF_compute = T / Φ;  ideal = 1 / (2N).
             * ε_c = TpF_compute / TpF_compute_ideal = 2N·T/Φ  (= MFU when
             *   Φ = peak_FLOP/s × wall_time, i.e. hardware is fully saturated).
             * ε_m = N·b_p·T/B  where b_p = bytes per param, B = total weight
             *   bytes streamed per token.  At batch=1, B/token = N·b_p →
             *   ε_m ≈ 1.0 (we stream each weight exactly once per token).
             * η_tok = ε_c^α · ε_m^(1-α).  Decode at batch=1 is memory-bound
             *   so α ≈ 0.0 and η_tok ≈ ε_m.
             *
             * Hardware peaks (RTX 4070 Laptop, 7001 MHz GDDR6X @ 192-bit):
             *   peak_compute_flops = 40e12 (FP16, tensor cores)
             *   peak_hbm_bw_bps    = 336e9 (192-bit × 7001 MHz × 2)
             *
             * Reference: https://github.com/NagusameCS/TpF
             */
            {
                extern uint64_t llm_param_count(void);
                uint64_t N = llm_param_count();
                if (N > 0 && decode_tok_s > 0.0f) {
                    const llm_model_t *tpf_m = llm_get_model();
                    /* b_p = data_size / N (average bytes per parameter) */
                    double b_p = tpf_m ? ((double)tpf_m->data_size / (double)N) : 0.5;
                    /* FLOPs per decode token = 2N (2N identity, §2.5 of TpF paper).
                     * Context-dependent attention term 4Lc_t·d is negligible at short
                     * context (Gemma4 E2B: L=35, d=1536, c_t≈50  → 4×35×50×1536/2N ≈ 3.6%). */
                    double flops_per_tok = 2.0 * (double)N;

                    /* TpF_compute = 1/(2N) tokens/FLOP  (hardware-agnostic ideal) */
                    double tpf_compute_ideal = 1.0 / flops_per_tok;
                    /* TpF_memory  = 1/(N·b_p) tokens/byte (hardware-agnostic ideal) */
                    double tpf_memory_ideal  = 1.0 / ((double)N * b_p);

                    /* Hardware peak rates for this GPU (RTX 4070 Laptop).
                     * FP16 tensor-core peak: 40 TFLOPS.
                     * HBM bandwidth: 192-bit bus × 7001 MHz × 2 (DDR) = 336 GB/s. */
                    const double peak_compute_flops = 40e12;  /* FP16 TFLOPS */
                    const double peak_hbm_bw        = 336e9;  /* GB/s */

                    /* ε_c = ratio of actual compute throughput to hardware peak.
                     * ε_c = (tok/s × FLOPs/tok) / peak_compute_FLOP/s
                     *      = TpF_compute_actual × peak_compute / 1.0 */
                    double actual_tflops = (double)decode_tok_s * flops_per_tok;
                    double eps_c = actual_tflops / peak_compute_flops; /* ε_c ∈ (0,1] */

                    /* ε_m = ratio of actual HBM BW used to hardware peak.
                     * At batch=1, we stream N×b_p bytes per token (weights loaded once).
                     * ε_m = (tok/s × N×b_p) / peak_hbm_bw */
                    double model_bytes = (double)N * b_p;
                    double actual_bw   = (double)decode_tok_s * model_bytes;
                    double eps_m = actual_bw / peak_hbm_bw; /* ε_m ∈ (0,1] */

                    /* α: fraction of time in compute-bound phases.
                     * At batch=1 decode, the arithmetic intensity I = 2N/(N×b_p) = 2/b_p
                     * = 2/1.245 ≈ 1.6 FLOP/byte.  The ridge point of this GPU (40T/336G)
                     * ≈ 119 FLOP/byte.  Since I << ridge_point, we are deep memory-bound.
                     * α ≈ 0.0  → η_tok = ε_c^0 × ε_m^1 = ε_m. */
                    double eta_tok = eps_m;

                    kprintf("[TpF] N=%.0fM  b_p=%.3f B/param  model=%.0fMB\n",
                            (double)N / 1e6, b_p,
                            model_bytes / (1024.0 * 1024.0));
                    kprintf("[TpF] ideal: TpF_c=%.3e tok/FLOP  TpF_m=%.3e tok/B\n",
                            tpf_compute_ideal, tpf_memory_ideal);
                    kprintf("[TpF] @ %.1f tok/s: %.2f GFLOPS (%.2f%% of %.0f TFLOPS peak)\n",
                            decode_tok_s,
                            actual_tflops / 1e9,
                            eps_c * 100.0, peak_compute_flops / 1e12);
                    kprintf("[TpF] @ %.1f tok/s: %.1f GB/s HBM (%.2f%% of %.0f GB/s peak)  "
                            "eta_tok=%.3f\n",
                            decode_tok_s, actual_bw / 1e9,
                            eps_m * 100.0, peak_hbm_bw / 1e9,
                            eta_tok);
                }
            }

            if (args.ott_ready_report_path) {
                FILE *rf = fopen(args.ott_ready_report_path, "wb");
                if (rf) {
                    int runtime_hits = GD_geodesic_last_runtime_hits;
                    int local_hits = GD_geodesic_last_local_hits;
                    int runtime_rejects = GD_geodesic_last_runtime_rejects;
                    int quality_pass = GD_geodesic_last_quality_gate_pass;
                    int geo_generated = GD_geodesic_last_generated;
                    int fallback_req = GD_geodesic_last_fallback_requested || geodesic_fallback_used;
                    int total_geo = runtime_hits + local_hits;
                    double runtime_share = (total_geo > 0) ? ((double)runtime_hits / (double)total_geo) : 0.0;
                    double runtime_reject_rate = (runtime_hits + runtime_rejects > 0)
                        ? ((double)runtime_rejects / (double)(runtime_hits + runtime_rejects))
                        : 0.0;
                    /* Minimum runtime participation to be considered "geodesic_ready" */
                    const double READY_RUNTIME_SHARE  = 0.08;  /* ≥8% tokens from runtime path */
                    const double DEGRADED_RUNTIME_SHARE = 0.01; /* ≥1% → degraded (not zero) */
                    int ready = 1;
                    int hybrid_ready = 1;
                    int degraded = 0; /* quality OK but runtime share too low */
                    if (args.axiom_geodesic_first && fallback_req) ready = 0;
                    if (args.axiom_geodesic_first && runtime_share < READY_RUNTIME_SHARE) ready = 0;
                    if (args.axiom_geodesic_first && runtime_share < DEGRADED_RUNTIME_SHARE && quality_pass && !fallback_req) degraded = 1;
                    if (args.axiom_beta_run && axiom_consistency < 0.85f) ready = 0;
                    if (args.axiom_beta_run && axiom_consistency < 0.80f) hybrid_ready = 0;
                    if (n <= 0) hybrid_ready = 0;

                    fprintf(rf, "{\n");
                    fprintf(rf, "  \"ott_full\": %s,\n", args.ott_full ? "true" : "false");
                    fprintf(rf, "  \"ott_theorem\": %s,\n", args.ott_theorem ? "true" : "false");
                    fprintf(rf, "  \"ott_perfect\": %s,\n", args.ott_perfect ? "true" : "false");
                    fprintf(rf, "  \"ready\": %s,\n", ready ? "true" : "false");
                    fprintf(rf, "  \"hybrid_ready\": %s,\n", hybrid_ready ? "true" : "false");
                    /* Status ladder: geodesic_ready > hybrid_ready > degraded_geodesic > not_ready */
                    const char *status_str;
                    if (ready) status_str = "geodesic_ready";
                    else if (degraded) status_str = "degraded_geodesic";
                    else if (hybrid_ready) status_str = "hybrid_ready";
                    else status_str = "not_ready";
                    fprintf(rf, "  \"degraded_geodesic\": %s,\n", degraded ? "true" : "false");
                    fprintf(rf, "  \"status\": \"%s\",\n", status_str);
                    fprintf(rf, "  \"model\": \"%s\",\n", llm_model_name());
                    fprintf(rf, "  \"generation\": {\n");
                    fprintf(rf, "    \"tokens\": %d,\n", n);
                    fprintf(rf, "    \"total_ms\": %llu,\n", (unsigned long long)total_ms);
                    fprintf(rf, "    \"tok_per_s\": %.2f,\n", tok_per_s);
                    fprintf(rf, "    \"prefill_ms\": %.2f,\n", prefill_ms);
                    fprintf(rf, "    \"decode_tok_per_s\": %.2f\n", decode_tok_s);
                    fprintf(rf, "  },\n");
                    fprintf(rf, "  \"geodesic\": {\n");
                    fprintf(rf, "    \"enabled\": %s,\n", args.axiom_geodesic_first ? "true" : "false");
                    fprintf(rf, "    \"runtime_tokens\": %d,\n", runtime_hits);
                    fprintf(rf, "    \"local_tokens\": %d,\n", local_hits);
                    fprintf(rf, "    \"runtime_rejects\": %d,\n", runtime_rejects);
                    fprintf(rf, "    \"runtime_share\": %.4f,\n", runtime_share);
                    fprintf(rf, "    \"runtime_reject_rate\": %.4f,\n", runtime_reject_rate);
                    fprintf(rf, "    \"quality_gate_pass\": %s,\n", quality_pass ? "true" : "false");
                    fprintf(rf, "    \"generated_tokens\": %d,\n", geo_generated);
                    fprintf(rf, "    \"fallback_used\": %s\n", fallback_req ? "true" : "false");
                    fprintf(rf, "  },\n");
                    fprintf(rf, "  \"axiom\": {\n");
                    fprintf(rf, "    \"ran\": %s,\n", axiom_ran_this_invocation ? "true" : "false");
                    fprintf(rf, "    \"intrinsic_dim\": %d,\n", axiom_intrinsic_dim);
                    fprintf(rf, "    \"consistency\": %.4f,\n", axiom_consistency);
                    fprintf(rf, "    \"projected_speedup\": %.2f,\n", axiom_geodesic_speedup);
                    fprintf(rf, "    \"total_ms\": %llu\n", axiom_total_ms);
                    fprintf(rf, "  },\n");
                    fprintf(rf, "  \"attnres\": {\n");
                    fprintf(rf, "    \"enabled\": %s,\n", args.attnres_enable ? "true" : "false");
                    fprintf(rf, "    \"strength\": %.3f\n", args.attnres_strength);
                    fprintf(rf, "  },\n");
                    fprintf(rf, "  \"depth_attention\": {\n");
                    fprintf(rf, "    \"enabled\": %s,\n", args.depth_attn_enable ? "true" : "false");
                    fprintf(rf, "    \"strength\": %.3f,\n", args.depth_attn_strength);
                    fprintf(rf, "    \"window\": %d\n", args.depth_attn_window);
                    fprintf(rf, "  }\n");
                    fprintf(rf, "}\n");
                    fclose(rf);
                    kprintf("[GD] OTT readiness report: %s\n", args.ott_ready_report_path);
                } else {
                    kprintf("[GD] WARNING: Could not write OTT readiness report: %s\n", args.ott_ready_report_path);
                }
            }
        } else {
            kprintf("[error generating response]\n");
        }
    }

    /* Cleanup */
    hal_munmap(&model);
    hal_shutdown();

    return 0;
}


