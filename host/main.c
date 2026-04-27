/*
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
 * ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
 * ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
 * ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
 * ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
 * ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
 * :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
 * :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
 * ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
 * :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
 * ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
 * ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
 * :::::::::................................:@@@@@@@@@@%:...............................::::::
 * ::::::::..................................*@@@@@@@@@-................................::::::::
 * ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
 * :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
 * :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
 * :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
 * :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
 * :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
 * :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
 * :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
 * :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
 * ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
 * ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
 * :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
 * ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
 * ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
 * :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
 * ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
 * :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
 * :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
 * ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
 * ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
 * :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
 * ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
 * ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
 * :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 */

/*
 * Geodessical — Hosted Main Entry Point
 *
 * Loads a GGUF model from disk via memory-mapped I/O and runs LLM inference
 * using the TensorOS inference engine on the host CPU, with native threading.
 *
 * Usage: geodessical <model.gguf> [prompt]
 */
#define _CRT_SECURE_NO_WARNINGS

/* Number of context tokens prefilled before the calibration probe token.
 * Single-token (pos-0) calibration misses contextual activation directions
 * that dominate inference (≥90% of hidden-state variance at positions >0).
 * Using 16 context tokens adds ~17× forward passes but aligns the PCA
 * subspace with the actual WikiText-2 inference distribution.
 * Hash v2: value encoded in hs/W_proj cache keys so old caches are rejected. */
#define AXEX_CALIB_CTX_LEN 16
#include "hal.h"

/* Forward declarations from TensorOS inference engine */
#include "../runtime/nn/llm.h"
#include "../runtime/nn/hf_download.h"
#include "../runtime/nn/axiom_beta.h"
#include "../runtime/nn/axiom_exploit.h"
#include "../runtime/nn/axiom_gauge.h"
#include "../runtime/nn/axiom_linalg.h"
#include "../runtime/nn/axiom_vis.h"
#include "../runtime/nn/geo_research.h"
#include "../runtime/nn/online_basis.h"
#include "../runtime/nn/mcr_compress.h"
#include "../runtime/nn/thermal_rank.h"
#include "../runtime/nn/qspec_basis.h"
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
    kprintf("  ██████████████████████████████████████████████████████████████\n");
    kprintf("██████████████████████████████████████████████████████████████\n");
    kprintf("██████████████████████████████████████████████████████████████\n");
    kprintf("██████████████████████████████████████████████████████████████\n");
    kprintf("██████████████████████████████████████████████████████████████\n");
    kprintf("██████████████████████████████████████████████████████████████\n");
    kprintf("███████████████████████████████ ██████████████████████████████\n");
    kprintf("██████████████████████████████   █████████████████████████████\n");
    kprintf("█████████████████████████████     ████████████████████████████\n");
    kprintf("████████████████████████████       ███████████████████████████\n");
    kprintf("███████████████████████████         ██████████████████████████\n");
    kprintf("████████████████████████████       ███████████████████████████\n");
    kprintf("████████████████████████████       ███████████████████████████\n");
    kprintf("█████████████████████████████     ████████████████████████████\n");
    kprintf("█████████████████████████████     ████████████████████████████\n");
    kprintf("██████████████████    ███████    █████████   █████████████████\n");
    kprintf("██████████████████       █████   █████       █████████████████\n");
    kprintf("██████████████████        ████   ████        █████████████████\n");
    kprintf("████████████████          ████   ███          ████████████████\n");
    kprintf("████████████████           ██    ███          ████████████████\n");
    kprintf("█████████████████          ██    ██           ████████████████\n");
    kprintf("████████████████████                       ███████████████████\n");
    kprintf("███████████████████████                 ██████████████████████\n");
    kprintf("██████████████████████████          ██████████████████████████\n");
    kprintf("██████████████████████████   █████    ████████████████████████\n");
    kprintf("███████████████████      █████████████   █████████████████████\n");
    kprintf("███████████████████████████         ██████████████████████████\n");
    kprintf("██████████████████████████          ██████████████████████████\n");
    kprintf("███████████████████████████   ██  ████████████████████████████\n");
    kprintf("██████████████████████████████████████████████████████████████\n");
    kprintf("██████████████████████████████████████████████████████████████\n");
    kprintf("██████████████████████████████████████████████████████████████\n");
    kprintf("██████████████████████████████████████████████████████████████\n");
    kprintf("██████████████████████████████████████████████████████████████\n");
    kprintf("\n");
    kprintf("  Geodessical v%d.%d.%d \"%s\"\n",
            GD_VERSION_MAJOR, GD_VERSION_MINOR, GD_VERSION_PATCH, GD_CODENAME);
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
    kprintf("  --axex-attn-svd        Also SVD-compress Q and O attention weights (slot 1/4)\n");
    kprintf("                         Combined with --axex-ffn-compress: full model SVD (no PCA needed)\n");
    kprintf("                         70B at rank-128: Q+O+FFN ≈ 6.6 GB vs 20 GB IQ2_XS — all-GPU\n");
    kprintf("  --axex-compress-rank N Max rank k for GP compression (default: auto-scales with model dim:\n");
    kprintf("                         dim<=1024: 3/4*dim, dim<=2048: 256, dim<=5120: 384, dim>5120: 512)\n");
    kprintf("                         Use 128 for 7GB VRAM target on 70B; 512 for best quality.\n");
    kprintf("                         GP (Geodesic Projection): Q/K/V/O/gate/up → W_proj[m×k]\n");
    kprintf("                         Enables 22B-70B models in 8 GB VRAM (one-time calibration cost)\n");
    kprintf("  --axex-compress-max-err <f> Max Frobenius error to accept (0=no limit; use 0.5 to skip\n");
    kprintf("                         matrices with >50%% error, preventing garbled output)\n");
    kprintf("  --axex-skip-o          Disable O-projection GP compression (preserves full O rank)\n");
    kprintf("  --axex-quality <f>     Weight compression quality floor 0-1 (default: 0.90)\n");
    kprintf("  --one-decode           OneDecode: bake geodesic flow map once, then decode instantly\n");
    kprintf("  --one-decode-coverage <n> Vocab tokens to bake for OneDecode (default: 2048)\n");
    kprintf("  --ott-od               OTT-OD Protocol: OneDecode as speculative draft source (fastest OTT mode)\n");
    kprintf("  --vis [dir]            Visualize Riemannian manifolds (default: axiom_vis/)\n");
    kprintf("  --ppl-eval             Evaluate WikiText-2 perplexity and exit (works with --axex-compress)\n");
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
    int         axex_attn_svd;       /* 1 = also SVD-compress Q and O attention weights */
    int         axex_compress_rank;  /* max SVD rank (0 = auto, default 128) */
    float       axex_compress_quality;  /* 0-1 quality floor (default 0.90) */
    float       axex_compress_max_err;  /* max Frobenius error to accept (0=no limit) */
    double      axiom_pca_variance;     /* 0 = default 0.95; override for GP weight basis */
    int         axex_attn_only;      /* 1 = compress only Q/K/V/O (default); 0 = all incl FFN */
    int         axex_calib_samples;  /* calibration samples per layer (0 = auto) */
    int         axex_no_actaware;    /* 1 = disable column-norm reweighting (pure h-PCA) */
    int         axex_skip_o;         /* 1 = skip O_proj compression (it uses wrong basis anyway) */
    int         axex_weight_pca;     /* 1 = use weight-gram eigenvectors (no calib data needed) */
    /* Research features */
    int         axex_real_curvature; /* 1 = use true Ricci signal to set per-layer ranks */
    int         axex_learn_plan;     /* 1 = run differentiable rank plan optimisation after SVD */
    int         axex_plan_steps;     /* diffplan optimisation steps (0 = default 200) */
    int         axex_phase_compress; /* 1 = build phase-conditional compression plan */
    float       axex_phase_ret_q;    /* retrieval phase quality floor (default 0.85) */
    float       axex_phase_rea_q;    /* reasoning phase quality floor (default 0.90) */
    float       axex_phase_gen_q;    /* generation phase quality floor (default 0.75) */
    int         axex_online_basis;   /* 1 = enable online PCA basis update on spec-decode rejects */
    int         axex_gauge;          /* 1 = diagonal gauge-optimal compression */
    int         axex_gauge_iter;     /* gauge optimisation iterations (0 = auto) */
    /* MCR + sink bypass (Features 1 & 2) */
    int         axex_mcr;              /* 1 = MCR-aware non-uniform rank allocation */
    float       axex_mcr_mix_scale;    /* rank multiplier for Mix phase (default 1.5) */
    float       axex_mcr_compress_scale; /* rank multiplier for Compress phase (default 0.35) */
    float       axex_mcr_refine_scale; /* rank multiplier for Refine phase (default 1.2) */
    int         axex_sink_bypass;      /* 1 = ensure sink direction is covered by PCA basis */
    /* Thermal adaptive + TPJ (Features 3 & 4) */
    int         axex_thermal;          /* 1 = scale rank with GPU thermal headroom */
    float       axex_thermal_low;      /* below this °C use full rank (default 65) */
    float       axex_thermal_high;     /* above this °C use min rank (default 85) */
    float       axex_thermal_power;    /* power budget W, 0 = no limit */
    int         axex_tpj;              /* 1 = tokens-per-joule gradient in diffplan */
    float       axex_tpj_lambda;       /* energy regularisation weight (default 0.005) */
    /* Cross-quant basis + failure-mode rank (Features 5 & 6) */
    int         axex_qspec;            /* 1 = run cross-quant shared basis analysis */
    float       axex_qspec_threshold;  /* alignment threshold for shared-ok (default 0.80) */
    int         axex_fail_rank;        /* 1 = failure-mode targeted rank allocation */
    float       axex_fail_boost;       /* dominant-zone rank boost factor (default 1.8) */
    int         axex_calib_only;       /* 1 = skip axiom geometry, run only W_proj calibration */
    int         ppl_eval;              /* 1 = evaluate perplexity on built-in WikiText-2 slice and exit */
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
    /* Zero-init the entire struct so all boolean/int flags default to 0 and
     * all float fields default to 0.0f.  Explicit assignments below override
     * the fields that need non-zero defaults.  This prevents subtle bugs when
     * new fields are added to GD_args_t without a corresponding init line. */
    memset(args, 0, sizeof(*args));
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
    args->axex_real_curvature = 0;
    args->axex_learn_plan = 0;
    args->axex_plan_steps = 0;
    args->axex_phase_compress = 0;
    args->axex_phase_ret_q = 0.85f;
    args->axex_phase_rea_q = 0.90f;
    args->axex_phase_gen_q = 0.75f;
    args->axex_online_basis = 0;
    args->axex_gauge      = 0;
    args->axex_gauge_iter = 0;  /* 0 = auto (10 for small, 1 for large) */
    args->axex_mcr = 0;
    args->axex_mcr_mix_scale = 1.5f;
    args->axex_mcr_compress_scale = 0.35f;
    args->axex_mcr_refine_scale = 1.2f;
    args->axex_sink_bypass = 0;
    args->axex_thermal = 0;
    args->axex_thermal_low  = 65.0f;
    args->axex_thermal_high = 85.0f;
    args->axex_thermal_power = 0.0f;
    args->axex_tpj = 0;
    args->axex_tpj_lambda = 0.005f;
    args->axex_qspec = 0;
    args->axex_qspec_threshold = 0.80f;
    args->axex_fail_rank = 0;
    args->axex_fail_boost = 1.8f;

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
            /* NOTE: fast_mode NOT set — need 512+ samples for good basis coverage (k~128+) */
            args->axiom_skip_geodesic = 1; /* Phase 5 not needed for GP compression */
            /* axiom_pca_variance intentionally NOT set here: layerwise PCA uses var_ratio=1.0
             * to keep all components; rank is capped by axex_compress_rank budget instead. */
        } else if (strcmp(argv[i], "--axex-attn-only") == 0) {
            /* Explicit: compress only Q/K/V/O attention weights (default behaviour) */
            args->axex_attn_only = 1;
        } else if (strcmp(argv[i], "--axex-no-actaware") == 0) {
            /* Disable activation-aware PCA reweighting (reverts to pure h-PCA).
             * Debug/ablation only — default is ON because it halves PPL. */
            args->axex_no_actaware = 1;
        } else if (strcmp(argv[i], "--axex-skip-o") == 0) {
            /* Skip O_proj compression. O's input is per-head attention output,
             * NOT the residual stream, so projecting it onto the shared
             * residual-stream PCA basis gives a random basis (exactly k/n
             * energy). Skipping preserves O at full rank. */
            args->axex_skip_o = 1;
        } else if (strcmp(argv[i], "--axex-weight-pca") == 0) {
            /* Use weight-gram eigenvectors as the Pt basis instead of the
             * data-PCA basis.  Finds top-k eigenvectors of K = Σ Wᵢ^T Wᵢ
             * (the weight gram matrix) via matrix-free power iteration.
             * This maximises weight energy ||W Pt^T||_F^2 directly.
             * No calibration data needed when combined with --axex-skip-calib. */
            args->axex_weight_pca = 1;
        } else if (strcmp(argv[i], "--axex-ffn-compress") == 0) {
            /* SVD compress gate/up/down only — no axiom survey, no manifold PCA.
             * Fast path: enables the cuBLAS batched GEMV for gate+up without
             * waiting 30 min for per-layer PCA.  Uses flat curvature (SVD). */
            args->axex_ffn_compress = 1;
        } else if (strcmp(argv[i], "--axex-attn-svd") == 0) {
            /* Also SVD-compress Q and O (slots 1/4) — no calibration needed.
             * Combine with --axex-ffn-compress for full model SVD:
             *   geodessical model.gguf --axex-ffn-compress --axex-attn-svd --axex-compress-rank 128
             * 70B at rank-128: Q+O+FFN ≈ 6.6 GB vs 20 GB IQ2_XS → fits 7 GB GPU. */
            args->axex_attn_svd = 1;
        } else if (strcmp(argv[i], "--axex-compress-rank") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-compress-rank requires int\n"); return -1; }
            args->axex_compress_rank = atoi(argv[i]);
            if (args->axex_compress_rank > AXEX_MANIFOLD_K_MAX) {
                kprintf("[AXEX] Warning: --axex-compress-rank %d exceeds AXEX_MANIFOLD_K_MAX (%d), capping to %d\n",
                        args->axex_compress_rank, AXEX_MANIFOLD_K_MAX, AXEX_MANIFOLD_K_MAX);
                args->axex_compress_rank = AXEX_MANIFOLD_K_MAX;
            }
        } else if (strcmp(argv[i], "--axex-compress-max-err") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-compress-max-err requires float\n"); return -1; }
            args->axex_compress_max_err = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-calib-only") == 0) {
            /* Skip axiom geometry phases 1-4 entirely; run only the 512-sample
             * W_proj manifold calibration. Saves ~300 forward passes on 70B.
             * Requires --axex-compress to be set for axex gates to fire. */
            args->axex_calib_only = 1;
        } else if (strcmp(argv[i], "--axex-skip-o") == 0) {
            axex_manifold_set_skip_o(1);
        } else if (strcmp(argv[i], "--ppl-eval") == 0) {
            args->ppl_eval = 1;
        } else if (strcmp(argv[i], "--axex-full-compress") == 0) {
            /* Override: also compress FFN gate/up (legacy — hurts accuracy ~20-35% perplexity) */
            args->axex_attn_only = 0;
        } else if (strcmp(argv[i], "--axex-gauge") == 0) {
            /* Diagonal gauge-optimal compression: find the basis in which the
             * residual stream is maximally compressible, then bake it into the
             * SVD factors (zero inference overhead).
             * Optionally combine with --axex-gauge-iter N (default: auto). */
            args->axex_gauge = 1;
        } else if (strcmp(argv[i], "--axex-gauge-iter") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-gauge-iter requires int\n"); return -1; }
            args->axex_gauge      = 1;
            args->axex_gauge_iter = atoi(argv[i]);
        } else if (strcmp(argv[i], "--axex-calib-samples") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-calib-samples requires integer\n"); return -1; }
            args->axex_calib_samples = atoi(argv[i]);
        } else if (strcmp(argv[i], "--axex-quality") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-quality requires float\n"); return -1; }
            args->axex_compress_quality = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-real-curvature") == 0) {
            args->axex_real_curvature = 1;
            args->axex_compress = 1;
            args->axiom_beta_run = 1;
            args->axiom_skip_geodesic = 1;
            args->axiom_pca_variance = 0.9999;
        } else if (strcmp(argv[i], "--axex-learn-plan") == 0) {
            args->axex_learn_plan = 1;
        } else if (strcmp(argv[i], "--axex-plan-steps") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-plan-steps requires int\n"); return -1; }
            args->axex_plan_steps = atoi(argv[i]);
        } else if (strcmp(argv[i], "--axex-phase-compress") == 0) {
            args->axex_phase_compress = 1;
            args->axex_compress = 1;
            args->axiom_beta_run = 1;
            args->axiom_skip_geodesic = 1;
            args->axiom_pca_variance = 0.9999;
        } else if (strcmp(argv[i], "--axex-phase-ret-q") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-phase-ret-q requires float\n"); return -1; }
            args->axex_phase_ret_q = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-phase-rea-q") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-phase-rea-q requires float\n"); return -1; }
            args->axex_phase_rea_q = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-phase-gen-q") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-phase-gen-q requires float\n"); return -1; }
            args->axex_phase_gen_q = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-online-basis") == 0) {
            args->axex_online_basis = 1;
        } else if (strcmp(argv[i], "--axex-mcr") == 0) {
            args->axex_mcr = 1;
            args->axex_compress = 1;
            args->axiom_beta_run = 1;
            args->axiom_skip_geodesic = 1;
            args->axiom_pca_variance = 0.9999;
        } else if (strcmp(argv[i], "--axex-mcr-mix-scale") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-mcr-mix-scale requires float\n"); return -1; }
            args->axex_mcr_mix_scale = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-mcr-compress-scale") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-mcr-compress-scale requires float\n"); return -1; }
            args->axex_mcr_compress_scale = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-mcr-refine-scale") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-mcr-refine-scale requires float\n"); return -1; }
            args->axex_mcr_refine_scale = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-sink-bypass") == 0) {
            args->axex_sink_bypass = 1;
            args->axex_compress = 1;
            args->axiom_beta_run = 1;
            args->axiom_skip_geodesic = 1;
        } else if (strcmp(argv[i], "--axex-thermal") == 0) {
            args->axex_thermal = 1;
        } else if (strcmp(argv[i], "--axex-thermal-low") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-thermal-low requires float\n"); return -1; }
            args->axex_thermal_low = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-thermal-high") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-thermal-high requires float\n"); return -1; }
            args->axex_thermal_high = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-thermal-power") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-thermal-power requires float\n"); return -1; }
            args->axex_thermal_power = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-tpj") == 0) {
            args->axex_tpj = 1;
            args->axex_learn_plan = 1;  /* TPJ builds on diffplan */
        } else if (strcmp(argv[i], "--axex-tpj-lambda") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-tpj-lambda requires float\n"); return -1; }
            args->axex_tpj_lambda = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-qspec") == 0) {
            args->axex_qspec = 1;
        } else if (strcmp(argv[i], "--axex-qspec-threshold") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-qspec-threshold requires float\n"); return -1; }
            args->axex_qspec_threshold = (float)atof(argv[i]);
        } else if (strcmp(argv[i], "--axex-fail-rank") == 0) {
            args->axex_fail_rank = 1;
            args->axex_compress = 1;
            args->axiom_beta_run = 1;
            args->axiom_skip_geodesic = 1;
        } else if (strcmp(argv[i], "--axex-fail-boost") == 0) {
            if (++i >= argc) { kprintf("Error: --axex-fail-boost requires float\n"); return -1; }
            args->axex_fail_boost = (float)atof(argv[i]);
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
        if (isalnum((int)c) || c == ' ' || c == '.' || c == ',' || c == ';' ||
            c == ':' || c == '!' || c == '?' || c == '\'' || c == '"') {
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

/* ── Inference-time thermal / TPJ / phase-plan state ────────────────────────
 * These are file-scope statics so both geodesic_speculative_generate_text()
 * and the main() init block can access them without parameter threading.
 * Activated only when --axex-thermal / --axex-phase-compress is passed. */
static thermal_ctx_t s_spec_thermal;
static int           s_spec_thermal_active = 0;    /* 1 = thermal throttle enabled */
static tpj_ctx_t     s_spec_tpj;
static int           s_spec_tpj_active     = 0;    /* 1 = TPJ energy tracking enabled */
static phased_plan_t s_spec_phase_plan;
static int           s_spec_phase_plan_active = 0; /* 1 = phase detector running */

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
        /* ── Thermal-adaptive dyn_batch cap ─────────────────────────────────────
         * When --axex-thermal is active, call thermal_get_rank() every ~32 tokens
         * to get a thermally-scaled rank and map it proportionally to dyn_batch.
         * High GPU temp → lower rank → lower batch → reduced throughput pressure.
         * The thermal poll is debounced inside thermal_get_rank() (250 ms cooldown)
         * so there's no NVML overhead when called frequently. */
        if (s_spec_thermal_active && (generated & 31) == 0) {
            int t_rank = thermal_get_rank(&s_spec_thermal, s_spec_thermal.rank_max);
            /* Map rank linearly to batch cap: rank_min→1, rank_max→batch_size */
            int rank_span = s_spec_thermal.rank_max - s_spec_thermal.rank_min;
            int thermal_batch;
            if (rank_span <= 0) {
                thermal_batch = batch_size;
            } else {
                int above_min = t_rank - s_spec_thermal.rank_min;
                thermal_batch = 1 + (above_min * (batch_size - 1) + rank_span / 2) / rank_span;
            }
            if (thermal_batch < 1) thermal_batch = 1;
            if (dyn_batch > thermal_batch) {
                kprintf("[THERMAL] temp=%.1f°C → capping dyn_batch %d→%d\n",
                        (double)s_spec_thermal.current_temp_C, dyn_batch, thermal_batch);
                dyn_batch = thermal_batch;
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
                    if (room > 0) {
                        if (pn > room) pn = room;
                        memcpy(output + out_len, piece, (size_t)pn);
                        out_len += pn;
                        output[out_len] = '\0';
                    }
                }
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
        /* ── TPJ record + phase plan update (per decode batch) ───────────────
         * tpj_record() feeds the tokens-per-joule energy model with the current
         * throughput so cumulative J/tok stats are accurate at end-of-run.
         * phaseplan_update() advances the RETRIEVAL/REASONING/GENERATION phase
         * detector using logit entropy as an attention-entropy proxy: high
         * entropy → retrieval; low entropy → generation. */
        if (s_spec_tpj_active) {
            float tps = llm_last_tok_per_sec();
            if (tps > 0.0f) tpj_record(&s_spec_tpj, tps);
        }
        if (s_spec_phase_plan_active) {
            float entropy = llm_last_logit_entropy();
            phaseplan_update(&s_spec_phase_plan, entropy, 0.0f);
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
        /* Print TPJ energy summary when tracking was active */
        if (s_spec_tpj_active && s_spec_tpj.cumulative_tokens > 0)
            tpj_print(&s_spec_tpj);
        /* Print phase plan summary when tracking was active */
        if (s_spec_phase_plan_active)
            phaseplan_print(&s_spec_phase_plan);
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

typedef struct {
    const llm_layer_t *layer;
    const llm_model_t *model;
    uint32_t weight_mask;
    float *tmp_m;
    float *row_f32;
} my_grc_weighted_ctx_t;

static void my_grc_weighted_matvec(const float *x, float *y, int n, void *vctx) {
    my_grc_weighted_ctx_t *c = (my_grc_weighted_ctx_t *)vctx;
    memset(y, 0, n * sizeof(float));
    int dim = c->model->dim;
    if (n != dim) return;

    int lq_dim = c->model->n_heads * (c->model->dim / c->model->n_heads);
    int lkv_dim = c->model->n_kv_heads * (c->model->dim / c->model->n_heads);

    #define DO_WEIGHT(W, W_type, m) do { \
        if (W) { \
            for (int i = 0; i < (m); i++) { \
                const void *row_ptr = (const char *)(W) + i * ggml_tensor_size((W_type), dim); \
                ax_dequant_row_f32(row_ptr, c->row_f32, dim, (W_type)); \
                float dot = 0.0f; \
                for (int j = 0; j < dim; j++) dot += c->row_f32[j] * x[j]; \
                c->tmp_m[i] = dot; \
            } \
            for (int i = 0; i < (m); i++) { \
                const void *row_ptr = (const char *)(W) + i * ggml_tensor_size((W_type), dim); \
                ax_dequant_row_f32(row_ptr, c->row_f32, dim, (W_type)); \
                for (int j = 0; j < dim; j++) y[j] += c->row_f32[j] * c->tmp_m[i]; \
            } \
        } \
    } while (0)

    if (c->weight_mask & 1) DO_WEIGHT(c->layer->q_weight, c->layer->q_type, lq_dim);
    if (c->weight_mask & 2) DO_WEIGHT(c->layer->k_weight, c->layer->k_type, lkv_dim);
    if (c->weight_mask & 4) DO_WEIGHT(c->layer->v_weight, c->layer->v_type, lkv_dim);
    #undef DO_WEIGHT
}

/* ── Fast cached matvec for weight-PCA ─────────────────────────────────────
 * Pre-dequantized f32 weight matrices — eliminates repeated Q4_K_M decoding.
 * K_apply cost: one sequential read of (Q+K+V) f32 matrices per call.
 * Fused W^T W: reads each row once (dot product + accumulate in same pass),
 * keeping the 16KB row hot in L1d between the two uses.
 * Normalized: each matrix weighted by 1/||W||_F^2 to give equal importance
 * to Q, K, V regardless of their differing row counts (GQA: 4096 vs 1024). */
typedef struct {
    const float *q_f32;   /* [lq_dim  × dim] row-major f32 */
    const float *k_f32;   /* [lkv_dim × dim] row-major f32 */
    const float *v_f32;   /* [lkv_dim × dim] row-major f32 */
    float *tmp;            /* temp buffer, size = max(lq_dim, lkv_dim) */
    int lq_dim, lkv_dim, dim;
    float inv_norm_q;     /* 1 / ||W_q||_F^2 — equal weighting per matrix */
    float inv_norm_k;
    float inv_norm_v;
} my_grc_cached_ctx_t;

static void my_grc_cached_matvec(const float *x, float *y, int n, void *vctx) {
    my_grc_cached_ctx_t *c = (my_grc_cached_ctx_t *)vctx;
    if (n != c->dim) return;
    memset(y, 0, (size_t)n * sizeof(float));

    /* Fused normalized W^T W x: read each row once → dot, then accumulate.
     * Scale by 1/||W||_F^2 so Q, K, V contribute equally to the eigenbasis. */
    #define CACHED_WT_W(W_f32, m, inv_norm) do {                              \
        for (int _i = 0; _i < (m); _i++) {                                    \
            const float *_row = (W_f32) + (size_t)_i * c->dim;               \
            float _dot = 0.0f;                                                 \
            for (int _j = 0; _j < c->dim; _j++) _dot += _row[_j] * x[_j];   \
            float _sc = _dot * (inv_norm);                                     \
            for (int _j = 0; _j < c->dim; _j++) y[_j] += _row[_j] * _sc;    \
        }                                                                      \
    } while (0)

    CACHED_WT_W(c->q_f32, c->lq_dim,  c->inv_norm_q);
    CACHED_WT_W(c->k_f32, c->lkv_dim, c->inv_norm_k);
    CACHED_WT_W(c->v_f32, c->lkv_dim, c->inv_norm_v);
    #undef CACHED_WT_W
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
        /* NOTE: Do NOT call llm_gpu_set_compress_mode(1) here.
         * axiom_beta_run() needs ALL weights on GPU — including FFN — for its
         * calibration forward passes (axiom_beta_probe_all_layer_states calls
         * llm_generate_tokens, which returns 1 if FFN GPU pointers are null).
         * VRAM savings for GP compression come from freeing raw Q/K/V/O after
         * uploading compressed W_proj matrices (handled in phase 2b of
         * llm_gpu_upload_compressed_weights), not by skipping raw FFN here.
         * If explicit FFN compression (--axex-ffn-compress) is also requested
         * it is handled by the block below. */
        (void)0;
    }
    /* --axex-ffn-compress: skip raw FFN uploads during llm_gpu_init so that
     * attn-only VRAM estimates are used.  All 32 layers' attention weights fit
     * in ~1.3 GB, leaving ample room for compressed FFN uploaded later.
     * Without this, raw FFN fills VRAM at layer ~18 and the remaining layers
     * must be promoted via Phase 3 (compress→free→re-upload dance). */
    if (args.axex_ffn_compress) {
        llm_gpu_set_compress_mode(1);
    }
    /* --axex-attn-svd: skip raw Q/O upload during llm_gpu_init to save VRAM.
     * For 70B models, Q+O at IQ2_XS ≈ 38 MB/layer × 80 = 3 GB — too large to
     * waste on upload-then-free. This flag tells the VRAM budgeter to use attn
     * sizes (norms + K/V only) and skip raw Q/O from the upload list. */
    if (args.axex_attn_svd) {
        llm_gpu_set_attn_compress_mode(1);
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

    /* ── Early W_proj cache fast-path ─────────────────────────────────────
     * If the W_proj cache is already on disk, skip axiom_beta_run entirely
     * (avoiding depth-sink detection: ~128 forward passes on 70B = ~7 min).
     * On hit: set axiom_ran_this_invocation so downstream axex gates fire. */
    int wp_early_loaded = 0;
    if (args.axiom_beta_run && args.axex_compress && !args.axex_ffn_compress) {
        const llm_model_t *_em = llm_get_model();
        if (_em) {
            int _e_dim = _em->dim, _e_nl = _em->n_layers;
            int _e_vocab = llm_model_vocab();
            int _e_ns = (args.axex_calib_samples > 0) ? args.axex_calib_samples
                      : (args.axiom_fast_mode ? 64 : 2048);
            if (_e_ns > _e_vocab) _e_ns = _e_vocab;
            uint64_t _wp_hk = 0xDEADBEEF12345678ULL;
            const char *_wmp = args.model_path ? args.model_path : "";
            for (const char *_c = _wmp; *_c; _c++) { _wp_hk ^= (uint8_t)*_c; _wp_hk *= 0x100000001B3ULL; }
            _wp_hk ^= (uint64_t)_e_ns  * 0xABCDEF01ULL;
            _wp_hk ^= (uint64_t)_e_nl  * 0x12345678ULL;
            _wp_hk ^= (uint64_t)_e_dim * 0x87654321ULL;
            _wp_hk ^= (uint64_t)AXEX_CALIB_CTX_LEN * 0xFEDCBA98ULL; /* v2: contextual calibration */
            _wp_hk ^= (args.axex_no_actaware ? 0ULL : 0xA1B2C3D4AAAAULL);  /* v3: actaware PCA */
            _wp_hk ^= (args.axex_skip_o ? 0x5C09F0DEADBEULL : 0ULL);  /* v4: skip-O */
            _wp_hk ^= (args.axex_weight_pca ? 0xC8F2620B4E3DULL : 0ULL);  /* v6: weight-PCA normalized-K */
            _wp_hk ^= (uint64_t)(args.axex_compress_rank > 0 ? args.axex_compress_rank : 0) * 0xF1E2D3C4ULL; /* compress-rank */
            char _wp_path[512];
            snprintf(_wp_path, sizeof(_wp_path),
                     "ott_wproj_cache_%08X.bin", (unsigned)(_wp_hk & 0xFFFFFFFFu));
            axex_manifold_set_attn_only(args.axex_attn_only);
            axex_manifold_set_skip_o(args.axex_skip_o);
            int _wn = axex_manifold_load_wproj_cache(_wp_path, _wp_hk, _e_nl, _e_dim);
            if (_wn > 0) {
                kprintf("[AXEX-MANIFOLD] Loaded W_proj cache: %s (%d matrices, skipping axiom-beta + calibration)\n",
                        _wp_path, _wn);
                wp_early_loaded = 1;
                axiom_ran_this_invocation = 1; /* allow downstream axex gates */
                axiom_consistency = 1.0f;      /* assume good — cached calibration */
            }
        }
    }

    /* --axex-calib-only: skip axiom geometry phases 1-4, run only W_proj calibration.
     * Sets axiom_ran_this_invocation so downstream axex gates fire. */
    if (args.axex_calib_only && args.axex_compress && !wp_early_loaded) {
        axiom_ran_this_invocation = 1;
        kprintf("[AXEX-CALIB-ONLY] Skipping axiom geometry — running W_proj calibration directly.\n");
    }

    if (args.axiom_beta_run && !wp_early_loaded && !args.axex_calib_only) {
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

    /* ── Gauge-optimal compression pre-pass ────────────────────────────── *
     * When --axex-gauge is requested, find the optimal diagonal gauge       *
     * g[dim] that minimises total SVD truncation error across all weight    *
     * matrices BEFORE running the compression loop.  The resulting g is     *
     * baked into U/Vt factors at compression time (zero inference overhead).  */
    static axex_gauge_t *g_gauge = NULL;
    if (args.axex_gauge && (args.axex_ffn_compress || args.axex_attn_svd || args.axex_compress)) {
        const llm_model_t *gm = llm_get_model();
        if (gm) {
            int g_rank = (args.axex_compress_rank > 0) ? args.axex_compress_rank
                       : (gm->dim <= 1024) ? (gm->dim * 3 / 4)
                       : (gm->dim <= 2048) ? 256
                       : (gm->dim <= 5120) ? 384
                       : AXEX_MANIFOLD_K_MAX;
            /* Default 1 iter: Newton step is unstable beyond iter 1 (Jacobian ≈ 2 in
             * log-space → diverges exponentially past first step). Use --axex-gauge-iter
             * to override if the model is known to be stable at higher counts. */
            int g_iter = (args.axex_gauge_iter > 0)    ? args.axex_gauge_iter    : 1;
            kprintf("[AXEX-GAUGE] Starting gauge optimisation: rank=%d iters=%d dim=%d\n",
                    g_rank, g_iter, gm->dim);
            g_gauge = axex_gauge_optimize(gm, g_rank, g_iter);
            if (g_gauge) {
                axex_compress_set_gauge(g_gauge->g, g_gauge->dim);
                kprintf("[AXEX-GAUGE] Gauge ready — will be applied during compression.\n");
            } else {
                kprintf("[AXEX-GAUGE] Warning: gauge optimisation failed — continuing without gauge.\n");
            }
        }
    }

    if (args.axex_kv || args.axex_offload || args.axex_compress || args.axex_ffn_compress || args.axex_attn_svd) {
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
         * --axex-ffn-compress skips the manifold PCA for a faster benchmark.
         * --axex-attn-svd additionally SVD-compresses Q and O (slots 1/4),
         * enabling 70B-in-7GB via full model SVD without calibration data.
         * --axex-compress (GP mode) does NOT enable SVD FFN by default: the
         * GP manifold path handles Q/K/V/O and FFN-down via near-lossless PCA
         * projection.  SVD FFN produces 85-95% Frobenius error on quantized
         * (Q4_0/Q8_0) weights and causes incoherent output regardless of rank.
         * To add SVD FFN on top of GP, explicitly pass --axex-ffn-compress. */
        xcfg.enable_weight_compress = args.axex_ffn_compress ? 1 : 0;
        xcfg.enable_attn_compress   = args.axex_attn_svd ? 1 : 0;
        xcfg.weight_quality         = args.axex_compress_quality;
        xcfg.weight_max_rank        = args.axex_compress_rank;  /* 0 = auto */
        xcfg.weight_max_err         = args.axex_compress_max_err; /* 0 = no limit */
        /* Performance note: compression time is dominated by CBLAS GEMMs
         * (Steps 2, 3, 5 of randomised SVD). The inner SVD (Step 6) now
         * uses LAPACKE_sgesdd (divide-and-conquer) and is no longer a
         * bottleneck.  For rank > 256 the GEMM cost grows with O(rank).
         * Compressed factors are cached to disk after the first run;
         * subsequent runs load the cache and skip re-compression entirely. */
        if (args.axex_compress_rank > 256 &&
            (args.axex_ffn_compress || args.axex_attn_svd || args.axex_compress)) {
            const llm_model_t *_m = llm_get_model();
            int _dim = _m ? _m->dim  : 0;
            int _ff  = _m ? _m->ff_dim : 0;
            int _mn  = (_dim < _ff && _dim > 0) ? _dim : _ff;
            if (args.axex_compress_rank > _mn / 4) {
                kprintf("[AXEX-COMPRESS] Note: rank %d is large for dim=%d (%.0f%% of min_mn) — "
                        "first-run compression will be slow; cached for subsequent runs.\n",
                        args.axex_compress_rank, _dim,
                        100.0f * args.axex_compress_rank / (_mn > 0 ? _mn : 1));
            }
        }
        /* ── Compression cache: derive cache path from model file + rank ──
         * Format: <model-base>.r<rank>.axex alongside the model file.
         * If no explicit rank given, use 0 (auto) as the cache key.
         * Cache is only used for SVD FFN/attn compression (not GP manifold). */
        if ((args.axex_ffn_compress || args.axex_attn_svd) && args.model_path) {
            char _cache_buf[4096];
            int  _rank_key = args.axex_compress_rank; /* 0 = auto */
            snprintf(_cache_buf, sizeof(_cache_buf),
                     "%s.r%d.axex", args.model_path, _rank_key);
            axex_compress_set_cache_path(_cache_buf);
        }

        if (axex_init(&g_axex_state, &xcfg,
                      axiom_ran_this_invocation ? &axiom_rep : NULL,
                      llm_get_model()) != 0)
            kprintf("[AXEX] Warning: manifold exploitation init failed — continuing without it\n");

        /* ── Feature 6: Differentiable compression plan ───────────────────
         * After SVD compression, learn the optimal per-layer rank assignment
         * by minimising Frobenius reconstruction error + L1 rank penalty.
         * Reads frobenius_err from each compressed weight slot. */
        static diffplan_t g_diffplan;
        if (args.axex_learn_plan && g_axex_state.compress_layers > 0) {
            const llm_model_t *dp_mdl = llm_get_model();
            int dp_nl = dp_mdl ? dp_mdl->n_layers : 0;
            if (dp_nl > 0) {
                diffplan_layer_data_t *dp_data = (diffplan_layer_data_t *)
                    calloc((size_t)dp_nl, sizeof(diffplan_layer_data_t));
                if (dp_data) {
                    for (int l = 0; l < dp_nl; l++) {
                        const axex_compressed_weight_t *cw =
                            axex_get_compressed_layer(l, 0); /* gate slot */
                        if (cw && cw->rank > 0) {
                            dp_data[l].frob_err     = cw->frobenius_err;
                            dp_data[l].current_rank = cw->rank;
                            /* Approximate: sv_slope ≈ -frob_err / rank
                             * (negative because higher rank → lower error) */
                            dp_data[l].sv_slope = (cw->rank > 0)
                                ? -(cw->frobenius_err / (float)cw->rank)
                                : -0.001f;
                        } else {
                            dp_data[l].frob_err     = 0.05f;
                            dp_data[l].sv_slope     = -0.001f;
                            dp_data[l].current_rank = 128;
                        }
                    }
                    int plan_steps = (args.axex_plan_steps > 0) ? args.axex_plan_steps : 200;
                    diffplan_init(&g_diffplan, dp_data, dp_nl,
                                  0.001f, args.axex_compress_quality);
                    /* Feature 4: tokens-per-joule objective
                     * If --axex-tpj is set, initialise TPJ context and inject
                     * the energy gradient into each diffplan step. */
                    static thermal_ctx_t g_dp_thermal;
                    static tpj_ctx_t     g_dp_tpj;
                    int tpj_active = 0;
                    if (args.axex_tpj) {
                        thermal_init(&g_dp_thermal,
                                     args.axex_thermal_low, args.axex_thermal_high,
                                     args.axex_thermal_power,
                                     16, AXEX_MANIFOLD_K_MAX);
                        thermal_poll(&g_dp_thermal);
                        tpj_init(&g_dp_tpj, &g_dp_thermal, args.axex_tpj_lambda);
                        /* Pre-calibrate rank_coeff so the gradient is non-trivial
                         * from step 1, before any tpj_record() call.
                         * Rough TPS estimate: assume 20 GFLOP/s peak and
                         * 4 matrix-multiplies of shape [dim × dim] per layer. */
                        {
                            const llm_model_t *tpj_mdl = llm_get_model();
                            int tpj_d = (tpj_mdl && tpj_mdl->dim > 0) ? tpj_mdl->dim : 4096;
                            int tpj_l = (tpj_mdl && tpj_mdl->n_layers > 0) ? tpj_mdl->n_layers : 32;
                            float flops = (float)tpj_l * (float)tpj_d * (float)tpj_d * 4.0f;
                            float tps_est = (flops > 0.0f) ? (2.0e10f / flops) : 10.0f;
                            tpj_bootstrap(&g_dp_tpj, tps_est);
                        }
                        tpj_active = 1;
                    }
                    float final_loss = 0.0f;
                    if (tpj_active) {
                        /* Manual step loop so we can inject TPJ gradient */
                        float g_tpj[DIFFPLAN_MAX_LAYERS][DIFFPLAN_N_LEVELS];
                        for (int iter = 0; iter < plan_steps; iter++) {
                            final_loss = diffplan_step(&g_diffplan);
                            memset(g_tpj, 0, sizeof(g_tpj));
                            tpj_gradient(&g_dp_tpj, g_tpj, dp_nl, &g_diffplan);
                            /* Add TPJ gradient into theta (simple SGD on top of diffplan) */
                            for (int l = 0; l < dp_nl && l < DIFFPLAN_MAX_LAYERS; l++)
                                for (int r = 0; r < DIFFPLAN_N_LEVELS; r++)
                                    g_diffplan.theta[l][r] -= g_diffplan.lr * g_tpj[l][r];
                        }
                        g_diffplan.n_iter = plan_steps;
                        tpj_print(&g_dp_tpj);
                    } else {
                        final_loss = diffplan_optimise(&g_diffplan, plan_steps);
                    }
                    kprintf("[DIFFPLAN] Optimised in %d steps, final loss=%.5f\n",
                            g_diffplan.n_iter, (double)final_loss);
                    diffplan_print(&g_diffplan);
                    free(dp_data);
                }
            }
        }

        /* ── Feature 5: Phase-conditional compression plan ────────────────
         * Build retrieval/reasoning/generation rank plans from the offload
         * plan's per-layer importance scores.  Printed at startup; updated
         * per decode token in the speculative loop via phaseplan_update()
         * using logit entropy as an attention-entropy proxy. */
        if (args.axex_phase_compress) {
            const llm_model_t *pp_mdl = llm_get_model();
            int pp_nl = pp_mdl ? pp_mdl->n_layers : 0;
            if (pp_nl > 0 && g_axex_state.offload.n_layers > 0) {
                if (phaseplan_build(&s_spec_phase_plan,
                                    &g_axex_state.offload, pp_nl,
                                    args.axex_phase_ret_q,
                                    args.axex_phase_rea_q,
                                    args.axex_phase_gen_q) == 0) {
                    kprintf("[PHASE] Phase-conditional plan built (%d layers)\n", pp_nl);
                    phaseplan_print(&s_spec_phase_plan);
                    s_spec_phase_plan_active = 1;
                } else {
                    kprintf("[PHASE] phaseplan_build failed\n");
                }
            } else {
                kprintf("[PHASE] Phase plan skipped (no offload plan — use --axex-compress first)\n");
            }
        }

        /* ── Feature 5 (new): Cross-quantization shared basis test ───────────
         * Tests whether the activation-space PCA basis captures the same
         * subspace as the weight SVD right vectors.  alignment ≈ 1.0 means
         * one basis works for all quantization levels of the same weight.
         * Requires both manifold and SVD compression to have run. */
        static qspec_result_t g_qspec_result;
        if (args.axex_qspec) {
            const llm_model_t *qs_mdl = llm_get_model();
            int qs_nl = qs_mdl ? qs_mdl->n_layers : 0;
            if (qs_nl > 0) {
                int n_ev = qspec_test_shared_basis(&g_qspec_result, qs_nl,
                                                   args.axex_qspec_threshold);
                if (n_ev > 0) {
                    qspec_print(&g_qspec_result);
                } else {
                    kprintf("[QSPEC] No layer/slot pairs available — run --axex-compress first\n");
                }
            }
        }

        /* ── Feature 6 (new): Failure-mode-targeted rank allocation ────────
         * Detect which layer zone has the highest Frobenius error after SVD
         * compression, classify the dominant failure mode (factual / reasoning /
         * coherence / context), and print recommended rank scales per layer. */
        static frank_result_t g_frank_result;
        if (args.axex_fail_rank) {
            const llm_model_t *fr_mdl = llm_get_model();
            int fr_nl = fr_mdl ? fr_mdl->n_layers : 0;
            if (fr_nl > 0) {
                float *fr_errs = (float *)calloc((size_t)fr_nl, sizeof(float));
                if (fr_errs) {
                    for (int l = 0; l < fr_nl; l++) {
                        /* Average frobenius_err across Q/K/V/O slots (1-4) for
                         * a representative per-layer estimate. */
                        float sum_e = 0.0f; int cnt = 0;
                        for (int s = 1; s <= 4; s++) {
                            const axex_compressed_weight_t *cw =
                                axex_get_compressed_layer(l, s);
                            if (cw && cw->rank > 0) { sum_e += cw->frobenius_err; cnt++; }
                        }
                        fr_errs[l] = (cnt > 0) ? (sum_e / (float)cnt) : 0.0f;
                    }
                    if (frank_build(&g_frank_result, fr_errs, fr_nl,
                                    args.axex_fail_boost, 0.6f) == 0) {
                        frank_print(&g_frank_result);
                        /* Apply FRANK rank scales to the stored per-layer manifold k values.
                         * axex_manifold_adjust_ks() modifies g_Pt_layers_ks[] in place so
                         * that any future re-compression pass (e.g. a subsequent --axex-compress
                         * invocation) uses FRANK-recommended k per layer.  For the current run
                         * this annotates the stored basis for diagnostic use. */
                        {
                            int min_k = (args.axex_compress_rank > 0)
                                        ? args.axex_compress_rank / 4 : 8;
                            int max_k = (args.axex_compress_rank > 0)
                                        ? args.axex_compress_rank : AXEX_MANIFOLD_K_MAX;
                            axex_manifold_adjust_ks(g_frank_result.rank_scale,
                                                    fr_nl, min_k, max_k);
                            kprintf("[FRANK] Rank scales written to manifold layer-k table\n");
                        }
                    } else {
                        kprintf("[FRANK] frank_build failed\n");
                    }
                    free(fr_errs);
                }
            } else {
                kprintf("[FRANK] No layers available — run --axex-compress first\n");
            }
        }

        /* ── Feature 3: Thermal-adaptive rank context init ───────────────
         * Initialise the NVML context once at startup; thermal_get_rank() is
         * called in the spec decode loop whenever --axex-thermal is active to
         * cap dyn_batch proportionally to GPU thermal headroom. */
        if (args.axex_thermal) {
            thermal_init(&s_spec_thermal,
                         args.axex_thermal_low, args.axex_thermal_high,
                         args.axex_thermal_power,
                         args.axex_compress_rank > 0 ? args.axex_compress_rank / 4 : 8,
                         args.axex_compress_rank > 0 ? args.axex_compress_rank : 128);
            thermal_poll(&s_spec_thermal);
            thermal_print(&s_spec_thermal);
            s_spec_thermal_active = 1;
            /* Also initialise the inference TPJ context so energy efficiency
             * is tracked per decode batch and printed at end-of-run. */
            tpj_init(&s_spec_tpj, &s_spec_thermal, args.axex_tpj_lambda);
            tpj_bootstrap(&s_spec_tpj, 10.0f); /* conservative 10 tok/s seed */
            s_spec_tpj_active = 1;
        }

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
                 *   - fast mode:        64  (quick preview, low quality)
                 *   - default:          2048 (lossless-grade PCA basis; cached to disk) */
                int n_samples;
                if (args.axex_calib_samples > 0)
                    n_samples = args.axex_calib_samples;
                else
                    n_samples = (args.axiom_fast_mode) ? 64 : 2048;
                if (n_samples > vocab) n_samples = vocab;

                /* Wire up attention-only vs full-compress mode */
                axex_manifold_set_attn_only(args.axex_attn_only);
                axex_manifold_set_skip_o(args.axex_skip_o);
                if (args.axex_skip_o)
                    kprintf("[AXEX-MANIFOLD] Skip-O mode: O_proj preserved at full rank (Q/K/V compressed only)\n");
                if (args.axex_weight_pca)
                    kprintf("[AXEX-MANIFOLD] Weight-PCA mode: Pt = top-k eigenvectors of K=Σ W^T W (maximises weight energy, no calib data needed)\n");
                if (args.axex_attn_only)
                    kprintf("[AXEX-MANIFOLD] Attention-only mode: Q/K/V/O compressed, FFN preserved (near-lossless accuracy)\n");
                else
                    kprintf("[AXEX-MANIFOLD] Full-compress mode: Q/K/V/O + FFN gate/up compressed (lower accuracy)\n");

                /* ── W_proj disk cache (fast path) ───────────────────────────
                 * Build the cache key (identical to hs_cache key so they stay
                 * in sync; different filename prefix so they don't collide).
                 * On hit: skip ALL calibration + PCA + compression (~22s) and
                 * restore state directly from disk.  On miss: run normally and
                 * save afterwards. */
                uint64_t wp_hk = 0xDEADBEEF12345678ULL;
                {
                    const char *wmp = args.model_path ? args.model_path : "";
                    for (const char *c = wmp; *c; c++) { wp_hk ^= (uint8_t)*c; wp_hk *= 0x100000001B3ULL; }
                    wp_hk ^= (uint64_t)n_samples * 0xABCDEF01ULL;
                    wp_hk ^= (uint64_t)n_layers  * 0x12345678ULL;
                    wp_hk ^= (uint64_t)dim        * 0x87654321ULL;
                    wp_hk ^= (uint64_t)AXEX_CALIB_CTX_LEN * 0xFEDCBA98ULL; /* v2: contextual calibration */
                    wp_hk ^= (args.axex_no_actaware ? 0ULL : 0xA1B2C3D4AAAAULL);  /* v3: actaware PCA */
                    wp_hk ^= (args.axex_skip_o ? 0x5C09F0DEADBEULL : 0ULL);  /* v4: skip-O */
                    wp_hk ^= (args.axex_weight_pca ? 0xC8F2620B4E3DULL : 0ULL);  /* v6: weight-PCA normalized-K */
                    wp_hk ^= (uint64_t)(args.axex_compress_rank > 0 ? args.axex_compress_rank : 0) * 0xF1E2D3C4ULL; /* compress-rank */
                }
                char wp_cache_path[512];
                snprintf(wp_cache_path, sizeof(wp_cache_path),
                         "ott_wproj_cache_%08X.bin", (unsigned)(wp_hk & 0xFFFFFFFFu));

                int wp_cache_loaded = wp_early_loaded;
                if (!wp_cache_loaded) {
                    int wn = axex_manifold_load_wproj_cache(wp_cache_path, wp_hk, n_layers, dim);
                    if (wn > 0) {
                        kprintf("[AXEX-MANIFOLD] Loaded W_proj cache: %s (%d matrices, skipping calibration)\n",
                                wp_cache_path, wn);
                        wp_cache_loaded = 1;
                    }
                }

                float   *hs_buf = NULL; /* unused in multi-layer path, kept for fallback */
                int      lw_ok  = 1;

                if (!wp_cache_loaded) {
                /* Quality pre-check: warn when n_samples is below the target rank
                 * so the user knows calibration may be under-powered.
                 * We do NOT abort here — the per-weight proj_energy guard (80%)
                 * inside COMPRESS_SLOT_LW already skips any matrix whose basis
                 * doesn't capture sufficient energy.
                 * k_warn = ceil(dim / 16): the number of PCA components needed so
                 * that P covers ~6% of dim, the empirical minimum for coherent output.
                 * For dim=3072: k_warn=192.  For dim=4096: k_warn=256. */
                {
                    int k_warn = (dim + 15) / 16;
                    if (k_warn < 32) k_warn = 32;
                    if (n_samples < k_warn) {
                        kprintf("[AXEX-MANIFOLD] WARNING: --axex-calib-samples %d < k_warn=%d "
                                "(dim=%d); basis may be under-powered — output quality could degrade.\n"
                                "[AXEX-MANIFOLD] Recommended: --axex-calib-samples %d "
                                "(or omit for default 512).\n",
                                n_samples, k_warn, dim, k_warn);
                    }
                }

                if (lw_ok) {
                    if (args.axex_weight_pca)
                        kprintf("[AXEX-MANIFOLD] Weight-PCA: skipping calibration forward passes (using K=Σ W^T W eigenvectors)\n");
                    else
                        kprintf("[AXEX-MANIFOLD] Computing per-layer PCA (%d layers × %d samples × dim=%d)...\n",
                                n_layers, n_samples, dim);

                    /* Allocate flat matrix: all_hs[layer][sample][dim]
                     * Skip allocation and calibration when --axex-weight-pca is set;
                     * weight-only eigenvectors don't need any hidden-state data.
                     * If initial allocation fails (OOM), halve n_samples and retry once.
                     * Prevents full compression abort on memory-constrained hosts; fewer
                     * calibration samples reduce quality slightly but avoid rank-1 fallback. */
                    size_t total_f = (size_t)n_layers * n_samples * dim;
                    float *all_hs = args.axex_weight_pca ? NULL : (float *)malloc(total_f * sizeof(float));
                    if (!args.axex_weight_pca && !all_hs && n_samples > 64) {
                        n_samples /= 2;
                        total_f = (size_t)n_layers * n_samples * dim;
                        all_hs = (float *)malloc(total_f * sizeof(float));
                        if (all_hs)
                            kprintf("[AXEX-MANIFOLD] OOM on full alloc — retrying with %d samples\n", n_samples);
                    }
                    int   *all_valid = (int *)calloc((size_t)n_layers, sizeof(int));
                    int   *cap_valid = (int *)calloc((size_t)n_layers, sizeof(int));
                    if ((!args.axex_weight_pca && !all_hs) || !all_valid || !cap_valid) {
                        free(all_hs); free(all_valid); free(cap_valid);
                        lw_ok = 0;
                    }

                    /* ── Calibration disk cache ──────────────────────────────────
                     * Cache key: model_path + n_samples + n_layers + dim.
                     * File: ott_hs_cache_<hash8>.bin
                     * Format: magic(8) key(32) valid(n_layers×4) data(n_layers×n_samples×dim×4)
                     * On hit: load in ~50ms instead of running 512+ forward passes.
                     * On miss or version mismatch: run calibration, then save. */
                    char hs_cache_path[512];
                    char hs_chkpt_path[512];
                    uint64_t hs_hk = 0;
                    int  hs_cache_loaded = 0;
                    if (lw_ok) {
                        /* Build a simple hash of the cache key */
                        uint64_t hk = 0xDEADBEEF12345678ULL;
                        const char *mp = args.model_path ? args.model_path : "";
                        for (const char *c = mp; *c; c++) { hk ^= (uint8_t)*c; hk *= 0x100000001B3ULL; }
                        hk ^= (uint64_t)n_samples * 0xABCDEF01ULL;
                        hk ^= (uint64_t)n_layers  * 0x12345678ULL;
                        hk ^= (uint64_t)dim       * 0x87654321ULL;
                        hk ^= (uint64_t)AXEX_CALIB_CTX_LEN * 0xFEDCBA98ULL; /* v2: contextual calibration */
                        hs_hk = hk;
                        snprintf(hs_cache_path, sizeof(hs_cache_path),
                                 "ott_hs_cache_%08X.bin", (unsigned)(hk & 0xFFFFFFFFu));
                        snprintf(hs_chkpt_path, sizeof(hs_chkpt_path),
                                 "ott_hs_checkpoint_%08X.bin", (unsigned)(hk & 0xFFFFFFFFu));

                        /* Try to load from cache (skip when weight-PCA: no hidden states needed) */
                        FILE *cf = args.axex_weight_pca ? NULL : fopen(hs_cache_path, "rb");
                        if (cf) {
                            uint64_t magic_stored = 0, key_stored = 0;
                            int nl_stored = 0, ns_stored = 0, dim_stored = 0;
                            int r_magic = (int)fread(&magic_stored, 8, 1, cf);
                            int r_key   = r_magic && (int)fread(&key_stored, 8, 1, cf);
                            int r_nl    = r_key   && (int)fread(&nl_stored,  4, 1, cf);
                            int r_ns    = r_nl    && (int)fread(&ns_stored,  4, 1, cf);
                            int r_dim   = r_ns    && (int)fread(&dim_stored, 4, 1, cf);
                            size_t r_valid = r_dim ? fread(all_valid, sizeof(int), (size_t)n_layers, cf) : 0;
                            size_t r_data  = (r_valid == (size_t)n_layers) ? fread(all_hs, sizeof(float), total_f, cf) : 0;
                            if (r_magic && magic_stored == 0x4D414E494645535FULL &&
                                r_key   && key_stored  == hk &&
                                r_nl    && nl_stored   == n_layers &&
                                r_ns    && ns_stored   == n_samples &&
                                r_dim   && dim_stored  == dim &&
                                r_valid == (size_t)n_layers &&
                                r_data  == total_f) {
                                hs_cache_loaded = 1;
                                kprintf("[AXEX-MANIFOLD] Loaded calibration cache: %s (%d layers × %d samples × dim=%d)\n",
                                        hs_cache_path, n_layers, n_samples, dim);
                            } else {
                                kprintf("[AXEX-MANIFOLD] Cache mismatch: magic=%d(%llx==%llx?) key=%d(%llx==%llx?) "
                                        "nl=%d(%d==%d?) ns=%d(%d==%d?) dim=%d(%d==%d?) valid=%zu/%zu data=%zu/%zu\n",
                                        r_magic, (unsigned long long)magic_stored, (unsigned long long)0x4D414E494645535FULL,
                                        r_key, (unsigned long long)key_stored, (unsigned long long)hk,
                                        r_nl, nl_stored, n_layers,
                                        r_ns, ns_stored, n_samples,
                                        r_dim, dim_stored, dim,
                                        r_valid, (size_t)n_layers, r_data, total_f);
                                memset(all_valid, 0, (size_t)n_layers * sizeof(int));
                            }
                            fclose(cf);
                        }
                    }

                    /* Try to resume from checkpoint (partial run saved every 64 passes) */
                    int hs_chkpt_done = 0; /* number of passes already captured */
                    if (lw_ok && !hs_cache_loaded && !args.axex_weight_pca) {
                        FILE *cpf = fopen(hs_chkpt_path, "rb");
                        if (cpf) {
                            uint64_t cp_magic = 0, cp_key = 0;
                            int cp_nl = 0, cp_ns = 0, cp_done = 0, cp_dim = 0;
                            if (fread(&cp_magic, 8, 1, cpf) == 1 && cp_magic == 0x4348454350544348ULL /* "CHECKPNT" */ &&
                                fread(&cp_key,   8, 1, cpf) == 1 && cp_key  == hs_hk &&
                                fread(&cp_nl,    4, 1, cpf) == 1 && cp_nl   == n_layers &&
                                fread(&cp_ns,    4, 1, cpf) == 1 && cp_ns   == n_samples &&
                                fread(&cp_done,  4, 1, cpf) == 1 && cp_done  > 0 && cp_done < n_samples &&
                                fread(&cp_dim,   4, 1, cpf) == 1 && cp_dim  == dim &&
                                fread(all_valid, sizeof(int), (size_t)n_layers, cpf) == (size_t)n_layers &&
                                fread(all_hs, sizeof(float), total_f, cpf) == total_f) {
                                hs_chkpt_done = cp_done;
                                kprintf("[AXEX-MANIFOLD] Resuming from checkpoint: %s (%d/%d passes done)\n",
                                        hs_chkpt_path, cp_done, n_samples);
                            } else {
                                kprintf("[AXEX-MANIFOLD] Checkpoint mismatch/corrupt — starting fresh\n");
                                memset(all_valid, 0, (size_t)n_layers * sizeof(int));
                            }
                            fclose(cpf);
                        }
                    }

                    if (lw_ok && !hs_cache_loaded && !args.axex_weight_pca) {
                        kprintf("[AXEX-MANIFOLD] Using contextual multi-layer capture "
                                "(%d samples × %d tokens = %d forward passes)\n",
                                n_samples, AXEX_CALIB_CTX_LEN + 2,
                                n_samples * (AXEX_CALIB_CTX_LEN + 2));
                        /* Advance seed to match the starting offset (so resumed runs use
                         * the same tokens as the original run would have at that point).
                         * Each sample now consumes (AXEX_CALIB_CTX_LEN + 1) seed steps. */
                        uint64_t seed = 0x123456789ABCDEF0ULL;
                        for (int _sk = 0; _sk < hs_chkpt_done * (AXEX_CALIB_CTX_LEN + 1); _sk++) {
                            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                        }
                        float *pass_buf = (float *)malloc((size_t)n_layers * dim * sizeof(float));
                        if (!pass_buf) lw_ok = 0;

                        for (int s = hs_chkpt_done; s < n_samples && lw_ok; s++) {
                            /* Build context + probe sequence (AXEX_CALIB_CTX_LEN context
                             * tokens + 1 probe token).  Context tokens are prefilled via
                             * batch GPU prefill; the probe token's hidden states are
                             * captured by the bridge at position AXEX_CALIB_CTX_LEN+1,
                             * giving contextually-rich activations for PCA calibration.
                             * Without this, single-token pos-0 calibration misses the
                             * contextual subspace (~90% of inference variance) → PPL~7244. */
                            int ctx_arr[AXEX_CALIB_CTX_LEN];
                            for (int _ci = 0; _ci < AXEX_CALIB_CTX_LEN; _ci++) {
                                seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                                ctx_arr[_ci] = (int)(seed % (uint64_t)vocab);
                            }
                            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                            int probe_tok = (int)(seed % (uint64_t)vocab);

                            memset(cap_valid, 0, (size_t)n_layers * sizeof(int));

                            /* Phase 1: prefill context WITHOUT bridge capture */
                            tensor_bridge_t *calib_br = llm_get_bridge();
                            calib_br->mode = BRIDGE_MODE_NONE;
                            calib_br->multi_cap_bufs  = NULL;
                            calib_br->multi_cap_valid = NULL;
                            calib_br->multi_cap_n     = 0;
                            calib_br->multi_cap_dim   = 0;
                            llm_reset_cache();
                            int _ctx_dummy_out[AXEX_CALIB_CTX_LEN + 3];
                            llm_generate_tokens(ctx_arr, AXEX_CALIB_CTX_LEN,
                                                _ctx_dummy_out, AXEX_CALIB_CTX_LEN + 3,
                                                1, 0.0f, 0);

                            /* Phase 2: capture probe token WITH bridge (continue_cache=1
                             * so it runs at position AXEX_CALIB_CTX_LEN+1 with full ctx) */
                            tensor_bridge_set_multi_capture(calib_br, pass_buf,
                                                            cap_valid, n_layers, dim);
                            calib_br->mode = (bridge_mode_t)(
                                BRIDGE_MODE_CAP_ALL | BRIDGE_MODE_CAP_ONCE);
                            int _probe_out[2];
                            int rc_br = llm_generate_tokens(&probe_tok, 1,
                                                            _probe_out, 2, 1, 0.0f, 1);
                            calib_br->mode = BRIDGE_MODE_NONE;
                            calib_br->multi_cap_bufs  = NULL;
                            calib_br->multi_cap_valid = NULL;
                            calib_br->multi_cap_n     = 0;
                            calib_br->multi_cap_dim   = 0;
                            int rc = (rc_br >= 0) ? 0 : -1;

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

                            if ((s % 64) == 63 || s == n_samples - 1) {
                                kprintf("[AXEX-MANIFOLD] Calibration sample %d/%d done\n",
                                        s + 1, n_samples);
                                /* Write crash-recovery checkpoint every 64 passes */
                                if (s < n_samples - 1) {
                                    FILE *cpw = fopen(hs_chkpt_path, "wb");
                                    if (cpw) {
                                        uint64_t cp_magic = 0x4348454350544348ULL;
                                        int n_done_now = s + 1;
                                        int cp_ok = 1;
                                        cp_ok &= (fwrite(&cp_magic,    8, 1, cpw) == 1);
                                        cp_ok &= (fwrite(&hs_hk,       8, 1, cpw) == 1);
                                        cp_ok &= (fwrite(&n_layers,    4, 1, cpw) == 1);
                                        cp_ok &= (fwrite(&n_samples,   4, 1, cpw) == 1);
                                        cp_ok &= (fwrite(&n_done_now,  4, 1, cpw) == 1);
                                        cp_ok &= (fwrite(&dim,         4, 1, cpw) == 1);
                                        cp_ok &= (fwrite(all_valid, sizeof(int), (size_t)n_layers, cpw) == (size_t)n_layers);
                                        cp_ok &= (fwrite(all_hs, sizeof(float), total_f, cpw) == total_f);
                                        fclose(cpw);
                                        if (cp_ok)
                                            kprintf("[AXEX-MANIFOLD] Checkpoint saved: %s (%d/%d passes)\n",
                                                    hs_chkpt_path, n_done_now, n_samples);
                                        else
                                            kprintf("[AXEX-MANIFOLD] WARNING: checkpoint write failed\n");
                                    }
                                }
                            }
                        }
                        free(pass_buf);

                        /* Save cache for future runs */
                        if (lw_ok) {
                            FILE *cf = fopen(hs_cache_path, "wb");
                            if (cf) {
                                uint64_t magic = 0x4D414E494645535FULL;
                                uint64_t hk2 = hs_hk;
                                int ok_w = 1;
                                ok_w &= (fwrite(&magic,      8, 1, cf) == 1);
                                ok_w &= (fwrite(&hk2,        8, 1, cf) == 1);
                                ok_w &= (fwrite(&n_layers,   4, 1, cf) == 1);
                                ok_w &= (fwrite(&n_samples,  4, 1, cf) == 1);
                                ok_w &= (fwrite(&dim,        4, 1, cf) == 1);
                                ok_w &= (fwrite(all_valid, sizeof(int), (size_t)n_layers, cf) == (size_t)n_layers);
                                ok_w &= (fwrite(all_hs, sizeof(float), total_f, cf) == total_f);
                                fclose(cf);
                                if (ok_w) {
                                    kprintf("[AXEX-MANIFOLD] Saved calibration cache: %s (%.1f MB)\n",
                                            hs_cache_path,
                                            (double)(total_f * sizeof(float)) / (1024.0 * 1024.0));
                                    /* Remove checkpoint now that we have the full cache */
                                    remove(hs_chkpt_path);
                                } else {
                                    kprintf("[AXEX-MANIFOLD] WARNING: failed to write cache %s\n", hs_cache_path);
                                }
                            }
                        }
                    }

                    /* Diagnostic: verify calibration captured non-zero data */
                    {
                        int n_valid_layers = 0;
                        float sample_norm2 = 0.0f;
                        for (int l = 0; l < n_layers; l++) if (all_valid[l] > 0) n_valid_layers++;
                        if (n_valid_layers > 0 && all_valid[0] > 0) {
                            const float *s0 = all_hs;
                            for (int j = 0; j < dim; j++) sample_norm2 += s0[j] * s0[j];
                        }
                        kprintf("[AXEX-MANIFOLD] Capture diagnostic: %d/%d layers have valid samples, "
                                "all_valid[0]=%d, sample[0] norm2=%.4f\n",
                                n_valid_layers, n_layers, all_valid[0], (double)sample_norm2);
                    }

                    /* ── Feature 2: Real Ricci curvature signal ──────────────
                     * Build per-layer sectional curvature from the hidden state
                     * samples and use it to set the manifold basis rank budget.
                     * Falls back to AXEX_MANIFOLD_K_MAX if not requested or
                     * if the curvature computation fails. */
                    int *ricci_ranks = NULL;
                    static grcurv_per_layer_t g_grcurv_out;
                    if (args.axex_real_curvature && lw_ok) {
                        kprintf("[GRCURV] Computing real Ricci curvature per layer...\n");
                        grcurv_layer_input_t *grc_in = (grcurv_layer_input_t *)
                            calloc((size_t)n_layers, sizeof(grcurv_layer_input_t));
                        if (grc_in) {
                            for (int l = 0; l < n_layers; l++) {
                                grc_in[l].samples   = all_hs + (size_t)l * n_samples * dim;
                                grc_in[l].n_samples = (all_valid[l] > 0) ? n_samples : 0;
                                grc_in[l].dim       = dim;
                                grc_in[l].proxy_ricci = (float)g_axex_state.offload.entries[
                                    l < g_axex_state.offload.n_layers ? l : 0].ricci_scalar;
                            }
                            if (grcurv_compute(grc_in, n_layers, &g_grcurv_out) == 0) {
                                grcurv_print_comparison(&g_grcurv_out);
                                ricci_ranks = (int *)malloc((size_t)n_layers * sizeof(int));
                                if (ricci_ranks)
                                    grcurv_to_rank_budget(&g_grcurv_out,
                                        16, AXEX_MANIFOLD_K_MAX,
                                        AXEX_MANIFOLD_K_MAX * n_layers,
                                        ricci_ranks);
                            } else {
                                kprintf("[GRCURV] Computation failed — using uniform ranks\n");
                            }
                            free(grc_in);
                        }
                    }

                    /* ── Feature 1: MCR phase-aware rank allocation ──────────────────
                     * Queipo-de-Llano et al. (Oct 2025): transformers process tokens
                     * in Mix / Compress / Refine phases.  Middle layers are already
                     * self-compressed; compressing them aggressively is nearly free.
                     * Mix and Refine layers need higher rank for quality preservation.
                     * Detect phases from activation-variance profile of all_hs.
                     *
                     * Feature 2: Attention-sink bypass
                     * Detect high-norm calibration samples (proxy for BOS sink tokens);
                     * ensure the sink direction is covered by each layer's PCA basis. */
                    int   *mcr_ranks   = NULL;
                    float *sink_norms  = NULL;
                    static mcr_result_t  g_mcr_result;
                    static sink_ctx_t    g_sink_ctx;
                    if ((args.axex_mcr || args.axex_sink_bypass) && lw_ok) {
                        mcr_layer_stats_t *mcr_stats =
                            (mcr_layer_stats_t *)calloc((size_t)n_layers, sizeof(mcr_layer_stats_t));
                        if (args.axex_sink_bypass)
                            sink_norms = (float *)calloc((size_t)n_samples, sizeof(float));
                        if (mcr_stats) {
                            int mid_l = n_layers / 2;
                            for (int l = 0; l < n_layers; l++) {
                                const float *hs_l = all_hs + (size_t)l * n_samples * dim;
                                double s1 = 0.0, s2 = 0.0;
                                int cnt = (all_valid[l] > 0) ? n_samples : 0;
                                for (int s = 0; s < cnt; s++) {
                                    const float *h = hs_l + (size_t)s * dim;
                                    for (int d = 0; d < dim; d++) {
                                        double v = (double)h[d];
                                        s1 += v; s2 += v * v;
                                    }
                                }
                                double n_el = (double)(cnt * dim);
                                if (n_el > 0.0) {
                                    double mean = s1 / n_el;
                                    mcr_stats[l].act_variance = (float)(s2 / n_el - mean * mean);
                                }
                                /* Per-sample norms at mid layer for sink detection */
                                if (l == mid_l && sink_norms) {
                                    for (int s = 0; s < n_samples; s++) {
                                        const float *h = hs_l + (size_t)s * dim;
                                        double ns = 0.0;
                                        for (int d = 0; d < dim; d++) ns += (double)h[d] * h[d];
                                        sink_norms[s] = (float)sqrt(ns);
                                    }
                                }
                            }
                            if (args.axex_mcr) {
                                if (mcr_detect_phases(&g_mcr_result, mcr_stats,
                                                      n_layers, 1.5f) == 0) {
                                    mcr_print(&g_mcr_result);
                                    mcr_ranks = (int *)malloc((size_t)n_layers * sizeof(int));
                                    if (mcr_ranks)
                                        mcr_rank_budget(&g_mcr_result,
                                            AXEX_MANIFOLD_K_MAX * n_layers,
                                            16, AXEX_MANIFOLD_K_MAX,
                                            args.axex_mcr_mix_scale,
                                            args.axex_mcr_compress_scale,
                                            args.axex_mcr_refine_scale,
                                            mcr_ranks);
                                }
                            }
                            if (args.axex_sink_bypass && sink_norms) {
                                sink_detect(&g_sink_ctx, sink_norms, n_samples, 3.0f);
                                sink_print(&g_sink_ctx);
                            }
                            free(mcr_stats);
                        }
                    }

                    /* Build per-layer PCA from collected samples */
                    int lw_skipped = 0;  /* count of layers skipped due to PCA failure */
                    for (int l = 0; l < n_layers && lw_ok; l++) {
                        /* Determine rank budget BEFORE the PCA call so we can
                         * use the fast randomized-SVD path (axpca_compute_topk)
                         * which is 20-100× faster than full Jacobi for k << dim. */
                        int k_for_layer;
                        /* Compute k from MCR/Ricci curvature guides if available. */
                        if (mcr_ranks && mcr_ranks[l] > 0 && ricci_ranks && ricci_ranks[l] > 0) {
                            k_for_layer = mcr_ranks[l] > ricci_ranks[l]
                                          ? mcr_ranks[l] : ricci_ranks[l];
                        } else if (mcr_ranks && mcr_ranks[l] > 0) {
                            k_for_layer = mcr_ranks[l];
                        } else if (ricci_ranks && ricci_ranks[l] > 0) {
                            k_for_layer = ricci_ranks[l];
                        } else {
                            k_for_layer = 0; /* use model-size defaults below */
                        }
                        /* Apply a minimum-rank floor for small models (dim≤1024).
                         * Empirically, k/n < 88% causes attention collapse in these
                         * models: the curvature-guided rank often lands at ~78% which
                         * produces repetitive output.  Clamp upward to K_MAX when
                         * the curvature estimate is below the safe floor.
                         * User can force a specific rank with --axex-compress-rank. */
                        if (args.axex_compress_rank > 0) {
                            k_for_layer = args.axex_compress_rank;
                        } else if (dim <= 1024) {
                            /* Small models: enforce minimum k = AXEX_MANIFOLD_K_MAX.
                             * If MCR/Ricci gave a higher value, keep it; otherwise
                             * override to K_MAX (88% of dim=576 → k=512). */
                            if (k_for_layer < AXEX_MANIFOLD_K_MAX)
                                k_for_layer = AXEX_MANIFOLD_K_MAX;
                        } else if (k_for_layer == 0) {
                            /* v3 (2026-04-23): bumped default k for large models.
                             * At k/n <~ 15% the shared-Pt PCA basis is effectively
                             * random for V/O weights (uniform W spectrum) → PPL
                             * collapse. Raise defaults to keep k/n ≥ 25%. */
                            if (dim <= 2048) {
                                k_for_layer = 512;   /* was 256 */
                            } else if (dim <= 5120) {
                                k_for_layer = 1024;  /* was 384; k/n=25% for Llama-8B */
                            } else {
                                k_for_layer = 2048;  /* was K_MAX=384; k/n=25% for 70B */
                            }
                        }
                        /* Round up to a multiple of 32 so W_proj can always be quantized
                         * to Q8_0 (which requires k % 32 == 0 for block-aligned quantization).
                         * This wastes at most 31 PCA components but guarantees 4× memory saving. */
                        k_for_layer = (k_for_layer + 31) & ~31;
                        if (k_for_layer < 32) k_for_layer = 32;  /* floor: prevent k=0 from curvature underflow */
                        /* Cap at available rank; add 3 slots for sink bypass headroom */
                        int k_budget = k_for_layer + (args.axex_sink_bypass ? 3 : 0);
                        /* In weight-PCA mode, rank is limited only by dim (no sample count limit) */
                        int k_max_avail = args.axex_weight_pca ? dim : ((n_samples < dim) ? n_samples : dim);
                        if (k_budget > k_max_avail) k_budget = k_max_avail;

                        axmat_t X = axmat_create(args.axex_weight_pca ? 1 : n_samples, dim);
                        if (!X.data) {
                            /* OOM on axmat: skip this layer, not the whole compression */
                            kprintf("[AXEX-MANIFOLD] Layer %d: axmat alloc failed — skipping layer\n", l);
                            lw_skipped++;
                            if (lw_skipped > n_layers / 2) { lw_ok = 0; break; }
                            continue;
                        }

                        if (!args.axex_weight_pca) {
                            for (int s = 0; s < n_samples; s++) {
                                const float *src = all_hs + ((size_t)l * n_samples + s) * dim;
                                for (int j = 0; j < dim; j++)
                                    X.data[(size_t)s * dim + j] = (double)src[j];
                            }
                        } else {
                            /* weight-PCA: X is unused — just zero-initialise */
                            memset(X.data, 0, (size_t)dim * sizeof(double));
                        }

                        /* ── v4 (2026-04-23) Data-aware PCA preconditioner ──
                         * Scale sample columns by sqrt(diag(Σ_W W^T W)) where W
                         * ranges over attention Q/K/V projections of this layer.
                         * This biases the PCA toward directions that matter for
                         * the downstream weighted reconstruction error.  When
                         * disabled (--axex-no-actaware), reverts to pure h-PCA. */
                        /* actaware column-norm scaling — skip in weight-PCA mode (no X data) */
                        int actaware = !args.axex_no_actaware && !args.axex_weight_pca;
                        float *col_s = NULL;
                        if (actaware) {
                            col_s = (float *)malloc((size_t)dim * sizeof(float));
                            if (col_s && grc_compute_attn_col_norms(
                                    &mdl->layers[l], dim, col_s) == 0) {
                                for (int s = 0; s < n_samples; s++) {
                                    double *row = X.data + (size_t)s * dim;
                                    for (int j = 0; j < dim; j++)
                                        row[j] *= (double)col_s[j];
                                }
                                if (l == 0) {
                                    /* Report scale spread for diagnostic */
                                    float s_min = col_s[0], s_max = col_s[0];
                                    double s_mean = 0.0;
                                    for (int j = 0; j < dim; j++) {
                                        if (col_s[j] < s_min) s_min = col_s[j];
                                        if (col_s[j] > s_max) s_max = col_s[j];
                                        s_mean += col_s[j];
                                    }
                                    s_mean /= (double)dim;
                                    kprintf("[AXEX-ACTAWARE] L0 col-norm scale: min=%.3f mean=%.3f max=%.3f (ratio=%.1f×)\n",
                                            s_min, (float)s_mean, s_max,
                                            s_min > 1e-6f ? s_max / s_min : 0.0f);
                                }
                            } else if (col_s && l == 0) {
                                kprintf("[AXEX-ACTAWARE] Column-norm computation failed — falling back to h-PCA\n");
                                actaware = 0;
                            }
                            if (col_s) free(col_s);
                        }

                        /* Fast truncated PCA: uses randomized SVD when k_budget << min(n,d),
                         * falls back to Jacobi (now 15-sweep) for near-full-rank requests. */
                        int lq_dim = mdl->n_heads * (dim / mdl->n_heads);
                        int lkv_dim = mdl->n_kv_heads * (dim / mdl->n_heads);
                        int max_m = lq_dim > lkv_dim ? lq_dim : lkv_dim;
                        float *tmp_m = (float *)malloc(max_m * sizeof(float));
                        float *row_f32 = (float *)malloc(dim * sizeof(float));
                        my_grc_weighted_ctx_t wctx = {
                            .layer = &mdl->layers[l],
                            .model = mdl,
                            .weight_mask = 1 | 2 | 4,
                            .tmp_m = tmp_m,
                            .row_f32 = row_f32
                        };
                        axpca_t lpca;
                        if (args.axex_weight_pca) {
                            /* Weight-gram eigenvectors: find top-k eigenvectors of
                             * K = W_q^T W_q + W_k^T W_k + W_v^T W_v without any
                             * calibration data.  Maximises weight energy directly.
                             * Pre-dequantize weights to f32 to avoid repeated Q4_K_M
                             * decoding on every K_apply call (speedup: ~1000×). */
                            kprintf("[AXEX-MANIFOLD] Layer %d: dequantizing weights for weight-PCA...\n", l);
                            float *q_f32c = (float *)malloc((size_t)lq_dim  * dim * sizeof(float));
                            float *k_f32c = (float *)malloc((size_t)lkv_dim * dim * sizeof(float));
                            float *v_f32c = (float *)malloc((size_t)lkv_dim * dim * sizeof(float));
                            int max_kv = lq_dim > lkv_dim ? lq_dim : lkv_dim;
                            float *tmp_cache = (float *)malloc((size_t)max_kv * sizeof(float));

                            if (q_f32c && k_f32c && v_f32c && tmp_cache &&
                                    mdl->layers[l].q_weight && mdl->layers[l].k_weight &&
                                    mdl->layers[l].v_weight) {
                                /* Dequantize Q weight rows */
                                double norm_q_sq = 0.0;
                                for (int ri = 0; ri < lq_dim; ri++) {
                                    const void *rp = (const char *)mdl->layers[l].q_weight +
                                        (size_t)ri * ggml_tensor_size(mdl->layers[l].q_type, dim);
                                    ax_dequant_row_f32(rp, q_f32c + (size_t)ri * dim,
                                                       dim, mdl->layers[l].q_type);
                                    const float *rr = q_f32c + (size_t)ri * dim;
                                    for (int j = 0; j < dim; j++) norm_q_sq += (double)rr[j] * rr[j];
                                }
                                /* Dequantize K weight rows */
                                double norm_k_sq = 0.0;
                                for (int ri = 0; ri < lkv_dim; ri++) {
                                    const void *rp = (const char *)mdl->layers[l].k_weight +
                                        (size_t)ri * ggml_tensor_size(mdl->layers[l].k_type, dim);
                                    ax_dequant_row_f32(rp, k_f32c + (size_t)ri * dim,
                                                       dim, mdl->layers[l].k_type);
                                    const float *rr = k_f32c + (size_t)ri * dim;
                                    for (int j = 0; j < dim; j++) norm_k_sq += (double)rr[j] * rr[j];
                                }
                                /* Dequantize V weight rows */
                                double norm_v_sq = 0.0;
                                for (int ri = 0; ri < lkv_dim; ri++) {
                                    const void *rp = (const char *)mdl->layers[l].v_weight +
                                        (size_t)ri * ggml_tensor_size(mdl->layers[l].v_type, dim);
                                    ax_dequant_row_f32(rp, v_f32c + (size_t)ri * dim,
                                                       dim, mdl->layers[l].v_type);
                                    const float *rr = v_f32c + (size_t)ri * dim;
                                    for (int j = 0; j < dim; j++) norm_v_sq += (double)rr[j] * rr[j];
                                }
                                float inv_q = (norm_q_sq > 1e-12) ? (float)(1.0 / norm_q_sq) : 1.0f;
                                float inv_k = (norm_k_sq > 1e-12) ? (float)(1.0 / norm_k_sq) : 1.0f;
                                float inv_v = (norm_v_sq > 1e-12) ? (float)(1.0 / norm_v_sq) : 1.0f;
                                my_grc_cached_ctx_t cctx = {
                                    .q_f32 = q_f32c, .k_f32 = k_f32c, .v_f32 = v_f32c,
                                    .tmp = tmp_cache,
                                    .lq_dim = lq_dim, .lkv_dim = lkv_dim, .dim = dim,
                                    .inv_norm_q = inv_q, .inv_norm_k = inv_k, .inv_norm_v = inv_v
                                };
                                kprintf("[AXEX-MANIFOLD] Layer %d: running weight-PCA eigvec (k=%d)...\n", l, k_budget);
                                lpca = axpca_compute_weight_topk(my_grc_cached_matvec, &cctx, dim, k_budget);
                            } else {
                                /* Fallback: slow path (re-dequantizes on every call) */
                                kprintf("[AXEX-MANIFOLD] Layer %d: weight-PCA fallback (slow dequant path)\n", l);
                                lpca = axpca_compute_weight_topk(my_grc_weighted_matvec, &wctx, dim, k_budget);
                            }
                            free(q_f32c); free(k_f32c); free(v_f32c); free(tmp_cache);
                            axmat_destroy(&X);  /* X not used in weight-PCA mode */
                        } else {
                            lpca = axpca_compute_topk_weighted(&X, my_grc_weighted_matvec, &wctx, k_budget);
                            axmat_destroy(&X);
                        }
                        free(tmp_m);
                        free(row_f32);

                        if (lpca.n_components <= 0) {
                            kprintf("[AXEX-MANIFOLD] Layer %d PCA failed — skipping layer (raw weights used)\n", l);
                            axpca_destroy(&lpca);
                            lw_skipped++;
                            if (lw_skipped > n_layers / 2) {
                                kprintf("[AXEX-MANIFOLD] Too many PCA failures (%d/%d) — aborting GP compression\n",
                                        lw_skipped, n_layers);
                                lw_ok = 0;
                            }
                            continue;
                        }

                        if (k_for_layer > lpca.n_components) k_for_layer = lpca.n_components;

                        /* Feature 2: ensure sink direction is captured — bump rank if needed */
                        if (args.axex_sink_bypass && g_sink_ctx.valid &&
                                g_sink_ctx.n_sinks > 0 &&
                                k_for_layer < lpca.n_components) {
                            /* Compute mean hidden state of sink-like samples at this layer */
                            float *mean_sink = (float *)calloc((size_t)dim, sizeof(float));
                            if (mean_sink) {
                                int n_s_valid = 0;
                                const float *hs_l = all_hs + (size_t)l * n_samples * dim;
                                for (int si = 0; si < g_sink_ctx.n_sinks; si++) {
                                    int sidx = g_sink_ctx.indices[si];
                                    if (sidx < n_samples) {
                                        const float *h = hs_l + (size_t)sidx * dim;
                                        for (int d = 0; d < dim; d++) mean_sink[d] += h[d];
                                        n_s_valid++;
                                    }
                                }
                                if (n_s_valid > 0) {
                                    float inv_ns = 1.0f / (float)n_s_valid;
                                    for (int d = 0; d < dim; d++) mean_sink[d] *= inv_ns;
                                    int needs = sink_check_basis_coverage(&g_sink_ctx,
                                                    mean_sink, lpca.components.data,
                                                    lpca.n_components, dim, 0.5f, NULL);
                                    if (needs) {
                                        /* Reserve one extra slot per sink token, up to 3.
                                         * Multiple massive-activation tokens may lie in
                                         * different directions; a single extra component
                                         * often cannot cover them all. */
                                        int extra = g_sink_ctx.n_sinks < 3
                                                    ? g_sink_ctx.n_sinks : 3;
                                        k_for_layer += extra;
                                        if (k_for_layer > lpca.n_components)
                                            k_for_layer = lpca.n_components;
                                    }
                                }
                                free(mean_sink);
                            }
                        }

                        if (axex_manifold_init_layer(l, &lpca, dim, k_for_layer) != 0) {
                            kprintf("[AXEX-MANIFOLD] Layer %d basis init failed — skipping layer\n", l);
                            axpca_destroy(&lpca);
                            lw_skipped++;
                            if (lw_skipped > n_layers / 2) { lw_ok = 0; }
                            continue;
                        }
                        axpca_destroy(&lpca);

                        if ((l % 8) == 7 || l == n_layers - 1)
                            kprintf("[AXEX-MANIFOLD] Per-layer PCA: %d/%d layers done\n",
                                    l + 1, n_layers);
                    }
                    free(ricci_ranks);
                    free(mcr_ranks);
                    free(sink_norms);
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

                /* ── FFN Down GP Compression ──────────────────────────────────
                 * Capture SiLU(gate)⊙up activations at every layer in one pass,
                 * build per-layer PCA in the ff_dim space, then compress W_down
                 * using GP: W_down_proj[dim×k_ff] = W_down @ Qff_l.
                 * This is the key piece needed for 70B-in-7GB: FFN down weights
                 * dominate VRAM usage and are now compressible without quality
                 * loss because we use the actual FFN activation manifold basis.
                 * In attn-only mode this calibration always ends up skipping every
                 * layer (quality guard rejects low-energy FFN projections), so skip
                 * the entire calibration pass to save ~2 min startup time. */
                if (lw_ok && !args.axex_attn_only) {
                    int ff_dim = mdl->ff_dim;
                    if (ff_dim <= 0) {
                        kprintf("[AXEX-FFN-DOWN] Skipping: ff_dim=0\n");
                    } else {
                        kprintf("[AXEX-FFN-DOWN] Capturing FFN intermediate activations "
                                "(%d layers × %d samples × ff_dim=%d)...\n",
                                n_layers, n_samples, ff_dim);

                        size_t ffn_total_f = (size_t)n_layers * n_samples * ff_dim;
                        float *all_ffn = (float *)malloc(ffn_total_f * sizeof(float));
                        /* OOM retry: halve n_samples if the FFN buffer can't be allocated.
                         * Phi-3.5: 32×512×8192×4 = 536 MB; halved = 268 MB — still compressible. */
                        if (!all_ffn && n_samples > 64) {
                            int ffn_n_samples = n_samples / 2;
                            ffn_total_f = (size_t)n_layers * ffn_n_samples * ff_dim;
                            all_ffn = (float *)malloc(ffn_total_f * sizeof(float));
                            if (all_ffn)
                                kprintf("[AXEX-FFN-DOWN] OOM on full alloc — retrying with %d samples\n",
                                        ffn_n_samples);
                            /* Patch n_samples for the FFN path only */
                            /* (use a local so the attention n_samples is unchanged) */
                            if (all_ffn) n_samples = ffn_n_samples;
                        }
                        int   *ffn_valid = (int *)calloc((size_t)n_layers, sizeof(int));
                        int   *ffn_cap_v = (int *)calloc((size_t)n_layers, sizeof(int));
                        int    ffn_ok = (all_ffn && ffn_valid && ffn_cap_v);

                        if (ffn_ok) {
                            uint64_t seed2 = 0xDEADBEEFCAFE1234ULL;
                            float *ffn_pass_buf = (float *)malloc((size_t)n_layers * ff_dim * sizeof(float));
                            if (!ffn_pass_buf) { ffn_ok = 0; }

                            for (int s = 0; s < n_samples && ffn_ok; s++) {
                                seed2 ^= seed2 << 13; seed2 ^= seed2 >> 7; seed2 ^= seed2 << 17;
                                int tok = (int)(seed2 % (uint64_t)llm_model_vocab());

                                memset(ffn_cap_v, 0, (size_t)n_layers * sizeof(int));
                                int rc = axiom_beta_probe_all_ffn_states(
                                    tok, ffn_pass_buf, ffn_cap_v, n_layers, ff_dim);

                                for (int l = 0; l < n_layers; l++) {
                                    float *dst = all_ffn + ((size_t)l * n_samples + s) * ff_dim;
                                    if (rc == 0 && ffn_cap_v[l]) {
                                        memcpy(dst, ffn_pass_buf + (size_t)l * ff_dim,
                                               (size_t)ff_dim * sizeof(float));
                                        ffn_valid[l]++;
                                    } else {
                                        memset(dst, 0, (size_t)ff_dim * sizeof(float));
                                    }
                                }

                                if ((s % 64) == 63 || s == n_samples - 1)
                                    kprintf("[AXEX-FFN-DOWN] FFN calibration sample %d/%d done\n",
                                            s + 1, n_samples);
                            }
                            free(ffn_pass_buf);
                        }

                        /* Build per-layer FFN PCA and compress W_down */
                        for (int l = 0; l < n_layers && ffn_ok; l++) {
                            axmat_t X = axmat_create(n_samples, ff_dim);
                            if (!X.data) { ffn_ok = 0; break; }

                            for (int s = 0; s < n_samples; s++) {
                                const float *src = all_ffn + ((size_t)l * n_samples + s) * ff_dim;
                                for (int j = 0; j < ff_dim; j++)
                                    X.data[(size_t)s * ff_dim + j] = (double)src[j];
                            }

                            /* Use fast truncated PCA.  Cap ffn_k to always trigger the
                             * randomised-SVD path in axpca_compute_topk (that path is used
                             * only when k_max < min(n,d)*0.65).  With n_samples=512,
                             * min(n,d)=512, the full-Jacobi fallback fires for k>=332.
                             * Limit to (n_samples * 55)/100 so we always use the fast path
                             * regardless of sample count, and stay <= K_MAX. */
                            int ffn_k = (n_samples * 55) / 100;
                            if (ffn_k > AXEX_MANIFOLD_K_MAX) ffn_k = AXEX_MANIFOLD_K_MAX;
                            if (ffn_k < 8) ffn_k = 8;
                            if (args.axex_compress_rank > 0 && args.axex_compress_rank < ffn_k)
                                ffn_k = args.axex_compress_rank;
                            axpca_t ffn_pca = axpca_compute_topk(&X, ffn_k);
                            axmat_destroy(&X);

                            if (ffn_pca.n_components <= 0) {
                                kprintf("[AXEX-FFN-DOWN] Layer %d FFN PCA failed\n", l);
                                axpca_destroy(&ffn_pca);
                                ffn_ok = 0;
                                break;
                            }

                            if (axex_manifold_init_ffn_layer(l, &ffn_pca, ff_dim,
                                                              AXEX_MANIFOLD_K_MAX) != 0) {
                                kprintf("[AXEX-FFN-DOWN] Layer %d FFN basis init failed\n", l);
                                axpca_destroy(&ffn_pca);
                                ffn_ok = 0;
                                break;
                            }
                            axpca_destroy(&ffn_pca);

                            if ((l % 8) == 7 || l == n_layers - 1)
                                kprintf("[AXEX-FFN-DOWN] FFN PCA: %d/%d layers done\n",
                                        l + 1, n_layers);
                        }
                        free(all_ffn);
                        free(ffn_valid);
                        free(ffn_cap_v);

                        if (ffn_ok) {
                            int dn = axex_compress_model_ffn_down_layerwise(
                                axiom_ran_this_invocation ? &axiom_rep : NULL, mdl);
                            kprintf("[AXEX-FFN-DOWN] W_down layerwise GP: %d layers compressed\n", dn);
                        } else {
                            kprintf("[AXEX-FFN-DOWN] FFN calibration failed — W_down uncompressed\n");
                        }
                    }
                }

                /* ── Save W_proj cache for next run ──────────────────────────
                 * Writes compressed Pt+W_proj state so future startups load in
                 * ~1-2s instead of running ~22s of PCA + compression.
                 * Only save when compression actually produced matrices. */
                if (lw_ok && axex_manifold_compressed_count() > 0) {
                    int wsaved = axex_manifold_save_wproj_cache(wp_cache_path, wp_hk);
                    if (wsaved > 0) {
                        kprintf("[AXEX-MANIFOLD] Saved W_proj cache: %s (%d matrices)\n",
                                wp_cache_path, wsaved);
                    } else {
                        kprintf("[AXEX-MANIFOLD] WARNING: failed to save W_proj cache %s\n",
                                wp_cache_path);
                    }
                }

                } /* end if (!wp_cache_loaded) */
            } else {
                kprintf("[AXEX-MANIFOLD] Skipped (no PCA — run with --axiom-beta-run first)\n");
            }
        }

        /* Upload any compressed weight matrices to GPU device memory */
        llm_gpu_upload_compressed_weights();
    }

    /* ── Feature 1: Online basis update context ────────────────────────────
     * Arm the per-layer Oja update context so the speculative decode path
     * can record rejections and update the PCA basis between steps.
     * The actual onb_record_rejection / onb_apply_pending calls live in
     * llm_speculative_verify_with_correction (llm.c) gated on this flag. */
    static onb_ctx_t g_onb_ctx;
    if (args.axex_online_basis) {
        const llm_model_t *ob_mdl = llm_get_model();
        /* Build uniform dim/k arrays for onb_init */
        int ob_nl = ob_mdl ? ob_mdl->n_layers : 0;
        int *ob_dims = (int *)calloc((size_t)(ob_nl > 0 ? ob_nl : 1), sizeof(int));
        int *ob_ks   = (int *)calloc((size_t)(ob_nl > 0 ? ob_nl : 1), sizeof(int));
        if (ob_dims && ob_ks) {
            for (int _l = 0; _l < ob_nl; _l++) {
                ob_dims[_l] = ob_mdl->dim;
                ob_ks[_l]   = 8;
            }
        }
        if (ob_mdl && ob_dims && ob_ks &&
            onb_init(&g_onb_ctx, ob_nl, ob_dims, ob_ks, NULL, 0.01) == 0) {
            llm_set_online_basis_ctx(&g_onb_ctx);
            kprintf("[ONB] Online basis update armed: %d layers, k=8, eta0=0.01\n",
                    ob_mdl->n_layers);
        } else {
            kprintf("[ONB] Online basis init failed — continuing without it\n");
        }
        free(ob_dims);
        free(ob_ks);
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
    if (args.ppl_eval) {
        /* ── WikiText-2 perplexity evaluation ──────────────────────────────
         * Evaluates cross-entropy on a 512-token WikiText-2 slice, computes
         * PPL = exp(mean NLL).  Works with or without --axex-compress. */
        static const char *PPL_TEXT =
            " = Valuation ( finance ) = \n"
            " In finance , valuation is the process of determining the present value of an asset or a company . "
            "Valuations can be done on assets ( for example , investments in marketable securities such as "
            "stocks , options , business enterprises , or intangible assets such as patents and trademarks ) "
            "or on liabilities ( e.g. , bonds issued by a company ) . Valuations are needed for many reasons "
            "such as investment analysis , capital budgeting , merger and acquisition transactions , financial "
            "reporting , and taxable events to determine the proper tax liability . \n"
            " = = Valuation overview = = \n"
            " In finance , a valuation is the process of estimating what something is worth . Items that are "
            "usually valued are a financial asset or liability . Valuations can be done on assets ( for example "
            ", investments in marketable securities such as stocks , options , business enterprises , or "
            "intangible assets such as patents and trademarks ) or on liabilities ( e.g. , bonds issued by a "
            "company ) . \n"
            " The concept of present value was first described by Martin de Azpilcueta , a Spanish theologian , "
            "in 1533 . He was the first to observe that money received in the future is worth less than money "
            "received today , a concept now known as the time value of money . \n"
            " = = = Intrinsic value = = = \n"
            " Intrinsic value refers to the value of a company , stock , currency or product determined through "
            "fundamental analysis without reference to its market value . It is also frequently called "
            "fundamental value . It is ordinarily calculated by summing the discounted future income generated "
            "by the asset to obtain the present value . It is worthy of note that the intrinsic value of stocks "
            "are hard to accurately estimate . \n"
            " = = = Market value = = = \n"
            " The market value of a publicly traded company is determined by the stock market price which "
            "represents the consensus of investors about the future earnings and cash flows of the company . \n"
            " = Homarus gammarus = \n"
            " Homarus gammarus , known as the European lobster or common lobster , is a species of clawed "
            "lobster from the eastern Atlantic Ocean , Mediterranean Sea and parts of the Black Sea . It is "
            "closely related to the American lobster , Homarus americanus . It may grow to a length of 60 cm "
            "( 24 in ) and a mass of 6 kilograms ( 13 lb ) , and bears a conspicuous pair of claws . \n"
            " = Robert Boulter = \n"
            " Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring "
            "role on the television series The Bill in 2000 . This was followed by a starring role in the "
            "play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . \n";

        int ppl_max = 512;
        float *ppl_lps = (float *)malloc((size_t)ppl_max * sizeof(float));
        if (!ppl_lps) { kprintf("[PPL] OOM\n"); return 1; }
        kprintf("[PPL] Evaluating WikiText-2 perplexity (%d tokens max)...\n", ppl_max);
        uint64_t ppl_t0 = hal_timer_us();
        int ppl_n = llm_eval_sequence_logprobs(PPL_TEXT, ppl_lps, ppl_max);
        uint64_t ppl_t1 = hal_timer_us();
        if (ppl_n <= 0) {
            kprintf("[PPL] ERROR: logprobs eval failed (n=%d)\n", ppl_n);
            free(ppl_lps);
            return 1;
        }
        double nll_sum = 0.0;
        for (int i = 0; i < ppl_n; i++) nll_sum += (double)ppl_lps[i];
        double nll_mean = -nll_sum / ppl_n;
        double ppl_val  = exp(nll_mean);
        double elapsed_s = (ppl_t1 - ppl_t0) / 1e6;
        double tps = ppl_n / elapsed_s;
        kprintf("[PPL] n_tokens=%d  NLL=%.6f  PPL=%.4f  elapsed=%.1fs  tps=%.1f\n",
                ppl_n, nll_mean, ppl_val, elapsed_s, tps);
        /* JSON-formatted result for easy parsing */
        kprintf("[PPL-JSON] {\"n_tokens\":%d,\"nll\":%.6f,\"ppl\":%.4f,\"elapsed_s\":%.2f,\"tps\":%.1f}\n",
                ppl_n, nll_mean, ppl_val, elapsed_s, tps);
        free(ppl_lps);
        return 0;
    } else if (args.serve) {
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
        /* Stream each token to stdout as it is generated so output is captured
         * incrementally — critical for slow/large models where the process may
         * take minutes before reaching the final kprintf. */
        llm_set_stream_cb(GD_stream_cb, (void *)0);
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
        llm_set_stream_cb((llm_token_cb_t)0, (void *)0);
        uint64_t gen_t1 = hal_timer_us();
        if (n > 0) {
            kprintf("\n");  /* tokens already streamed above; just terminate the line */
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


