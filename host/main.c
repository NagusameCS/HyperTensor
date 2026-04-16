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
#include "api_server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#ifdef _WIN32
#  include <windows.h>
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
    kprintf("  --ott-spec-thresh N    Confidence threshold for draft acceptance (default: 0.65)\n");
    kprintf("  --ott-spec-batch N     Draft batch size for speculative decode (default: 4)\n");
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
    float       ott_speculative_thresh; /* geodesic confidence threshold [0,1], default 0.65 */
    int         ott_speculative_batch;  /* draft batch size (default 4) */
    int         ctx_size;   /* 0 = default (2048); user override via --ctx-size */
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
    args->max_tokens  = 128;
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
    args->ott_speculative_thresh = 0.65f;
    args->ott_speculative_batch = 4;
    args->ctx_size = 0;

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
            /* Speed-first OTT: axiom geometry primed, fast decode, no deep theorem mode. */
            args->ott_full = 1;
            args->axiom_beta_run = 1;
            args->axiom_fast_mode = 1;
            args->axiom_geodesic_first = 1;
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
            args->attnres_enable = 1;
            args->attnres_strength = 0.25f;
            args->top_k = 20;
            args->top_p = 0.9f;
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
    int   geodesic_mode = args->axiom_geodesic_first;
    static int geo_ctx_tokens[32768];
    int   geo_n_ctx = 0;

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
            n = llm_chat_turn(line, output, (int)sizeof(output), max_tokens, temperature);
        }
        uint64_t t1 = hal_timer_us();

        printf("\n");

        if (n > 0) {
            uint64_t total_ms   = (t1 - t0) / 1000;
            float    tok_per_s  = total_ms > 0 ? (float)n * 1000.0f / (float)total_ms : 0.0f;
                 int ctx    = geodesic_mode ? geo_n_ctx : llm_chat_context_tokens();
                 int ctxmax = geodesic_mode ? (int)(sizeof(geo_ctx_tokens) / sizeof(geo_ctx_tokens[0]))
                                : llm_chat_context_max();
            printf(GD_DIM "  [%d tok  %.1f tok/s  %llu ms  ctx %d/%d]\n\n" GD_RESET,
                   n, tok_per_s, (unsigned long long)total_ms, ctx, ctxmax);
        } else {
            printf(GD_RED "  [error generating response (code %d)]\n\n" GD_RESET, n);
        }
    }

    llm_set_stream_cb((llm_token_cb_t)0, (void *)0);
    printf("\n" GD_DIM "  [Session ended]\n" GD_RESET "\n");
}

/* Forward declaration — defined later in this file. */
static int geodesic_piece_quality_ok(const char *piece, int piece_n);

static int geodesic_next_token_local(const int *ctx, int n_ctx, int *out_tok) {
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
    for (int i = 0; i < piece_n; i++) {
        unsigned char c = (unsigned char)piece[i];
        if (c < 32 && c != '\n' && c != '\r' && c != '\t') control++;
        if (c >= 128) high++;
        if (isalnum((int)c) || c == ' ' || c == '.' || c == ',' || c == ';' ||
            c == ':' || c == '!' || c == '?' || c == '\'' || c == '"' ||
            c == '(' || c == ')' || c == '-' || c == '_') useful++;
    }
    if (control > 0) return 0;
    /* Allow short high-byte pieces (UTF-8 sub-word tokens like diacritics) */
    if (piece_n >= 4 && useful == 0 && high * 2 >= piece_n) return 0;
    if (piece_n >= 6 && useful <= 1 && high * 4 >= piece_n * 3) return 0;
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
            kprintf("[GD] Geodesic output quality gate failed; requesting standard decode fallback.\n");
            s_runtime_hits = runtime_hits;
            s_local_hits = local_hits;
            s_runtime_rejects = runtime_rejects;
            s_quality_gate_pass = 0;
            s_generated = generated;
            s_fallback_requested = 1;
            GD_geodesic_last_runtime_hits = s_runtime_hits;
            GD_geodesic_last_local_hits = s_local_hits;
            GD_geodesic_last_runtime_rejects = s_runtime_rejects;
            GD_geodesic_last_quality_gate_pass = s_quality_gate_pass;
            GD_geodesic_last_generated = s_generated;
            GD_geodesic_last_fallback_requested = s_fallback_requested;
            free(ctx);
            return -1;
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
                                              char *output,
                                              int max_output)
{
    const llm_model_t *m = llm_get_model();
    if (!m || !prompt || !output || max_output <= 0 || max_tokens <= 0)
        return -1;

    if (batch_size < 1) batch_size = 1;
    if (batch_size > 16) batch_size = 16;

    int max_ctx = max_tokens + 2048;
    int *ctx     = (int *)malloc((size_t)max_ctx * sizeof(int));
    int *drafts  = (int *)malloc((size_t)batch_size * sizeof(int));
    if (!ctx || !drafts) { free(ctx); free(drafts); return -1; }

    int n_ctx = llm_test_tokenize(prompt, (int)strlen(prompt), ctx, max_ctx);
    if (n_ctx <= 0 || n_ctx >= max_ctx) {
        free(ctx); free(drafts); return -1;
    }

    output[0] = '\0';
    int out_len   = 0;
    int generated = 0;
    int geo_accepted = 0;
    int xfmr_accepted = 0;
    int eos = m->eos_id;

    kprintf("[SPEC] Starting OTT speculative decode (batch=%d, thresh=%.2f)\n",
            batch_size, (double)conf_thresh);

    while (generated < max_tokens && n_ctx < max_ctx) {
        /* ── Step 1: collect up to batch_size geodesic draft tokens ────────────
         * Use axiom_beta_geodesic_rollout() which integrates a trajectory-coherent
         * geodesic path — carrying the velocity vector between steps so curvature
         * corrections accumulate properly.  Falls back to individual step_fast calls
         * if the rollout is unavailable.
         * ─────────────────────────────────────────────────────────────────────── */
        int n_drafts = 0;
        float min_conf = 1.0f;

        /* Attempt multi-step rollout first */
        {
            int   roll_toks[16];
            float roll_conf[16];
            int   roll_n = 0;
            int   need   = batch_size;
            if (need > 16) need = 16;

            if (axiom_beta_geodesic_rollout(ctx, n_ctx, need,
                                            roll_toks, roll_conf, &roll_n)
                == AXIOM_BETA_OK && roll_n > 0) {
                /* Accept rollout tokens that pass threshold and quality gate */
                for (int d = 0; d < roll_n && n_drafts < batch_size; d++) {
                    if (roll_conf[d] < conf_thresh) break;
                    int draft_tok = roll_toks[d];
                    if (draft_tok < 0 || draft_tok >= m->vocab_size) break;
                    char piece[256];
                    int pn = llm_test_decode_token(draft_tok, piece, (int)sizeof(piece));
                    if (pn <= 0 || !geodesic_piece_quality_ok(piece, pn)) break;
                    drafts[n_drafts++] = draft_tok;
                    if (roll_conf[d] < min_conf) min_conf = roll_conf[d];
                }
            }
        }

        /* Fallback: fill remaining slots with individual step_fast calls */
        for (int d = n_drafts; d < batch_size && n_ctx + d < max_ctx - 1; d++) {
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

        /* ── Step 2: verify drafts (or get a fresh transformer token) ── */
        if (n_drafts > 0) {
            int correction = -1;
            int n_acc = llm_speculative_verify_with_correction(
                ctx, n_ctx, drafts, n_drafts, &correction);

            if (n_acc < 0) {
                /* verification error — fall back to standard generation */
                break;
            }

            /* Emit accepted draft tokens */
            for (int i = 0; i < n_acc && generated < max_tokens; i++) {
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
            }

            /* If not all drafts were accepted, emit correction token */
            if (n_acc < n_drafts && correction >= 0 &&
                correction < m->vocab_size && generated < max_tokens) {
                int tok = correction;
                if (eos >= 0 && tok == eos) goto spec_done;
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
            /* No high-confidence drafts — run transformer directly for one token */
            int out_tok = -1;
            int n_gen = llm_generate_tokens(ctx, n_ctx, &out_tok, 1, 1, 0.0f, 0);
            if (n_gen <= 0 || out_tok < 0 || out_tok >= m->vocab_size)
                break;

            int tok = out_tok;
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
            xfmr_accepted++;
        }
    }

spec_done:
    kprintf("[SPEC] Done: %d tokens (geo_accepted=%d xfmr=%d, acceptance_rate=%.1f%%)\n",
            generated, geo_accepted, xfmr_accepted,
            generated > 0 ? 100.0f * (float)geo_accepted / (float)generated : 0.0f);

    GD_geodesic_last_runtime_hits  = geo_accepted;
    GD_geodesic_last_local_hits    = 0;
    GD_geodesic_last_runtime_rejects = xfmr_accepted;
    GD_geodesic_last_quality_gate_pass = (generated > 0) ? 1 : 0;
    GD_geodesic_last_generated     = generated;
    GD_geodesic_last_fallback_requested = 0;

    free(ctx);
    free(drafts);
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

    /* Apply context-size override before loading (0 = use 2048-token default) */
    llm_set_max_ctx(args.ctx_size);

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

    if (args.axiom_beta_run) {
        axiom_beta_config_t cfg;
        axiom_beta_report_t rep;
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

        st = axiom_beta_run(&cfg, &rep);
        if (st != AXIOM_BETA_OK) {
            kprintf("[AXIOM-BETA-3] ERROR: %s\n", axiom_beta_status_string(st));
            hal_munmap(&model);
            hal_shutdown();
            return 1;
        }
        axiom_ran_this_invocation = 1;
        axiom_intrinsic_dim = rep.phase1.intrinsic_dim;
        axiom_consistency = rep.phase4.consistency_score;
        axiom_geodesic_speedup = rep.phase5.projected_speedup;
        axiom_total_ms = (unsigned long long)(rep.total_us / 1000);

        kprintf("\n[AXIOM-BETA-3] === Summary ==========================\n");
        kprintf("  Intrinsic dim : %d (TwoNN=%.2f, PCA=%d components)\n",
                rep.phase1.intrinsic_dim, rep.phase1.twonn_raw,
                rep.phase1.pca_components_kept);
        kprintf("  Symmetry      : score=%.4f, generators=%d, invariant=%d\n",
                rep.phase2.symmetry_score, rep.phase2.generators_found,
                rep.phase2.permutation_invariant_heads);
        kprintf("  Curvature     : mean=%.6f, max=%.6f, high-curv=%d\n",
                rep.phase3.mean_scalar_curvature,
                rep.phase3.max_scalar_curvature,
                rep.phase3.high_curvature_loci);
        if (rep.uses_fisher_metric)
            kprintf("  Fisher metric : trace_mean=%.4f, det_log_mean=%.4f\n",
                    rep.phase3.fisher_trace_mean,
                    rep.phase3.fisher_det_log_mean);
        kprintf("  Axioms        : %d (consistency=%.4f, oracle=%d)\n",
                rep.phase4.axiom_count, rep.phase4.consistency_score,
                rep.phase4.oracle_calls_used);
        kprintf("  Geodesic pilot: speedup=%.1fx\n",
                rep.phase5.projected_speedup);
        kprintf("  Cache reuse   : %s\n",
            rep.reused_geometry_cache ? "yes" : "no");
        if (rep.supports_geodesic_pilot)
            kprintf("  Geodesic sim  : cos=%.4f, L2_err=%.4f\n",
                    rep.phase5.geodesic_cosine_similarity,
                    rep.phase5.geodesic_reconstruction_error);
        if (rep.supports_geodesic_pilot)
            kprintf("  Geodesic tok  : top1=%.3f (%d/%d), mrr=%.3f, probe=%d, targets(o/r)=%d/%d, gpu=%s\n",
                    rep.phase5.geodesic_top1_match_rate,
                    rep.phase5.geodesic_top1_hits,
                    rep.phase5.pilot_tokens_tested,
                    rep.phase5.geodesic_target_mrr,
                rep.phase5.geodesic_vocab_probe,
                rep.phase5.oracle_target_count,
                rep.phase5.random_target_count,
                rep.phase5.used_gpu_scoring ? "yes" : "no");
        kprintf("  Total time    : %llu ms\n",
                (unsigned long long)(rep.total_us / 1000));
        kprintf("[AXIOM-BETA-3] ======================================\n");

        if (args.axiom_report_path) {
            axiom_beta_status_t wr = axiom_beta_write_json(args.axiom_report_path,
                                                           &rep, &cfg);
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
