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
#ifdef _WIN32
#  include <windows.h>
#endif

#define GD_VERSION_MAJOR 0
#define GD_VERSION_MINOR 5
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
    kprintf("  --axiom-beta-run       Run autonomous axiomatic beta-3 survey\n");
    kprintf("  --axiom-beta-only      Run axiomatic beta survey then exit\n");
    kprintf("  --axiom-report <path>  Write beta report JSON (default: axiom_beta_report.json)\n");
    kprintf("  --axiom-samples <n>    Embedding samples for manifold identification (default: 256)\n");
    kprintf("  --axiom-probe <n>      Phase 5 endpoint token probe count (default: 1024)\n");
    kprintf("  --axiom-gpu            Enable GPU acceleration for Phase 5 scorer\n");
    kprintf("  --axiom-no-gpu         Force CPU-only Phase 5 scorer\n");
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
    int         axiom_beta_run;
    int         axiom_beta_only;
    int         axiom_samples;
    int         axiom_vocab_probe;
    int         axiom_use_gpu_phase5;
    uint64_t    axiom_seed;
    int         axiom_skip_geodesic;
    const char *axiom_report_path;
} GD_args_t;

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
    args->axiom_beta_run = 0;
    args->axiom_beta_only = 0;
    args->axiom_samples = 256;
    args->axiom_vocab_probe = 1024;
    args->axiom_use_gpu_phase5 = -1;
    args->axiom_seed = 0;
    args->axiom_skip_geodesic = 0;
    args->axiom_report_path = "axiom_beta_report.json";

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
        } else if (strcmp(argv[i], "--axiom-seed") == 0) {
            if (++i >= argc) { kprintf("Error: --axiom-seed requires number\n"); return -1; }
#ifdef _WIN32
            args->axiom_seed = _strtoui64(argv[i], NULL, 10);
#else
            args->axiom_seed = strtoull(argv[i], NULL, 10);
#endif
        } else if (strcmp(argv[i], "--axiom-skip-geodesic") == 0) {
            args->axiom_skip_geodesic = 1;
        } else if (argv[i][0] != '-' && !args->model_path) {
            args->model_path = argv[i];
        } else {
            kprintf("Unknown option: %s\n", argv[i]);
            return -1;
        }
    }

    if (!args->model_path && !args->download_repo) {
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

static void interactive_loop(const char *model_path, GD_args_t *args) {
    (void)model_path;
    char line[2048];
    static char output[131072]; /* 128 KB — holds full reply if needed */
    float temperature = args->temperature;
    int   max_tokens  = args->max_tokens;

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
                printf(GD_DIM "  [Conversation reset — new context started]\n\n" GD_RESET);
                continue;
            }
            if (strcmp(line, "/stats") == 0) {
                int ctx    = llm_chat_context_tokens();
                int ctxmax = llm_chat_context_max();
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
        int n = llm_chat_turn(line, output, (int)sizeof(output), max_tokens, temperature);
        uint64_t t1 = hal_timer_us();

        printf("\n");

        if (n > 0) {
            uint64_t total_ms   = (t1 - t0) / 1000;
            float    tok_per_s  = total_ms > 0 ? (float)n * 1000.0f / (float)total_ms : 0.0f;
            int ctx    = llm_chat_context_tokens();
            int ctxmax = llm_chat_context_max();
            printf(GD_DIM "  [%d tok  %.1f tok/s  %llu ms  ctx %d/%d]\n\n" GD_RESET,
                   n, tok_per_s, (unsigned long long)total_ms, ctx, ctxmax);
        } else {
            printf(GD_RED "  [error generating response (code %d)]\n\n" GD_RESET, n);
        }
    }

    llm_set_stream_cb((llm_token_cb_t)0, (void *)0);
    printf("\n" GD_DIM "  [Session ended]\n" GD_RESET "\n");
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

    if (args.axiom_beta_run) {
        axiom_beta_config_t cfg;
        axiom_beta_report_t rep;
        axiom_beta_status_t st;
        axiom_beta_default_config(&cfg);
        if (args.axiom_samples > 0) cfg.embedding_samples = args.axiom_samples;
        if (args.axiom_vocab_probe > 0) cfg.geodesic_vocab_probe = args.axiom_vocab_probe;
        if (args.axiom_use_gpu_phase5 >= 0) cfg.use_gpu_phase5 = args.axiom_use_gpu_phase5;
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
        if (rep.supports_geodesic_pilot)
            kprintf("  Geodesic sim  : cos=%.4f, L2_err=%.4f\n",
                    rep.phase5.geodesic_cosine_similarity,
                    rep.phase5.geodesic_reconstruction_error);
        if (rep.supports_geodesic_pilot)
            kprintf("  Geodesic tok  : top1=%.3f (%d/%d), mrr=%.3f, probe=%d, gpu=%s\n",
                    rep.phase5.geodesic_top1_match_rate,
                    rep.phase5.geodesic_top1_hits,
                    rep.phase5.pilot_tokens_tested,
                    rep.phase5.geodesic_target_mrr,
                rep.phase5.geodesic_vocab_probe,
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

    /* Run inference */
    if (args.serve) {
        kprintf("[GD] Starting API server on port %d...\n", args.port);
        GD_api_serve(args.port);
    } else if (args.interactive) {
        interactive_loop(args.model_path, &args);
    } else {
        const char *prompt = args.prompt ? args.prompt : "Hello";
        kprintf("[GD] Prompt: \"%s\"\n", prompt);
        kprintf("[GD] Generating %d tokens...\n\n", args.max_tokens);

        static char output[65536];
        uint64_t gen_t0 = hal_timer_us();
        int n = llm_prompt_n(prompt, output, (int)sizeof(output), args.max_tokens);
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
        } else {
            kprintf("[error generating response]\n");
        }
    }

    /* Cleanup */
    hal_munmap(&model);
    hal_shutdown();

    return 0;
}
