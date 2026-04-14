/*
 * HyperTensor — Hosted Main Entry Point
 *
 * Loads a GGUF model from disk via memory-mapped I/O and runs LLM inference
 * using the TensorOS inference engine on the host CPU, with native threading.
 *
 * Usage: hypertensor <model.gguf> [prompt]
 */
#define _CRT_SECURE_NO_WARNINGS
#include "hal.h"

/* Forward declarations from TensorOS inference engine */
#include "../runtime/nn/llm.h"
#include "../runtime/nn/hf_download.h"
#include "api_server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#  include <windows.h>
#endif

#define HT_VERSION_MAJOR 0
#define HT_VERSION_MINOR 5
#define HT_VERSION_PATCH 0
#define HT_CODENAME      "Synapse"

static void print_banner(void) {
    kprintf("\n");
    kprintf("  ╔═══════════════════════════════════════════╗\n");
    kprintf("  ║  HyperTensor v%d.%d.%d \"%s\"              ║\n",
            HT_VERSION_MAJOR, HT_VERSION_MINOR, HT_VERSION_PATCH, HT_CODENAME);
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
    kprintf("  --serve                Start HyperTensor HTTP API server\n");
    kprintf("  --port <num>           API server port (default: 8080)\n");
    kprintf("  --download <repo>      Download model from HuggingFace\n");
    kprintf("  --quant <type>         Quantization hint for download (default: q4_0)\n");
    kprintf("  -v, --verbose          Enable debug logging\n");
    kprintf("  --log-level <n>        Log level: 0=error 1=warn 2=info 3=debug 4=trace\n");
    kprintf("  --no-think             Disable thinking (strip <|think|> blocks)\n");
    kprintf("  --force-think          Force thinking on all prompts\n");
    kprintf("  --show-think           Show thinking tokens in output\n");
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
} ht_args_t;

static int parse_args(int argc, char **argv, ht_args_t *args) {
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
static void ht_stream_cb(const char *text, int len, void *ud) {
    (void)ud;
    fwrite(text, 1, (size_t)len, stdout);
    fflush(stdout);
}

/* ANSI escape helpers */
#define HT_RESET   "\033[0m"
#define HT_BOLD    "\033[1m"
#define HT_DIM     "\033[2m"
#define HT_GREEN   "\033[32m"
#define HT_CYAN    "\033[36m"
#define HT_LBLUE   "\033[94m"
#define HT_YELLOW  "\033[33m"
#define HT_RED     "\033[31m"

static void ht_ansi_enable(void) {
#ifdef _WIN32
    HANDLE h    = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD  mode = 0;
    if (GetConsoleMode(h, &mode))
        SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif
}

static void print_chat_help(void) {
    printf(HT_CYAN "  Commands:\n" HT_RESET);
    printf(HT_CYAN "    /help       " HT_RESET "show this message\n");
    printf(HT_CYAN "    /reset      " HT_RESET "clear conversation (new context)\n");
    printf(HT_CYAN "    /stats      " HT_RESET "show context usage\n");
    printf(HT_CYAN "    /temp <n>   " HT_RESET "set sampling temperature   (e.g. /temp 0.8)\n");
    printf(HT_CYAN "    /tokens <n> " HT_RESET "set max tokens per reply   (e.g. /tokens 512)\n");
    printf(HT_CYAN "    /quit       " HT_RESET "exit\n\n");
}

static void interactive_loop(const char *model_path, ht_args_t *args) {
    (void)model_path;
    char line[2048];
    static char output[131072]; /* 128 KB — holds full reply if needed */
    float temperature = args->temperature;
    int   max_tokens  = args->max_tokens;

    ht_ansi_enable();

    /* Welcome header */
    printf("\n"
           HT_CYAN HT_BOLD
           "  ╔══════════════════════════════════════════════════╗\n"
           "  ║  HyperTensor Chat                                ║\n"
           HT_RESET);
    printf(HT_CYAN HT_BOLD "  ║  Model : %-41s║\n" HT_RESET, llm_model_name());
    printf(HT_CYAN HT_BOLD "  ║  Context: %-6d tokens  |  All CPUs active       ║\n"
           HT_RESET, llm_chat_context_max());
    printf(HT_CYAN HT_BOLD
           "  ╚══════════════════════════════════════════════════╝\n"
           HT_RESET "\n");
    printf(HT_DIM "  Type /help for commands.\n\n" HT_RESET);

    llm_set_stream_cb(ht_stream_cb, (void *)0);

    for (;;) {
        /* User input prompt */
        printf(HT_GREEN HT_BOLD "You: " HT_RESET);
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
                printf(HT_DIM "  [Conversation reset — new context started]\n\n" HT_RESET);
                continue;
            }
            if (strcmp(line, "/stats") == 0) {
                int ctx    = llm_chat_context_tokens();
                int ctxmax = llm_chat_context_max();
                int think  = llm_thinking_tokens();
                printf(HT_DIM "  [Context: %d / %d tokens (%.1f%%)  |  "
                       "temp=%.2f  max_tok=%d  think_last=%d]\n\n" HT_RESET,
                       ctx, ctxmax,
                       ctxmax > 0 ? 100.0f * ctx / ctxmax : 0.0f,
                       temperature, max_tokens, think);
                continue;
            }
            if (strncmp(line, "/temp ", 6) == 0) {
                temperature = (float)atof(line + 6);
                if (temperature < 0.0f) temperature = 0.0f;
                if (temperature > 2.0f) temperature = 2.0f;
                printf(HT_DIM "  [Temperature set to %.2f]\n\n" HT_RESET, temperature);
                continue;
            }
            if (strncmp(line, "/tokens ", 8) == 0) {
                max_tokens = atoi(line + 8);
                if (max_tokens < 1)    max_tokens = 1;
                if (max_tokens > 8192) max_tokens = 8192;
                printf(HT_DIM "  [Max tokens set to %d]\n\n" HT_RESET, max_tokens);
                continue;
            }
            printf(HT_YELLOW "  [Unknown command: %s — try /help]\n\n" HT_RESET, line);
            continue;
        }

        /* ── Generate response ─────────────────────────────── */
        printf("\n" HT_LBLUE HT_BOLD "AI: " HT_RESET);
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
            printf(HT_DIM "  [%d tok  %.1f tok/s  %llu ms  ctx %d/%d]\n\n" HT_RESET,
                   n, tok_per_s, (unsigned long long)total_ms, ctx, ctxmax);
        } else {
            printf(HT_RED "  [error generating response (code %d)]\n\n" HT_RESET, n);
        }
    }

    llm_set_stream_cb((llm_token_cb_t)0, (void *)0);
    printf("\n" HT_DIM "  [Session ended]\n" HT_RESET "\n");
}

int main(int argc, char **argv) {
    ht_args_t args;

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
            kprintf("[HT] ERROR: Download failed: %s\n", hf_download_error(&dl_ctx));
            hf_download_free(&dl_ctx);
            hal_shutdown();
            return 1;
        }

        /* If no model_path was given, use the downloaded file */
        if (!args.model_path) {
            static char dl_path[256];
            snprintf(dl_path, sizeof(dl_path), "%s", dl_ctx.output_path);
            args.model_path = dl_path;
            kprintf("[HT] Using downloaded model: %s\n", args.model_path);
        }
        hf_download_free(&dl_ctx);
    }

    if (!args.model_path) {
        kprintf("[HT] ERROR: No model path available.\n");
        hal_shutdown();
        return 1;
    }

    /* Memory-map the model file */
    kprintf("[HT] Loading model: %s\n", args.model_path);
    hal_mmap_t model = hal_mmap_file(args.model_path);
    if (!model.data) {
        kprintf("[HT] ERROR: Could not open model file: %s\n", args.model_path);
        hal_shutdown();
        return 1;
    }
    kprintf("[HT] Mapped %llu MB\n", (unsigned long long)(model.size / (1024 * 1024)));

    /* Load model via GGUF parser + LLM engine */
    uint64_t t0 = hal_timer_us();
    int rc = llm_load_from_buffer(model.data, model.size);
    uint64_t t1 = hal_timer_us();

    if (rc < 0) {
        kprintf("[HT] ERROR: Failed to load model (rc=%d)\n", rc);
        hal_munmap(&model);
        hal_shutdown();
        return 1;
    }

    kprintf("[HT] Model loaded in %llu ms\n", (unsigned long long)((t1 - t0) / 1000));
    kprintf("[HT] Model: %s\n", llm_model_name());

    /* Run inference */
    if (args.serve) {
        kprintf("[HT] Starting API server on port %d...\n", args.port);
        ht_api_serve(args.port);
    } else if (args.interactive) {
        interactive_loop(args.model_path, &args);
    } else {
        const char *prompt = args.prompt ? args.prompt : "Hello";
        kprintf("[HT] Prompt: \"%s\"\n", prompt);
        kprintf("[HT] Generating %d tokens...\n\n", args.max_tokens);

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
            kprintf("\n[HT] %d tokens in %llu ms (%.1f tok/s)\n",
                    n, (unsigned long long)total_ms, tok_per_s);
            if (decode_tok_s > 0.0f) {
                kprintf("[HT] Decode-only: prefill %.1f ms, %.1f tok/s\n",
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
