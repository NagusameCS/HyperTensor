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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HT_VERSION_MAJOR 0
#define HT_VERSION_MINOR 4
#define HT_VERSION_PATCH 0
#define HT_CODENAME      "Axon"

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
    kprintf("  -h, --help             Show this help\n");
    kprintf("\nExamples:\n");
    kprintf("  %s phi3.5.gguf -p \"What is an OS?\"\n", argv0);
    kprintf("  %s llama3.gguf -i\n", argv0);
}

typedef struct {
    const char *model_path;
    const char *prompt;
    int         max_tokens;
    int         n_threads;
    float       temperature;
    int         top_k;
    float       top_p;
    int         interactive;
} ht_args_t;

static int parse_args(int argc, char **argv, ht_args_t *args) {
    args->model_path  = NULL;
    args->prompt      = NULL;
    args->max_tokens  = 128;
    args->n_threads   = 0;  /* 0 = auto */
    args->temperature = 0.7f;
    args->top_k       = 40;
    args->top_p       = 0.9f;
    args->interactive = 0;

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
        } else if (argv[i][0] != '-' && !args->model_path) {
            args->model_path = argv[i];
        } else {
            kprintf("Unknown option: %s\n", argv[i]);
            return -1;
        }
    }

    if (!args->model_path) {
        kprintf("Error: no model file specified\n\n");
        return -1;
    }

    return 0;
}

static void interactive_loop(const char *model_path, ht_args_t *args) {
    char line[2048];
    static char output[65536];

    kprintf("\nInteractive mode. Type 'quit' to exit.\n\n");

    for (;;) {
        kprintf("> ");
        fflush(stdout);

        if (!fgets(line, sizeof(line), stdin)) break;

        /* Strip trailing newline */
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';

        if (len == 0) continue;
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) break;

        /* Run inference */
        int n = llm_prompt_n(line, output, (int)sizeof(output), args->max_tokens);
        if (n > 0) {
            kprintf("%s\n\n", output);
        } else {
            kprintf("[error generating response]\n\n");
        }
    }
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
    if (args.interactive) {
        interactive_loop(args.model_path, &args);
    } else {
        const char *prompt = args.prompt ? args.prompt : "Hello";
        kprintf("[HT] Prompt: \"%s\"\n", prompt);
        kprintf("[HT] Generating %d tokens...\n\n", args.max_tokens);

        static char output[65536];
        int n = llm_prompt_n(prompt, output, (int)sizeof(output), args.max_tokens);
        if (n > 0) {
            kprintf("%s\n", output);
        } else {
            kprintf("[error generating response]\n");
        }
    }

    /* Cleanup */
    hal_munmap(&model);
    hal_shutdown();

    return 0;
}
