/*
 * HyperTensor Eval Harness
 * Deterministic prompt/output regression checks + quality metrics.
 *
 * Build: zig cc -target x86_64-windows-gnu -O2 -mavx2 -mfma
 *        -DHYPERTENSOR_HOSTED=1 -DENABLE_CUDA -DENABLE_MLIR
 *        -Ihost/shims -I. -Ihost -Wno-unused-function -Wno-unused-variable
 *        -Wno-format -Wno-incompatible-pointer-types -Wno-int-conversion
 *        -Wno-sign-compare -Wno-missing-field-initializers -Wno-unused-parameter
 *        host/hal.c runtime/nn/llm.c runtime/nn/gguf.c runtime/nn/backend.c
 *        runtime/nn/model_meta.c runtime/nn/tensor_bridge.c runtime/nn/mod_package.c
 *        runtime/nn/token_comm.c runtime/nn/hf_download.c runtime/jit/x86_jit.c
 *        runtime/jit/llm_jit.c runtime/nn/backend_cuda.c runtime/nn/mlir_ops.c
 *        tests/eval_harness.c
 *        -o build_host/eval_harness.exe -ladvapi32 -lws2_32 -lwinhttp
 *
 * Run: eval_harness.exe <model.gguf> [--save baseline.json] [--check baseline.json]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifdef HYPERTENSOR_HOSTED
#include "hal.h"
#endif

/* Forward decls from llm.c */
extern int  llm_load_from_buffer(void *data, uint64_t size);
extern int  llm_prompt_n(const char *user_text, char *output, int max_output,
                          int max_tokens);
extern int  llm_chat_turn(const char *user_text, char *output, int max_output,
                          int max_tokens, float temperature);
extern void llm_chat_reset(void);
extern int  llm_is_loaded(void);
/* ─── Eval Test Case ─── */
typedef struct {
    const char *name;
    const char *prompt;
    int         max_tokens;
    float       temperature;   /* 0 = greedy/deterministic */
    const char *expected_contains;   /* substring that MUST appear */
    const char *expected_not_contains; /* substring that MUST NOT appear */
    int         min_tokens;    /* minimum output length in tokens */
    int         max_repetition; /* max allowed consecutive identical lines */
} eval_case_t;

/* ─── Test Suite ─── */
static const eval_case_t eval_suite[] = {
    /* Math correctness */
    {
        .name = "math_basic",
        .prompt = "What is 2+2? Answer with just the number.",
        .max_tokens = 8,
        .temperature = 0.0f,
        .expected_contains = "4",
        .expected_not_contains = NULL,
        .min_tokens = 1,
        .max_repetition = 3,
    },
    {
        .name = "math_multiply",
        .prompt = "What is 7 times 8? Answer with just the number.",
        .max_tokens = 8,
        .temperature = 0.0f,
        .expected_contains = "56",
        .expected_not_contains = NULL,
        .min_tokens = 1,
        .max_repetition = 3,
    },
    /* Factual knowledge */
    {
        .name = "fact_capital",
        .prompt = "What is the capital of France? Answer in one word.",
        .max_tokens = 8,
        .temperature = 0.0f,
        .expected_contains = "Paris",
        .expected_not_contains = NULL,
        .min_tokens = 1,
        .max_repetition = 3,
    },
    /* Coherence - no repetition */
    {
        .name = "coherence_explain",
        .prompt = "Explain what gravity is in 2 sentences.",
        .max_tokens = 64,
        .temperature = 0.7f,
        .expected_contains = NULL,
        .expected_not_contains = NULL,
        .min_tokens = 10,
        .max_repetition = 2,
    },
    /* Code generation */
    {
        .name = "code_hello",
        .prompt = "Write a Python hello world program. Just the code, nothing else.",
        .max_tokens = 32,
        .temperature = 0.3f,
        .expected_contains = "print",
        .expected_not_contains = NULL,
        .min_tokens = 3,
        .max_repetition = 3,
    },
    /* Instruction following */
    {
        .name = "instruct_list",
        .prompt = "List 3 colors. One per line.",
        .max_tokens = 32,
        .temperature = 0.5f,
        .expected_contains = NULL,
        .expected_not_contains = NULL,
        .min_tokens = 3,
        .max_repetition = 2,
    },
    /* EOS/stop token handling */
    {
        .name = "eos_short",
        .prompt = "Say just the word 'hello' and nothing else.",
        .max_tokens = 16,
        .temperature = 0.0f,
        .expected_contains = NULL,
        .expected_not_contains = NULL,
        .min_tokens = 1,
        .max_repetition = 3,
    },
    /* Long-form coherence */
    {
        .name = "coherence_long",
        .prompt = "Write a short paragraph about the ocean.",
        .max_tokens = 128,
        .temperature = 0.7f,
        .expected_contains = NULL,
        .expected_not_contains = NULL,
        .min_tokens = 20,
        .max_repetition = 2,
    },
};

#define NUM_EVAL_CASES (sizeof(eval_suite) / sizeof(eval_suite[0]))

/* ─── Quality Metrics ─── */
typedef struct {
    int    n_tokens;
    int    max_consec_repeat;
    float  avg_token_len;
    int    has_expected;
    int    has_forbidden;
    int    passed;
    char   output[4096];
    double latency_ms;
    float  tok_per_s;
} eval_result_t;

/* Count consecutive repeated lines */
static int count_max_repeat(const char *text) {
    /* Split by newline and check consecutive duplicates */
    char buf[4096];
    strncpy(buf, text, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';

    char *lines[256];
    int n_lines = 0;
    char *p = buf;
    while (*p && n_lines < 256) {
        lines[n_lines++] = p;
        char *nl = strchr(p, '\n');
        if (nl) { *nl = '\0'; p = nl + 1; }
        else break;
    }

    int max_rep = 1, cur_rep = 1;
    for (int i = 1; i < n_lines; i++) {
        if (strlen(lines[i]) > 0 && strcmp(lines[i], lines[i-1]) == 0) {
            cur_rep++;
            if (cur_rep > max_rep) max_rep = cur_rep;
        } else {
            cur_rep = 1;
        }
    }

    /* Also check for repeated multi-word patterns within a single line */
    const char *src = text;
    int len = (int)strlen(src);
    int max_word_rep = 1;
    /* Only flag patterns of 10+ chars repeating 3+ times - avoids false positives */
    for (int wlen = 10; wlen <= 40 && wlen <= len / 3; wlen++) {
        for (int pos = 0; pos + wlen * 2 <= len; pos++) {
            int reps = 1;
            while (pos + wlen * (reps + 1) <= len &&
                   memcmp(src + pos, src + pos + wlen * reps, wlen) == 0) {
                reps++;
            }
            if (reps >= 3 && reps > max_word_rep) max_word_rep = reps;
        }
    }

    return max_rep > max_word_rep ? max_rep : max_word_rep;
}

/* Count approximate tokens (split by space/punctuation) */
static int approx_tokens(const char *text) {
    int count = 0;
    int in_word = 0;
    for (const char *p = text; *p; p++) {
        if (*p == ' ' || *p == '\n' || *p == '\t') {
            if (in_word) { count++; in_word = 0; }
        } else {
            in_word = 1;
        }
    }
    if (in_word) count++;
    return count;
}

/* Case-insensitive substring search */
static int contains_ci(const char *haystack, const char *needle) {
    int hlen = (int)strlen(haystack);
    int nlen = (int)strlen(needle);
    if (nlen > hlen) return 0;
    for (int i = 0; i <= hlen - nlen; i++) {
        int match = 1;
        for (int j = 0; j < nlen && match; j++) {
            char hc = haystack[i+j], nc = needle[j];
            if (hc >= 'A' && hc <= 'Z') hc += 32;
            if (nc >= 'A' && nc <= 'Z') nc += 32;
            if (hc != nc) match = 0;
        }
        if (match) return 1;
    }
    return 0;
}

/* ─── Performance timer ─── */
#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return (double)cnt.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <time.h>
static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

/* ─── Run single eval case ─── */
static void run_eval(const eval_case_t *tc, eval_result_t *res) {
    memset(res, 0, sizeof(*res));

    /* Reset chat for each test so KV cache is clean */
    llm_chat_reset();

    fprintf(stderr, "[eval] calling llm for prompt: %s\n", tc->prompt);
    fflush(stderr);

    double t0 = get_time_ms();
    int n;
    /* Always use chat_turn which applies proper chat templates */
    float temp = tc->temperature > 0.001f ? tc->temperature : 0.01f; /* near-greedy */
    n = llm_chat_turn(tc->prompt, res->output, sizeof(res->output),
                      tc->max_tokens, temp);
    double t1 = get_time_ms();
    fprintf(stderr, "[eval] done, n=%d\n", n);
    fflush(stderr);

    res->latency_ms = t1 - t0;
    res->n_tokens = n;
    res->tok_per_s = (t1 - t0) > 0 ? (float)n * 1000.0f / (float)(t1 - t0) : 0;
    res->max_consec_repeat = count_max_repeat(res->output);

    /* Check expected substring */
    if (tc->expected_contains) {
        res->has_expected = contains_ci(res->output, tc->expected_contains);
    } else {
        res->has_expected = 1; /* no constraint */
    }

    /* Check forbidden substring */
    if (tc->expected_not_contains) {
        res->has_forbidden = contains_ci(res->output, tc->expected_not_contains);
    } else {
        res->has_forbidden = 0;
    }

    /* Overall pass/fail */
    res->passed = 1;
    if (!res->has_expected) res->passed = 0;
    if (res->has_forbidden) res->passed = 0;
    if (res->n_tokens < tc->min_tokens) res->passed = 0;
    if (res->max_consec_repeat > tc->max_repetition) res->passed = 0;
}

/* ─── Main ─── */
int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: eval_harness <model.gguf> [--save results.txt] [--check results.txt]\n");
        return 1;
    }

    const char *model_path = argv[1];
    const char *save_path = NULL;
    const char *check_path = NULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--save") == 0 && i + 1 < argc)
            save_path = argv[++i];
        else if (strcmp(argv[i], "--check") == 0 && i + 1 < argc)
            check_path = argv[++i];
    }

    /* Load model via mmap (like main.c) */
    printf("=== HyperTensor Eval Harness ===\n");
    printf("Model: %s\n", model_path);

    /* Initialize HAL (CPU detection, thread pool) - must come before model load */
    extern void hal_init(void);
    hal_init();

#ifdef _WIN32
    HANDLE hFile = CreateFileA(model_path, GENERIC_READ, FILE_SHARE_READ,
                               NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) { fprintf(stderr, "Cannot open model\n"); return 1; }
    LARGE_INTEGER fsize;
    GetFileSizeEx(hFile, &fsize);
    uint64_t msize = (uint64_t)fsize.QuadPart;
    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) { fprintf(stderr, "Cannot map model\n"); CloseHandle(hFile); return 1; }
    void *mbuf = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!mbuf) { fprintf(stderr, "Cannot view model\n"); CloseHandle(hMap); CloseHandle(hFile); return 1; }
    printf("Mapped %llu MB\n", (unsigned long long)(msize / (1024*1024)));
#else
    FILE *mf = fopen(model_path, "rb");
    if (!mf) { fprintf(stderr, "Cannot open model\n"); return 1; }
    fseek(mf, 0, SEEK_END);
    uint64_t msize = (uint64_t)ftell(mf);
    fseek(mf, 0, SEEK_SET);
    void *mbuf = malloc(msize);
    fread(mbuf, 1, msize, mf);
    fclose(mf);
#endif

    int lret = llm_load_from_buffer(mbuf, msize);
    if (lret != 0) { fprintf(stderr, "Model load failed: %d\n", lret); return 1; }

    /* Run eval suite */
    printf("\nRunning %zu eval cases...\n\n", NUM_EVAL_CASES);

    eval_result_t results[NUM_EVAL_CASES];
    int total_pass = 0, total_fail = 0;
    double total_time = 0;
    float total_tps = 0;

    for (int i = 0; i < (int)NUM_EVAL_CASES; i++) {
        const eval_case_t *tc = &eval_suite[i];
        eval_result_t *r = &results[i];

        printf("[%d/%zu] %-24s ", i + 1, NUM_EVAL_CASES, tc->name);
        fflush(stdout);
        fflush(stderr);

        run_eval(tc, r);

        if (r->passed) {
            printf("PASS  %5.1f tok/s  %4d ms  %d tok",
                   r->tok_per_s, (int)r->latency_ms, r->n_tokens);
            total_pass++;
        } else {
            printf("FAIL ");
            if (!r->has_expected)
                printf("[missing '%s'] ", tc->expected_contains);
            if (r->has_forbidden)
                printf("[has '%s'] ", tc->expected_not_contains);
            if (r->n_tokens < tc->min_tokens)
                printf("[too short: %d < %d] ", r->n_tokens, tc->min_tokens);
            if (r->max_consec_repeat > tc->max_repetition)
                printf("[repeat: %d > %d] ", r->max_consec_repeat, tc->max_repetition);
            total_fail++;
        }
        printf("\n");

        total_time += r->latency_ms;
        total_tps += r->tok_per_s;
    }

    /* Summary */
    printf("\n══════════════════════════════════════════\n");
    printf("Results: %d/%zu passed, %d failed\n", total_pass, NUM_EVAL_CASES, total_fail);
    printf("Avg tok/s: %.1f\n", total_tps / (float)NUM_EVAL_CASES);
    printf("Total time: %.1f ms\n", total_time);
    printf("══════════════════════════════════════════\n");

    /* Save results */
    if (save_path) {
        FILE *sf = fopen(save_path, "w");
        if (sf) {
            fprintf(sf, "# HyperTensor Eval Baseline\n");
            for (int i = 0; i < (int)NUM_EVAL_CASES; i++) {
                fprintf(sf, "[%s]\n", eval_suite[i].name);
                fprintf(sf, "passed=%d\n", results[i].passed);
                fprintf(sf, "tokens=%d\n", results[i].n_tokens);
                fprintf(sf, "tps=%.1f\n", results[i].tok_per_s);
                fprintf(sf, "repeat=%d\n", results[i].max_consec_repeat);
                /* Save first 200 chars of output for regression check */
                fprintf(sf, "output=%.200s\n\n", results[i].output);
            }
            fclose(sf);
            printf("Baseline saved to: %s\n", save_path);
        }
    }

    /* Check against baseline */
    if (check_path) {
        FILE *cf = fopen(check_path, "r");
        if (cf) {
            printf("\nRegression check against: %s\n", check_path);
            char line[4096];
            int case_idx = -1;
            int regressions = 0;
            while (fgets(line, sizeof(line), cf)) {
                if (line[0] == '[') {
                    char name[64];
                    if (sscanf(line, "[%63[^]]", name) == 1) {
                        for (int i = 0; i < (int)NUM_EVAL_CASES; i++) {
                            if (strcmp(eval_suite[i].name, name) == 0) {
                                case_idx = i;
                                break;
                            }
                        }
                    }
                }
                if (case_idx >= 0 && strncmp(line, "passed=", 7) == 0) {
                    int baseline_pass = atoi(line + 7);
                    if (baseline_pass && !results[case_idx].passed) {
                        printf("  REGRESSION: %s (was PASS, now FAIL)\n",
                               eval_suite[case_idx].name);
                        regressions++;
                    }
                }
            }
            fclose(cf);
            if (regressions == 0)
                printf("  No regressions detected!\n");
            else
                printf("  %d regression(s) found!\n", regressions);
        }
    }

#ifdef _WIN32
    UnmapViewOfFile(mbuf);
    CloseHandle(hMap);
    CloseHandle(hFile);
#else
    free(mbuf);
#endif
    return total_fail;
}
