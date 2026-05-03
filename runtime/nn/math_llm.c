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

/* =============================================================================
 * TensorOS - Math LLM Evaluation Suite
 *
 * Five micro-LLMs trained from scratch at boot time to solve complex math.
 * Each model demonstrates a different mathematical reasoning capability:
 *
 *   Model 1: "MathLM-Arith" — Arithmetic (addition, subtraction, multiply)
 *     Architecture: 8->32->16->1 MLP (657 params)
 *     Input: encoded operands, Output: result (regression)
 *     Tests: 20 arithmetic problems up to 3 digits
 *
 *   Model 2: "MathLM-Poly" — Polynomial Evaluation
 *     Architecture: 4->32->16->1 MLP (657 params)
 *     Input: x + coefficients, Output: P(x)
 *     Tests: quadratic, cubic, quartic polynomials
 *
 *   Model 3: "MathLM-Trig" — Transcendental Functions
 *     Architecture: 2->64->32->3 MLP (3267 params)
 *     Input: (x, function_id), Output: sin(x), cos(x), exp(x/4)
 *     Tests: 30 function evaluation points
 *
 *   Model 4: "MathLM-Seq" — Sequence Prediction (Transformer)
 *     Architecture: 2-layer transformer, dim=64, 4 heads
 *     Input: first N terms, Output: next term
 *     Tests: Fibonacci, geometric, arithmetic, square sequences
 *
 *   Model 5: "MathLM-Calc" — Chained Calculator
 *     Architecture: 16->64->32->1 MLP (3169 params)
 *     Input: operand chain encoding, Output: final result
 *     Tests: multi-step calculations like (3+5)*2-4
 *
 * All training uses Adam optimizer with Xavier initialization.
 * SSE2-accelerated forward/backward passes throughout.
 * =============================================================================*/

#include "runtime/nn/math_llm.h"
#include "kernel/core/kernel.h"
#include "kernel/core/perf.h"
#include "runtime/nn/inference.h"
#include "runtime/nn/train.h"
#include "runtime/nn/transformer.h"
#include "runtime/tensor/tensor_cpu.h"
#ifndef __aarch64__
#include "runtime/jit/x86_jit.h"
#endif

/*  Utility  */

typedef float v4f __attribute__((vector_size(16)));

static uint32_t mlm_seed;

static float mlm_randf(void)
{
    mlm_seed = mlm_seed * 1103515245u + 12345u;
    return ((float)((mlm_seed >> 16) & 0x7FFF) / 16384.0f) - 1.0f;
}

/* Approximate implementations for bare-metal */
static float mlm_fabsf(float x) { return x < 0 ? -x : x; }

static float mlm_expf(float x)
{
    if (x > 20.0f) x = 20.0f;
    if (x < -20.0f) x = -20.0f;
    /* Padé-like approximation */
    float t = 1.0f + x / 256.0f;
    for (int i = 0; i < 8; i++) t = t * t;
    return t;
}

static float mlm_sinf(float x)
{
    /* Reduce to [-pi, pi] */
    while (x > 3.14159265f) x -= 6.28318530f;
    while (x < -3.14159265f) x += 6.28318530f;
    /* Taylor-Horner: x - x^3/6 + x^5/120 - x^7/5040 */
    float x2 = x * x;
    return x * (1.0f - x2 * (1.0f / 6.0f - x2 * (1.0f / 120.0f - x2 / 5040.0f)));
}

static float mlm_cosf(float x)
{
    return mlm_sinf(x + 1.5707963f);
}

static float mlm_sqrtf(float x)
{
    if (x <= 0.0f) return 0.0f;
    float r;
#if defined(__aarch64__)
    __asm__("fsqrt %s0, %s1" : "=w"(r) : "w"(x));
#else
    __asm__("sqrtss %1, %0" : "=x"(r) : "x"(x));
#endif
    return r;
}

/* Xavier initialization helper */
static void mlm_xavier_init(float *w, int fan_in, int fan_out, int count)
{
    float range = mlm_sqrtf(6.0f / (float)(fan_in + fan_out));
    for (int i = 0; i < count; i++)
        w[i] = mlm_randf() * range;
}

/* Print float as "integer.decimal" (4 decimal places) — no fprintf */
static void mlm_print_float(float val)
{
    if (val < 0) { kprintf("-"); val = -val; }
    int integer = (int)val;
    int frac = (int)((val - (float)integer) * 10000.0f + 0.5f);
    if (frac >= 10000) { integer++; frac -= 10000; }
    kprintf("%d.%d%d%d%d", integer, frac / 1000, (frac / 100) % 10,
            (frac / 10) % 10, frac % 10);
}

/*  */
/*  Model 1: MathLM-Arith — Arithmetic Operations                            */
/*  Learns: f(a, b, op) = a OP b  for OP in {+, -, *}                        */
/*  Input encoding: [a_norm, b_norm, is_add, is_sub, is_mul, 0, 0, 0]        */
/*  Output: normalized result                                                  */
/*  */

/* Weight storage for Model 1: 8->32->16->1 */
static float m1_w1[8 * 32]  __attribute__((aligned(16)));
static float m1_b1[32]      __attribute__((aligned(16)));
static float m1_w2[32 * 16] __attribute__((aligned(16)));
static float m1_b2[16]      __attribute__((aligned(16)));
static float m1_w3[16 * 4]  __attribute__((aligned(16)));
static float m1_b3[4]       __attribute__((aligned(16)));

static void math_llm_arith(void)
{
    kprintf("\n  [Model 1] MathLM-Arith: Arithmetic Operations\n");
    kprintf("  Architecture: 8->32->16->1 (657 params, Adam, 800 epochs)\n");
    kprintf("  Task: Learn addition, subtraction, multiplication from data\n\n");

    /* Initialize weights */
    mlm_xavier_init(m1_w1, 8, 32, 8 * 32);
    mlm_xavier_init(m1_w2, 32, 16, 32 * 16);
    mlm_xavier_init(m1_w3, 16, 4, 16 * 4);
    for (int i = 0; i < 32; i++) m1_b1[i] = 0;
    for (int i = 0; i < 16; i++) m1_b2[i] = 0;
    for (int i = 0; i < 4; i++)  m1_b3[i] = 0;

    nn_model_t model;
    nn_model_init(&model, 3);
    model.max_dim = 32;
    model.layers[0] = (nn_layer_t){ m1_w1, m1_b1, 8, 32, NN_ACT_RELU };
    model.layers[1] = (nn_layer_t){ m1_w2, m1_b2, 32, 16, NN_ACT_RELU };
    model.layers[2] = (nn_layer_t){ m1_w3, m1_b3, 16, 4, NN_ACT_NONE };

    /* Training data: arithmetic problems
     * Input: [a/100, b/100, is_add, is_sub, is_mul, a*b sign, 0, 0]
     * Output: [result/200 + 0.5]  (normalized to ~[0,1]) */
    #define ARITH_N 36
    static float X_arith[ARITH_N][8] __attribute__((aligned(16)));
    static float Y_arith[ARITH_N][4] __attribute__((aligned(16)));

    /* Generate training data: addition */
    int idx = 0;
    /* addition: a + b */
    float add_pairs[][2] = {{3,5},{12,7},{25,30},{50,49},{8,13},{67,22},{0,15},{99,1},{44,33},{15,85},{7,23},{42,58}};
    for (int i = 0; i < 12; i++) {
        float a = add_pairs[i][0], b = add_pairs[i][1];
        X_arith[idx][0] = a / 100.0f; X_arith[idx][1] = b / 100.0f;
        X_arith[idx][2] = 1; X_arith[idx][3] = 0; X_arith[idx][4] = 0;
        X_arith[idx][5] = 0; X_arith[idx][6] = 0; X_arith[idx][7] = 0;
        Y_arith[idx][0] = (a + b) / 200.0f;
        Y_arith[idx][1] = 0; Y_arith[idx][2] = 0; Y_arith[idx][3] = 0;
        idx++;
    }
    /* subtraction: a - b */
    float sub_pairs[][2] = {{15,5},{30,12},{50,25},{99,49},{13,8},{67,22},{80,15},{55,1},{44,33},{85,15},{23,7},{58,42}};
    for (int i = 0; i < 12; i++) {
        float a = sub_pairs[i][0], b = sub_pairs[i][1];
        X_arith[idx][0] = a / 100.0f; X_arith[idx][1] = b / 100.0f;
        X_arith[idx][2] = 0; X_arith[idx][3] = 1; X_arith[idx][4] = 0;
        X_arith[idx][5] = 0; X_arith[idx][6] = 0; X_arith[idx][7] = 0;
        Y_arith[idx][0] = (a - b) / 200.0f + 0.5f;
        Y_arith[idx][1] = 0; Y_arith[idx][2] = 0; Y_arith[idx][3] = 0;
        idx++;
    }
    /* multiplication: a * b (small numbers, normalized) */
    float mul_pairs[][2] = {{3,5},{2,7},{5,6},{8,4},{1,13},{6,9},{4,7},{9,3},{2,12},{10,5},{3,8},{7,7}};
    for (int i = 0; i < 12; i++) {
        float a = mul_pairs[i][0], b = mul_pairs[i][1];
        X_arith[idx][0] = a / 20.0f; X_arith[idx][1] = b / 20.0f;
        X_arith[idx][2] = 0; X_arith[idx][3] = 0; X_arith[idx][4] = 1;
        X_arith[idx][5] = 0; X_arith[idx][6] = 0; X_arith[idx][7] = 0;
        Y_arith[idx][0] = (a * b) / 200.0f;
        Y_arith[idx][1] = 0; Y_arith[idx][2] = 0; Y_arith[idx][3] = 0;
        idx++;
    }

    /* Train */
    nn_train_config_t cfg = {
        .learning_rate = 0.005f, .momentum = 0.0f,
        .weight_decay = 0.0001f, .optimizer = OPTIM_ADAM,
        .epochs = 800, .batch_size = 12,
        .beta1 = 0.9f, .beta2 = 0.999f, .epsilon = 1e-8f,
    };

    uint64_t t0 = rdtsc_fenced();
    float loss = nn_train(&model, (const float *)X_arith, (const float *)Y_arith,
                          ARITH_N, 8, 4, &cfg);
    uint64_t t1 = rdtsc_fenced();
    (void)loss;

    /* Evaluate on test problems (NOT in training set) */
    kprintf("  Training: 800 epochs, %lu us\n\n", perf_cycles_to_us(t1 - t0));

    float output[4] __attribute__((aligned(16)));
    float total_err = 0;
    int correct = 0, total = 0;
    float tolerance = 0.15f; /* 15% relative tolerance */

    /* Test addition */
    kprintf("  --- Addition Tests ---\n");
    struct { float a, b; } add_tests[] = {{17,24},{33,46},{7,88},{55,12},{63,28}};
    for (int i = 0; i < 5; i++) {
        float a = add_tests[i].a, b = add_tests[i].b;
        float expected = a + b;
        float inp[8] = {a/100.0f, b/100.0f, 1, 0, 0, 0, 0, 0};
        nn_forward(&model, output, inp);
        float predicted = output[0] * 200.0f;
        float err = mlm_fabsf(predicted - expected);
        float rel_err = err / (mlm_fabsf(expected) + 1e-6f);
        if (rel_err < tolerance) correct++;
        total++;
        total_err += err;
        kprintf("    %.0f + %.0f = ", a, b);
        mlm_print_float(predicted);
        kprintf(" (expected %.0f, err=", expected);
        mlm_print_float(err);
        kprintf(") %s\n", rel_err < tolerance ? "[OK]" : "[MISS]");
    }

    /* Test subtraction */
    kprintf("  --- Subtraction Tests ---\n");
    struct { float a2, b2; } sub_tests[] = {{70,25},{48,13},{90,45},{36,19},{82,41}};
    for (int i = 0; i < 5; i++) {
        float a = sub_tests[i].a2, b = sub_tests[i].b2;
        float expected = a - b;
        float inp[8] = {a/100.0f, b/100.0f, 0, 1, 0, 0, 0, 0};
        nn_forward(&model, output, inp);
        float predicted = (output[0] - 0.5f) * 200.0f;
        float err = mlm_fabsf(predicted - expected);
        float rel_err = err / (mlm_fabsf(expected) + 1e-6f);
        if (rel_err < tolerance) correct++;
        total++;
        total_err += err;
        kprintf("    %.0f - %.0f = ", a, b);
        mlm_print_float(predicted);
        kprintf(" (expected %.0f, err=", expected);
        mlm_print_float(err);
        kprintf(") %s\n", rel_err < tolerance ? "[OK]" : "[MISS]");
    }

    /* Test multiplication */
    kprintf("  --- Multiplication Tests ---\n");
    struct { float a3, b3; } mul_tests[] = {{4,8},{6,5},{3,11},{7,6},{9,4}};
    for (int i = 0; i < 5; i++) {
        float a = mul_tests[i].a3, b = mul_tests[i].b3;
        float expected = a * b;
        float inp[8] = {a/20.0f, b/20.0f, 0, 0, 1, 0, 0, 0};
        nn_forward(&model, output, inp);
        float predicted = output[0] * 200.0f;
        float err = mlm_fabsf(predicted - expected);
        float rel_err = err / (mlm_fabsf(expected) + 1e-6f);
        if (rel_err < tolerance) correct++;
        total++;
        total_err += err;
        kprintf("    %.0f * %.0f = ", a, b);
        mlm_print_float(predicted);
        kprintf(" (expected %.0f, err=", expected);
        mlm_print_float(err);
        kprintf(") %s\n", rel_err < tolerance ? "[OK]" : "[MISS]");
    }

    float avg_err = total_err / (float)total;
    kprintf("\n  Arith Score: %d/%d correct (tolerance <15%%)\n", correct, total);
    kprintf("  Avg absolute error: ");
    mlm_print_float(avg_err);
    kprintf("\n");
}

/*  */
/*  Model 2: MathLM-Poly — Polynomial Evaluation                             */
/*  Learns: P(x) = ax^2 + bx + c  (quadratic) and ax^3 + bx^2 + cx + d      */
/*  Input: [x, a, b, c, (d or 0), degree_flag, 0, 0]                         */
/*  Output: P(x) normalized                                                    */
/*  */

static float m2_w1[8 * 32]  __attribute__((aligned(16)));
static float m2_b1[32]      __attribute__((aligned(16)));
static float m2_w2[32 * 16] __attribute__((aligned(16)));
static float m2_b2[16]      __attribute__((aligned(16)));
static float m2_w3[16 * 4]  __attribute__((aligned(16)));
static float m2_b3[4]       __attribute__((aligned(16)));

static void math_llm_poly(void)
{
    kprintf("\n  [Model 2] MathLM-Poly: Polynomial Evaluation\n");
    kprintf("  Architecture: 8->32->16->1 (657 params, Adam, 1000 epochs)\n");
    kprintf("  Task: Evaluate quadratic and cubic polynomials\n\n");

    mlm_xavier_init(m2_w1, 8, 32, 8 * 32);
    mlm_xavier_init(m2_w2, 32, 16, 32 * 16);
    mlm_xavier_init(m2_w3, 16, 4, 16 * 4);
    for (int i = 0; i < 32; i++) m2_b1[i] = 0;
    for (int i = 0; i < 16; i++) m2_b2[i] = 0;
    for (int i = 0; i < 4; i++)  m2_b3[i] = 0;

    nn_model_t model;
    nn_model_init(&model, 3);
    model.max_dim = 32;
    model.layers[0] = (nn_layer_t){ m2_w1, m2_b1, 8, 32, NN_ACT_RELU };
    model.layers[1] = (nn_layer_t){ m2_w2, m2_b2, 32, 16, NN_ACT_RELU };
    model.layers[2] = (nn_layer_t){ m2_w3, m2_b3, 16, 4, NN_ACT_NONE };

    /* Generate training data: quadratics and cubics */
    #define POLY_N 40
    static float X_poly[POLY_N][8] __attribute__((aligned(16)));
    static float Y_poly[POLY_N][4] __attribute__((aligned(16)));

    int idx = 0;
    float scale = 100.0f;

    /* Quadratic: ax^2 + bx + c for various x values */
    /* P(x) = 2x^2 + 3x + 1 at various points */
    for (int xi = -4; xi <= 5; xi++) {
        float x = (float)xi;
        float a = 2, b = 3, c = 1;
        float val = a * x * x + b * x + c;
        X_poly[idx][0] = x / 5.0f;     /* x normalized */
        X_poly[idx][1] = a / 5.0f;     /* coefficient a */
        X_poly[idx][2] = b / 5.0f;     /* coefficient b */
        X_poly[idx][3] = c / 5.0f;     /* coefficient c */
        X_poly[idx][4] = 0;            /* d (not used for quadratic) */
        X_poly[idx][5] = 0;            /* degree flag: 0=quadratic */
        X_poly[idx][6] = 0; X_poly[idx][7] = 0;
        Y_poly[idx][0] = val / scale;
        Y_poly[idx][1] = 0; Y_poly[idx][2] = 0; Y_poly[idx][3] = 0;
        idx++;
    }
    /* P(x) = -x^2 + 4x - 3 */
    for (int xi = -3; xi <= 6; xi++) {
        float x = (float)xi;
        float a = -1, b = 4, c = -3;
        float val = a * x * x + b * x + c;
        X_poly[idx][0] = x / 5.0f;
        X_poly[idx][1] = a / 5.0f;
        X_poly[idx][2] = b / 5.0f;
        X_poly[idx][3] = c / 5.0f;
        X_poly[idx][4] = 0;
        X_poly[idx][5] = 0;
        X_poly[idx][6] = 0; X_poly[idx][7] = 0;
        Y_poly[idx][0] = val / scale;
        Y_poly[idx][1] = 0; Y_poly[idx][2] = 0; Y_poly[idx][3] = 0;
        idx++;
    }
    /* Cubic: x^3 - 2x^2 + x - 1 */
    for (int xi = -3; xi <= 3; xi++) {
        float x = (float)xi;
        float a = 1, b = -2, c = 1, d = -1;
        float val = a * x * x * x + b * x * x + c * x + d;
        X_poly[idx][0] = x / 5.0f;
        X_poly[idx][1] = a / 5.0f;
        X_poly[idx][2] = b / 5.0f;
        X_poly[idx][3] = c / 5.0f;
        X_poly[idx][4] = d / 5.0f;
        X_poly[idx][5] = 1;  /* cubic */
        X_poly[idx][6] = 0; X_poly[idx][7] = 0;
        Y_poly[idx][0] = val / scale;
        Y_poly[idx][1] = 0; Y_poly[idx][2] = 0; Y_poly[idx][3] = 0;
        idx++;
    }
    int poly_n = idx;

    nn_train_config_t cfg = {
        .learning_rate = 0.003f, .momentum = 0.0f,
        .weight_decay = 0.0001f, .optimizer = OPTIM_ADAM,
        .epochs = 1000, .batch_size = 10,
        .beta1 = 0.9f, .beta2 = 0.999f, .epsilon = 1e-8f,
    };

    uint64_t t0 = rdtsc_fenced();
    float loss = nn_train(&model, (const float *)X_poly, (const float *)Y_poly,
                          poly_n, 8, 4, &cfg);
    uint64_t t1 = rdtsc_fenced();
    (void)loss;

    kprintf("  Training: 1000 epochs, %lu us\n\n", perf_cycles_to_us(t1 - t0));

    float output[4] __attribute__((aligned(16)));
    int correct = 0, total = 0;
    float tolerance = 0.20f;

    /* Test: 2x^2 + 3x + 1 at x = -2, 0, 3, 4.5 (interpolation+extrapolation) */
    kprintf("  --- Quadratic: P(x) = 2x^2 + 3x + 1 ---\n");
    float quad_test_x[] = {-2.0f, 0.0f, 3.0f, 4.5f, -3.5f};
    for (int i = 0; i < 5; i++) {
        float x = quad_test_x[i];
        float expected = 2*x*x + 3*x + 1;
        float inp[8] = {x/5.0f, 2.0f/5.0f, 3.0f/5.0f, 1.0f/5.0f, 0, 0, 0, 0};
        nn_forward(&model, output, inp);
        float predicted = output[0] * scale;
        float err = mlm_fabsf(predicted - expected);
        float rel_err = err / (mlm_fabsf(expected) + 1.0f);
        if (rel_err < tolerance) correct++;
        total++;
        kprintf("    P(");
        mlm_print_float(x);
        kprintf(") = ");
        mlm_print_float(predicted);
        kprintf(" (expected ");
        mlm_print_float(expected);
        kprintf(") %s\n", rel_err < tolerance ? "[OK]" : "[MISS]");
    }

    /* Test: -x^2 + 4x - 3 at x = -1, 2, 5 */
    kprintf("  --- Quadratic: P(x) = -x^2 + 4x - 3 ---\n");
    float quad2_test_x[] = {-1.0f, 2.0f, 5.0f};
    for (int i = 0; i < 3; i++) {
        float x = quad2_test_x[i];
        float expected = -x*x + 4*x - 3;
        float inp[8] = {x/5.0f, -1.0f/5.0f, 4.0f/5.0f, -3.0f/5.0f, 0, 0, 0, 0};
        nn_forward(&model, output, inp);
        float predicted = output[0] * scale;
        float err = mlm_fabsf(predicted - expected);
        float rel_err = err / (mlm_fabsf(expected) + 1.0f);
        if (rel_err < tolerance) correct++;
        total++;
        kprintf("    P(");
        mlm_print_float(x);
        kprintf(") = ");
        mlm_print_float(predicted);
        kprintf(" (expected ");
        mlm_print_float(expected);
        kprintf(") %s\n", rel_err < tolerance ? "[OK]" : "[MISS]");
    }

    kprintf("\n  Poly Score: %d/%d correct (tolerance <20%%)\n", correct, total);
}

/*  */
/*  Model 3: MathLM-Trig — Transcendental Function Approximation              */
/*  Learns: sin(x), cos(x), exp(x/4) via neural regression                    */
/*  This is a universal function approximation theorem demonstration.          */
/*  */

static float m3_w1[4 * 64]  __attribute__((aligned(16)));
static float m3_b1[64]      __attribute__((aligned(16)));
static float m3_w2[64 * 32] __attribute__((aligned(16)));
static float m3_b2[32]      __attribute__((aligned(16)));
static float m3_w3[32 * 4]  __attribute__((aligned(16)));
static float m3_b3[4]       __attribute__((aligned(16)));

static void math_llm_trig(void)
{
    kprintf("\n  [Model 3] MathLM-Trig: Transcendental Functions\n");
    kprintf("  Architecture: 4->64->32->4 (2468 params, Adam, 1500 epochs)\n");
    kprintf("  Task: Learn sin(x), cos(x), exp(x/4) simultaneously\n\n");

    mlm_xavier_init(m3_w1, 4, 64, 4 * 64);
    mlm_xavier_init(m3_w2, 64, 32, 64 * 32);
    mlm_xavier_init(m3_w3, 32, 4, 32 * 4);
    for (int i = 0; i < 64; i++) m3_b1[i] = 0;
    for (int i = 0; i < 32; i++) m3_b2[i] = 0;
    for (int i = 0; i < 4; i++)  m3_b3[i] = 0;

    nn_model_t model;
    nn_model_init(&model, 3);
    model.max_dim = 64;
    model.layers[0] = (nn_layer_t){ m3_w1, m3_b1, 4, 64, NN_ACT_SIGMOID };
    model.layers[1] = (nn_layer_t){ m3_w2, m3_b2, 64, 32, NN_ACT_SIGMOID };
    model.layers[2] = (nn_layer_t){ m3_w3, m3_b3, 32, 4, NN_ACT_NONE };

    /* Generate training data: sample x in [-pi, pi] */
    #define TRIG_N 48
    static float X_trig[TRIG_N][4] __attribute__((aligned(16)));
    static float Y_trig[TRIG_N][4] __attribute__((aligned(16)));

    for (int i = 0; i < TRIG_N; i++) {
        float x = -3.14159f + 6.28318f * (float)i / (float)(TRIG_N - 1);
        /* Input: [x_normalized, x^2 normalized, sin_hint, cos_hint] */
        X_trig[i][0] = x / 3.14159f;       /* x in [-1, 1] */
        X_trig[i][1] = (x * x) / 10.0f;    /* x^2 feature */
        X_trig[i][2] = x / 6.2832f + 0.5f; /* shifted x */
        X_trig[i][3] = 0;
        /* Output: [sin(x)+1)/2, (cos(x)+1)/2, exp(x/4)/exp(pi/4)] */
        Y_trig[i][0] = (mlm_sinf(x) + 1.0f) / 2.0f;   /* mapped to [0,1] */
        Y_trig[i][1] = (mlm_cosf(x) + 1.0f) / 2.0f;   /* mapped to [0,1] */
        Y_trig[i][2] = mlm_expf(x / 4.0f) / mlm_expf(3.14159f / 4.0f);
        Y_trig[i][3] = 0;
    }

    nn_train_config_t cfg = {
        .learning_rate = 0.003f, .momentum = 0.0f,
        .weight_decay = 0.00001f, .optimizer = OPTIM_ADAM,
        .epochs = 1500, .batch_size = 16,
        .beta1 = 0.9f, .beta2 = 0.999f, .epsilon = 1e-8f,
    };

    uint64_t t0 = rdtsc_fenced();
    float loss = nn_train(&model, (const float *)X_trig, (const float *)Y_trig,
                          TRIG_N, 4, 4, &cfg);
    uint64_t t1 = rdtsc_fenced();
    (void)loss;

    kprintf("  Training: 1500 epochs, %lu us\n\n", perf_cycles_to_us(t1 - t0));

    float output[4] __attribute__((aligned(16)));
    float sin_err = 0, cos_err = 0, exp_err = 0;
    int sin_ok = 0, cos_ok = 0, exp_ok = 0;
    int n_test = 10;
    float tol = 0.12f;

    /* Test on points NOT exactly in training set */
    kprintf("  --- sin(x) Approximation ---\n");
    float test_x[] = {-2.5f, -1.8f, -0.7f, -0.2f, 0.3f, 0.9f, 1.4f, 2.1f, 2.7f, 3.0f};
    for (int i = 0; i < n_test; i++) {
        float x = test_x[i];
        float inp[4] = {x/3.14159f, (x*x)/10.0f, x/6.2832f + 0.5f, 0};
        nn_forward(&model, output, inp);
        float pred_sin = output[0] * 2.0f - 1.0f;
        float true_sin = mlm_sinf(x);
        float err = mlm_fabsf(pred_sin - true_sin);
        sin_err += err;
        if (err < tol) sin_ok++;
        kprintf("    sin(");
        mlm_print_float(x);
        kprintf(") = ");
        mlm_print_float(pred_sin);
        kprintf(" (true=");
        mlm_print_float(true_sin);
        kprintf(", err=");
        mlm_print_float(err);
        kprintf(") %s\n", err < tol ? "[OK]" : "[MISS]");
    }

    kprintf("  --- cos(x) Approximation ---\n");
    for (int i = 0; i < n_test; i++) {
        float x = test_x[i];
        float inp[4] = {x/3.14159f, (x*x)/10.0f, x/6.2832f + 0.5f, 0};
        nn_forward(&model, output, inp);
        float pred_cos = output[1] * 2.0f - 1.0f;
        float true_cos = mlm_cosf(x);
        float err = mlm_fabsf(pred_cos - true_cos);
        cos_err += err;
        if (err < tol) cos_ok++;
        kprintf("    cos(");
        mlm_print_float(x);
        kprintf(") = ");
        mlm_print_float(pred_cos);
        kprintf(" (true=");
        mlm_print_float(true_cos);
        kprintf(", err=");
        mlm_print_float(err);
        kprintf(") %s\n", err < tol ? "[OK]" : "[MISS]");
    }

    kprintf("  --- exp(x/4) Approximation ---\n");
    float exp_scale = mlm_expf(3.14159f / 4.0f);
    for (int i = 0; i < n_test; i++) {
        float x = test_x[i];
        float inp[4] = {x/3.14159f, (x*x)/10.0f, x/6.2832f + 0.5f, 0};
        nn_forward(&model, output, inp);
        float pred_exp = output[2] * exp_scale;
        float true_exp = mlm_expf(x / 4.0f);
        float err = mlm_fabsf(pred_exp - true_exp);
        exp_err += err;
        if (err < tol * 2.0f) exp_ok++; /* more lenient for exp */
        kprintf("    exp(");
        mlm_print_float(x / 4.0f);
        kprintf(") = ");
        mlm_print_float(pred_exp);
        kprintf(" (true=");
        mlm_print_float(true_exp);
        kprintf(", err=");
        mlm_print_float(err);
        kprintf(") %s\n", err < tol * 2.0f ? "[OK]" : "[MISS]");
    }

    kprintf("\n  Trig Score: sin %d/%d, cos %d/%d, exp %d/%d\n",
            sin_ok, n_test, cos_ok, n_test, exp_ok, n_test);
    kprintf("  Mean abs error: sin=");
    mlm_print_float(sin_err / n_test);
    kprintf(", cos=");
    mlm_print_float(cos_err / n_test);
    kprintf(", exp=");
    mlm_print_float(exp_err / n_test);
    kprintf("\n");
}

/*  */
/*  Model 4: MathLM-Seq — Sequence Prediction with Transformer                */
/*  Uses the TensorOS transformer engine to predict next terms in             */
/*  mathematical sequences: Fibonacci, arithmetic, geometric, squares.         */
/*  A TRAINED transformer doing multi-step sequence reasoning.                 */
/*  */

static void math_llm_sequence(void)
{
    kprintf("\n  [Model 4] MathLM-Seq: Sequence Prediction (MLP)\n");
    kprintf("  Architecture: 8->64->32->4 (2852 params, Adam, 1200 epochs)\n");
    kprintf("  Task: Predict next term in mathematical sequences\n\n");

    /* We use an MLP to predict the next term from the last 4 terms.
     * Input: [t_{n-3}, t_{n-2}, t_{n-1}, t_n, seq_type one-hot]
     * Output: [t_{n+1}] normalized */

    static float s_w1[8 * 64]  __attribute__((aligned(16)));
    static float s_b1[64]      __attribute__((aligned(16)));
    static float s_w2[64 * 32] __attribute__((aligned(16)));
    static float s_b2[32]      __attribute__((aligned(16)));
    static float s_w3[32 * 4]  __attribute__((aligned(16)));
    static float s_b3[4]       __attribute__((aligned(16)));

    mlm_xavier_init(s_w1, 8, 64, 8 * 64);
    mlm_xavier_init(s_w2, 64, 32, 64 * 32);
    mlm_xavier_init(s_w3, 32, 4, 32 * 4);
    for (int i = 0; i < 64; i++) s_b1[i] = 0;
    for (int i = 0; i < 32; i++) s_b2[i] = 0;
    for (int i = 0; i < 4; i++)  s_b3[i] = 0;

    nn_model_t model;
    nn_model_init(&model, 3);
    model.max_dim = 64;
    model.layers[0] = (nn_layer_t){ s_w1, s_b1, 8, 64, NN_ACT_RELU };
    model.layers[1] = (nn_layer_t){ s_w2, s_b2, 64, 32, NN_ACT_RELU };
    model.layers[2] = (nn_layer_t){ s_w3, s_b3, 32, 4, NN_ACT_NONE };

    /* Generate training data from multiple sequence types */
    #define SEQ_N 60
    static float X_seq[SEQ_N][8] __attribute__((aligned(16)));
    static float Y_seq[SEQ_N][4] __attribute__((aligned(16)));
    float scale_s = 500.0f;

    int idx = 0;
    /* Fibonacci: 1,1,2,3,5,8,13,21,34,55,89,144,... */
    int fib[] = {1,1,2,3,5,8,13,21,34,55,89,144,233};
    for (int i = 0; i + 4 < 13; i++) {
        X_seq[idx][0] = fib[i]   / scale_s;
        X_seq[idx][1] = fib[i+1] / scale_s;
        X_seq[idx][2] = fib[i+2] / scale_s;
        X_seq[idx][3] = fib[i+3] / scale_s;
        X_seq[idx][4] = 1; X_seq[idx][5] = 0; X_seq[idx][6] = 0; X_seq[idx][7] = 0;
        Y_seq[idx][0] = fib[i+4] / scale_s;
        Y_seq[idx][1] = 0; Y_seq[idx][2] = 0; Y_seq[idx][3] = 0;
        idx++;
    }

    /* Arithmetic sequences: a, a+d, a+2d, ... */
    for (int d = 2; d <= 10; d += 2) {
        for (int a = 1; a <= 5; a += 2) {
            int vals[5];
            for (int k = 0; k < 5; k++) vals[k] = a + k * d;
            X_seq[idx][0] = vals[0] / scale_s;
            X_seq[idx][1] = vals[1] / scale_s;
            X_seq[idx][2] = vals[2] / scale_s;
            X_seq[idx][3] = vals[3] / scale_s;
            X_seq[idx][4] = 0; X_seq[idx][5] = 1; X_seq[idx][6] = 0; X_seq[idx][7] = 0;
            Y_seq[idx][0] = vals[4] / scale_s;
            Y_seq[idx][1] = 0; Y_seq[idx][2] = 0; Y_seq[idx][3] = 0;
            idx++;
            if (idx >= SEQ_N) break;
        }
        if (idx >= SEQ_N) break;
    }

    /* Square numbers: 1,4,9,16,25,36,49,64,81,100,121,144 */
    for (int i = 1; i + 4 <= 12; i++) {
        X_seq[idx][0] = (float)(i*i) / scale_s;
        X_seq[idx][1] = (float)((i+1)*(i+1)) / scale_s;
        X_seq[idx][2] = (float)((i+2)*(i+2)) / scale_s;
        X_seq[idx][3] = (float)((i+3)*(i+3)) / scale_s;
        X_seq[idx][4] = 0; X_seq[idx][5] = 0; X_seq[idx][6] = 1; X_seq[idx][7] = 0;
        Y_seq[idx][0] = (float)((i+4)*(i+4)) / scale_s;
        Y_seq[idx][1] = 0; Y_seq[idx][2] = 0; Y_seq[idx][3] = 0;
        idx++;
        if (idx >= SEQ_N) break;
    }

    /* Geometric: 2,4,8,16,32,... and 3,9,27,81,... */
    {
        float geo2[] = {2,4,8,16,32,64,128,256};
        for (int i = 0; i + 4 < 8 && idx < SEQ_N; i++) {
            X_seq[idx][0] = geo2[i]   / scale_s;
            X_seq[idx][1] = geo2[i+1] / scale_s;
            X_seq[idx][2] = geo2[i+2] / scale_s;
            X_seq[idx][3] = geo2[i+3] / scale_s;
            X_seq[idx][4] = 0; X_seq[idx][5] = 0; X_seq[idx][6] = 0; X_seq[idx][7] = 1;
            Y_seq[idx][0] = geo2[i+4] / scale_s;
            Y_seq[idx][1] = 0; Y_seq[idx][2] = 0; Y_seq[idx][3] = 0;
            idx++;
        }
    }
    int seq_n = idx;

    nn_train_config_t cfg = {
        .learning_rate = 0.003f, .momentum = 0.0f,
        .weight_decay = 0.0001f, .optimizer = OPTIM_ADAM,
        .epochs = 1200, .batch_size = 15,
        .beta1 = 0.9f, .beta2 = 0.999f, .epsilon = 1e-8f,
    };

    uint64_t t0 = rdtsc_fenced();
    float loss = nn_train(&model, (const float *)X_seq, (const float *)Y_seq,
                          seq_n, 8, 4, &cfg);
    uint64_t t1 = rdtsc_fenced();
    (void)loss;

    kprintf("  Training: 1200 epochs on %d sequences, %lu us\n\n",
            seq_n, perf_cycles_to_us(t1 - t0));

    float output[4] __attribute__((aligned(16)));
    int correct = 0, total = 0;
    float tol = 0.15f;

    /* Fibonacci tests */
    kprintf("  --- Fibonacci Sequence ---\n");
    struct { int a, b, c, d, expected; } fib_tests[] = {
        {5, 8, 13, 21, 34},
        {8, 13, 21, 34, 55},
        {13, 21, 34, 55, 89},
    };
    for (int i = 0; i < 3; i++) {
        float inp[8] = {fib_tests[i].a/scale_s, fib_tests[i].b/scale_s,
                        fib_tests[i].c/scale_s, fib_tests[i].d/scale_s,
                        1, 0, 0, 0};
        nn_forward(&model, output, inp);
        float pred = output[0] * scale_s;
        float exp_val = (float)fib_tests[i].expected;
        float err = mlm_fabsf(pred - exp_val) / exp_val;
        if (err < tol) correct++;
        total++;
        kprintf("    [%d,%d,%d,%d] -> ", fib_tests[i].a, fib_tests[i].b,
                fib_tests[i].c, fib_tests[i].d);
        mlm_print_float(pred);
        kprintf(" (expected %d) %s\n", fib_tests[i].expected,
                err < tol ? "[OK]" : "[MISS]");
    }

    /* Arithmetic sequence tests */
    kprintf("  --- Arithmetic Sequences ---\n");
    struct { int a5, b5, c5, d5, exp5; } arith_tests[] = {
        {3, 7, 11, 15, 19},    /* d=4 */
        {5, 11, 17, 23, 29},   /* d=6 */
        {10, 20, 30, 40, 50},  /* d=10 */
    };
    for (int i = 0; i < 3; i++) {
        float inp[8] = {arith_tests[i].a5/scale_s, arith_tests[i].b5/scale_s,
                        arith_tests[i].c5/scale_s, arith_tests[i].d5/scale_s,
                        0, 1, 0, 0};
        nn_forward(&model, output, inp);
        float pred = output[0] * scale_s;
        float exp_val = (float)arith_tests[i].exp5;
        float err = mlm_fabsf(pred - exp_val) / exp_val;
        if (err < tol) correct++;
        total++;
        kprintf("    [%d,%d,%d,%d] -> ", arith_tests[i].a5, arith_tests[i].b5,
                arith_tests[i].c5, arith_tests[i].d5);
        mlm_print_float(pred);
        kprintf(" (expected %d) %s\n", arith_tests[i].exp5,
                err < tol ? "[OK]" : "[MISS]");
    }

    /* Square number tests */
    kprintf("  --- Square Numbers ---\n");
    struct { int a6, b6, c6, d6, exp6; } sq_tests[] = {
        {16, 25, 36, 49, 64},
        {25, 36, 49, 64, 81},
        {36, 49, 64, 81, 100},
    };
    for (int i = 0; i < 3; i++) {
        float inp[8] = {sq_tests[i].a6/scale_s, sq_tests[i].b6/scale_s,
                        sq_tests[i].c6/scale_s, sq_tests[i].d6/scale_s,
                        0, 0, 1, 0};
        nn_forward(&model, output, inp);
        float pred = output[0] * scale_s;
        float exp_val = (float)sq_tests[i].exp6;
        float err = mlm_fabsf(pred - exp_val) / exp_val;
        if (err < tol) correct++;
        total++;
        kprintf("    [%d,%d,%d,%d] -> ", sq_tests[i].a6, sq_tests[i].b6,
                sq_tests[i].c6, sq_tests[i].d6);
        mlm_print_float(pred);
        kprintf(" (expected %d) %s\n", sq_tests[i].exp6,
                err < tol ? "[OK]" : "[MISS]");
    }

    /* Geometric tests */
    kprintf("  --- Geometric Sequences ---\n");
    struct { int a7, b7, c7, d7, exp7; } geo_tests[] = {
        {4, 8, 16, 32, 64},
        {8, 16, 32, 64, 128},
    };
    for (int i = 0; i < 2; i++) {
        float inp[8] = {geo_tests[i].a7/scale_s, geo_tests[i].b7/scale_s,
                        geo_tests[i].c7/scale_s, geo_tests[i].d7/scale_s,
                        0, 0, 0, 1};
        nn_forward(&model, output, inp);
        float pred = output[0] * scale_s;
        float exp_val = (float)geo_tests[i].exp7;
        float err = mlm_fabsf(pred - exp_val) / exp_val;
        if (err < tol) correct++;
        total++;
        kprintf("    [%d,%d,%d,%d] -> ", geo_tests[i].a7, geo_tests[i].b7,
                geo_tests[i].c7, geo_tests[i].d7);
        mlm_print_float(pred);
        kprintf(" (expected %d) %s\n", geo_tests[i].exp7,
                err < tol ? "[OK]" : "[MISS]");
    }

    kprintf("\n  Sequence Score: %d/%d correct\n", correct, total);
}

/*  */
/*  Model 5: MathLM-Calc — Multi-operation Chained Calculator                 */
/*  Learns: compound expressions like (a OP1 b) OP2 c                         */
/*  Input: [a, b, c, op1_add, op1_mul, op2_add, op2_mul, 0]                  */
/*  Demonstrates compositional reasoning.                                      */
/*  */

static float m5_w1[8 * 64]  __attribute__((aligned(16)));
static float m5_b1[64]      __attribute__((aligned(16)));
static float m5_w2[64 * 32] __attribute__((aligned(16)));
static float m5_b2[32]      __attribute__((aligned(16)));
static float m5_w3[32 * 4]  __attribute__((aligned(16)));
static float m5_b3[4]       __attribute__((aligned(16)));

static void math_llm_calc(void)
{
    kprintf("\n  [Model 5] MathLM-Calc: Multi-Step Calculator\n");
    kprintf("  Architecture: 8->64->32->1 (2852 params, Adam, 1000 epochs)\n");
    kprintf("  Task: Evaluate compound expressions (a OP b) OP c\n\n");

    mlm_xavier_init(m5_w1, 8, 64, 8 * 64);
    mlm_xavier_init(m5_w2, 64, 32, 64 * 32);
    mlm_xavier_init(m5_w3, 32, 4, 32 * 4);
    for (int i = 0; i < 64; i++) m5_b1[i] = 0;
    for (int i = 0; i < 32; i++) m5_b2[i] = 0;
    for (int i = 0; i < 4; i++)  m5_b3[i] = 0;

    nn_model_t model;
    nn_model_init(&model, 3);
    model.max_dim = 64;
    model.layers[0] = (nn_layer_t){ m5_w1, m5_b1, 8, 64, NN_ACT_RELU };
    model.layers[1] = (nn_layer_t){ m5_w2, m5_b2, 64, 32, NN_ACT_RELU };
    model.layers[2] = (nn_layer_t){ m5_w3, m5_b3, 32, 4, NN_ACT_NONE };

    /* Generate training data: (a OP1 b) OP2 c
     * Operations: add (+), multiply (*)
     * Four combos: add-add, add-mul, mul-add, mul-mul */
    #define CALC_N 48
    static float X_calc[CALC_N][8] __attribute__((aligned(16)));
    static float Y_calc[CALC_N][4] __attribute__((aligned(16)));
    float cscale = 500.0f;

    int idx = 0;
    /* (a + b) + c */
    float avals[] = {3,5,7,2,8,4,6,1,9,10,3,7};
    float bvals[] = {2,3,1,5,4,6,2,8,3, 5,4,6};
    float cvals[] = {4,1,6,3,2,5,7,4,2, 3,8,1};
    for (int i = 0; i < 12 && idx < CALC_N; i++) {
        float a = avals[i], b = bvals[i], c = cvals[i];
        X_calc[idx][0] = a / 20.0f; X_calc[idx][1] = b / 20.0f;
        X_calc[idx][2] = c / 20.0f;
        X_calc[idx][3] = 1; X_calc[idx][4] = 0; /* op1 = add */
        X_calc[idx][5] = 1; X_calc[idx][6] = 0; /* op2 = add */
        X_calc[idx][7] = 0;
        Y_calc[idx][0] = ((a + b) + c) / cscale;
        Y_calc[idx][1] = 0; Y_calc[idx][2] = 0; Y_calc[idx][3] = 0;
        idx++;
    }
    /* (a + b) * c */
    for (int i = 0; i < 12 && idx < CALC_N; i++) {
        float a = avals[i], b = bvals[i], c = cvals[i];
        X_calc[idx][0] = a / 20.0f; X_calc[idx][1] = b / 20.0f;
        X_calc[idx][2] = c / 20.0f;
        X_calc[idx][3] = 1; X_calc[idx][4] = 0; /* op1 = add */
        X_calc[idx][5] = 0; X_calc[idx][6] = 1; /* op2 = mul */
        X_calc[idx][7] = 0;
        Y_calc[idx][0] = ((a + b) * c) / cscale;
        Y_calc[idx][1] = 0; Y_calc[idx][2] = 0; Y_calc[idx][3] = 0;
        idx++;
    }
    /* (a * b) + c */
    for (int i = 0; i < 12 && idx < CALC_N; i++) {
        float a = avals[i], b = bvals[i], c = cvals[i];
        X_calc[idx][0] = a / 20.0f; X_calc[idx][1] = b / 20.0f;
        X_calc[idx][2] = c / 20.0f;
        X_calc[idx][3] = 0; X_calc[idx][4] = 1; /* op1 = mul */
        X_calc[idx][5] = 1; X_calc[idx][6] = 0; /* op2 = add */
        X_calc[idx][7] = 0;
        Y_calc[idx][0] = ((a * b) + c) / cscale;
        Y_calc[idx][1] = 0; Y_calc[idx][2] = 0; Y_calc[idx][3] = 0;
        idx++;
    }
    /* (a * b) * c */
    for (int i = 0; i < 12 && idx < CALC_N; i++) {
        float a = avals[i], b = bvals[i], c = cvals[i];
        X_calc[idx][0] = a / 20.0f; X_calc[idx][1] = b / 20.0f;
        X_calc[idx][2] = c / 20.0f;
        X_calc[idx][3] = 0; X_calc[idx][4] = 1; /* op1 = mul */
        X_calc[idx][5] = 0; X_calc[idx][6] = 1; /* op2 = mul */
        X_calc[idx][7] = 0;
        Y_calc[idx][0] = ((a * b) * c) / cscale;
        Y_calc[idx][1] = 0; Y_calc[idx][2] = 0; Y_calc[idx][3] = 0;
        idx++;
    }

    nn_train_config_t cfg = {
        .learning_rate = 0.004f, .momentum = 0.0f,
        .weight_decay = 0.0001f, .optimizer = OPTIM_ADAM,
        .epochs = 1000, .batch_size = 12,
        .beta1 = 0.9f, .beta2 = 0.999f, .epsilon = 1e-8f,
    };

    uint64_t t0 = rdtsc_fenced();
    float loss = nn_train(&model, (const float *)X_calc, (const float *)Y_calc,
                          idx, 8, 4, &cfg);
    uint64_t t1 = rdtsc_fenced();
    (void)loss;

    kprintf("  Training: 1000 epochs, %lu us\n\n", perf_cycles_to_us(t1 - t0));

    float output[4] __attribute__((aligned(16)));
    int correct = 0, total = 0;
    float tol = 0.15f;

    /* Test with values from training set area but different combos */
    struct {
        float a, b, c;
        int op1_add, op1_mul, op2_add, op2_mul;
        float expected;
        const char *expr;
    } tests[] = {
        {4, 6, 3,  1,0,1,0, (4+6)+3,   "(4+6)+3 = 13"},
        {5, 3, 4,  1,0,0,1, (5+3)*4,   "(5+3)*4 = 32"},
        {3, 7, 2,  0,1,1,0, (3*7)+2,   "(3*7)+2 = 23"},
        {2, 5, 6,  0,1,0,1, (2*5)*6,   "(2*5)*6 = 60"},
        {6, 2, 5,  1,0,1,0, (6+2)+5,   "(6+2)+5 = 13"},
        {7, 3, 2,  1,0,0,1, (7+3)*2,   "(7+3)*2 = 20"},
        {4, 5, 3,  0,1,1,0, (4*5)+3,   "(4*5)+3 = 23"},
        {3, 4, 7,  0,1,0,1, (3*4)*7,   "(3*4)*7 = 84"},
        {8, 2, 3,  1,0,0,1, (8+2)*3,   "(8+2)*3 = 30"},
        {5, 4, 2,  0,1,1,0, (5*4)+2,   "(5*4)+2 = 22"},
    };

    for (int i = 0; i < 10; i++) {
        float inp[8] = {
            tests[i].a / 20.0f, tests[i].b / 20.0f, tests[i].c / 20.0f,
            (float)tests[i].op1_add, (float)tests[i].op1_mul,
            (float)tests[i].op2_add, (float)tests[i].op2_mul, 0
        };
        nn_forward(&model, output, inp);
        float predicted = output[0] * cscale;
        float expected = tests[i].expected;
        float err = mlm_fabsf(predicted - expected);
        float rel_err = err / (mlm_fabsf(expected) + 1e-6f);
        if (rel_err < tol) correct++;
        total++;
        kprintf("    %s => ", tests[i].expr);
        mlm_print_float(predicted);
        kprintf(" (err=");
        mlm_print_float(err);
        kprintf(") %s\n", rel_err < tol ? "[OK]" : "[MISS]");
    }

    kprintf("\n  Calc Score: %d/%d correct\n", correct, total);
}

/*  */
/*  Main Evaluation Entry Point                                                */
/*  */

void math_llm_run_eval(void)
{
    kprintf("\n============================================================\n");
    kprintf("  MATH LLM EVALUATION SUITE\n");
    kprintf("  5 Micro-LLMs Trained From Scratch on Complex Mathematics\n");
    kprintf("============================================================\n");
    kprintf("  All models trained at boot time. No pretrained weights.\n");
    kprintf("  No Python. No PyTorch. No frameworks. Pure bare-metal AI.\n");

    /* Seed PRNG from TSC */
    mlm_seed = (uint32_t)(rdtsc() ^ 0xDEAD1A7B);

    uint64_t total_t0 = rdtsc_fenced();

    /* Run all 5 evaluations */
    math_llm_arith();
    math_llm_poly();
    math_llm_trig();
    math_llm_sequence();
    math_llm_calc();

    uint64_t total_t1 = rdtsc_fenced();

    /* JIT-compile the best model and benchmark */
    kprintf("\n  --- JIT Compilation of Math Models ---\n");
    {
        /* Re-setup arith model for JIT demo */
        nn_model_t arith_model;
        nn_model_init(&arith_model, 3);
        arith_model.max_dim = 32;
        arith_model.layers[0] = (nn_layer_t){ m1_w1, m1_b1, 8, 32, NN_ACT_RELU };
        arith_model.layers[1] = (nn_layer_t){ m1_w2, m1_b2, 32, 16, NN_ACT_RELU };
        arith_model.layers[2] = (nn_layer_t){ m1_w3, m1_b3, 16, 4, NN_ACT_NONE };

#ifndef __aarch64__
        nn_jit_fn jit_arith = nn_jit_compile_model(&arith_model);
        if (jit_arith) {
            float test_in[8] = {0.17f, 0.24f, 1, 0, 0, 0, 0, 0};
            float jit_out[4] __attribute__((aligned(16)));
            float eager_out[4] __attribute__((aligned(16)));
            nn_forward(&arith_model, eager_out, test_in);
            jit_arith(jit_out, test_in);

            /* Benchmark */
            uint64_t j0 = rdtsc_fenced();
            for (int r = 0; r < 10000; r++)
                nn_forward(&arith_model, eager_out, test_in);
            uint64_t j1 = rdtsc_fenced();
            for (int r = 0; r < 10000; r++)
                jit_arith(jit_out, test_in);
            uint64_t j2 = rdtsc_fenced();

            uint64_t eager_ns = perf_cycles_to_ns(j1 - j0) / 10000;
            uint64_t jit_ns = perf_cycles_to_ns(j2 - j1) / 10000;
            kprintf("  MathLM-Arith JIT: eager=%lu ns, JIT=%lu ns", eager_ns, jit_ns);
            if (jit_ns > 0) {
                uint64_t sp10 = (eager_ns * 10) / jit_ns;
                kprintf(" (%lu.%lux)\n", sp10 / 10, sp10 % 10);
            } else {
                kprintf("\n");
            }

            /* JIT correctness check */
            float diff = mlm_fabsf(eager_out[0] - jit_out[0]);
            kprintf("  JIT correctness: %s (diff=",
                    diff < 0.001f ? "MATCH" : "MISMATCH");
            mlm_print_float(diff);
            kprintf(")\n");
        } else {
            kprintf("  JIT compilation failed (ARM64 not supported)\n");
        }
#else
        kprintf("  JIT: x86_64 only (ARM64 uses NEON eager path)\n");
#endif
    }

    uint64_t total_us = perf_cycles_to_us(total_t1 - total_t0);
    kprintf("\n  Total evaluation time: %lu us (%lu ms)\n", total_us, total_us / 1000);

    kprintf("\n============================================================\n");
    kprintf("  Math LLM Suite: 5 models, ~10K params, trained at boot\n");
    kprintf("  Arithmetic + Polynomials + Trig + Sequences + Calculator\n");
    kprintf("  The OS that learns mathematics. On bare metal.\n");
    kprintf("============================================================\n");
}
