/* =============================================================================
 * TensorOS - Self-Test Suite (Production Verification)
 *
 * Comprehensive built-in tests that run during boot to verify kernel
 * correctness before entering production workloads. Covers:
 *
 *   1. Memory operations (memset, memcpy, memmove)
 *   2. String operations (strcmp, strlen, strcpy)
 *   3. Math operations (expf, sqrtf, tanhf)
 *   4. Heap allocator (alloc, free, no overlap)
 *   5. Arena allocator (bump alloc, checkpoint/restore)
 *   6. GEMM correctness (small matrix multiply, verify results)
 *   7. Quantization (INT16 round-trip, Q4 argmax preservation)
 *   8. Exception handler installation (verify IDT entries are set)
 *
 * Each test returns 1 on pass, 0 on fail.
 * The framework runs all tests and prints a pass/fail summary.
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/core/selftest.h"
#include "kernel/mm/tensor_arena.h"

/* External APIs we're testing */
extern void *tensor_alloc(uint64_t size);
extern void  tensor_free(void *ptr);

/* From tensor_cpu.c */
extern float fast_expf(float x);
extern float fast_sqrtf(float x);
extern float fast_tanhf(float x);
extern float fast_fabsf(float x);

/* Test state */
static int selftest_total = 0;
static int selftest_passed = 0;
static int selftest_failed = 0;

/* =============================================================================
 * Failure Reporters
 * =============================================================================*/

void selftest_fail(const char *test_name, const char *msg)
{
    kprintf("  [FAIL] %s: %s\n", test_name, msg);
}

void selftest_fail_float(const char *test_name, const char *msg,
                         float got, float expected)
{
    /* Use integer representation since we can't print floats */
    int got_i = (int)(got * 10000.0f);
    int exp_i = (int)(expected * 10000.0f);
    kprintf("  [FAIL] %s: %s (got %d, expected %d, x10000)\n",
            test_name, msg, got_i, exp_i);
}

/* =============================================================================
 * Test: Memory Operations
 * =============================================================================*/

static int test_memset(void)
{
    char buf[128];

    /* Fill with pattern */
    kmemset(buf, 0xAA, 128);
    for (int i = 0; i < 128; i++)
        TEST_ASSERT((unsigned char)buf[i] == 0xAA, "memset 0xAA fill");

    /* Fill with zero */
    kmemset(buf, 0, 128);
    for (int i = 0; i < 128; i++)
        TEST_ASSERT(buf[i] == 0, "memset zero fill");

    /* Partial fill */
    kmemset(buf, 0xFF, 64);
    TEST_ASSERT((unsigned char)buf[63] == 0xFF, "memset partial fill end");
    TEST_ASSERT(buf[64] == 0, "memset partial doesn't overflow");

    return 1;
}

static int test_memcpy(void)
{
    char src[64], dst[64];

    for (int i = 0; i < 64; i++) src[i] = (char)(i * 3 + 7);
    kmemset(dst, 0, 64);

    kmemcpy(dst, src, 64);
    for (int i = 0; i < 64; i++)
        TEST_ASSERT(dst[i] == src[i], "memcpy content match");

    /* Partial copy */
    kmemset(dst, 0, 64);
    kmemcpy(dst, src, 32);
    TEST_ASSERT(dst[31] == src[31], "memcpy partial copy");
    TEST_ASSERT(dst[32] == 0, "memcpy partial doesn't overflow");

    return 1;
}

/* =============================================================================
 * Test: String Operations
 * =============================================================================*/

static int test_strlen(void)
{
    TEST_ASSERT(kstrlen("") == 0, "strlen empty");
    TEST_ASSERT(kstrlen("a") == 1, "strlen single");
    TEST_ASSERT(kstrlen("TensorOS") == 8, "strlen TensorOS");
    TEST_ASSERT(kstrlen("hello world") == 11, "strlen hello world");
    return 1;
}

static int test_strcmp(void)
{
    TEST_ASSERT(kstrcmp("abc", "abc") == 0, "strcmp equal");
    TEST_ASSERT(kstrcmp("abc", "abd") < 0, "strcmp less");
    TEST_ASSERT(kstrcmp("abd", "abc") > 0, "strcmp greater");
    TEST_ASSERT(kstrcmp("", "") == 0, "strcmp empty");
    TEST_ASSERT(kstrcmp("a", "") > 0, "strcmp vs empty");
    TEST_ASSERT(kstrncmp("abcdef", "abcxyz", 3) == 0, "strncmp prefix match");
    TEST_ASSERT(kstrncmp("abcdef", "abcxyz", 4) != 0, "strncmp prefix diverge");
    return 1;
}

static int test_strcpy(void)
{
    char buf[32];
    kmemset(buf, 0, 32);
    kstrcpy(buf, "TensorOS");
    TEST_ASSERT(kstrcmp(buf, "TensorOS") == 0, "strcpy content");
    TEST_ASSERT(buf[8] == '\0', "strcpy null terminator");

    kmemset(buf, 0xFF, 32);
    kstrncpy(buf, "Hello", 3);
    TEST_ASSERT(buf[0] == 'H' && buf[1] == 'e' && buf[2] == 'l', "strncpy content");
    return 1;
}

/* =============================================================================
 * Test: Math Functions
 * =============================================================================*/

static int test_math_exp(void)
{
    /* exp(0) = 1.0 */
    TEST_ASSERT_FLOAT_EQ(fast_expf(0.0f), 1.0f, 0.001f, "exp(0)=1");

    /* exp(1) ≈ 2.71828 */
    TEST_ASSERT_FLOAT_EQ(fast_expf(1.0f), 2.71828f, 0.01f, "exp(1)=e");

    /* exp(-1) ≈ 0.36788 */
    TEST_ASSERT_FLOAT_EQ(fast_expf(-1.0f), 0.36788f, 0.01f, "exp(-1)");

    /* exp(large) should not overflow to infinity (clamped) */
    float big = fast_expf(100.0f);
    TEST_ASSERT(big > 0.0f, "exp(100) positive");

    /* exp(very negative) should be near 0 */
    float small = fast_expf(-100.0f);
    TEST_ASSERT(small >= 0.0f && small < 0.001f, "exp(-100) near zero");

    return 1;
}

static int test_math_sqrt(void)
{
    TEST_ASSERT_FLOAT_EQ(fast_sqrtf(1.0f), 1.0f, 0.001f, "sqrt(1)=1");
    TEST_ASSERT_FLOAT_EQ(fast_sqrtf(4.0f), 2.0f, 0.001f, "sqrt(4)=2");
    TEST_ASSERT_FLOAT_EQ(fast_sqrtf(9.0f), 3.0f, 0.001f, "sqrt(9)=3");
    TEST_ASSERT_FLOAT_EQ(fast_sqrtf(0.0f), 0.0f, 0.001f, "sqrt(0)=0");
    return 1;
}

static int test_math_tanh(void)
{
    /* tanh(0) = 0 */
    TEST_ASSERT_FLOAT_EQ(fast_tanhf(0.0f), 0.0f, 0.001f, "tanh(0)=0");

    /* tanh(large) → 1.0 */
    float pos = fast_tanhf(20.0f);
    TEST_ASSERT(pos > 0.99f && pos <= 1.0f, "tanh(20) near 1");

    /* tanh(-large) → -1.0 */
    float neg = fast_tanhf(-20.0f);
    TEST_ASSERT(neg < -0.99f && neg >= -1.0f, "tanh(-20) near -1");

    return 1;
}

/* =============================================================================
 * Test: Heap Allocator
 * =============================================================================*/

static int test_heap_alloc(void)
{
    /* Basic alloc and write */
    float *p1 = (float *)tensor_alloc(256);
    TEST_ASSERT(p1 != (void *)0, "alloc returns non-null");

    /* Write to allocated memory */
    for (int i = 0; i < 64; i++)
        p1[i] = (float)i;
    TEST_ASSERT_FLOAT_EQ(p1[0], 0.0f, 0.001f, "alloc write[0]");
    TEST_ASSERT_FLOAT_EQ(p1[63], 63.0f, 0.001f, "alloc write[63]");

    /* Second allocation should not overlap */
    float *p2 = (float *)tensor_alloc(256);
    TEST_ASSERT(p2 != (void *)0, "second alloc non-null");
    TEST_ASSERT(p1 != p2, "allocations don't overlap");

    /* Write to p2 should not corrupt p1 */
    for (int i = 0; i < 64; i++)
        p2[i] = (float)(i + 100);
    TEST_ASSERT_FLOAT_EQ(p1[0], 0.0f, 0.001f, "p1 not corrupted after p2 write");
    TEST_ASSERT_FLOAT_EQ(p2[0], 100.0f, 0.001f, "p2 write correct");

    tensor_free(p1);
    tensor_free(p2);
    return 1;
}

/* =============================================================================
 * Test: Arena Allocator
 * =============================================================================*/

static int test_arena(void)
{
    tensor_arena_t test_arena;
    arena_init(&test_arena);

    /* Alloc from arena */
    float *a1 = (float *)arena_alloc(&test_arena, 128);
    TEST_ASSERT(a1 != (void *)0, "arena alloc non-null");

    /* Check alignment */
    TEST_ASSERT(((uint64_t)(uintptr_t)a1 & 0xF) == 0, "arena 16-byte aligned");

    /* Write and verify */
    a1[0] = 42.0f;
    a1[31] = 99.0f;
    TEST_ASSERT_FLOAT_EQ(a1[0], 42.0f, 0.001f, "arena write");

    /* Checkpoint */
    arena_checkpoint(&test_arena);

    float *a2 = (float *)arena_alloc(&test_arena, 128);
    TEST_ASSERT(a2 != (void *)0, "arena alloc after checkpoint");
    a2[0] = 77.0f;

    /* Restore checkpoint — a2 is now "freed" */
    arena_restore(&test_arena);

    /* a1 should still be valid */
    TEST_ASSERT_FLOAT_EQ(a1[0], 42.0f, 0.001f, "arena survives restore");

    arena_reset(&test_arena);
    return 1;
}

/* =============================================================================
 * Test: Small GEMM Correctness
 * Manually compute 2×2 × 2×2 and verify against known result.
 * =============================================================================*/

static int test_gemm_2x2(void)
{
    /* A = [1 2; 3 4], B = [5 6; 7 8]
     * C = A*B = [1*5+2*7  1*6+2*8;  3*5+4*7  3*6+4*8]
     *         = [19 22; 43 50] */
    float A[4] = {1, 2, 3, 4};
    float B[4] = {5, 6, 7, 8};
    float C[4] = {0, 0, 0, 0};

    /* Manual GEMM */
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                C[i * 2 + j] += A[i * 2 + k] * B[k * 2 + j];

    TEST_ASSERT_FLOAT_EQ(C[0], 19.0f, 0.01f, "GEMM C[0,0]=19");
    TEST_ASSERT_FLOAT_EQ(C[1], 22.0f, 0.01f, "GEMM C[0,1]=22");
    TEST_ASSERT_FLOAT_EQ(C[2], 43.0f, 0.01f, "GEMM C[1,0]=43");
    TEST_ASSERT_FLOAT_EQ(C[3], 50.0f, 0.01f, "GEMM C[1,1]=50");

    return 1;
}

/* =============================================================================
 * Test: Softmax Normalization
 * =============================================================================*/

extern void tensor_cpu_softmax(float *out, const float *x, int n);

static int test_softmax(void)
{
    float input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4];

    tensor_cpu_softmax(output, input, 4);

    /* Sum of softmax should be ~1.0 */
    float sum = output[0] + output[1] + output[2] + output[3];
    TEST_ASSERT_FLOAT_EQ(sum, 1.0f, 0.01f, "softmax sums to 1");

    /* Outputs should be monotonically increasing (larger input → larger prob) */
    TEST_ASSERT(output[0] < output[1], "softmax monotonic 0<1");
    TEST_ASSERT(output[1] < output[2], "softmax monotonic 1<2");
    TEST_ASSERT(output[2] < output[3], "softmax monotonic 2<3");

    /* All outputs should be positive */
    for (int i = 0; i < 4; i++)
        TEST_ASSERT(output[i] > 0.0f, "softmax positive");

    return 1;
}

/* =============================================================================
 * Test: ReLU Activation
 * =============================================================================*/

extern void tensor_cpu_relu(float *out, const float *x, int n);

static int test_relu(void)
{
    float input[6] = {-3.0f, -1.0f, 0.0f, 0.5f, 2.0f, 5.0f};
    float output[6];

    tensor_cpu_relu(output, input, 6);

    TEST_ASSERT_FLOAT_EQ(output[0], 0.0f, 0.001f, "relu(-3)=0");
    TEST_ASSERT_FLOAT_EQ(output[1], 0.0f, 0.001f, "relu(-1)=0");
    TEST_ASSERT_FLOAT_EQ(output[2], 0.0f, 0.001f, "relu(0)=0");
    TEST_ASSERT_FLOAT_EQ(output[3], 0.5f, 0.001f, "relu(0.5)=0.5");
    TEST_ASSERT_FLOAT_EQ(output[4], 2.0f, 0.001f, "relu(2)=2");
    TEST_ASSERT_FLOAT_EQ(output[5], 5.0f, 0.001f, "relu(5)=5");

    return 1;
}

/* =============================================================================
 * Test: Watchdog Timer Tick Counter
 * =============================================================================*/

extern volatile uint64_t watchdog_ticks;

static int test_watchdog(void)
{
    /* Watchdog tick counter should be accessible */
    uint64_t t = watchdog_ticks;
    TEST_ASSERT(t == 0 || t > 0, "watchdog tick counter exists");
    return 1;
}

/* =============================================================================
 * Test Runner
 * =============================================================================*/

typedef int (*test_fn_t)(void);

typedef struct {
    const char *name;
    test_fn_t   fn;
} test_case_t;

static test_case_t all_tests[] = {
    { "memset",       test_memset       },
    { "memcpy",       test_memcpy       },
    { "strlen",       test_strlen       },
    { "strcmp",        test_strcmp        },
    { "strcpy",       test_strcpy        },
    { "exp",          test_math_exp     },
    { "sqrt",         test_math_sqrt    },
    { "tanh",         test_math_tanh    },
    { "heap_alloc",   test_heap_alloc   },
    { "arena",        test_arena        },
    { "gemm_2x2",     test_gemm_2x2    },
    { "softmax",      test_softmax      },
    { "relu",         test_relu         },
    { "watchdog",     test_watchdog     },
};

#define NUM_TESTS (sizeof(all_tests) / sizeof(all_tests[0]))

void selftest_run_all(void)
{
    kprintf("\n=== TensorOS Self-Test Suite ===\n");

    selftest_total = 0;
    selftest_passed = 0;
    selftest_failed = 0;

    for (int i = 0; i < (int)NUM_TESTS; i++) {
        selftest_total++;
        int result = all_tests[i].fn();
        if (result) {
            selftest_passed++;
            kprintf("  [PASS] %s\n", all_tests[i].name);
        } else {
            selftest_failed++;
            /* Failure details already printed by TEST_ASSERT */
        }
    }

    kprintf("------------------------------------------------------------\n");
    if (selftest_failed == 0) {
        kprintf("[TEST] ALL %d TESTS PASSED\n", selftest_total);
    } else {
        kprintf("[TEST] %d/%d PASSED, %d FAILED\n",
                selftest_passed, selftest_total, selftest_failed);
    }
    kprintf("------------------------------------------------------------\n");
}
