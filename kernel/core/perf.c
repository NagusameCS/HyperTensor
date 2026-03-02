/* =============================================================================
 * TensorOS - Performance Measurement & Benchmarks
 *
 * Uses RDTSC for cycle-accurate timing, calibrated against PIT.
 * Runs tensor operation benchmarks during boot to verify performance.
 * =============================================================================*/

#include "kernel/core/perf.h"
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#include "runtime/tensor/tensor_cpu.h"
#include "runtime/nn/inference.h"
#ifndef __aarch64__
#include "runtime/jit/x86_jit.h"
#endif

/* =============================================================================
 * TSC / Timer Calibration
 * =============================================================================*/

static uint64_t tsc_mhz = 0; /* TSC/counter frequency in MHz */

void perf_calibrate(void)
{
#if defined(__aarch64__)
    /* ARM64: Read CNTFRQ_EL0 directly — no calibration needed! */
    uint64_t freq;
    __asm__ volatile ("mrs %0, cntfrq_el0" : "=r"(freq));
    tsc_mhz = freq / 1000000;
    if (tsc_mhz == 0) tsc_mhz = 54; /* Fallback: BCM2711 default 54 MHz */
#else
    /* Use PIT channel 2 for calibration.
     * Count = 11932 ticks at 1.193182 MHz ≈ 10ms */
    uint16_t pit_count = 11932;

    /* Enable PIT channel 2 gate, disable speaker */
    uint8_t gate = inb(0x61);
    gate = (gate & 0xFD) | 0x01;
    outb(0x61, gate);

    /* Program channel 2: mode 0 (one-shot), binary, lobyte/hibyte */
    outb(0x43, 0xB0);
    outb(0x42, pit_count & 0xFF);
    outb(0x42, (pit_count >> 8) & 0xFF);

    /* Reload: toggle gate off then on */
    gate = inb(0x61);
    outb(0x61, gate & 0xFE);
    outb(0x61, gate | 0x01);

    /* Wait for OUT pin to go high (bit 5 of port 0x61) */
    uint64_t start = rdtsc();
    while (!(inb(0x61) & 0x20))
        ;
    uint64_t end = rdtsc();

    uint64_t elapsed_cycles = end - start;
    /* pit_count ticks at 1193182 Hz → time_us = pit_count * 1000000 / 1193182 */
    uint64_t time_us = (uint64_t)pit_count * 1000000ULL / 1193182ULL;

    if (time_us > 0)
        tsc_mhz = elapsed_cycles / time_us;

    if (tsc_mhz == 0)
        tsc_mhz = 1000; /* Fallback: assume 1 GHz */
#endif /* __aarch64__ */
}

uint64_t perf_tsc_mhz(void)    { return tsc_mhz; }

uint64_t perf_cycles_to_ns(uint64_t cycles)
{
    if (tsc_mhz == 0) return 0;
    return cycles * 1000ULL / tsc_mhz;
}

uint64_t perf_cycles_to_us(uint64_t cycles)
{
    if (tsc_mhz == 0) return 0;
    return cycles / tsc_mhz;
}

/* =============================================================================
 * Helper: Print float as "integer.fraction" (2 decimal places)
 * Since kprintf doesn't support %f yet we do it manually.
 * =============================================================================*/

static void print_float2(const char *label, float val, const char *unit)
{
    int neg = 0;
    if (val < 0.0f) { neg = 1; val = -val; }
    uint32_t ipart = (uint32_t)val;
    uint32_t fpart = (uint32_t)((val - (float)ipart) * 100.0f + 0.5f);
    if (fpart >= 100) { ipart++; fpart -= 100; }

    if (neg)
        kprintf("%s-%u.%02u %s", label, ipart, fpart, unit);
    else
        kprintf("%s%u.%02u %s", label, ipart, fpart, unit);
}

/* =============================================================================
 * Benchmarks
 * =============================================================================*/

/* Benchmark a single matmul size and print results.
 * Runs multiple iterations and reports the BEST (minimum-cycle) result
 * to filter out QEMU-TCG host noise. */
static void bench_matmul(int M, int N, int K, float *A, float *B, float *C)
{
    /* Adaptive iteration count based on problem size */
    uint64_t flops = 2ULL * M * N * K;
    int iters = (flops < 20000) ? 50 : (flops < 200000) ? 10 : 5;

    /* Warm up the TCG cache */
    for (int w = 0; w < 3; w++)
        tensor_cpu_matmul(C, A, B, M, N, K);

    /* Best of N runs */
    uint64_t best_cycles = (uint64_t)-1;
    for (int r = 0; r < iters; r++) {
        uint64_t t0 = rdtsc_fenced();
        tensor_cpu_matmul(C, A, B, M, N, K);
        uint64_t t1 = rdtsc_fenced();
        uint64_t c = t1 - t0;
        if (c < best_cycles) best_cycles = c;
    }

    uint64_t us = perf_cycles_to_us(best_cycles);
    /* MFLOPS = flops / us */
    uint64_t mflops = (us > 0) ? (flops / us) : 0;

    kprintf("  MatMul %dx%dx%d:  ", M, N, K);
    if (mflops >= 1000)
        print_float2("", (float)mflops / 1000.0f, "GFLOPS");
    else
        kprintf("%lu MFLOPS", mflops);
    kprintf("  (%lu us, %lu Kcycles, best-of-%d)\n", us, best_cycles / 1000, iters);
}

#ifndef __aarch64__
static void bench_jit_matmul(int M, int N, int K, float *A, float *B, float *C,
                              float *C2)
{
    /* C version */
    uint64_t t0 = rdtsc_fenced();
    tensor_cpu_matmul(C, A, B, M, N, K);
    uint64_t t1 = rdtsc_fenced();
    uint64_t c_cycles = t1 - t0;

    /* JIT compile */
    uint64_t jt0 = rdtsc_fenced();
    jit_matmul_fn fn = jit_compile_matmul_kernel(M, N, K);
    uint64_t jt1 = rdtsc_fenced();
    uint64_t compile_us = perf_cycles_to_us(jt1 - jt0);

    if (!fn) {
        kprintf("  JIT %dx%d: FAILED to compile\n", M, N);
        return;
    }

    /* JIT execute */
    tensor_cpu_zero(C2, M * N);
    uint64_t t2 = rdtsc_fenced();
    fn(C2, A, B, M, N, K);
    uint64_t t3 = rdtsc_fenced();
    uint64_t jit_cycles = t3 - t2;

    /* Verify correctness */
    int correct = 1;
    for (int i = 0; i < M * N; i++) {
        float diff = C[i] - C2[i];
        if (diff > 0.1f || diff < -0.1f) { correct = 0; break; }
    }

    uint64_t c_us = perf_cycles_to_us(c_cycles);
    uint64_t j_us = perf_cycles_to_us(jit_cycles);

    kprintf("  JIT %dx%d: C=%lu JIT=%lu us %s",
            M, N, c_us, j_us, correct ? "OK" : "BAD");

    if (jit_cycles > 0 && c_cycles > 0) {
        uint32_t ratio = (uint32_t)((c_cycles * 100ULL) / jit_cycles);
        kprintf(" (%u.%u%ux)", ratio / 100, (ratio / 10) % 10, ratio % 10);
    }
    kprintf("\n");
}
#endif /* !__aarch64__ */

/* =============================================================================
 * Main Benchmark Suite
 * Called during boot (Phase 5) to demonstrate real performance.
 * =============================================================================*/

void run_benchmarks(void)
{
    kprintf("\n[BENCH] TensorOS Performance Benchmarks\n");
    kprintf("  TSC frequency: %lu MHz\n", tsc_mhz);

    /* Allocate working buffers (use 16-byte aligned tensor heap) */
    /* Max size: 512×512 = 256K floats = 1MB per matrix */
    int max_dim = 512;
    int buf_size = max_dim * max_dim * sizeof(float);

    float *A  = (float *)tensor_alloc((uint64_t)buf_size);
    float *B  = (float *)tensor_alloc((uint64_t)buf_size);
    float *C  = (float *)tensor_alloc((uint64_t)buf_size);
    float *C2 = (float *)tensor_alloc((uint64_t)buf_size);

    if (!A || !B || !C || !C2) {
        kprintf("  [ERR] Failed to allocate benchmark buffers\n");
        return;
    }

    /* Fill A and B with small values */
    for (int i = 0; i < max_dim * max_dim; i++) {
        /* Use simple deterministic pattern */
        A[i] = (float)(i % 7) * 0.1f + 0.1f;
        B[i] = (float)(i % 5) * 0.1f + 0.1f;
    }

    kprintf("  --- Matrix Multiply (BLIS-style packed GEMM, SSE2) ---\n");
    bench_matmul(16, 16, 16, A, B, C);
    bench_matmul(32, 32, 32, A, B, C);
    bench_matmul(64, 64, 64, A, B, C);
    bench_matmul(128, 128, 128, A, B, C);
    bench_matmul(256, 256, 256, A, B, C);
    bench_matmul(512, 512, 512, A, B, C);

    /* Batch GEMV benchmark (turns into GEMM for batch>1) */
    kprintf("  --- Batch GEMV (inference throughput) ---\n");
    {
        int K = 256, N = 128;
        float *W = A;  /* reuse: 256x128 weights */
        float *bias = (float *)tensor_alloc(N * sizeof(float));
        float *batch_in = (float *)tensor_alloc(64 * K * sizeof(float));
        float *batch_out = (float *)tensor_alloc(64 * N * sizeof(float));
        if (bias && batch_in && batch_out) {
            for (int i = 0; i < N; i++) bias[i] = 0.01f;
            for (int i = 0; i < 64 * K; i++) batch_in[i] = (float)(i % 11) * 0.1f;

            /* Single inference (GEMV) */
            uint64_t t0 = rdtsc_fenced();
            for (int r = 0; r < 100; r++)
                tensor_cpu_batch_gemv(batch_out, batch_in, W, bias, 1, K, N, NN_ACT_RELU);
            uint64_t t1 = rdtsc_fenced();
            uint64_t us1 = perf_cycles_to_us(t1 - t0);
            uint64_t mflops1 = (us1 > 0) ? (2ULL * K * N * 100 / us1) : 0;
            kprintf("  GEMV 256x128 (batch=1): %lu MFLOPS (%lu us/100)\n", mflops1, us1);

            /* Batch=32 inference (GEMM) */
            uint64_t t2 = rdtsc_fenced();
            tensor_cpu_batch_gemv(batch_out, batch_in, W, bias, 32, K, N, NN_ACT_RELU);
            uint64_t t3 = rdtsc_fenced();
            uint64_t us32 = perf_cycles_to_us(t3 - t2);
            uint64_t mflops32 = (us32 > 0) ? (2ULL * 32 * K * N / us32) : 0;
            kprintf("  GEMM 32x256x128 (batch=32): %lu MFLOPS (%lu us)\n", mflops32, us32);

            tensor_free(batch_out);
            tensor_free(batch_in);
            tensor_free(bias);
        }
    }

    /* Conv2D benchmark */
    kprintf("  --- Conv2D (im2col + GEMM) ---\n");
    {
        int H = 16, W_ = 16, IC = 3, OC = 16, KH = 3, KW = 3;
        int OH = H - KH + 1, OW = W_ - KW + 1; /* stride=1, pad=0 */
        float *img = (float *)tensor_alloc(IC * H * W_ * sizeof(float));
        float *kern = (float *)tensor_alloc(OC * IC * KH * KW * sizeof(float));
        float *cbias = (float *)tensor_alloc(OC * sizeof(float));
        float *conv_out = (float *)tensor_alloc(OC * OH * OW * sizeof(float));
        if (img && kern && cbias && conv_out) {
            for (int i = 0; i < IC * H * W_; i++) img[i] = (float)(i % 7) * 0.1f;
            for (int i = 0; i < OC * IC * KH * KW; i++) kern[i] = (float)(i % 5) * 0.05f;
            for (int i = 0; i < OC; i++) cbias[i] = 0.01f;

            uint64_t t0 = rdtsc_fenced();
            for (int r = 0; r < 10; r++)
                tensor_cpu_conv2d(conv_out, img, kern, cbias, H, W_, IC, OC, KH, KW, 1, 0);
            uint64_t t1 = rdtsc_fenced();
            uint64_t us = perf_cycles_to_us(t1 - t0);
            uint64_t flops = 2ULL * OC * OH * OW * IC * KH * KW * 10;
            uint64_t mflops = (us > 0) ? (flops / us) : 0;
            kprintf("  Conv2D 16x16 3ch->16ch 3x3: %lu MFLOPS (%lu us/10)\n", mflops, us);

            /* Winograd F(2,3) benchmark on same workload
             * kern is already [OC, KH, KW, IC] = same layout Winograd expects.
             * Use pad=1 so output is same spatial size as input (16x16). */
            float *wino_out = (float *)tensor_alloc(OC * H * W_ * sizeof(float));
            if (wino_out) {
                    uint64_t wt0 = rdtsc_fenced();
                    for (int r = 0; r < 10; r++)
                        tensor_cpu_conv2d_winograd(wino_out, img, kern, cbias, H, W_, IC, OC, 1);
                    uint64_t wt1 = rdtsc_fenced();
                    uint64_t wus = perf_cycles_to_us(wt1 - wt0);
                    /* Same effective FLOPs for comparison (measures throughput, not raw muls) */
                    uint64_t wflops = 2ULL * OC * H * W_ * IC * 3 * 3 * 10;
                    uint64_t wmflops = (wus > 0) ? (wflops / wus) : 0;
                    kprintf("  Winograd F(2,3) 16x16 3->16ch: %lu MFLOPS (%lu us/10)\n", wmflops, wus);
                    if (us > 0 && wus > 0) {
                        uint64_t sp100 = us * 100 / wus;
                        kprintf("  Winograd speedup: %lu.%lux vs im2col\n",
                                sp100 / 100, (sp100 / 10) % 10);
                    }
                tensor_free(wino_out);
            }

            tensor_free(conv_out);
            tensor_free(cbias);
            tensor_free(kern);
            tensor_free(img);
        }
    }

    /* Activation function benchmarks */
    kprintf("  --- Activation Functions ---\n");
    {
        int n = 4096;
        float *x = A; /* reuse buffer */
        float *y = C;

        uint64_t t0 = rdtsc_fenced();
        tensor_cpu_relu(y, x, n);
        uint64_t t1 = rdtsc_fenced();
        kprintf("  ReLU %d:        %lu ns/elem  (%lu us total)\n",
                n, perf_cycles_to_ns(t1 - t0) / n, perf_cycles_to_us(t1 - t0));

        t0 = rdtsc_fenced();
        tensor_cpu_softmax(y, x, 1024);
        t1 = rdtsc_fenced();
        kprintf("  Softmax 1024:   %lu us\n", perf_cycles_to_us(t1 - t0));

        t0 = rdtsc_fenced();
        tensor_cpu_gelu(y, x, n);
        t1 = rdtsc_fenced();
        kprintf("  GELU %d:        %lu ns/elem\n",
                n, perf_cycles_to_ns(t1 - t0) / n);

        t0 = rdtsc_fenced();
        tensor_cpu_layernorm(y, x, 1024, 1e-5f);
        t1 = rdtsc_fenced();
        kprintf("  LayerNorm 1024: %lu us\n", perf_cycles_to_us(t1 - t0));
    }

    /* Self-tests */
    kprintf("  --- Correctness Verification ---\n");
    {
        int rc = tensor_cpu_selftest();
        kprintf("  CPU tensor math: %s", rc == 0 ? "PASS" : "FAIL");
        if (rc != 0) kprintf(" (code %d)", rc);
        kprintf("\n");
    }

    /* JIT benchmarks (x86_64 only) */
#if defined(__aarch64__)
    kprintf("  --- JIT Compiler (ARM64 \u2014 not yet ported) ---\n");
    kprintf("  JIT: skipped (x86_64 JIT backend only)\n");
#else
    kprintf("  --- JIT Compiler (x86_64 native codegen) ---\n");

    /* Minimal execution test: prologue+epilogue only */
    {
        jit_buf_t *tb = jit_create(256);
        if (tb) {
            jit_prologue(tb);
            jit_epilogue(tb);
            jit_void_fn tfn = jit_get_fn(tb);
            kprintf("  JIT exec test: ");
            tfn();
            kprintf("PASS (code at %p, %d bytes)\n", (void*)(uintptr_t)tfn, tb->len);
        }
    }

    bench_jit_matmul(16, 16, 16, A, B, C, C2);
    bench_jit_matmul(32, 32, 32, A, B, C, C2);
    bench_jit_matmul(64, 64, 64, A, B, C, C2);

    {
        int rc = jit_selftest();
        kprintf("  JIT self-test:   %s", rc == 0 ? "PASS" : "FAIL");
        if (rc != 0) kprintf(" (code %d)", rc);
        kprintf("\n");
    }

    kprintf("  JIT stats: %d kernels, %d bytes machine code\n",
            jit_kernel_count(), jit_code_bytes());
#endif

    /* Attention benchmark */
    kprintf("  --- Transformer Attention ---\n");
    {
        int seq = 32, dim = 64;
        int sz = seq * dim;
        float *Q = A, *K_m = B, *V = C;
        /* Fill V with values (Q and K are already filled) */
        for (int i = 0; i < sz; i++)
            V[i] = (float)(i % 3) * 0.1f + 0.1f;

        uint64_t t0 = rdtsc_fenced();
        tensor_cpu_attention(C2, Q, K_m, V, seq, dim);
        uint64_t t1 = rdtsc_fenced();
        kprintf("  Attention seq=%d dim=%d: %lu us\n",
                seq, dim, perf_cycles_to_us(t1 - t0));
    }

    /* Memory allocator benchmark */
    kprintf("  --- Memory Allocator ---\n");
    {
        uint64_t t0 = rdtsc_fenced();
        for (int i = 0; i < 100; i++) {
            void *p = tensor_alloc(4096);
            tensor_free(p);
        }
        uint64_t t1 = rdtsc_fenced();
        kprintf("  Alloc+Free 4KB (100x): %lu ns/op\n",
                perf_cycles_to_ns(t1 - t0) / 100);
    }

    kprintf("[BENCH] Complete\n\n");
}
