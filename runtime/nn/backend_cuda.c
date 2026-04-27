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
 * TensorOS CUDA Backend — Dynamic Loading
 *
 * Loads cuda_kernels.dll at runtime (compiled separately by nvcc).
 * Falls back gracefully if the DLL is not found.
 *
 * Build: compile with main project (no CUDA deps needed at compile time)
 * Runtime: needs cuda_kernels.dll next to the executable
 */

#ifdef ENABLE_CUDA

#include "runtime/nn/backend.h"

#ifdef _WIN32
#include <windows.h>
typedef HMODULE lib_handle_t;
#define LIB_OPEN(path)       LoadLibraryA(path)
#define LIB_SYM(h, name)    GetProcAddress(h, name)
#define LIB_CLOSE(h)        FreeLibrary(h)
#else
#include <dlfcn.h>
typedef void *lib_handle_t;
#define LIB_OPEN(path)       dlopen(path, RTLD_NOW)
#define LIB_SYM(h, name)    dlsym(h, name)
#define LIB_CLOSE(h)        dlclose(h)
#endif

#ifdef GEODESSICAL_HOSTED
#include "hal.h"
#include <stdlib.h>
#include <math.h>
#else
#include "kernel/core/kernel.h"
#endif

/* ─── Function pointer typedefs matching cuda_kernels.h ─── */
typedef int      (*fn_init)(void);
typedef void     (*fn_shutdown)(void);
typedef int      (*fn_device_count)(void);
typedef uint64_t (*fn_free_memory)(int);
typedef void    *(*fn_alloc)(uint64_t);
typedef void     (*fn_free)(void *);
typedef int      (*fn_upload)(void *, const void *, uint64_t);
typedef int      (*fn_download)(void *, const void *, uint64_t);
typedef void     (*fn_sync)(void);
typedef void     (*fn_gemv)(float *, const void *, const float *, int, int, int);
typedef void     (*fn_gemm)(float *, const float *, const float *, int, int, int);
typedef void     (*fn_rmsnorm)(float *, const float *, const float *, int, float);
typedef void     (*fn_layernorm)(float *, const float *, const float *, const float *, int, float);
typedef void     (*fn_rope)(float *, float *, int, int, int, int, float, const float *);
typedef void     (*fn_softmax)(float *, int);
typedef void     (*fn_silu)(float *, int);
typedef void     (*fn_gelu)(float *, int);
typedef void     (*fn_mul)(float *, const float *, const float *, int);
typedef void     (*fn_add)(float *, const float *, const float *, int);
typedef void     (*fn_scale)(float *, const float *, float, int);
typedef float    (*fn_dot)(const float *, const float *, int);
typedef void     (*fn_dequantize)(float *, const void *, int, int);
typedef void     (*fn_attention)(float *, const float *, const float *, const float *,
                                 int, int, int, int, int, float, float);
typedef void     (*fn_kv_update)(float *, float *, const float *, const float *,
                                  int, int, int, int, int);
typedef void     (*fn_embed)(float *, const void *, int, int, int);
typedef void     (*fn_softcap)(float *, int, float);
typedef int      (*fn_upload_async)(void *, const void *, uint64_t);
typedef int      (*fn_download_async)(void *, const void *, uint64_t);
typedef void     (*fn_stream_sync)(void);
typedef void     (*fn_fused_qk_norm_rope)(float *, float *, const float *, const float *,
                                           int, int, int, int, float, const float *, float, int);
typedef void     (*fn_v_norm)(float *, int, int, float);
typedef void     (*fn_gemv_async)(float *, const void *, const float *, int, int, int);
typedef void     (*fn_fused_geglu)(float *, const float *, int);
typedef void     (*fn_batched_rmsnorm)(float *, const float *, int, int, float);
typedef void     (*fn_iswa_combine)(float *, const float *, const float *, float, int);
typedef void     (*fn_dequant_q4_f16)(void *, const void *, int, int);
typedef void     (*fn_add_rmsnorm)(float *, float *, const float *, const float *, int, float);
typedef void     (*fn_rmsnorm_add)(float *, const float *, const float *, int, float);
typedef void     (*fn_gelu_mul)(float *, const float *, int);
typedef void     (*fn_gemv_dual_q4_0)(float *, float *, const void *, const void *,
                                       const float *, int, int);
typedef void     (*fn_gemv_dual_q8_0)(float *, float *, const void *, const void *,
                                       const float *, int, int);
typedef void     (*fn_gemv_triple_q4_0)(float *, float *, float *,
                                         const void *, const void *, const void *,
                                         const float *, int, int, int, int);
typedef void     (*fn_fused_rmsnorm_triple_q4_0)(float *, float *, float *,
                                                  const void *, const void *, const void *,
                                                  const float *, const float *, float,
                                                  int, int, int, int);
typedef int      (*fn_graph_op)(void);
typedef void     (*fn_graph_destroy)(void);
typedef void     (*fn_set_decode_pos)(int, int);
typedef int      (*fn_argmax)(const float *, int);
/* Batch Prefill */
typedef void     (*fn_prefill_batch_presized)(int, int);
typedef void     (*fn_prefill_batch_quant)(const float *, int, int);
typedef void     (*fn_prefill_batch_gemv_q4)(float *, const void *, int, int, int);
typedef void     (*fn_batched_rmsnorm_out)(float *, const float *, const float *, int, int, float);
/* Batched prefill attention kernels */
typedef void     (*fn_batch_fused_qk_norm_rope)(float *, float *, const float *, const float *,
                                                int, int, int, int, int, float, const float *,
                                                float, int);
typedef void     (*fn_batch_v_norm)(float *, int, int, int, float);
typedef void     (*fn_batch_kv_update)(float *, float *, const float *, const float *,
                                       int, int, int, int, int);
typedef void     (*fn_prefill_attn_batched)(float *, const float *, const float *, const float *,
                                            int, int, int, int, int, int, float, float);
typedef void     (*fn_sgemm_batched_f32)(int, int,
                                          const float * const *, const float * const *,
                                          float * const *, int);
typedef void     (*fn_l2_persist)(const void *, size_t);

/* ─── Dynamic dispatch table ─── */
static struct {
    lib_handle_t    lib;
    fn_init         init;
    fn_shutdown     shutdown;
    fn_device_count device_count;
    fn_free_memory  free_memory;
    fn_alloc        alloc;
    fn_free         free;
    fn_upload       upload;
    fn_download     download;
    fn_sync         sync;
    fn_gemv         gemv;
    fn_gemm         gemm;
    fn_rmsnorm      rmsnorm;
    fn_layernorm    layernorm;
    fn_rope         rope;
    fn_softmax      softmax;
    fn_silu         silu;
    fn_gelu         gelu;
    fn_mul          mul;
    fn_add          add;
    fn_scale        scale;
    fn_dot          dot;
    fn_dequantize   dequantize;
    fn_attention    attention;
    fn_kv_update    kv_update;
    fn_embed        embed;
    fn_softcap      softcap;
    fn_upload_async upload_async;
    fn_download_async download_async;
    fn_stream_sync  stream_sync_transfer;
    fn_stream_sync  stream_sync_compute;
    fn_fused_qk_norm_rope fused_qk_norm_rope;
    fn_v_norm       v_norm;
    fn_gemv_async   gemv_async;
    fn_fused_geglu  fused_geglu;
    fn_fused_geglu  fused_swiglu;
    fn_batched_rmsnorm batched_rmsnorm;
    fn_iswa_combine iswa_combine;
    fn_dequant_q4_f16 dequant_q4_0_to_f16;
    fn_add_rmsnorm  add_rmsnorm;
    fn_rmsnorm_add  rmsnorm_add;
    fn_gelu_mul     gelu_mul;
    fn_gemv_dual_q4_0              gemv_dual_q4_0;
    fn_gemv_dual_q8_0              gemv_dual_q8_0;
    fn_gemv_triple_q4_0            gemv_triple_q4_0;
    fn_fused_rmsnorm_triple_q4_0   fused_rmsnorm_triple_q4_0;
    fn_graph_op     graph_begin_capture;
    fn_graph_op     graph_end_capture;
    fn_graph_op     graph_launch;
    fn_graph_destroy graph_destroy;
    fn_set_decode_pos set_decode_pos;
    fn_argmax       argmax;
    /* Batch Prefill */
    fn_prefill_batch_presized  prefill_batch_presized;
    fn_prefill_batch_quant     prefill_batch_quant;
    fn_prefill_batch_gemv_q4   prefill_batch_gemv_q4;
    fn_batched_rmsnorm_out     batched_rmsnorm_out;
    /* Batched prefill attention */
    fn_batch_fused_qk_norm_rope batch_fused_qk_norm_rope;
    fn_batch_v_norm             batch_v_norm;
    fn_batch_kv_update          batch_kv_update;
    fn_prefill_attn_batched     prefill_attn_batched;
    /* cuBLAS batched GEMV — optional, soft-loaded */
    fn_sgemm_batched_f32        sgemm_batched_f32;
    /* L2 persistent cache pinning — optional, soft-loaded */
    fn_l2_persist               l2_persist;
} ck;

/* Load a symbol, return 0 on success */
#define LOAD_SYM(field, sym_name) do { \
    ck.field = (typeof(ck.field))LIB_SYM(ck.lib, sym_name); \
    if (!ck.field) { kprintf("[CUDA] Missing symbol: %s\n", sym_name); return -1; } \
} while (0)

static int cuda_load_library(void) {
    /* Allow override via environment variable for testing alternate builds */
    const char *env_path = getenv("GD_CUDA_KERNELS_PATH");

    /* Try several paths */
    const char *paths[] = {
#ifdef _WIN32
        "cuda_kernels.dll",
        "build_host/cuda_kernels.dll",
        "./cuda_kernels.dll",
#else
        "./cuda_kernels.so",
        "/root/cuda_kernels.so",
        "/usr/local/lib/cuda_kernels.so",
#endif
        NULL
    };

    /* Env var takes highest priority */
    if (env_path) {
        ck.lib = LIB_OPEN(env_path);
        if (ck.lib) {
            kprintf("[CUDA] Loaded kernels from GD_CUDA_KERNELS_PATH: %s\n", env_path);
        } else {
            kprintf("[CUDA] WARNING: GD_CUDA_KERNELS_PATH=%s failed to load, trying defaults\n", env_path);
        }
    }

    if (!ck.lib) {
        for (int i = 0; paths[i]; i++) {
            ck.lib = LIB_OPEN(paths[i]);
            if (ck.lib) break;
        }
    }

    if (!ck.lib) {
#ifdef _WIN32
        kprintf("[CUDA] cuda_kernels.dll not found\n");
#else
        kprintf("[CUDA] cuda_kernels.so not found\n");
#endif
        return -1;
    }

    LOAD_SYM(init,         "ck_init");
    LOAD_SYM(shutdown,     "ck_shutdown");
    LOAD_SYM(device_count, "ck_device_count");
    LOAD_SYM(free_memory,  "ck_free_memory");
    LOAD_SYM(alloc,        "ck_alloc");
    LOAD_SYM(free,         "ck_free");
    LOAD_SYM(upload,       "ck_upload");
    LOAD_SYM(download,     "ck_download");
    LOAD_SYM(sync,         "ck_sync");
    LOAD_SYM(gemv,         "ck_gemv");
    LOAD_SYM(gemm,         "ck_gemm");
    LOAD_SYM(rmsnorm,      "ck_rmsnorm");
    LOAD_SYM(layernorm,    "ck_layernorm");
    LOAD_SYM(rope,         "ck_rope");
    LOAD_SYM(softmax,      "ck_softmax");
    LOAD_SYM(silu,         "ck_silu");
    LOAD_SYM(gelu,         "ck_gelu");
    LOAD_SYM(mul,          "ck_mul");
    LOAD_SYM(add,          "ck_add");
    LOAD_SYM(scale,        "ck_scale");
    LOAD_SYM(dot,          "ck_dot");
    LOAD_SYM(dequantize,   "ck_dequantize");
    LOAD_SYM(attention,    "ck_attention");
    LOAD_SYM(kv_update,    "ck_kv_update");
    LOAD_SYM(embed,        "ck_embed_lookup");
    LOAD_SYM(softcap,      "ck_softcap");    LOAD_SYM(upload_async,      "ck_upload_async");
    LOAD_SYM(download_async,    "ck_download_async");
    LOAD_SYM(stream_sync_transfer, "ck_stream_sync_transfer");
    LOAD_SYM(stream_sync_compute,  "ck_stream_sync_compute");
    LOAD_SYM(fused_qk_norm_rope,   "ck_fused_qk_norm_rope");
    LOAD_SYM(v_norm,               "ck_v_norm");
    LOAD_SYM(gemv_async,           "ck_gemv_async");
    LOAD_SYM(fused_geglu,          "ck_fused_geglu");
    LOAD_SYM(fused_swiglu,         "ck_fused_swiglu");
    LOAD_SYM(batched_rmsnorm,      "ck_batched_rmsnorm");
    LOAD_SYM(iswa_combine,         "ck_iswa_combine");
    LOAD_SYM(dequant_q4_0_to_f16,  "ck_dequant_q4_0_to_f16");

    /* Optional fused kernels — soft-load (don't fail if missing) */
    ck.add_rmsnorm = (fn_add_rmsnorm)LIB_SYM(ck.lib, "ck_add_rmsnorm");
    ck.rmsnorm_add = (fn_rmsnorm_add)LIB_SYM(ck.lib, "ck_rmsnorm_add");
    ck.gelu_mul    = (fn_gelu_mul)LIB_SYM(ck.lib, "ck_gelu_mul");
    ck.gemv_dual_q4_0              = (fn_gemv_dual_q4_0)LIB_SYM(ck.lib, "ck_gemv_dual_q4_0");
    ck.gemv_dual_q8_0              = (fn_gemv_dual_q8_0)LIB_SYM(ck.lib, "ck_gemv_dual_q8_0");
    ck.gemv_triple_q4_0            = (fn_gemv_triple_q4_0)LIB_SYM(ck.lib, "ck_gemv_triple_q4_0");
    ck.fused_rmsnorm_triple_q4_0   = (fn_fused_rmsnorm_triple_q4_0)LIB_SYM(ck.lib, "ck_fused_rmsnorm_triple_q4_0");
    ck.graph_begin_capture = (fn_graph_op)LIB_SYM(ck.lib, "ck_graph_begin_capture");
    ck.graph_end_capture   = (fn_graph_op)LIB_SYM(ck.lib, "ck_graph_end_capture");
    ck.graph_launch        = (fn_graph_op)LIB_SYM(ck.lib, "ck_graph_launch");
    ck.graph_destroy       = (fn_graph_destroy)LIB_SYM(ck.lib, "ck_graph_destroy");
    ck.set_decode_pos      = (fn_set_decode_pos)LIB_SYM(ck.lib, "ck_set_decode_pos");
    ck.argmax              = (fn_argmax)LIB_SYM(ck.lib, "ck_argmax");
    /* Batch Prefill — optional (soft-load) */
    ck.prefill_batch_presized = (fn_prefill_batch_presized)LIB_SYM(ck.lib, "ck_prefill_batch_presized");
    ck.prefill_batch_quant    = (fn_prefill_batch_quant)LIB_SYM(ck.lib, "ck_prefill_batch_quant");
    ck.prefill_batch_gemv_q4  = (fn_prefill_batch_gemv_q4)LIB_SYM(ck.lib, "ck_prefill_batch_gemv_q4");
    ck.batched_rmsnorm_out    = (fn_batched_rmsnorm_out)LIB_SYM(ck.lib, "ck_batched_rmsnorm_out");
    /* Batched prefill attention — soft-load */
    ck.batch_fused_qk_norm_rope = (fn_batch_fused_qk_norm_rope)LIB_SYM(ck.lib, "ck_batch_fused_qk_norm_rope");
    ck.batch_v_norm             = (fn_batch_v_norm)LIB_SYM(ck.lib, "ck_batch_v_norm");
    ck.batch_kv_update          = (fn_batch_kv_update)LIB_SYM(ck.lib, "ck_batch_kv_update");
    ck.prefill_attn_batched     = (fn_prefill_attn_batched)LIB_SYM(ck.lib, "ck_prefill_attn_batched");
    /* cuBLAS batched GEMV — soft-load (only present when built with -lcublas) */
    ck.sgemm_batched_f32 = (fn_sgemm_batched_f32)LIB_SYM(ck.lib, "ck_sgemm_batched_f32");
    /* L2 persistent cache — soft-load */
    ck.l2_persist = (fn_l2_persist)LIB_SYM(ck.lib, "ck_l2_persist");
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Backend vtable wrappers — delegate to loaded DLL
 * ════════════════════════════════════════════════════════════════════════ */

static void *cuda_alloc(uint64_t size)                    { return ck.alloc ? ck.alloc(size) : NULL; }
static void  cuda_free(void *ptr)                         { if (ck.free) ck.free(ptr); }
static int   cuda_upload(void *d, const void *s, uint64_t sz)   { return ck.upload ? ck.upload(d, s, sz) : -1; }
static int   cuda_download(void *d, const void *s, uint64_t sz) { return ck.download ? ck.download(d, s, sz) : -1; }
static void  cuda_sync(void)                              { if (ck.sync) ck.sync(); }

/* When Q4_0 weights are dequantized to F16 on GPU, remap GEMV type */
static int cuda_q4_as_f16 = 0;

void cuda_set_q4_dequant_flag(int v) { cuda_q4_as_f16 = v; }

static void cuda_gemv(float *o, const void *w, const float *x,
                      int od, int id, ggml_type_t t) {
    int type = (int)t;
    if (cuda_q4_as_f16 && type == 2) type = 1;  /* Q4_0(2) → F16(1) */
    ck.gemv(o, w, x, od, id, type);
}

static void cuda_gemm(float *C, const float *A, const float *B,
                      int M, int N, int K) {
    ck.gemm(C, A, B, M, N, K);
}

static void cuda_rmsnorm(float *o, const float *x, const float *w,
                         int d, float e) {
    ck.rmsnorm(o, x, w, d, e);
}

static void cuda_layernorm(float *o, const float *x, const float *w,
                           const float *b, int d, float e) {
    ck.layernorm(o, x, w, b, d, e);
}

static void cuda_rope(float *q, float *k, int hd, int nh,
                      int nkv, int p, float b, const float *f) {
    ck.rope(q, k, hd, nh, nkv, p, b, f);
}

static void cuda_softmax(float *x, int n)  { ck.softmax(x, n); }
static void cuda_silu(float *x, int n)     { ck.silu(x, n); }
static void cuda_gelu(float *x, int n)     { ck.gelu(x, n); }

static void cuda_mul(float *o, const float *a, const float *b, int n) {
    ck.mul(o, a, b, n);
}

static void cuda_add(float *o, const float *a, const float *b, int n) {
    ck.add(o, a, b, n);
}

static void cuda_scale(float *o, const float *x, float s, int n) {
    ck.scale(o, x, s, n);
}

static float cuda_dot(const float *a, const float *b, int n) {
    return ck.dot(a, b, n);
}

static void cuda_dequant(float *o, const void *d, int n, ggml_type_t t) {
    ck.dequantize(o, d, n, (int)t);
}

static void cuda_attention(float *o, const float *Q, const float *K, const float *V,
                           int nh, int nkv, int hd, int sl, int ms, float sc, float cap) {
    ck.attention(o, Q, K, V, nh, nkv, hd, sl, ms, sc, cap);
}

static void cuda_kv_update(float *K, float *V, const float *Kn, const float *Vn,
                           int nkv, int hd, int p, int ms, int l) {
    ck.kv_update(K, V, Kn, Vn, nkv, hd, p, ms, l);
}

static void cuda_embed(float *o, const void *t, int id, int d, ggml_type_t ty) {
    ck.embed(o, t, id, d, (int)ty);
}

static void cuda_softcap_fn(float *x, int n, float c) {
    ck.softcap(x, n, c);
}

/* ═══════════════════════════════════════════════════════════════════════
 * CUDA-specific fused kernels — bypass vtable for performance
 * ════════════════════════════════════════════════════════════════════════ */

void cuda_fused_qk_norm_rope(float *Q, float *K,
    const float *q_norm_w, const float *k_norm_w,
    int n_heads, int n_kv_heads, int head_dim,
    int pos, float rope_base, const float *rope_freqs,
    float eps, int rope_dim)
{
    ck.fused_qk_norm_rope(Q, K, q_norm_w, k_norm_w,
        n_heads, n_kv_heads, head_dim,
        pos, rope_base, rope_freqs, eps, rope_dim);
}

void cuda_v_norm(float *V, int n_kv_heads, int head_dim, float eps) {
    ck.v_norm(V, n_kv_heads, head_dim, eps);
}

void cuda_fused_geglu(float *gate, const float *up, int n) {
    ck.fused_geglu(gate, up, n);
}

void cuda_fused_swiglu(float *gate, const float *up, int n) {
    ck.fused_swiglu(gate, up, n);
}

/* cuBLAS batched GEMV: y[i] = A[i]*x[i], A[i] is [M×K] row-major.
 * All pointer arrays (d_Aarray, d_xarray, d_yarray) must be in device memory.
 * Silently falls back to nothing if symbol not available (build without cuBLAS). */
void cuda_sgemm_batched_f32(int M, int K,
                              const float * const *d_Aarray,
                              const float * const *d_xarray,
                              float * const       *d_yarray,
                              int batch_count) {
    if (ck.sgemm_batched_f32)
        ck.sgemm_batched_f32(M, K, d_Aarray, d_xarray, d_yarray, batch_count);
}

void cuda_batched_rmsnorm(float *data, const float *w,
                           int n_slices, int slice_dim, float eps) {
    ck.batched_rmsnorm(data, w, n_slices, slice_dim, eps);
}

void cuda_iswa_combine(float *out, const float *tok_embd,
                        const float *proj, float scale, int n) {
    ck.iswa_combine(out, tok_embd, proj, scale, n);
}

void cuda_add_rmsnorm(float *norm_out, float *x_inout,
                       const float *residual, const float *norm_w,
                       int dim, float eps) {
    if (ck.add_rmsnorm) {
        ck.add_rmsnorm(norm_out, x_inout, residual, norm_w, dim, eps);
    } else {
        /* Fallback: separate add + rmsnorm */
        ck.add(x_inout, x_inout, residual, dim);
        ck.rmsnorm(norm_out, x_inout, norm_w, dim, eps);
    }
}

void cuda_rmsnorm_add(float *x_inout, const float *data,
                       const float *norm_w, int dim, float eps) {
    if (ck.rmsnorm_add) {
        ck.rmsnorm_add(x_inout, data, norm_w, dim, eps);
    } else {
        /* Fallback: rmsnorm in-place + add
         * Note: we need a temp buffer for in-place rmsnorm into add.
         * Since data is const, we use the simple fallback approach. */
        float *tmp = (float *)ck.alloc((uint64_t)dim * sizeof(float));
        if (tmp) {
            ck.rmsnorm(tmp, data, norm_w, dim, eps);
            ck.add(x_inout, x_inout, tmp, dim);
            ck.free(tmp);
        }
    }
}

void cuda_gelu_mul(float *a, const float *b, int n) {
    if (ck.gelu_mul) {
        ck.gelu_mul(a, b, n);
    } else {
        ck.gelu(a, n);
        ck.mul(a, a, b, n);
    }
}

int cuda_gemv_dual_q4_0(float *out_a, float *out_b,
                         const void *W_a, const void *W_b,
                         const float *x, int out_dim, int in_dim) {
    if (ck.gemv_dual_q4_0) {
        ck.gemv_dual_q4_0(out_a, out_b, W_a, W_b, x, out_dim, in_dim);
        return 1;
    }
    return 0;
}

int cuda_gemv_dual_q8_0(float *out_a, float *out_b,
                         const void *W_a, const void *W_b,
                         const float *x, int out_dim, int in_dim) {
    if (ck.gemv_dual_q8_0) {
        ck.gemv_dual_q8_0(out_a, out_b, W_a, W_b, x, out_dim, in_dim);
        return 1;
    }
    return 0;
}

int cuda_gemv_triple_q4_0(float *out_q, float *out_k, float *out_v,
                            const void *W_q, const void *W_k, const void *W_v,
                            const float *x,
                            int q_dim, int k_dim, int v_dim, int in_dim) {
    if (ck.gemv_triple_q4_0) {
        ck.gemv_triple_q4_0(out_q, out_k, out_v, W_q, W_k, W_v, x,
                             q_dim, k_dim, v_dim, in_dim);
        return 1;
    }
    return 0;
}

/* Fused RMSNorm + triple Q4_0 GEMV: computes Q, K, V in one kernel
 * launch, normalizing x on-the-fly with norm_w/eps (no d_xn write).
 * Returns 1 if the kernel was available, 0 if caller must fall back. */
int cuda_fused_rmsnorm_triple_q4_0(
        float *out_q, float *out_k, float *out_v,
        const void *W_q, const void *W_k, const void *W_v,
        const float *x, const float *norm_w, float eps,
        int q_dim, int k_dim, int v_dim, int in_dim) {
    if (ck.fused_rmsnorm_triple_q4_0) {
        ck.fused_rmsnorm_triple_q4_0(out_q, out_k, out_v, W_q, W_k, W_v,
                                      x, norm_w, eps,
                                      q_dim, k_dim, v_dim, in_dim);
        return 1;
    }
    return 0;
}

void cuda_dequant_q4_0_to_f16(void *out, const void *q4_data, int n_rows, int in_dim) {
    ck.dequant_q4_0_to_f16(out, q4_data, n_rows, in_dim);
}

int cuda_graph_begin_capture(void) {
    return ck.graph_begin_capture ? ck.graph_begin_capture() : -1;
}

int cuda_graph_end_capture(void) {
    return ck.graph_end_capture ? ck.graph_end_capture() : -1;
}

int cuda_graph_launch(void) {
    return ck.graph_launch ? ck.graph_launch() : -1;
}

void cuda_graph_destroy(void) {
    if (ck.graph_destroy) ck.graph_destroy();
}

void cuda_set_decode_pos(int pos, int seq_len) {
    if (ck.set_decode_pos) ck.set_decode_pos(pos, seq_len);
}

int cuda_argmax(const float *data, int n) {
    return ck.argmax ? ck.argmax(data, n) : -1;
}

/* ─── Batched prefill attention wrappers ─── */
void cuda_batch_fused_qk_norm_rope(float *Q, float *K,
    const float *q_norm_w, const float *k_norm_w,
    int n_heads, int n_kv_heads, int head_dim,
    int n, int start_pos, float rope_base, const float *rope_freqs,
    float eps, int rope_dim)
{
    if (ck.batch_fused_qk_norm_rope)
        ck.batch_fused_qk_norm_rope(Q, K, q_norm_w, k_norm_w,
            n_heads, n_kv_heads, head_dim,
            n, start_pos, rope_base, rope_freqs, eps, rope_dim);
}

void cuda_batch_v_norm(float *V, int n_kv_heads, int head_dim, int n, float eps) {
    if (ck.batch_v_norm) ck.batch_v_norm(V, n_kv_heads, head_dim, n, eps);
}

void cuda_batch_kv_update(float *K_cache, float *V_cache,
    const float *K_new, const float *V_new,
    int n_kv_heads, int head_dim, int n, int start_pos, int max_seq)
{
    if (ck.batch_kv_update)
        ck.batch_kv_update(K_cache, V_cache, K_new, V_new,
                           n_kv_heads, head_dim, n, start_pos, max_seq);
}

void cuda_prefill_attn_batched(float *O, const float *Q,
    const float *K_cache, const float *V_cache,
    int n_heads, int n_kv_heads, int head_dim,
    int n, int start_pos, int max_seq, float scale, float softcap)
{
    if (ck.prefill_attn_batched)
        ck.prefill_attn_batched(O, Q, K_cache, V_cache,
                                n_heads, n_kv_heads, head_dim,
                                n, start_pos, max_seq, scale, softcap);
}

int cuda_have_batch_attn(void) {
    return (ck.prefill_attn_batched != NULL) ? 1 : 0;
}

int cuda_have_sgemm_batched_f32(void) {
    return (ck.sgemm_batched_f32 != NULL) ? 1 : 0;
}

/* ─── Batch Prefill wrappers ─── */
void cuda_prefill_batch_presized(int max_batch, int max_dim) {
    if (ck.prefill_batch_presized) ck.prefill_batch_presized(max_batch, max_dim);
}

void cuda_prefill_batch_quant(const float *X, int batch, int in_dim) {
    if (ck.prefill_batch_quant) ck.prefill_batch_quant(X, batch, in_dim);
}

int cuda_prefill_batch_gemv_q4(float *C, const void *W,
                                int out_dim, int in_dim, int batch) {
    if (!ck.prefill_batch_gemv_q4) return 0;
    ck.prefill_batch_gemv_q4(C, W, out_dim, in_dim, batch);
    return 1;
}

void cuda_batched_rmsnorm_out(float *out, const float *in, const float *w,
                               int n, int d, float eps) {
    if (ck.batched_rmsnorm_out) ck.batched_rmsnorm_out(out, in, w, n, d, eps);
}

int cuda_upload_async(void *dst, const void *src, uint64_t size) {
    return ck.upload_async(dst, src, size);
}

int cuda_download_async(void *dst, const void *src, uint64_t size) {
    return ck.download_async(dst, src, size);
}

void cuda_stream_sync_transfer(void) {
    ck.stream_sync_transfer();
}

void cuda_stream_sync_compute(void) {
    ck.stream_sync_compute();
}

void cuda_l2_persist(const void *ptr, size_t bytes) {
    if (ck.l2_persist) ck.l2_persist(ptr, bytes);
}

/* CPU fallback implementations for tensor operations */
static void cpu_gemv(float *out, const float *matrix, const float *vector, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        out[i] = 0.0f;
        for (int j = 0; j < cols; j++) {
            out[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

static void cpu_rmsnorm(float *out, const float *input, const float *gamma, int dim, float epsilon) {
    float mean = 0.0f, variance = 0.0f;
    for (int i = 0; i < dim; i++) {
        mean += input[i];
    }
    mean /= dim;
    for (int i = 0; i < dim; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance = sqrtf(variance / dim + epsilon);
    for (int i = 0; i < dim; i++) {
        out[i] = gamma[i] * (input[i] - mean) / variance;
    }
}

/* Modify dispatch logic to use CPU fallbacks when GPU is unavailable */
static void dispatch_gemv(float *out, const float *matrix, const float *vector, int rows, int cols) {
    if (ck.gemv) {
        ck.gemv(out, matrix, vector, rows, cols, 0);
    } else {
        cpu_gemv(out, matrix, vector, rows, cols);
    }
}

static void dispatch_rmsnorm(float *out, const float *input, const float *gamma, int dim, float epsilon) {
    if (ck.rmsnorm) {
        ck.rmsnorm(out, input, gamma, dim, epsilon);
    } else {
        cpu_rmsnorm(out, input, gamma, dim, epsilon);
    }
}

/* ═══════════════════════════════════════════════════════════════════════
 * Init / Shutdown
 * ════════════════════════════════════════════════════════════════════════ */

static int cuda_init(void) {
    if (cuda_load_library() != 0) return -1;
    int ret = ck.init();
    if (ret == 0) {
        int count = ck.device_count();
        uint64_t free_mb = ck.free_memory(0) / (1024 * 1024);
        kprintf("[CUDA] Initialized: %d device(s), %llu MB free\n",
                count, (unsigned long long)free_mb);
    }
    return ret;
}

static void cuda_shutdown(void) {
    if (ck.shutdown) ck.shutdown();
    if (ck.lib) { LIB_CLOSE(ck.lib); ck.lib = 0; }
}

static int cuda_device_count(void) {
    return ck.device_count ? ck.device_count() : 0;
}

static uint64_t cuda_free_memory_fn(int dev) {
    return ck.free_memory ? ck.free_memory(dev) : 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Backend Definition
 * ════════════════════════════════════════════════════════════════════════ */

const backend_t backend_cuda = {
    .id   = BACKEND_CUDA,
    .name = "cuda",
    .init = cuda_init,
    .shutdown = cuda_shutdown,
    .get_device_count = cuda_device_count,
    .get_free_memory  = cuda_free_memory_fn,
    .mem = {
        .alloc    = cuda_alloc,
        .free     = cuda_free,
        .upload   = cuda_upload,
        .download = cuda_download,
        .sync     = cuda_sync,
    },
    .compute = {
        .gemv         = cuda_gemv,
        .gemm         = cuda_gemm,
        .rmsnorm      = cuda_rmsnorm,
        .layernorm    = cuda_layernorm,
        .rope         = cuda_rope,
        .softmax      = cuda_softmax,
        .silu         = cuda_silu,
        .gelu         = cuda_gelu,
        .mul          = cuda_mul,
        .add          = cuda_add,
        .scale        = cuda_scale,
        .dot          = cuda_dot,
        .dequantize   = cuda_dequant,
        .attention    = cuda_attention,
        .kv_update    = cuda_kv_update,
        .embed_lookup = cuda_embed,
        .softcap      = cuda_softcap_fn,
    },
};

#endif /* ENABLE_CUDA */
