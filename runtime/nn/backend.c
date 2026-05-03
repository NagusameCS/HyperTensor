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
 * TensorOS Backend Registry + CPU Reference Implementation
 *
 * The CPU backend provides gold-standard reference implementations
 * of all tensor operations. Other backends must match CPU output
 * within specified numerical tolerances.
 */

#include "runtime/nn/backend.h"
#include <string.h>
#include <math.h>

#ifdef GEODESSICAL_HOSTED
#include "hal.h"
#else
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#endif

/* 
 * Backend Registry
 *  */

static const backend_t *registered_backends[BACKEND_COUNT] = {0};
static backend_id_t active_backend_id = BACKEND_CPU;

void backend_register(const backend_t *be) {
    if (be && be->id < BACKEND_COUNT)
        registered_backends[be->id] = be;
}

const backend_t *backend_get(void) {
    return registered_backends[active_backend_id];
}

int backend_set(backend_id_t id) {
    if (id >= BACKEND_COUNT) return -1;
    if (!registered_backends[id]) return -2;
    active_backend_id = id;
    return 0;
}

const backend_t *backend_get_by_id(backend_id_t id) {
    if (id >= BACKEND_COUNT) return (void *)0;
    return registered_backends[id];
}

void backend_init_all(void) {
    /* CPU is always available */
    backend_register(&backend_cpu);
    if (backend_cpu.init) backend_cpu.init();

#ifdef ENABLE_CUDA
    backend_register(&backend_cuda);
    if (backend_cuda.init) backend_cuda.init();
#endif

#ifdef ENABLE_MLIR
    backend_register(&backend_mlir);
    if (backend_mlir.init) backend_mlir.init();
#endif
}

/* 
 * CPU Backend Implementation
 *  */

/*  FP16 helper  */
static float cpu_fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else { exp = 1; while (!(mant&0x400)){mant<<=1;exp--;} mant&=0x3FF;
               bits = sign | ((exp+127-15)<<23) | (mant<<13); }
    } else if (exp == 31) bits = sign | 0x7F800000 | (mant<<13);
    else bits = sign | ((exp+127-15)<<23) | (mant<<13);
    float f; memcpy(&f, &bits, 4); return f;
}

/*  Memory ops  */
static void *cpu_alloc(uint64_t size) { return tensor_alloc(size); }
static void  cpu_free(void *ptr)      { tensor_free(ptr); }
static int   cpu_upload(void *d, const void *s, uint64_t sz) {
    kmemcpy(d, s, sz); return 0;
}
static int   cpu_download(void *d, const void *s, uint64_t sz) {
    kmemcpy(d, s, sz); return 0;
}
static void  cpu_sync(void) { /* CPU is synchronous */ }

/*  Dequantize  */
static void cpu_dequantize(float *out, const void *data, int n, ggml_type_t type) {
    if (type == GGML_TYPE_F32) {
        kmemcpy(out, data, n * sizeof(float));
        return;
    }
    if (type == GGML_TYPE_F16) {
        const uint16_t *h = (const uint16_t *)data;
        for (int i = 0; i < n; i++) out[i] = cpu_fp16_to_f32(h[i]);
        return;
    }
    if (type == GGML_TYPE_Q4_0) {
        typedef struct { uint16_t d; uint8_t qs[16]; } q4_0_t;
        int nb = n / 32;
        const q4_0_t *blocks = (const q4_0_t *)data;
        for (int b = 0; b < nb; b++) {
            float d = cpu_fp16_to_f32(blocks[b].d);
            for (int j = 0; j < 16; j++) {
                int lo = (blocks[b].qs[j] & 0x0F) - 8;
                int hi = (blocks[b].qs[j] >> 4) - 8;
                out[b*32 + j]      = d * (float)lo;
                out[b*32 + j + 16] = d * (float)hi;
            }
        }
        return;
    }
    if (type == GGML_TYPE_Q8_0) {
        typedef struct { uint16_t d; int8_t qs[32]; } q8_0_t;
        int nb = n / 32;
        const q8_0_t *blocks = (const q8_0_t *)data;
        for (int b = 0; b < nb; b++) {
            float d = cpu_fp16_to_f32(blocks[b].d);
            for (int j = 0; j < 32; j++)
                out[b*32 + j] = d * (float)blocks[b].qs[j];
        }
        return;
    }
    if (type == GGML_TYPE_Q2_K) {
        /* Q2_K: 256 elements/super-block, 84 bytes.
         * Layout: scales[16](nibble-packed) + qs[64] + d(fp16) + dmin(fp16). */
        const uint8_t *bptr = (const uint8_t *)data;
        int nb = n / 256;
        for (int b = 0; b < nb; b++) {
            const uint8_t *scales = bptr;
            const uint8_t *qs     = bptr + 16;
            uint16_t dh, dmh;
            kmemcpy(&dh,  bptr + 80, 2);
            kmemcpy(&dmh, bptr + 82, 2);
            float d    = cpu_fp16_to_f32(dh);
            float dmin = cpu_fp16_to_f32(dmh);
            bptr += 84;
            float *y = out + b * 256;
            int is = 0;
            const uint8_t *qb = qs;
            for (int half = 0; half < 2; half++) {
                int shift = 0;
                for (int j = 0; j < 4; j++) {
                    uint8_t sc = scales[is++];
                    float dl = d * (float)(sc & 0xF);
                    float ml = dmin * (float)(sc >> 4);
                    for (int l = 0; l < 16; l++)
                        *y++ = dl * (float)((qb[l] >> shift) & 3) - ml;
                    sc = scales[is++];
                    dl = d * (float)(sc & 0xF);
                    ml = dmin * (float)(sc >> 4);
                    for (int l = 0; l < 16; l++)
                        *y++ = dl * (float)((qb[l + 16] >> shift) & 3) - ml;
                    shift += 2;
                }
                qb += 32;
            }
        }
        return;
    }
    if (type == GGML_TYPE_IQ2_XS) {
        /* IQ2_XS: 256 elements/super-block, 74 bytes.
         * Layout: d(fp16) + qs[32](uint16_t) + scales[8](uint8_t).
         * Scales packed 2 nibbles/byte: group ib32 uses (sc[ib32/2]>>(4*(ib32&1)))&0xf.
         * Each qs[k]: bits[8:0]=grid_idx, bits[15:9]=7 sign bits (parity-extended to 8).
         * Grid byte values for 2-bit index v: {8, 25, 43, 60}. */
        static const uint8_t iq2xs_vals[4] = {1, 3, 5, 7};
        static const uint16_t kgrid[512] = {
               0,     2,     5,     8,    10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
              73,    80,    82,    85,    88,    97,   100,   128,   130,   133,   136,   145,   148,   153,   160,   257,
             260,   262,   265,   272,   274,   277,   280,   282,   289,   292,   320,   322,   325,   328,   337,   340,
             352,   360,   385,   388,   400,   512,   514,   517,   520,   529,   532,   544,   577,   580,   592,   597,
             640,   650,  1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1088,  1090,  1093,  1096,
            1105,  1108,  1110,  1120,  1153,  1156,  1168,  1280,  1282,  1285,  1288,  1297,  1300,  1312,  1345,  1348,
            1360,  1377,  1408,  1537,  1540,  1552,  1574,  1600,  1602,  1668,  2048,  2050,  2053,  2056,  2058,  2065,
            2068,  2080,  2085,  2113,  2116,  2128,  2136,  2176,  2208,  2218,  2305,  2308,  2320,  2368,  2433,  2441,
            2560,  2592,  2600,  2710,  2720,  4097,  4100,  4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4160,
            4162,  4165,  4168,  4177,  4180,  4192,  4202,  4225,  4228,  4240,  4352,  4354,  4357,  4360,  4369,  4372,
            4384,  4417,  4420,  4432,  4480,  4500,  4502,  4609,  4612,  4614,  4624,  4672,  4704,  5120,  5122,  5125,
            5128,  5137,  5140,  5152,  5185,  5188,  5193,  5200,  5220,  5248,  5377,  5380,  5392,  5440,  5632,  5652,
            5705,  6145,  6148,  6160,  6162,  6208,  6228,  6278,  6400,  6405,  6502,  6737,  6825,  8192,  8194,  8197,
            8200,  8202,  8209,  8212,  8224,  8257,  8260,  8272,  8320,  8352,  8449,  8452,  8464,  8512,  8520,  8549,
            8704,  8738,  8832,  8872,  9217,  9220,  9232,  9257,  9280,  9472,  9537,  9554,  9625,  9729,  9754,  9894,
           10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882, 10888, 10890, 16385, 16388,
           16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16480,
           16485, 16513, 16516, 16528, 16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773,
           16802, 16897, 16900, 16912, 16914, 16937, 16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473,
           17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818, 17920, 17930, 17988, 18000, 18433, 18436,
           18448, 18496, 18501, 18516, 18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488, 20497,
           20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752, 20757, 20800, 20802, 20992, 21060, 21162,
           21505, 21508, 21520, 21537, 21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528,
           22548, 22593, 22608, 22681, 22810, 22848, 22850, 23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745,
           24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625, 26730, 26917, 27142, 27220, 27234,
           32768, 32770, 32773, 32776, 32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025,
           33028, 33030, 33040, 33088, 33105, 33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793, 33796, 33808, 33810,
           33813, 33856, 33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906, 34944, 34946, 34984,
           35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868, 36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462,
           37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445, 39520, 40960,
           40962, 40968, 40970, 40992, 41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600, 41632, 42048,
           42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562, 43665, 43690,
        };
        const uint8_t *bptr = (const uint8_t *)data;
        int nb = n / 256;
        for (int b = 0; b < nb; b++) {
            uint16_t dh;
            kmemcpy(&dh, bptr, 2);
            float d = cpu_fp16_to_f32(dh);
            const uint16_t *qs = (const uint16_t *)(bptr + 2);
            const uint8_t  *sc = bptr + 66;
            bptr += 74;
            float *y = out + b * 256;
            for (int ib32 = 0; ib32 < 8; ib32++) {
                float dl0 = d * (0.5f + (sc[ib32] & 0xf)) * 0.25f;
                float dl1 = d * (0.5f + (sc[ib32] >> 4))  * 0.25f;
                for (int l = 0; l < 4; l++) {
                    float dl = (l < 2) ? dl0 : dl1;
                    uint16_t qv = qs[ib32 * 4 + l];
                    uint16_t gi = qv & 511;
                    uint8_t  s7 = (uint8_t)(qv >> 9);
                    /* extend sign to 8 bits using even parity */
                    int pc = 0;
                    uint8_t tmp = s7;
                    while (tmp) { pc ^= 1; tmp &= tmp - 1; }
                    uint8_t s8 = s7 | (uint8_t)(pc << 7);
                    uint16_t gv16 = kgrid[gi];
                    for (int k = 0; k < 8; k++) {
                        float gk = (float)iq2xs_vals[(gv16 >> (2 * k)) & 3];
                        if (s8 & (1u << k)) gk = -gk;
                        *y++ = dl * gk;
                    }
                }
            }
        }
        return;
    }
    /* Unsupported type: zero fill */
    kmemset(out, 0, n * sizeof(float));
}

/*  Dot product  */
static float cpu_dot(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

/*  GEMV (scalar reference)  */
static void cpu_gemv(float *out, const void *weight, const float *x,
                     int out_dim, int in_dim, ggml_type_t weight_type) {
    if (weight_type == GGML_TYPE_F32) {
        const float *w = (const float *)weight;
        for (int i = 0; i < out_dim; i++) {
            float s = 0.0f;
            for (int j = 0; j < in_dim; j++)
                s += w[i * in_dim + j] * x[j];
            out[i] = s;
        }
        return;
    }
    /* Quantized: dequant row then dot */
    float *row_buf = (float *)cpu_alloc(in_dim * sizeof(float));
    if (!row_buf) return;
    uint64_t row_bytes = 0;
    if (weight_type == GGML_TYPE_Q4_0) row_bytes = (in_dim / 32) * 18;
    else if (weight_type == GGML_TYPE_Q8_0) row_bytes = (in_dim / 32) * 34;
    else if (weight_type == GGML_TYPE_F16) row_bytes = in_dim * 2;
    else row_bytes = in_dim * 4;

    const uint8_t *base = (const uint8_t *)weight;
    for (int i = 0; i < out_dim; i++) {
        cpu_dequantize(row_buf, base + (uint64_t)i * row_bytes, in_dim, weight_type);
        float s = 0.0f;
        for (int j = 0; j < in_dim; j++) s += row_buf[j] * x[j];
        out[i] = s;
    }
    cpu_free(row_buf);
}

/*  GEMM  */
static void cpu_gemm(float *C, const float *A, const float *B,
                     int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] += s;
        }
}

/*  RMSNorm  */
static void cpu_rmsnorm(float *out, const float *x, const float *w,
                        int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / (float)dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * ss * w[i];
}

/*  LayerNorm  */
static void cpu_layernorm(float *out, const float *x, const float *w,
                          const float *bias, int dim, float eps) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= (float)dim;
    float var = 0.0f;
    for (int i = 0; i < dim; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)dim;
    float inv = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv * w[i];
        if (bias) out[i] += bias[i];
    }
}

/*  RoPE  */
static void cpu_rope(float *q, float *k, int head_dim, int n_heads,
                     int n_kv_heads, int pos, float base,
                     const float *freq_factors) {
    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(base, (float)i / (float)head_dim);
            if (freq_factors) freq *= freq_factors[i / 2];
            float angle = (float)pos * freq;
            float cs = cosf(angle), sn = sinf(angle);
            float q0 = qh[i], q1 = qh[i + 1];
            qh[i]     = q0 * cs - q1 * sn;
            qh[i + 1] = q0 * sn + q1 * cs;
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * head_dim;
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(base, (float)i / (float)head_dim);
            if (freq_factors) freq *= freq_factors[i / 2];
            float angle = (float)pos * freq;
            float cs = cosf(angle), sn = sinf(angle);
            float k0 = kh[i], k1 = kh[i + 1];
            kh[i]     = k0 * cs - k1 * sn;
            kh[i + 1] = k0 * sn + k1 * cs;
        }
    }
}

/*  Softmax  */
static void cpu_softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/*  SiLU  */
static void cpu_silu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

/*  GELU  */
static void cpu_gelu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

/*  Element-wise ops  */
static void cpu_mul(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}
static void cpu_add(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}
static void cpu_scale(float *out, const float *x, float s, int n) {
    for (int i = 0; i < n; i++) out[i] = x[i] * s;
}

/*  Attention  */
static void cpu_attention(float *out, const float *Q,
                          const float *K_cache, const float *V_cache,
                          int n_heads, int n_kv_heads, int head_dim,
                          int seq_len, int max_seq, float scale, float softcap) {
    int kv_group = n_heads / n_kv_heads;
    /* Allocate scratch for attention scores */
    float *scores = (float *)cpu_alloc(seq_len * sizeof(float));
    if (!scores) return;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / kv_group;
        const float *qh = Q + h * head_dim;

        /* Compute attention scores: Q·K^T / sqrt(d) */
        for (int t = 0; t < seq_len; t++) {
            const float *kh = K_cache + (kv_h * max_seq + t) * head_dim;
            float s = 0.0f;
            for (int d = 0; d < head_dim; d++) s += qh[d] * kh[d];
            s *= scale;
            if (softcap > 0.0f) s = softcap * tanhf(s / softcap);
            scores[t] = s;
        }

        /* Softmax */
        cpu_softmax(scores, seq_len);

        /* Weighted sum of V */
        float *oh = out + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
            float v = 0.0f;
            for (int t = 0; t < seq_len; t++)
                v += scores[t] * V_cache[(kv_h * max_seq + t) * head_dim + d];
            oh[d] = v;
        }
    }
    cpu_free(scores);
}

/*  KV cache update  */
static void cpu_kv_update(float *K_cache, float *V_cache,
                          const float *K_new, const float *V_new,
                          int n_kv_heads, int head_dim, int pos,
                          int max_seq, int layer) {
    (void)layer;
    for (int h = 0; h < n_kv_heads; h++) {
        kmemcpy(K_cache + (h * max_seq + pos) * head_dim,
                K_new + h * head_dim, head_dim * sizeof(float));
        kmemcpy(V_cache + (h * max_seq + pos) * head_dim,
                V_new + h * head_dim, head_dim * sizeof(float));
    }
}

/*  Embedding lookup  */
static void cpu_embed_lookup(float *out, const void *embd_table,
                             int token_id, int dim, ggml_type_t type) {
    if (type == GGML_TYPE_F32) {
        const float *t = (const float *)embd_table;
        kmemcpy(out, t + (uint64_t)token_id * dim, dim * sizeof(float));
    } else {
        /* Quantized embedding: dequant the row */
        uint64_t row_bytes = 0;
        if (type == GGML_TYPE_Q4_0) row_bytes = (dim / 32) * 18;
        else if (type == GGML_TYPE_Q8_0) row_bytes = (dim / 32) * 34;
        else if (type == GGML_TYPE_F16) row_bytes = dim * 2;
        else row_bytes = dim * 4;
        const uint8_t *base = (const uint8_t *)embd_table;
        cpu_dequantize(out, base + (uint64_t)token_id * row_bytes, dim, type);
    }
}

/*  Softcap  */
static void cpu_softcap(float *x, int n, float cap) {
    for (int i = 0; i < n; i++)
        x[i] = cap * tanhf(x[i] / cap);
}

/*  Init/Shutdown  */
static int  cpu_init(void) { return 0; }
static void cpu_shutdown(void) {}
static int  cpu_device_count(void) { return 1; }
static uint64_t cpu_free_memory(int dev) {
    (void)dev;
    return tensor_mm_free_bytes();
}

/* 
 * CPU Backend Definition
 *  */

const backend_t backend_cpu = {
    .id   = BACKEND_CPU,
    .name = "cpu",
    .init = cpu_init,
    .shutdown = cpu_shutdown,
    .get_device_count = cpu_device_count,
    .get_free_memory  = cpu_free_memory,
    .mem = {
        .alloc    = cpu_alloc,
        .free     = cpu_free,
        .upload   = cpu_upload,
        .download = cpu_download,
        .sync     = cpu_sync,
    },
    .compute = {
        .gemv         = cpu_gemv,
        .gemm         = cpu_gemm,
        .rmsnorm      = cpu_rmsnorm,
        .layernorm    = cpu_layernorm,
        .rope         = cpu_rope,
        .softmax      = cpu_softmax,
        .silu         = cpu_silu,
        .gelu         = cpu_gelu,
        .mul          = cpu_mul,
        .add          = cpu_add,
        .scale        = cpu_scale,
        .dot          = cpu_dot,
        .dequantize   = cpu_dequantize,
        .attention    = cpu_attention,
        .kv_update    = cpu_kv_update,
        .embed_lookup = cpu_embed_lookup,
        .softcap      = cpu_softcap,
    },
};
