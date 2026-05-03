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
 * thermal_rank.c - Thermal-adaptive rank and tokens-per-joule objective.
 *
 * See thermal_rank.h for design documentation.
 */

#include "runtime/nn/thermal_rank.h"
#include "runtime/nn/geo_research.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef GEODESSICAL_HOSTED
#  include "host/hal.h"
#  define TR_PRINTF   kprintf
#  define TR_TIMER_US hal_timer_us
#else
#  include "kernel/core/kernel.h"
#  define TR_PRINTF   kprintf
#  define TR_TIMER_US() ((uint64_t)0)
#endif

/* =========================================================================
 * Platform NVML dynamic loader
 * ========================================================================= */

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>

typedef unsigned int nvml_ret_t;
#define NVML_SUCCESS          0u
#define NVML_TEMPERATURE_GPU  0u

typedef void *nvml_dev_t;

typedef nvml_ret_t (__cdecl *pfn_nvmlInit)(void);
typedef nvml_ret_t (__cdecl *pfn_nvmlShutdown)(void);
typedef nvml_ret_t (__cdecl *pfn_nvmlDeviceGetHandleByIndex)(unsigned int idx,
                                                              nvml_dev_t *dev);
typedef nvml_ret_t (__cdecl *pfn_nvmlDeviceGetTemperature)(nvml_dev_t dev,
                                                            unsigned int sensor,
                                                            unsigned int *temp_C);
typedef nvml_ret_t (__cdecl *pfn_nvmlDeviceGetPowerUsage)(nvml_dev_t dev,
                                                           unsigned int *mw);

typedef struct {
    HMODULE                          lib;
    nvml_dev_t                       device;
    pfn_nvmlInit                     fnInit;
    pfn_nvmlShutdown                 fnShutdown;
    pfn_nvmlDeviceGetHandleByIndex   fnGetHandle;
    pfn_nvmlDeviceGetTemperature     fnGetTemp;
    pfn_nvmlDeviceGetPowerUsage      fnGetPower;
} nvml_priv_t;

static int nvml_load(nvml_priv_t *p)
{
    memset(p, 0, sizeof(*p));
    const char *paths[] = {
        "nvml.dll",
        "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll",
        "C:\\Windows\\System32\\nvml.dll",
        NULL
    };
    for (int i = 0; paths[i]; i++) {
        p->lib = LoadLibraryA(paths[i]);
        if (!p->lib) continue;

        p->fnInit = (pfn_nvmlInit)GetProcAddress(p->lib, "nvmlInit_v2");
        if (!p->fnInit)
            p->fnInit = (pfn_nvmlInit)GetProcAddress(p->lib, "nvmlInit");

        p->fnShutdown = (pfn_nvmlShutdown)GetProcAddress(p->lib, "nvmlShutdown");

        p->fnGetHandle = (pfn_nvmlDeviceGetHandleByIndex)
            GetProcAddress(p->lib, "nvmlDeviceGetHandleByIndex_v2");
        if (!p->fnGetHandle)
            p->fnGetHandle = (pfn_nvmlDeviceGetHandleByIndex)
                GetProcAddress(p->lib, "nvmlDeviceGetHandleByIndex");

        p->fnGetTemp  = (pfn_nvmlDeviceGetTemperature)
            GetProcAddress(p->lib, "nvmlDeviceGetTemperature");
        p->fnGetPower = (pfn_nvmlDeviceGetPowerUsage)
            GetProcAddress(p->lib, "nvmlDeviceGetPowerUsage");

        if (p->fnInit && p->fnGetHandle && p->fnGetTemp && p->fnGetPower) {
            if (p->fnInit() == NVML_SUCCESS) {
                if (p->fnGetHandle(0, &p->device) == NVML_SUCCESS)
                    return 0; /* success */
            }
        }
        FreeLibrary(p->lib);
        /* Clear all pointers so nvml_unload() is a safe no-op if we fail all paths */
        p->lib         = NULL;
        p->fnInit      = NULL;
        p->fnShutdown  = NULL;
        p->fnGetHandle = NULL;
        p->fnGetTemp   = NULL;
        p->fnGetPower  = NULL;
        p->device      = NULL;
    }
    return -1;
}

static void nvml_unload(nvml_priv_t *p)
{
    if (p->fnShutdown) p->fnShutdown();
    if (p->lib) { FreeLibrary(p->lib); p->lib = NULL; }
}

static int nvml_read(nvml_priv_t *p, float *out_temp_C, float *out_power_W)
{
    unsigned int temp = 0, mw = 0;
    nvml_ret_t r1 = p->fnGetTemp(p->device, NVML_TEMPERATURE_GPU, &temp);
    nvml_ret_t r2 = p->fnGetPower(p->device, &mw);
    if (r1 != NVML_SUCCESS || r2 != NVML_SUCCESS) return -1;
    *out_temp_C  = (float)temp;
    *out_power_W = (float)mw * 0.001f;
    return 0;
}

#else /* !_WIN32 */

/* Non-Windows stub — NVML not supported on this platform */
typedef struct { int dummy; } nvml_priv_t;
static int  nvml_load(nvml_priv_t *p)                                   { (void)p; return -1; }
static void nvml_unload(nvml_priv_t *p)                                 { (void)p; }
static int  nvml_read(nvml_priv_t *p, float *t, float *w)               { (void)p; (void)t; (void)w; return -1; }

#endif /* _WIN32 */

/* =========================================================================
 * Feature 3: Thermal-adaptive rank
 * ========================================================================= */

int thermal_init(thermal_ctx_t *ctx,
                 float temp_low_C, float temp_high_C,
                 float power_budget_W,
                 int rank_min, int rank_max)
{
    if (!ctx) return -1;
    memset(ctx, 0, sizeof(*ctx));

    ctx->temp_low_C       = (temp_low_C  >  0.0f) ? temp_low_C  : 65.0f;
    ctx->temp_high_C      = (temp_high_C >  0.0f) ? temp_high_C : 85.0f;
    ctx->power_budget_W   = power_budget_W;
    ctx->rank_min         = (rank_min > 0)         ? rank_min    : 8;
    ctx->rank_max         = (rank_max > rank_min)  ? rank_max    : 256;
    ctx->poll_interval_ms = 500;
    ctx->current_temp_C   = ctx->temp_low_C; /* optimistic default */
    ctx->current_power_W  = 0.0f;

    /* Allocate private NVML state */
    nvml_priv_t *priv = (nvml_priv_t *)malloc(sizeof(nvml_priv_t));
    if (!priv) return -1;
    ctx->_nvml_lib    = priv;
    ctx->_nvml_device = NULL;

    if (nvml_load(priv) == 0) {
        ctx->nvml_ok = 1;
        TR_PRINTF("[THERMAL] NVML loaded — GPU 0 handle acquired\n");
    } else {
        ctx->nvml_ok = 0;
        TR_PRINTF("[THERMAL] NVML not available — thermal rank adaptation disabled\n");
    }
    return 0;
}

void thermal_destroy(thermal_ctx_t *ctx)
{
    if (!ctx || !ctx->_nvml_lib) return;
    nvml_unload((nvml_priv_t *)ctx->_nvml_lib);
    free(ctx->_nvml_lib);
    ctx->_nvml_lib    = NULL;
    ctx->_nvml_device = NULL;
    ctx->nvml_ok      = 0;
}

int thermal_poll(thermal_ctx_t *ctx)
{
    if (!ctx || !ctx->nvml_ok) return -1;

    uint64_t now = TR_TIMER_US();
    uint64_t elapsed_ms = (now - ctx->last_poll_us) / 1000ULL;
    if (ctx->last_poll_us != 0 && elapsed_ms < (uint64_t)ctx->poll_interval_ms)
        return 1; /* within cooldown */

    float t = 0.0f, p = 0.0f;
    if (nvml_read((nvml_priv_t *)ctx->_nvml_lib, &t, &p) != 0)
        return -1;
    ctx->current_temp_C  = t;
    ctx->current_power_W = p;
    ctx->last_poll_us    = now;
    return 0;
}

int thermal_get_rank(thermal_ctx_t *ctx, int base_rank)
{
    if (!ctx || !ctx->nvml_ok) return base_rank;
    thermal_poll(ctx); /* best-effort; ignore cooldown result */

    float t    = ctx->current_temp_C;
    float lo   = ctx->temp_low_C;
    float hi   = ctx->temp_high_C;
    float span = hi - lo;

    int scaled_rank;
    if (span < 0.1f || t <= lo) {
        scaled_rank = ctx->rank_max;
    } else if (t >= hi) {
        scaled_rank = ctx->rank_min;
    } else {
        /* Linear interpolation from rank_max → rank_min */
        float frac  = (t - lo) / span;
        float fval  = (float)ctx->rank_max
                    - frac * (float)(ctx->rank_max - ctx->rank_min);
        scaled_rank = (int)(fval + 0.5f);
    }

    /* Also honour power budget if set */
    if (ctx->power_budget_W > 0.0f && ctx->current_power_W > ctx->power_budget_W) {
        float pscale  = ctx->power_budget_W / ctx->current_power_W;
        int   prank   = (int)((float)scaled_rank * pscale + 0.5f);
        if (prank < scaled_rank) scaled_rank = prank;
    }

    if (scaled_rank < ctx->rank_min) scaled_rank = ctx->rank_min;
    if (scaled_rank > ctx->rank_max) scaled_rank = ctx->rank_max;
    return scaled_rank;
}

void thermal_print(const thermal_ctx_t *ctx)
{
    if (!ctx) return;
    TR_PRINTF("[THERMAL] nvml_ok=%d  temp=%.1f°C  power=%.1fW\n",
              ctx->nvml_ok,
              (double)ctx->current_temp_C,
              (double)ctx->current_power_W);
    TR_PRINTF("[THERMAL] thresholds: [%.0f, %.0f]°C → rank [%d, %d]\n",
              (double)ctx->temp_low_C, (double)ctx->temp_high_C,
              ctx->rank_min, ctx->rank_max);
    if (ctx->power_budget_W > 0.0f)
        TR_PRINTF("[THERMAL] power_budget=%.1fW\n", (double)ctx->power_budget_W);
}

/* =========================================================================
 * Feature 4: Tokens-per-joule objective
 * ========================================================================= */

int tpj_init(tpj_ctx_t *ctx, thermal_ctx_t *thermal, float lambda)
{
    if (!ctx) return -1;
    memset(ctx, 0, sizeof(*ctx));
    ctx->thermal     = thermal;
    ctx->lambda      = (lambda > 0.0f) ? lambda : 0.005f;
    ctx->rank_coeff  = 0.0f; /* unknown until first measurement */
    return 0;
}

float tpj_record(tpj_ctx_t *ctx, float tokens_per_second)
{
    if (!ctx || tokens_per_second <= 0.0f) return 0.0f;

    float power_W = 0.0f;
    if (ctx->thermal && ctx->thermal->nvml_ok) {
        thermal_poll(ctx->thermal);
        power_W = ctx->thermal->current_power_W;
    }
    if (power_W <= 0.0f) return 0.0f;

    float joules_per_token = power_W / tokens_per_second;

    /* Store in history */
    int idx = ctx->n_history < TPJ_HISTORY_LEN
              ? ctx->n_history : (TPJ_HISTORY_LEN - 1);
    if (ctx->n_history < TPJ_HISTORY_LEN) {
        ctx->joules_history[idx] = joules_per_token;
        ctx->n_history++;
    } else {
        /* Shift window */
        memmove(ctx->joules_history, ctx->joules_history + 1,
                (TPJ_HISTORY_LEN - 1) * sizeof(float));
        ctx->joules_history[TPJ_HISTORY_LEN - 1] = joules_per_token;
    }

    ctx->cumulative_joules += joules_per_token;
    ctx->cumulative_tokens += 1;

    /*
     * Update rank_coeff estimate: we don't know the exact joules-per-rank
     * without an A/B measurement, but we assume joules is roughly linear in
     * rank (higher rank = more FLOPs).  As a starting calibration we scale
     * by the ratio power_W / mean_rank across the history.  After the first
     * few observations the value converges quickly enough for gradient use.
     */
    if (ctx->n_history >= 4) {
        double sum = 0.0;
        int n = ctx->n_history < TPJ_HISTORY_LEN ? ctx->n_history : TPJ_HISTORY_LEN;
        for (int i = 0; i < n; i++) sum += ctx->joules_history[i];
        /* rank_coeff ≈ (mean joules/tok) / DIFFPLAN_RANK_LEVELS[mid] */
        float mean_j = (float)(sum / n);
        float mid_rank = (float)DIFFPLAN_RANK_LEVELS[DIFFPLAN_N_LEVELS / 2];
        ctx->rank_coeff = mean_j / (mid_rank > 0.0f ? mid_rank : 1.0f);
    }
    return joules_per_token;
}

void tpj_gradient(const tpj_ctx_t *ctx,
                  float grads[][DIFFPLAN_N_LEVELS],
                  int n_layers,
                  const diffplan_t *dp)
{
    if (!ctx || !grads || !dp || n_layers < 1) return;
    if (ctx->rank_coeff <= 0.0f || ctx->lambda <= 0.0f) return;

    float lam_k = ctx->lambda * ctx->rank_coeff;
    for (int l = 0; l < n_layers && l < DIFFPLAN_MAX_LAYERS; l++) {
        float rs = dp->rank_soft[l];
        for (int r = 0; r < DIFFPLAN_N_LEVELS; r++) {
            /*
             * Policy-gradient energy term:
             *   dL/dtheta[l][r] += lambda * rank_coeff * p[l][r] * (R_r - rank_soft[l])
             *
             * This nudges the softmax distribution towards lower rank when
             * rank_coeff (joules/rank) is significant relative to lambda.
             */
            float advantage = (float)DIFFPLAN_RANK_LEVELS[r] - rs;
            grads[l][r] += lam_k * dp->p[l][r] * advantage;
        }
    }
}

void tpj_bootstrap(tpj_ctx_t *ctx, float tps_estimate)
{
    if (!ctx || tps_estimate <= 0.0f) return;
    if (!ctx->thermal || !ctx->thermal->nvml_ok) return;

    thermal_poll(ctx->thermal); /* best-effort read */
    float power_W = ctx->thermal->current_power_W;
    if (power_W <= 0.0f) return;

    /* rank_coeff ≈ joules_per_token / mid_rank
     * joules_per_token = power_W / tps_estimate                          */
    float mid_rank = (float)DIFFPLAN_RANK_LEVELS[DIFFPLAN_N_LEVELS / 2];
    ctx->rank_coeff = (power_W / tps_estimate) / (mid_rank > 0.0f ? mid_rank : 1.0f);

    TR_PRINTF("[TPJ] Bootstrap: power=%.1fW  tps_est=%.1f → rank_coeff=%.6f\n",
              (double)power_W, (double)tps_estimate, (double)ctx->rank_coeff);
}

void tpj_print(const tpj_ctx_t *ctx)
{
    if (!ctx) return;
    TR_PRINTF("[TPJ] lambda=%.4f  rank_coeff=%.6f  n_history=%d\n",
              (double)ctx->lambda, (double)ctx->rank_coeff, ctx->n_history);
    if (ctx->cumulative_tokens > 0) {
        TR_PRINTF("[TPJ] cumulative: %.4f J  over %d tokens  (%.5f J/tok)\n",
                  ctx->cumulative_joules,
                  ctx->cumulative_tokens,
                  ctx->cumulative_joules / ctx->cumulative_tokens);
        TR_PRINTF("[TPJ] To update: call tpj_record(ctx, tokens_per_second) after each eval.\n");
    }
}
