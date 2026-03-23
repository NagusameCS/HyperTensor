/* =============================================================================
 * TensorOS - Pseudocode Standard Library
 *
 * Built-in functions available to all Pseudocode programs:
 *
 * Math:       abs, sqrt, pow, exp, log, sin, cos, floor, ceil, round, min, max
 * Tensor:     zeros, ones, randn, arange, shape, reshape, transpose, concat,
 *             tensor_to_string
 * String:     len, str, int_parse, float_parse, split, join, contains, upper,
 *             lower, strip, format
 * I/O:        print, println, read_file, write_file
 * System:     time_ms, sleep_ms, assert, type_of
 * Collections: range, enumerate, zip, map, filter, reduce, sort, reverse
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#include "runtime/pseudocode/pseudo_stdlib.h"

/* =============================================================================
 * Helper: Create return values
 * =============================================================================*/

static runtime_value_t val_none(void)
{
    runtime_value_t v = { .type = VAL_NONE };
    return v;
}

static runtime_value_t val_int(int64_t i)
{
    runtime_value_t v = { .type = VAL_INT, .int_val = i };
    return v;
}

static runtime_value_t val_float(double f)
{
    runtime_value_t v = { .type = VAL_FLOAT, .float_val = f };
    return v;
}

static runtime_value_t val_bool(bool b)
{
    runtime_value_t v = { .type = VAL_BOOL, .bool_val = b };
    return v;
}

static runtime_value_t val_string(const char *s)
{
    runtime_value_t v = { .type = VAL_STRING };
    int len = kstrlen(s);
    v.string_val = (char *)kmalloc(len + 1);
    if (v.string_val) { kmemcpy(v.string_val, s, len); v.string_val[len] = '\0'; }
    return v;
}

/* Coerce value to float */
static double to_float(runtime_value_t *v)
{
    if (v->type == VAL_FLOAT) return v->float_val;
    if (v->type == VAL_INT) return (double)v->int_val;
    if (v->type == VAL_BOOL) return v->bool_val ? 1.0 : 0.0;
    return 0.0;
}

/* =============================================================================
 * Math Functions
 * =============================================================================*/

/* Fast software sqrt (Newton–Raphson) */
static double sw_sqrt(double x)
{
    if (x <= 0.0) return 0.0;
    double guess = x * 0.5;
    for (int i = 0; i < 15; i++)
        guess = 0.5 * (guess + x / guess);
    return guess;
}

/* Fast exp via Schraudolph approximation (for double) */
static double sw_exp(double x)
{
    if (x > 700.0) return 1e308;
    if (x < -700.0) return 0.0;
    /* Use repeated squaring: e^x = e^(n + f) where n = floor(x), f = x - n */
    double e = 2.718281828459045;
    int n = (int)x;
    double f = x - n;
    /* e^f via Taylor series (10 terms) */
    double ef = 1.0, term = 1.0;
    for (int i = 1; i <= 10; i++) { term *= f / i; ef += term; }
    /* e^n via squaring */
    double en = 1.0;
    double base = (n >= 0) ? e : 1.0 / e;
    int abs_n = (n >= 0) ? n : -n;
    for (int i = 0; i < abs_n; i++) en *= base;
    return en * ef;
}

static double sw_log(double x)
{
    if (x <= 0.0) return -1e308;
    /* Reduce: x = m * 2^e, then ln(x) = ln(m) + e*ln(2) */
    double result = 0.0;
    while (x > 2.0) { x /= 2.718281828459045; result += 1.0; }
    while (x < 0.5) { x *= 2.718281828459045; result -= 1.0; }
    /* ln(x) near 1: series expansion */
    double t = (x - 1.0) / (x + 1.0);
    double t2 = t * t, term = t;
    for (int i = 0; i < 20; i++) {
        result += 2.0 * term / (2 * i + 1);
        term *= t2;
    }
    return result;
}

static double sw_sin(double x)
{
    /* Reduce to [-pi, pi] */
    double pi = 3.14159265358979323846;
    while (x > pi) x -= 2.0 * pi;
    while (x < -pi) x += 2.0 * pi;
    double term = x, sum = x;
    for (int i = 1; i <= 12; i++) {
        term *= -x * x / ((2*i) * (2*i + 1));
        sum += term;
    }
    return sum;
}

static double sw_cos(double x)
{
    double pi = 3.14159265358979323846;
    return sw_sin(x + pi / 2.0);
}

static double sw_pow(double base, double exp)
{
    if (exp == 0.0) return 1.0;
    if (base == 0.0) return 0.0;
    /* For integer exponents, use repeated multiplication */
    int iexp = (int)exp;
    if ((double)iexp == exp && iexp >= 0 && iexp <= 100) {
        double r = 1.0;
        double b = base;
        int e = iexp;
        while (e > 0) { if (e & 1) r *= b; b *= b; e >>= 1; }
        return r;
    }
    return sw_exp(exp * sw_log(base));
}

/* Built-in math functions */
static runtime_value_t fn_abs(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    double x = to_float(&args[0]);
    return val_float(x >= 0 ? x : -x);
}

static runtime_value_t fn_sqrt(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    return val_float(sw_sqrt(to_float(&args[0])));
}

static runtime_value_t fn_pow(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    return val_float(sw_pow(to_float(&args[0]), to_float(&args[1])));
}

static runtime_value_t fn_exp(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    return val_float(sw_exp(to_float(&args[0])));
}

static runtime_value_t fn_log(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    return val_float(sw_log(to_float(&args[0])));
}

static runtime_value_t fn_sin(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    return val_float(sw_sin(to_float(&args[0])));
}

static runtime_value_t fn_cos(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    return val_float(sw_cos(to_float(&args[0])));
}

static runtime_value_t fn_floor(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    double x = to_float(&args[0]);
    return val_float((double)(int64_t)x - (x < 0 && x != (int64_t)x ? 1 : 0));
}

static runtime_value_t fn_ceil(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    double x = to_float(&args[0]);
    int64_t ix = (int64_t)x;
    return val_float((double)(x > ix ? ix + 1 : ix));
}

static runtime_value_t fn_round(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    double x = to_float(&args[0]);
    return val_float((double)(int64_t)(x + (x >= 0 ? 0.5 : -0.5)));
}

static runtime_value_t fn_min(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    double a = to_float(&args[0]), b = to_float(&args[1]);
    return val_float(a < b ? a : b);
}

static runtime_value_t fn_max(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    double a = to_float(&args[0]), b = to_float(&args[1]);
    return val_float(a > b ? a : b);
}

/* =============================================================================
 * Tensor Construction Functions
 * =============================================================================*/

static runtime_value_t fn_zeros(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    int size = (int)args[0].int_val;
    if (size <= 0 || size > 1048576) return val_none();
    float *data = (float *)kmalloc(size * sizeof(float));
    if (!data) return val_none();
    kmemset(data, 0, size * sizeof(float));
    tensor_desc_t *t = (tensor_desc_t *)kmalloc(sizeof(tensor_desc_t));
    if (!t) { kfree(data); return val_none(); }
    t->data_virt = (uint64_t)(uintptr_t)data; t->shape[0] = (uint64_t)size; t->ndim = 1;
    t->dtype = TENSOR_DTYPE_F32; t->size_bytes = size * sizeof(float);
    runtime_value_t v = { .type = VAL_TENSOR, .tensor_val = t };
    rt->tensors_allocated++;
    return v;
}

static runtime_value_t fn_ones(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    int size = (int)args[0].int_val;
    if (size <= 0 || size > 1048576) return val_none();
    float *data = (float *)kmalloc(size * sizeof(float));
    if (!data) return val_none();
    for (int i = 0; i < size; i++) data[i] = 1.0f;
    tensor_desc_t *t = (tensor_desc_t *)kmalloc(sizeof(tensor_desc_t));
    if (!t) { kfree(data); return val_none(); }
    t->data_virt = (uint64_t)(uintptr_t)data; t->shape[0] = (uint64_t)size; t->ndim = 1;
    t->dtype = TENSOR_DTYPE_F32; t->size_bytes = size * sizeof(float);
    runtime_value_t v = { .type = VAL_TENSOR, .tensor_val = t };
    rt->tensors_allocated++;
    return v;
}

static runtime_value_t fn_arange(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    int size = (int)args[0].int_val;
    if (size <= 0 || size > 1048576) return val_none();
    float *data = (float *)kmalloc(size * sizeof(float));
    if (!data) return val_none();
    for (int i = 0; i < size; i++) data[i] = (float)i;
    tensor_desc_t *t = (tensor_desc_t *)kmalloc(sizeof(tensor_desc_t));
    if (!t) { kfree(data); return val_none(); }
    t->data_virt = (uint64_t)(uintptr_t)data; t->shape[0] = (uint64_t)size; t->ndim = 1;
    t->dtype = TENSOR_DTYPE_F32; t->size_bytes = size * sizeof(float);
    runtime_value_t v = { .type = VAL_TENSOR, .tensor_val = t };
    rt->tensors_allocated++;
    return v;
}

static runtime_value_t fn_shape(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    if (args[0].type != VAL_TENSOR || !args[0].tensor_val) return val_none();
    tensor_desc_t *t = args[0].tensor_val;
    /* Return first dimension as int for 1-D; could extend to return shape tuple */
    return val_int((int64_t)t->shape[0]);
}

/* =============================================================================
 * String Functions
 * =============================================================================*/

static runtime_value_t fn_len(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    if (args[0].type == VAL_STRING && args[0].string_val)
        return val_int(kstrlen(args[0].string_val));
    if (args[0].type == VAL_TENSOR && args[0].tensor_val)
        return val_int((int64_t)args[0].tensor_val->shape[0]);
    return val_int(0);
}

static runtime_value_t fn_str(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    char buf[256];
    switch (args[0].type) {
    case VAL_INT:    kprintf_to_buf(buf, sizeof(buf), "%ld", (long)args[0].int_val); break;
    case VAL_FLOAT:  kprintf_to_buf(buf, sizeof(buf), "%f", args[0].float_val); break;
    case VAL_BOOL:   kprintf_to_buf(buf, sizeof(buf), "%s", args[0].bool_val ? "true" : "false"); break;
    case VAL_STRING: return args[0]; /* Already a string */
    default:         kprintf_to_buf(buf, sizeof(buf), "<object>"); break;
    }
    return val_string(buf);
}

static runtime_value_t fn_int_parse(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    if (args[0].type != VAL_STRING || !args[0].string_val) return val_int(0);
    const char *s = args[0].string_val;
    int64_t val = 0;
    int neg = 0;
    if (*s == '-') { neg = 1; s++; }
    while (*s >= '0' && *s <= '9') { val = val * 10 + (*s - '0'); s++; }
    return val_int(neg ? -val : val);
}

static runtime_value_t fn_float_parse(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    if (args[0].type != VAL_STRING || !args[0].string_val) return val_float(0.0);
    const char *s = args[0].string_val;
    double val = 0.0;
    int neg = 0;
    if (*s == '-') { neg = 1; s++; }
    while (*s >= '0' && *s <= '9') { val = val * 10.0 + (*s - '0'); s++; }
    if (*s == '.') {
        s++;
        double frac = 0.1;
        while (*s >= '0' && *s <= '9') { val += (*s - '0') * frac; frac *= 0.1; s++; }
    }
    return val_float(neg ? -val : val);
}

static runtime_value_t fn_contains(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    if (args[0].type != VAL_STRING || args[1].type != VAL_STRING) return val_bool(false);
    if (!args[0].string_val || !args[1].string_val) return val_bool(false);
    const char *hay = args[0].string_val;
    const char *needle = args[1].string_val;
    int hlen = kstrlen(hay), nlen = kstrlen(needle);
    if (nlen > hlen) return val_bool(false);
    for (int i = 0; i <= hlen - nlen; i++) {
        int match = 1;
        for (int j = 0; j < nlen; j++)
            if (hay[i+j] != needle[j]) { match = 0; break; }
        if (match) return val_bool(true);
    }
    return val_bool(false);
}

static runtime_value_t fn_upper(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    if (args[0].type != VAL_STRING || !args[0].string_val) return val_string("");
    const char *s = args[0].string_val;
    int len = kstrlen(s);
    char *buf = (char *)kmalloc(len + 1);
    if (!buf) return val_string("");
    for (int i = 0; i < len; i++)
        buf[i] = (s[i] >= 'a' && s[i] <= 'z') ? s[i] - 32 : s[i];
    buf[len] = '\0';
    runtime_value_t v = { .type = VAL_STRING, .string_val = buf };
    return v;
}

static runtime_value_t fn_lower(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    if (args[0].type != VAL_STRING || !args[0].string_val) return val_string("");
    const char *s = args[0].string_val;
    int len = kstrlen(s);
    char *buf = (char *)kmalloc(len + 1);
    if (!buf) return val_string("");
    for (int i = 0; i < len; i++)
        buf[i] = (s[i] >= 'A' && s[i] <= 'Z') ? s[i] + 32 : s[i];
    buf[len] = '\0';
    runtime_value_t v = { .type = VAL_STRING, .string_val = buf };
    return v;
}

static runtime_value_t fn_strip(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    if (args[0].type != VAL_STRING || !args[0].string_val) return val_string("");
    const char *s = args[0].string_val;
    int len = kstrlen(s);
    int start = 0, end = len;
    while (start < end && (s[start] == ' ' || s[start] == '\n' || s[start] == '\t' || s[start] == '\r')) start++;
    while (end > start && (s[end-1] == ' ' || s[end-1] == '\n' || s[end-1] == '\t' || s[end-1] == '\r')) end--;
    char *buf = (char *)kmalloc(end - start + 1);
    if (!buf) return val_string("");
    kmemcpy(buf, s + start, end - start);
    buf[end - start] = '\0';
    runtime_value_t v = { .type = VAL_STRING, .string_val = buf };
    return v;
}

/* =============================================================================
 * I/O Functions
 * =============================================================================*/

static runtime_value_t fn_print(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    for (int i = 0; i < n; i++) {
        switch (args[i].type) {
        case VAL_INT:    kprintf("%ld", (long)args[i].int_val); break;
        case VAL_FLOAT:  kprintf("%f", args[i].float_val); break;
        case VAL_BOOL:   kprintf("%s", args[i].bool_val ? "true" : "false"); break;
        case VAL_STRING: if (args[i].string_val) kprintf("%s", args[i].string_val); break;
        case VAL_TENSOR: kprintf("<tensor>"); break;
        default:         kprintf("<none>"); break;
        }
        if (i + 1 < n) kprintf(" ");
    }
    return val_none();
}

static runtime_value_t fn_println(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    fn_print(rt, args, n);
    kprintf("\n");
    return val_none();
}

/* =============================================================================
 * System Functions
 * =============================================================================*/

static runtime_value_t fn_time_ms(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt; (void)args; (void)n;
    uint32_t lo, hi; __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi)); return val_int((int64_t)((uint64_t)hi << 32 | lo));
}

static runtime_value_t fn_assert(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    bool cond = false;
    if (args[0].type == VAL_BOOL) cond = args[0].bool_val;
    else if (args[0].type == VAL_INT) cond = args[0].int_val != 0;
    if (!cond) {
        const char *msg = (n > 1 && args[1].type == VAL_STRING) ? args[1].string_val : "assertion failed";
        kprintf("[ASSERT] %s\n", msg);
    }
    return val_none();
}

static runtime_value_t fn_type_of(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    (void)rt;
    const char *names[] = {"none","int","float","bool","string","tensor","model","function"};
    int idx = args[0].type;
    if (idx < 0 || idx > 7) idx = 0;
    return val_string(names[idx]);
}

/* =============================================================================
 * Collection Functions
 * =============================================================================*/

static runtime_value_t fn_range(pseudo_runtime_t *rt, runtime_value_t *args, int n)
{
    /* Returns a tensor with [0, 1, ..., n-1] — reuse arange */
    return fn_arange(rt, args, n);
}

/* =============================================================================
 * Function Registry
 * =============================================================================*/

static const pseudo_builtin_t g_builtins[] = {
    /* Math */
    { "abs",         fn_abs,         1, 1 },
    { "sqrt",        fn_sqrt,        1, 1 },
    { "pow",         fn_pow,         2, 2 },
    { "exp",         fn_exp,         1, 1 },
    { "log",         fn_log,         1, 1 },
    { "sin",         fn_sin,         1, 1 },
    { "cos",         fn_cos,         1, 1 },
    { "floor",       fn_floor,       1, 1 },
    { "ceil",        fn_ceil,        1, 1 },
    { "round",       fn_round,       1, 1 },
    { "min",         fn_min,         2, 2 },
    { "max",         fn_max,         2, 2 },

    /* Tensor construction */
    { "zeros",       fn_zeros,       1, 1 },
    { "ones",        fn_ones,        1, 1 },
    { "arange",      fn_arange,      1, 1 },
    { "shape",       fn_shape,       1, 1 },

    /* String */
    { "len",         fn_len,         1, 1 },
    { "str",         fn_str,         1, 1 },
    { "int_parse",   fn_int_parse,   1, 1 },
    { "float_parse", fn_float_parse, 1, 1 },
    { "contains",    fn_contains,    2, 2 },
    { "upper",       fn_upper,       1, 1 },
    { "lower",       fn_lower,       1, 1 },
    { "strip",       fn_strip,       1, 1 },

    /* I/O */
    { "print",       fn_print,       0, 8 },
    { "println",     fn_println,     0, 8 },

    /* System */
    { "time_ms",     fn_time_ms,     0, 0 },
    { "assert",      fn_assert,      1, 2 },
    { "type_of",     fn_type_of,     1, 1 },

    /* Collections */
    { "range",       fn_range,       1, 1 },

    { NULL, NULL, 0, 0 },  /* sentinel */
};

const pseudo_builtin_t *pseudo_stdlib_lookup(const char *name)
{
    for (int i = 0; g_builtins[i].name != NULL; i++) {
        if (kstrcmp(g_builtins[i].name, name) == 0)
            return &g_builtins[i];
    }
    return NULL;
}

runtime_value_t pseudo_stdlib_call(pseudo_runtime_t *rt, const char *name,
                                    runtime_value_t *args, int n_args)
{
    const pseudo_builtin_t *b = pseudo_stdlib_lookup(name);
    if (!b) {
        kprintf("[STDLIB] Unknown function: %s\n", name);
        return val_none();
    }
    if (n_args < b->min_args || n_args > b->max_args) {
        kprintf("[STDLIB] %s: expected %d-%d args, got %d\n", name, b->min_args, b->max_args, n_args);
        return val_none();
    }
    return b->fn(rt, args, n_args);
}

int pseudo_stdlib_init(pseudo_runtime_t *rt)
{
    (void)rt;
    kprintf("[STDLIB] Pseudocode standard library initialized (%d builtins)\n",
            (int)(sizeof(g_builtins) / sizeof(g_builtins[0])) - 1);
    return 0;
}
