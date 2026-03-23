/* =============================================================================
 * TensorOS - safetensors Native Loader
 *
 * Parses the safetensors format: a simple JSON header followed by raw tensor
 * data.  Format:
 *   [8 bytes]  header_size (little-endian uint64)
 *   [header_size bytes]  JSON metadata describing tensors
 *   [remainder]  raw tensor data (contiguous, aligned)
 *
 * JSON header example:
 *   {"tensor_name": {"dtype": "F32", "shape": [768, 768],
 *                     "data_offsets": [0, 2359296]}, ...}
 *
 * Supported dtypes: F32, F16, BF16, I32, I16, I8, U8
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#include "kernel/core/perf.h"

/* dtype enum matching safetensors specification */
typedef enum {
    ST_DTYPE_F32  = 0,
    ST_DTYPE_F16  = 1,
    ST_DTYPE_BF16 = 2,
    ST_DTYPE_F64  = 3,
    ST_DTYPE_I32  = 4,
    ST_DTYPE_I16  = 5,
    ST_DTYPE_I8   = 6,
    ST_DTYPE_U8   = 7,
    ST_DTYPE_BOOL = 8,
    ST_DTYPE_UNKNOWN = 255,
} st_dtype_t;

static int st_dtype_size(st_dtype_t dt)
{
    switch (dt) {
    case ST_DTYPE_F64:  return 8;
    case ST_DTYPE_F32:
    case ST_DTYPE_I32:  return 4;
    case ST_DTYPE_F16:
    case ST_DTYPE_BF16:
    case ST_DTYPE_I16:  return 2;
    case ST_DTYPE_I8:
    case ST_DTYPE_U8:
    case ST_DTYPE_BOOL: return 1;
    default:            return 0;
    }
}

/* Parsed tensor info */
#define ST_MAX_TENSORS  512
#define ST_MAX_DIMS     8
#define ST_MAX_NAME     128

typedef struct {
    char        name[ST_MAX_NAME];
    st_dtype_t  dtype;
    int         ndim;
    uint64_t    shape[ST_MAX_DIMS];
    uint64_t    data_offset;    /* Offset from start of data section */
    uint64_t    data_size;      /* Size in bytes */
    float      *data;           /* Pointer to loaded/converted data */
} st_tensor_info_t;

typedef struct {
    st_tensor_info_t tensors[ST_MAX_TENSORS];
    int              n_tensors;
    uint64_t         header_size;
    uint64_t         data_size;
    const uint8_t   *data_base;  /* Pointer to start of raw data */
} st_file_t;

/* =============================================================================
 * Minimal JSON Parser (sufficient for safetensors headers)
 * =============================================================================*/

typedef struct {
    const char *p;
    const char *end;
} json_ctx_t;

static void json_skip_ws(json_ctx_t *j)
{
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\n' || *j->p == '\r' || *j->p == '\t'))
        j->p++;
}

static int json_expect(json_ctx_t *j, char c)
{
    json_skip_ws(j);
    if (j->p >= j->end || *j->p != c) return -1;
    j->p++;
    return 0;
}

/* Parse a JSON string, copy into buf (max buflen-1 chars) */
static int json_parse_string(json_ctx_t *j, char *buf, int buflen)
{
    json_skip_ws(j);
    if (j->p >= j->end || *j->p != '"') return -1;
    j->p++;
    int pos = 0;
    while (j->p < j->end && *j->p != '"') {
        if (*j->p == '\\') {
            j->p++;
            if (j->p >= j->end) return -1;
        }
        if (pos < buflen - 1) buf[pos++] = *j->p;
        j->p++;
    }
    buf[pos] = '\0';
    if (j->p < j->end) j->p++; /* skip closing quote */
    return pos;
}

/* Parse a JSON integer */
static int json_parse_uint64(json_ctx_t *j, uint64_t *val)
{
    json_skip_ws(j);
    *val = 0;
    if (j->p >= j->end) return -1;
    while (j->p < j->end && *j->p >= '0' && *j->p <= '9') {
        *val = *val * 10 + (*j->p - '0');
        j->p++;
    }
    return 0;
}

/* Parse dtype string → enum */
static st_dtype_t parse_dtype(const char *s)
{
    if (s[0] == 'F' && s[1] == '3' && s[2] == '2') return ST_DTYPE_F32;
    if (s[0] == 'F' && s[1] == '1' && s[2] == '6') return ST_DTYPE_F16;
    if (s[0] == 'B' && s[1] == 'F' && s[2] == '1' && s[3] == '6') return ST_DTYPE_BF16;
    if (s[0] == 'F' && s[1] == '6' && s[2] == '4') return ST_DTYPE_F64;
    if (s[0] == 'I' && s[1] == '3' && s[2] == '2') return ST_DTYPE_I32;
    if (s[0] == 'I' && s[1] == '1' && s[2] == '6') return ST_DTYPE_I16;
    if (s[0] == 'I' && s[1] == '8') return ST_DTYPE_I8;
    if (s[0] == 'U' && s[1] == '8') return ST_DTYPE_U8;
    if (s[0] == 'B' && s[1] == 'O') return ST_DTYPE_BOOL;
    return ST_DTYPE_UNKNOWN;
}

/* =============================================================================
 * Parse safetensors Header
 * =============================================================================*/

int safetensors_parse(const uint8_t *data, uint64_t file_size, st_file_t *st)
{
    if (!data || file_size < 8 || !st) return -1;

    kmemset(st, 0, sizeof(*st));

    /* Read header size (first 8 bytes, little-endian) */
    uint64_t hdr_size = 0;
    for (int i = 0; i < 8; i++)
        hdr_size |= (uint64_t)data[i] << (i * 8);

    if (hdr_size > file_size - 8 || hdr_size > 100 * 1024 * 1024) {
        kprintf("[SAFETENSORS] Invalid header size: %lu\n", (unsigned long)hdr_size);
        return -1;
    }

    st->header_size = hdr_size;
    st->data_base = data + 8 + hdr_size;
    st->data_size = file_size - 8 - hdr_size;

    /* Parse JSON header */
    json_ctx_t j = { .p = (const char *)(data + 8), .end = (const char *)(data + 8 + hdr_size) };

    if (json_expect(&j, '{') != 0) return -1;

    while (j.p < j.end && *j.p != '}') {
        if (st->n_tensors >= ST_MAX_TENSORS) break;

        /* Key: tensor name */
        st_tensor_info_t *t = &st->tensors[st->n_tensors];
        if (json_parse_string(&j, t->name, ST_MAX_NAME) < 0) break;

        json_skip_ws(&j);
        if (json_expect(&j, ':') != 0) break;

        /* Skip __metadata__ key */
        if (t->name[0] == '_' && t->name[1] == '_') {
            /* Skip value (scan for matching brace) */
            int depth = 0;
            while (j.p < j.end) {
                if (*j.p == '{') depth++;
                else if (*j.p == '}') { if (--depth <= 0) { j.p++; break; } }
                j.p++;
            }
            json_skip_ws(&j);
            if (j.p < j.end && *j.p == ',') j.p++;
            continue;
        }

        /* Value: {"dtype": ..., "shape": [...], "data_offsets": [...]} */
        if (json_expect(&j, '{') != 0) break;

        while (j.p < j.end && *j.p != '}') {
            char key[32];
            if (json_parse_string(&j, key, sizeof(key)) < 0) break;
            if (json_expect(&j, ':') != 0) break;

            if (key[0] == 'd' && key[1] == 't') {
                /* "dtype" */
                char dtype_str[16];
                json_parse_string(&j, dtype_str, sizeof(dtype_str));
                t->dtype = parse_dtype(dtype_str);
            } else if (key[0] == 's' && key[1] == 'h') {
                /* "shape" */
                json_expect(&j, '[');
                t->ndim = 0;
                while (j.p < j.end && *j.p != ']') {
                    if (t->ndim < ST_MAX_DIMS)
                        json_parse_uint64(&j, &t->shape[t->ndim++]);
                    json_skip_ws(&j);
                    if (j.p < j.end && *j.p == ',') j.p++;
                }
                if (j.p < j.end) j.p++; /* skip ] */
            } else if (key[0] == 'd' && key[5] == 'o') {
                /* "data_offsets" */
                json_expect(&j, '[');
                uint64_t off_start = 0, off_end = 0;
                json_parse_uint64(&j, &off_start);
                json_skip_ws(&j);
                if (j.p < j.end && *j.p == ',') j.p++;
                json_parse_uint64(&j, &off_end);
                json_skip_ws(&j);
                if (j.p < j.end && *j.p == ']') j.p++;
                t->data_offset = off_start;
                t->data_size = off_end - off_start;
            }
            json_skip_ws(&j);
            if (j.p < j.end && *j.p == ',') j.p++;
        }
        if (j.p < j.end) j.p++; /* skip } */

        st->n_tensors++;
        json_skip_ws(&j);
        if (j.p < j.end && *j.p == ',') j.p++;
    }

    kprintf("[SAFETENSORS] Parsed: %d tensors, header=%lu bytes, data=%lu bytes\n",
            st->n_tensors, (unsigned long)hdr_size, (unsigned long)st->data_size);
    return 0;
}

/* =============================================================================
 * Load tensors into memory (with optional FP16/BF16 → FP32 conversion)
 * =============================================================================*/

static float bf16_to_fp32(uint16_t bf)
{
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    kmemcpy(&f, &bits, 4);
    return f;
}

static float fp16_to_fp32(uint16_t h)
{
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) { float f; kmemcpy(&f, &sign, 4); return f; }
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= 0x3FF;
    } else if (exp == 31) {
        uint32_t bits = sign | 0x7F800000u | (mant << 13);
        float f; kmemcpy(&f, &bits, 4); return f;
    }
    exp = exp + 127 - 15;
    uint32_t bits = sign | (exp << 23) | (mant << 13);
    float f; kmemcpy(&f, &bits, 4); return f;
}

int safetensors_load_tensor(st_file_t *st, int tensor_idx, float **out_data, uint64_t *out_count)
{
    if (!st || tensor_idx < 0 || tensor_idx >= st->n_tensors) return -1;

    st_tensor_info_t *t = &st->tensors[tensor_idx];
    if (t->data_offset + t->data_size > st->data_size) return -1;

    /* Calculate element count */
    uint64_t count = 1;
    for (int d = 0; d < t->ndim; d++) count *= t->shape[d];

    /* Allocate FP32 output buffer */
    float *buf = (float *)kmalloc(count * sizeof(float));
    if (!buf) return -1;

    const uint8_t *src = st->data_base + t->data_offset;

    switch (t->dtype) {
    case ST_DTYPE_F32:
        kmemcpy(buf, src, count * sizeof(float));
        break;
    case ST_DTYPE_F16:
        for (uint64_t i = 0; i < count; i++) {
            uint16_t h;
            kmemcpy(&h, src + i * 2, 2);
            buf[i] = fp16_to_fp32(h);
        }
        break;
    case ST_DTYPE_BF16:
        for (uint64_t i = 0; i < count; i++) {
            uint16_t bf;
            kmemcpy(&bf, src + i * 2, 2);
            buf[i] = bf16_to_fp32(bf);
        }
        break;
    case ST_DTYPE_I8:
        for (uint64_t i = 0; i < count; i++)
            buf[i] = (float)((int8_t)src[i]);
        break;
    case ST_DTYPE_U8:
        for (uint64_t i = 0; i < count; i++)
            buf[i] = (float)src[i];
        break;
    default:
        kfree(buf);
        return -1;
    }

    t->data = buf;
    *out_data = buf;
    *out_count = count;
    return 0;
}

/* Lookup tensor by name */
int safetensors_find(const st_file_t *st, const char *name)
{
    for (int i = 0; i < st->n_tensors; i++) {
        if (kstrcmp(st->tensors[i].name, name) == 0)
            return i;
    }
    return -1;
}

/* Print summary of all tensors */
void safetensors_print_info(const st_file_t *st)
{
    kprintf("[SAFETENSORS] %d tensors:\n", st->n_tensors);
    for (int i = 0; i < st->n_tensors && i < 20; i++) {
        const st_tensor_info_t *t = &st->tensors[i];
        const char *dtype_names[] = {"F32","F16","BF16","F64","I32","I16","I8","U8","BOOL"};
        const char *dt = t->dtype < 9 ? dtype_names[t->dtype] : "?";
        kprintf("  [%d] %s: %s [", i, t->name, dt);
        for (int d = 0; d < t->ndim; d++)
            kprintf("%lu%s", (unsigned long)t->shape[d], d + 1 < t->ndim ? "×" : "");
        kprintf("] %lu bytes\n", (unsigned long)t->data_size);
    }
    if (st->n_tensors > 20)
        kprintf("  ... and %d more\n", st->n_tensors - 20);
}
