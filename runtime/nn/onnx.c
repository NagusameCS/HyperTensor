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
 * TensorOS - ONNX Model Format Parser and Runtime
 *
 * Parses ONNX protobuf format (subset sufficient for inference models):
 *   - ModelProto → GraphProto → NodeProto, TensorProto
 *   - Supports: Gemm, MatMul, Add, Relu, Sigmoid, Softmax, Reshape, Transpose
 *   - Initializer tensors loaded as FP32
 *   - Sequential execution through topologically-ordered nodes
 *
 * ONNX uses Protocol Buffers encoding. We implement a minimal protobuf decoder
 * sufficient to parse the ONNX schema without requiring protoc or external libs.
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#include "runtime/nn/onnx.h"

/* =============================================================================
 * Minimal Protobuf Decoder
 * =============================================================================*/

typedef struct {
    const uint8_t *p;
    const uint8_t *end;
} pb_reader_t;

static uint64_t pb_read_varint(pb_reader_t *r)
{
    uint64_t val = 0;
    int shift = 0;
    while (r->p < r->end) {
        uint8_t b = *r->p++;
        val |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
        if (shift > 63) break;
    }
    return val;
}

static int64_t pb_read_svarint(pb_reader_t *r)
{
    uint64_t v = pb_read_varint(r);
    return (int64_t)((v >> 1) ^ -(v & 1));  /* ZigZag decode */
}

static uint32_t pb_read_fixed32(pb_reader_t *r)
{
    if (r->p + 4 > r->end) return 0;
    uint32_t v = r->p[0] | (r->p[1] << 8) | (r->p[2] << 16) | ((uint32_t)r->p[3] << 24);
    r->p += 4;
    return v;
}

static uint64_t pb_read_fixed64(pb_reader_t *r)
{
    uint32_t lo = pb_read_fixed32(r);
    uint32_t hi = pb_read_fixed32(r);
    return (uint64_t)hi << 32 | lo;
}

static int pb_read_tag(pb_reader_t *r, int *field, int *wire_type)
{
    if (r->p >= r->end) return -1;
    uint64_t tag = pb_read_varint(r);
    *field = (int)(tag >> 3);
    *wire_type = (int)(tag & 7);
    return 0;
}

static int pb_read_bytes(pb_reader_t *r, const uint8_t **data, uint64_t *len)
{
    *len = pb_read_varint(r);
    if (r->p + *len > r->end) return -1;
    *data = r->p;
    r->p += *len;
    return 0;
}

static int pb_read_string(pb_reader_t *r, char *buf, int buflen)
{
    const uint8_t *data;
    uint64_t len;
    if (pb_read_bytes(r, &data, &len) != 0) return -1;
    int copy = (int)(len < (uint64_t)(buflen - 1) ? len : (uint64_t)(buflen - 1));
    kmemcpy(buf, data, copy);
    buf[copy] = '\0';
    return copy;
}

static void pb_skip_field(pb_reader_t *r, int wire_type)
{
    switch (wire_type) {
    case PB_VARINT: pb_read_varint(r); break;
    case PB_64BIT:  r->p += 8; break;
    case PB_LENGTH: { uint64_t l = pb_read_varint(r); r->p += l; } break;
    case PB_32BIT:  r->p += 4; break;
    }
}

/* =============================================================================
 * Parse TensorProto (ONNX field numbers from onnx.proto3)
 * =============================================================================*/

/* TensorProto field numbers */
#define TP_DIMS         1  /* repeated int64 */
#define TP_DATA_TYPE    2  /* int32 */
#define TP_FLOAT_DATA   4  /* repeated float (packed) */
#define TP_INT32_DATA   5  /* repeated int32 (packed) */
#define TP_NAME         8  /* string */
#define TP_RAW_DATA     13 /* bytes */

static int parse_tensor_proto(pb_reader_t *r, onnx_tensor_t *t)
{
    const uint8_t *msg_end = r->end;
    t->ndim = 0;
    t->dtype = ONNX_FLOAT;
    t->n_elements = 0;
    t->data = NULL;

    const uint8_t *raw_data = NULL;
    uint64_t raw_len = 0;
    const uint8_t *float_data = NULL;
    uint64_t float_len = 0;

    while (r->p < msg_end) {
        int field, wtype;
        if (pb_read_tag(r, &field, &wtype) != 0) break;

        switch (field) {
        case TP_NAME:
            pb_read_string(r, t->name, ONNX_MAX_NAME);
            break;
        case TP_DIMS:
            if (wtype == PB_LENGTH) {
                /* Packed repeated int64 */
                const uint8_t *d; uint64_t dlen;
                pb_read_bytes(r, &d, &dlen);
                pb_reader_t sub = { .p = d, .end = d + dlen };
                while (sub.p < sub.end && t->ndim < ONNX_MAX_DIMS)
                    t->shape[t->ndim++] = (int64_t)pb_read_varint(&sub);
            } else {
                if (t->ndim < ONNX_MAX_DIMS)
                    t->shape[t->ndim++] = (int64_t)pb_read_varint(r);
            }
            break;
        case TP_DATA_TYPE:
            t->dtype = (onnx_dtype_t)pb_read_varint(r);
            break;
        case TP_FLOAT_DATA:
            if (wtype == PB_LENGTH) {
                pb_read_bytes(r, &float_data, &float_len);
            } else {
                pb_skip_field(r, wtype);
            }
            break;
        case TP_RAW_DATA:
            pb_read_bytes(r, &raw_data, &raw_len);
            break;
        default:
            pb_skip_field(r, wtype);
        }
    }

    /* Calculate element count */
    uint64_t count = 1;
    for (int d = 0; d < t->ndim; d++) count *= (uint64_t)t->shape[d];
    t->n_elements = count;

    /* Load data as FP32 */
    if (count > 0) {
        t->data = (float *)kmalloc(count * sizeof(float));
        if (!t->data) return -1;

        if (raw_data && raw_len > 0) {
            if (t->dtype == ONNX_FLOAT && raw_len >= count * 4) {
                kmemcpy(t->data, raw_data, count * 4);
            } else if (t->dtype == ONNX_FLOAT16 && raw_len >= count * 2) {
                /* FP16 → FP32 */
                for (uint64_t i = 0; i < count; i++) {
                    uint16_t h = raw_data[i*2] | ((uint16_t)raw_data[i*2+1] << 8);
                    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
                    uint32_t exp = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;
                    uint32_t bits;
                    if (exp == 0) bits = sign;
                    else if (exp == 31) bits = sign | 0x7F800000u | (mant << 13);
                    else bits = sign | ((exp + 112) << 23) | (mant << 13);
                    kmemcpy(&t->data[i], &bits, 4);
                }
            } else {
                kmemset(t->data, 0, count * 4);
            }
        } else if (float_data && float_len >= count * 4) {
            kmemcpy(t->data, float_data, count * 4);
        } else {
            kmemset(t->data, 0, count * 4);
        }
    }

    return 0;
}

/* =============================================================================
 * Parse NodeProto
 * =============================================================================*/

/* NodeProto field numbers */
#define NP_INPUT    1
#define NP_OUTPUT   2
#define NP_NAME     3
#define NP_OP_TYPE  4

static int parse_node_proto(pb_reader_t *r, onnx_node_t *n)
{
    const uint8_t *msg_end = r->end;
    n->n_inputs = 0;
    n->n_outputs = 0;

    while (r->p < msg_end) {
        int field, wtype;
        if (pb_read_tag(r, &field, &wtype) != 0) break;

        switch (field) {
        case NP_INPUT:
            if (n->n_inputs < ONNX_MAX_NODE_IO)
                pb_read_string(r, n->inputs[n->n_inputs++], ONNX_MAX_NAME);
            else
                pb_skip_field(r, wtype);
            break;
        case NP_OUTPUT:
            if (n->n_outputs < ONNX_MAX_NODE_IO)
                pb_read_string(r, n->outputs[n->n_outputs++], ONNX_MAX_NAME);
            else
                pb_skip_field(r, wtype);
            break;
        case NP_NAME:
            pb_read_string(r, n->name, ONNX_MAX_NAME);
            break;
        case NP_OP_TYPE:
            pb_read_string(r, n->op_type, 64);
            break;
        default:
            pb_skip_field(r, wtype);
        }
    }
    return 0;
}

/* =============================================================================
 * Parse GraphProto
 * =============================================================================*/

/* GraphProto field numbers */
#define GP_NODE         1
#define GP_NAME         2
#define GP_INITIALIZER  5
#define GP_INPUT        11
#define GP_OUTPUT       12

static int parse_graph_proto(pb_reader_t *r, onnx_model_t *m)
{
    const uint8_t *msg_end = r->end;

    while (r->p < msg_end) {
        int field, wtype;
        if (pb_read_tag(r, &field, &wtype) != 0) break;

        if (field == GP_NODE && wtype == PB_LENGTH) {
            const uint8_t *d; uint64_t dlen;
            pb_read_bytes(r, &d, &dlen);
            if (m->n_nodes < ONNX_MAX_NODES) {
                pb_reader_t sub = { .p = d, .end = d + dlen };
                parse_node_proto(&sub, &m->nodes[m->n_nodes++]);
            }
        } else if (field == GP_INITIALIZER && wtype == PB_LENGTH) {
            const uint8_t *d; uint64_t dlen;
            pb_read_bytes(r, &d, &dlen);
            if (m->n_initializers < ONNX_MAX_TENSORS) {
                pb_reader_t sub = { .p = d, .end = d + dlen };
                parse_tensor_proto(&sub, &m->initializers[m->n_initializers++]);
            }
        } else if (field == GP_NAME && wtype == PB_LENGTH) {
            pb_read_string(r, m->model_name, ONNX_MAX_NAME);
        } else {
            pb_skip_field(r, wtype);
        }
    }
    return 0;
}

/* =============================================================================
 * Parse ModelProto (top-level)
 * =============================================================================*/

/* ModelProto field numbers */
#define MP_IR_VERSION   1
#define MP_GRAPH        7
#define MP_OPSET        8

int onnx_parse(const uint8_t *data, uint64_t size, onnx_model_t *model)
{
    if (!data || size < 4 || !model) return -1;
    kmemset(model, 0, sizeof(*model));

    pb_reader_t r = { .p = data, .end = data + size };

    while (r.p < r.end) {
        int field, wtype;
        if (pb_read_tag(&r, &field, &wtype) != 0) break;

        if (field == MP_IR_VERSION && wtype == PB_VARINT) {
            model->ir_version = (int64_t)pb_read_varint(&r);
        } else if (field == MP_GRAPH && wtype == PB_LENGTH) {
            const uint8_t *d; uint64_t dlen;
            pb_read_bytes(&r, &d, &dlen);
            pb_reader_t sub = { .p = d, .end = d + dlen };
            parse_graph_proto(&sub, model);
        } else if (field == MP_OPSET && wtype == PB_LENGTH) {
            /* OpsetIdProto — field 2 is version */
            const uint8_t *d; uint64_t dlen;
            pb_read_bytes(&r, &d, &dlen);
            pb_reader_t sub = { .p = d, .end = d + dlen };
            while (sub.p < sub.end) {
                int sf, sw;
                if (pb_read_tag(&sub, &sf, &sw) != 0) break;
                if (sf == 2 && sw == PB_VARINT)
                    model->opset_version = (int64_t)pb_read_varint(&sub);
                else
                    pb_skip_field(&sub, sw);
            }
        } else {
            pb_skip_field(&r, wtype);
        }
    }

    kprintf("[ONNX] Parsed: ir_version=%ld, opset=%ld, %d nodes, %d initializers\n",
            (long)model->ir_version, (long)model->opset_version,
            model->n_nodes, model->n_initializers);
    return 0;
}

/* =============================================================================
 * Execution Engine - Sequential Node Dispatch
 * =============================================================================*/

/* Simple tensor value store for intermediate results */
#define ONNX_MAX_VALUES 256
typedef struct {
    char    name[ONNX_MAX_NAME];
    float  *data;
    int64_t shape[ONNX_MAX_DIMS];
    int     ndim;
    uint64_t count;
} onnx_value_t;

static onnx_value_t g_values[ONNX_MAX_VALUES];
static int g_n_values;

static onnx_value_t *find_value(const char *name)
{
    for (int i = 0; i < g_n_values; i++)
        if (kstrcmp(g_values[i].name, name) == 0)
            return &g_values[i];
    return NULL;
}

static onnx_value_t *alloc_value(const char *name, uint64_t count)
{
    if (g_n_values >= ONNX_MAX_VALUES) return NULL;
    onnx_value_t *v = &g_values[g_n_values++];
    int len = kstrlen(name);
    if (len >= ONNX_MAX_NAME) len = ONNX_MAX_NAME - 1;
    kmemcpy(v->name, name, len);
    v->name[len] = '\0';
    v->count = count;
    v->data = (float *)kmalloc(count * sizeof(float));
    if (!v->data) return NULL;
    return v;
}

static void exec_gemm(const onnx_node_t *node)
{
    onnx_value_t *A = find_value(node->inputs[0]);
    onnx_value_t *B = find_value(node->inputs[1]);
    if (!A || !B || !A->data || !B->data) return;

    int M = (int)(A->ndim >= 2 ? A->shape[0] : 1);
    int K = (int)(A->ndim >= 2 ? A->shape[1] : A->count);
    int N = (int)(B->ndim >= 2 ? B->shape[1] : B->count / K);

    onnx_value_t *out = alloc_value(node->outputs[0], (uint64_t)M * N);
    if (!out) return;
    out->ndim = 2; out->shape[0] = M; out->shape[1] = N;

    /* Bias (input[2]) */
    onnx_value_t *C = (node->n_inputs > 2) ? find_value(node->inputs[2]) : NULL;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = (C && C->data && (uint64_t)j < C->count) ? C->data[j] : 0.0f;
            for (int k = 0; k < K; k++)
                sum += A->data[i * K + k] * B->data[k * N + j];
            out->data[i * N + j] = sum;
        }
    }
}

static void exec_add(const onnx_node_t *node)
{
    onnx_value_t *A = find_value(node->inputs[0]);
    onnx_value_t *B = find_value(node->inputs[1]);
    if (!A || !B || !A->data || !B->data) return;

    uint64_t count = A->count;
    onnx_value_t *out = alloc_value(node->outputs[0], count);
    if (!out) return;
    out->ndim = A->ndim;
    for (int d = 0; d < A->ndim; d++) out->shape[d] = A->shape[d];

    for (uint64_t i = 0; i < count; i++)
        out->data[i] = A->data[i] + B->data[i % B->count];  /* broadcast */
}

static void exec_relu(const onnx_node_t *node)
{
    onnx_value_t *A = find_value(node->inputs[0]);
    if (!A || !A->data) return;

    onnx_value_t *out = alloc_value(node->outputs[0], A->count);
    if (!out) return;
    out->ndim = A->ndim;
    for (int d = 0; d < A->ndim; d++) out->shape[d] = A->shape[d];

    for (uint64_t i = 0; i < A->count; i++)
        out->data[i] = A->data[i] > 0.0f ? A->data[i] : 0.0f;
}

static void exec_sigmoid(const onnx_node_t *node)
{
    onnx_value_t *A = find_value(node->inputs[0]);
    if (!A || !A->data) return;

    onnx_value_t *out = alloc_value(node->outputs[0], A->count);
    if (!out) return;
    out->ndim = A->ndim;
    for (int d = 0; d < A->ndim; d++) out->shape[d] = A->shape[d];

    for (uint64_t i = 0; i < A->count; i++) {
        float x = A->data[i];
        if (x > 20.0f) out->data[i] = 1.0f;
        else if (x < -20.0f) out->data[i] = 0.0f;
        else {
            /* Schraudolph fast exp */
            union { float f; int32_t i; } u;
            u.i = (int32_t)(12102203.0f * (-x) + 1065353216.0f);
            out->data[i] = 1.0f / (1.0f + u.f);
        }
    }
}

static void exec_softmax(const onnx_node_t *node)
{
    onnx_value_t *A = find_value(node->inputs[0]);
    if (!A || !A->data) return;

    onnx_value_t *out = alloc_value(node->outputs[0], A->count);
    if (!out) return;
    out->ndim = A->ndim;
    for (int d = 0; d < A->ndim; d++) out->shape[d] = A->shape[d];

    /* Last-axis softmax */
    int64_t axis_size = (A->ndim > 0) ? A->shape[A->ndim - 1] : (int64_t)A->count;
    int64_t batches = (int64_t)A->count / axis_size;

    for (int64_t b = 0; b < batches; b++) {
        float *src = A->data + b * axis_size;
        float *dst = out->data + b * axis_size;
        float maxval = src[0];
        for (int64_t i = 1; i < axis_size; i++)
            if (src[i] > maxval) maxval = src[i];
        float sum = 0.0f;
        for (int64_t i = 0; i < axis_size; i++) {
            union { float f; int32_t v; } u;
            u.v = (int32_t)(12102203.0f * (src[i] - maxval) + 1065353216.0f);
            dst[i] = u.f;
            sum += dst[i];
        }
        float inv = 1.0f / (sum + 1e-10f);
        for (int64_t i = 0; i < axis_size; i++)
            dst[i] *= inv;
    }
}

static void exec_matmul(const onnx_node_t *node)
{
    /* MatMul is like Gemm but without bias and alpha/beta */
    exec_gemm(node);
}

int onnx_execute(onnx_model_t *model, const float *input, int input_size,
                 float *output, int output_size)
{
    if (!model || !input || !output) return -1;

    /* Reset value store */
    g_n_values = 0;

    /* Register initializers as values */
    for (int i = 0; i < model->n_initializers; i++) {
        onnx_tensor_t *t = &model->initializers[i];
        if (g_n_values >= ONNX_MAX_VALUES) break;
        onnx_value_t *v = &g_values[g_n_values++];
        int len = kstrlen(t->name);
        if (len >= ONNX_MAX_NAME) len = ONNX_MAX_NAME - 1;
        kmemcpy(v->name, t->name, len);
        v->name[len] = '\0';
        v->data = t->data;  /* Point to existing data, not a copy */
        v->count = t->n_elements;
        v->ndim = t->ndim;
        for (int d = 0; d < t->ndim; d++) v->shape[d] = t->shape[d];
    }

    /* Register input as the first graph input */
    if (model->n_nodes > 0 && model->nodes[0].n_inputs > 0) {
        onnx_value_t *v = alloc_value(model->nodes[0].inputs[0], (uint64_t)input_size);
        if (v) {
            kmemcpy(v->data, input, input_size * sizeof(float));
            v->ndim = 1;
            v->shape[0] = input_size;
        }
    }

    /* Execute nodes sequentially */
    for (int i = 0; i < model->n_nodes; i++) {
        const onnx_node_t *n = &model->nodes[i];

        if (kstrcmp(n->op_type, "Gemm") == 0)         exec_gemm(n);
        else if (kstrcmp(n->op_type, "MatMul") == 0)   exec_matmul(n);
        else if (kstrcmp(n->op_type, "Add") == 0)      exec_add(n);
        else if (kstrcmp(n->op_type, "Relu") == 0)     exec_relu(n);
        else if (kstrcmp(n->op_type, "Sigmoid") == 0)  exec_sigmoid(n);
        else if (kstrcmp(n->op_type, "Softmax") == 0)  exec_softmax(n);
        else kprintf("[ONNX] Unsupported op: %s\n", n->op_type);
    }

    /* Find the last output value */
    if (model->n_nodes > 0) {
        const onnx_node_t *last = &model->nodes[model->n_nodes - 1];
        if (last->n_outputs > 0) {
            onnx_value_t *out_v = find_value(last->outputs[0]);
            if (out_v && out_v->data) {
                int copy = output_size < (int)out_v->count ? output_size : (int)out_v->count;
                kmemcpy(output, out_v->data, copy * sizeof(float));
                return copy;
            }
        }
    }

    return -1;
}

int onnx_find_initializer(const onnx_model_t *model, const char *name)
{
    for (int i = 0; i < model->n_initializers; i++)
        if (kstrcmp(model->initializers[i].name, name) == 0) return i;
    return -1;
}

void onnx_print_info(const onnx_model_t *model)
{
    kprintf("[ONNX] Model: %s (IR v%ld, opset %ld)\n",
            model->model_name, (long)model->ir_version, (long)model->opset_version);
    kprintf("[ONNX] %d nodes, %d initializers\n", model->n_nodes, model->n_initializers);
    for (int i = 0; i < model->n_nodes && i < 20; i++) {
        kprintf("  [%d] %s: %s", i, model->nodes[i].op_type, model->nodes[i].name);
        kprintf("  (");
        for (int j = 0; j < model->nodes[i].n_inputs; j++)
            kprintf("%s%s", model->nodes[i].inputs[j], j + 1 < model->nodes[i].n_inputs ? ", " : "");
        kprintf(" -> ");
        for (int j = 0; j < model->nodes[i].n_outputs; j++)
            kprintf("%s%s", model->nodes[i].outputs[j], j + 1 < model->nodes[i].n_outputs ? ", " : "");
        kprintf(")\n");
    }
}
