/* onnx.h - ONNX model format parser and runtime */
#ifndef ONNX_H
#define ONNX_H

#include <stdint.h>

/* ONNX tensor data types (subset matching ONNX spec) */
typedef enum {
    ONNX_FLOAT   = 1,
    ONNX_UINT8   = 2,
    ONNX_INT8    = 3,
    ONNX_UINT16  = 4,
    ONNX_INT16   = 5,
    ONNX_INT32   = 6,
    ONNX_INT64   = 7,
    ONNX_FLOAT16 = 10,
    ONNX_DOUBLE  = 11,
    ONNX_BFLOAT16 = 16,
} onnx_dtype_t;

/* Protobuf wire types */
#define PB_VARINT   0
#define PB_64BIT    1
#define PB_LENGTH   2
#define PB_32BIT    5

#define ONNX_MAX_DIMS       8
#define ONNX_MAX_NAME       128
#define ONNX_MAX_TENSORS    256
#define ONNX_MAX_NODES      512
#define ONNX_MAX_NODE_IO    8

typedef struct {
    char        name[ONNX_MAX_NAME];
    onnx_dtype_t dtype;
    int         ndim;
    int64_t     shape[ONNX_MAX_DIMS];
    float      *data;
    uint64_t    n_elements;
} onnx_tensor_t;

typedef struct {
    char    op_type[64];
    char    name[ONNX_MAX_NAME];
    char    inputs[ONNX_MAX_NODE_IO][ONNX_MAX_NAME];
    char    outputs[ONNX_MAX_NODE_IO][ONNX_MAX_NAME];
    int     n_inputs;
    int     n_outputs;
} onnx_node_t;

typedef struct {
    onnx_tensor_t  initializers[ONNX_MAX_TENSORS];
    int            n_initializers;
    onnx_node_t    nodes[ONNX_MAX_NODES];
    int            n_nodes;
    char           model_name[ONNX_MAX_NAME];
    int64_t        ir_version;
    int64_t        opset_version;
} onnx_model_t;

int  onnx_parse(const uint8_t *data, uint64_t size, onnx_model_t *model);
int  onnx_execute(onnx_model_t *model, const float *input, int input_size,
                  float *output, int output_size);
int  onnx_find_initializer(const onnx_model_t *model, const char *name);
void onnx_print_info(const onnx_model_t *model);

#endif /* ONNX_H */
