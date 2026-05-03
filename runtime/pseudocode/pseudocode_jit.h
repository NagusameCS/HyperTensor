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
 * TensorOS - Pseudocode JIT Runtime
 *
 * This is the default runtime for AI tasks in TensorOS. It executes
 * Pseudocode (by NaguSamecs on GitHub) as the native scripting language,
 * with JIT compilation for hot tensor operation paths.
 *
 * Architecture:
 * 
 *   Pseudocode Source (.pseudo files)              
 * 
 *   Lexer → Parser → AST                          
 * 
 *   Tensor IR (Intermediate Representation)        
 * 
 *   Optimization Passes:                           
 *     - Operator Fusion (matmul+bias+relu → fused) 
 *     - Tensor Layout Optimization                 
 *     - Precision Auto-tuning (FP32→FP16→INT8)    
 *     - Batch Dimension Inference                  
 * 
 *   Backend Code Generation:                       
 *     - x86_64 AVX-512 (CPU)                      
 *     - GPU Compute Dispatch                       
 *     - TPU IR                                     
 * 
 *
 * The JIT monitors execution and recompiles hot paths with
 * increasingly aggressive optimizations.
 * =============================================================================*/

#ifndef TENSOROS_PSEUDOCODE_JIT_H
#define TENSOROS_PSEUDOCODE_JIT_H

#include "kernel/core/kernel.h"

/* =============================================================================
 * Token Types for Pseudocode Lexer
 * =============================================================================*/

typedef enum {
    /* Literals */
    TOK_INT_LIT,
    TOK_FLOAT_LIT,
    TOK_STRING_LIT,
    TOK_BOOL_LIT,

    /* Identifiers and keywords */
    TOK_IDENT,
    TOK_KW_MODEL,          /* model MyModel: */
    TOK_KW_LAYER,          /* layer dense: */
    TOK_KW_TENSOR,         /* tensor x = ... */
    TOK_KW_TRAIN,          /* train model on data */
    TOK_KW_INFER,          /* infer model(input) */
    TOK_KW_LOAD,           /* load "model.safetensors" */
    TOK_KW_SAVE,           /* save model to "path" */
    TOK_KW_PIPELINE,       /* pipeline { ... } */
    TOK_KW_IF,
    TOK_KW_ELSE,
    TOK_KW_FOR,
    TOK_KW_WHILE,
    TOK_KW_RETURN,
    TOK_KW_FUNCTION,
    TOK_KW_IMPORT,
    TOK_KW_FROM,
    TOK_KW_AS,
    TOK_KW_PRINT,
    TOK_KW_GIT,            /* git commit "message" */
    TOK_KW_DEPLOY,         /* deploy model on port 8080 */
    TOK_KW_MONITOR,        /* monitor model { ... } */

    /* Tensor operations */
    TOK_KW_MATMUL,
    TOK_KW_CONV,
    TOK_KW_ATTENTION,
    TOK_KW_SOFTMAX,
    TOK_KW_RELU,
    TOK_KW_SIGMOID,
    TOK_KW_LAYERNORM,
    TOK_KW_DROPOUT,
    TOK_KW_RESHAPE,
    TOK_KW_TRANSPOSE,

    /* Operators */
    TOK_PLUS,
    TOK_MINUS,
    TOK_STAR,
    TOK_SLASH,
    TOK_AT,             /* @ for matrix multiply */
    TOK_PERCENT,
    TOK_EQUAL,
    TOK_EQUAL_EQUAL,
    TOK_NOT_EQUAL,
    TOK_LESS,
    TOK_GREATER,
    TOK_LESS_EQUAL,
    TOK_GREATER_EQUAL,
    TOK_ARROW,          /* -> for type annotations */
    TOK_FAT_ARROW,      /* => for lambdas */

    /* Delimiters */
    TOK_LPAREN,
    TOK_RPAREN,
    TOK_LBRACKET,
    TOK_RBRACKET,
    TOK_LBRACE,
    TOK_RBRACE,
    TOK_COMMA,
    TOK_COLON,
    TOK_SEMICOLON,
    TOK_DOT,
    TOK_NEWLINE,

    /* Special */
    TOK_EOF,
    TOK_ERROR,
} token_type_t;

typedef struct {
    token_type_t type;
    const char  *start;
    uint32_t     length;
    uint32_t     line;
    uint32_t     column;
    union {
        int64_t  int_val;
        double   float_val;
        bool     bool_val;
    } value;
} token_t;

/* =============================================================================
 * AST Node Types
 * =============================================================================*/

typedef enum {
    AST_PROGRAM,
    AST_MODEL_DEF,
    AST_LAYER_DEF,
    AST_FUNCTION_DEF,
    AST_TENSOR_DECL,
    AST_ASSIGNMENT,
    AST_BINARY_OP,
    AST_UNARY_OP,
    AST_CALL,
    AST_INDEX,
    AST_MEMBER,
    AST_IF,
    AST_FOR,
    AST_WHILE,
    AST_RETURN,
    AST_PRINT,
    AST_BLOCK,
    AST_LITERAL,
    AST_IDENT,
    AST_TENSOR_OP,      /* Built-in tensor operation */
    AST_TRAIN,
    AST_INFER,
    AST_LOAD,
    AST_SAVE,
    AST_PIPELINE,
    AST_GIT_OP,
    AST_DEPLOY,
    AST_MONITOR,
    AST_SHAPE_EXPR,     /* Tensor shape expression: [batch, seq_len, hidden] */
} ast_node_type_t;

typedef struct ast_node {
    ast_node_type_t     type;
    token_t             token;       /* Associated token */
    struct ast_node    *children[8]; /* Up to 8 children */
    uint32_t            child_count;
    struct ast_node    *next;        /* For linked lists (statements, etc.) */

    /* Type information (filled during type checking) */
    tensor_dtype_t      result_dtype;
    uint64_t            result_shape[8];
    uint32_t            result_ndim;
} ast_node_t;

/* =============================================================================
 * Tensor IR (Intermediate Representation)
 * Lower-level representation optimized for tensor operations
 * =============================================================================*/

typedef enum {
    TIR_ALLOC,          /* Allocate tensor */
    TIR_FREE,           /* Free tensor */
    TIR_LOAD,           /* Load from memory */
    TIR_STORE,          /* Store to memory */
    TIR_MATMUL,         /* Matrix multiply */
    TIR_CONV2D,         /* 2D convolution */
    TIR_ATTENTION,      /* Attention mechanism */
    TIR_SOFTMAX,        /* Softmax */
    TIR_LAYERNORM,      /* Layer normalization */
    TIR_RELU,           /* ReLU activation */
    TIR_GELU,           /* GELU activation */
    TIR_SIGMOID,        /* Sigmoid activation */
    TIR_ADD,            /* Element-wise add */
    TIR_MUL,            /* Element-wise multiply */
    TIR_SUB,            /* Element-wise subtract */
    TIR_DIV,            /* Element-wise divide */
    TIR_RESHAPE,        /* Reshape tensor */
    TIR_TRANSPOSE,      /* Transpose tensor */
    TIR_CONCAT,         /* Concatenate tensors */
    TIR_SPLIT,          /* Split tensor */
    TIR_REDUCE_SUM,     /* Sum reduction */
    TIR_REDUCE_MEAN,    /* Mean reduction */
    TIR_DROPOUT,        /* Dropout */
    TIR_EMBEDDING,      /* Embedding lookup */
    TIR_FUSED_OP,       /* Fused operator (multiple ops in one) */
    TIR_CALL,           /* Call external function */
    TIR_BRANCH,         /* Conditional branch */
    TIR_LOOP,           /* Loop construct */
    TIR_RETURN,         /* Return value */
    TIR_NOP,            /* No operation */
} tir_opcode_t;

#define TIR_MAX_OPERANDS 4

typedef struct tir_instruction {
    tir_opcode_t    opcode;
    uint32_t        dest;           /* Destination register */
    uint32_t        operands[TIR_MAX_OPERANDS]; /* Source registers */
    uint32_t        num_operands;

    /* Tensor metadata for this operation */
    tensor_dtype_t  dtype;
    uint64_t        shape[8];
    uint32_t        ndim;

    /* Device hint */
    uint32_t        target_device;  /* 0=CPU, 1+=GPU */

    /* Fused operation chain */
    struct tir_instruction *fused_next;

    /* Profiling data */
    uint64_t        exec_count;     /* Times executed */
    uint64_t        total_ns;       /* Total execution time */
} tir_instruction_t;

#define TIR_MAX_INSTRUCTIONS 4096
#define TIR_MAX_REGISTERS    256

typedef struct {
    tir_instruction_t   instructions[TIR_MAX_INSTRUCTIONS];
    uint32_t            inst_count;
    tensor_desc_t       registers[TIR_MAX_REGISTERS]; /* Virtual tensor regs */
    uint32_t            reg_count;
} tir_program_t;

/* =============================================================================
 * JIT Compilation State
 * =============================================================================*/

typedef enum {
    JIT_TIER_INTERP  = 0,  /* Interpreted (first execution) */
    JIT_TIER_BASIC   = 1,  /* Basic JIT (after 10 executions) */
    JIT_TIER_OPT     = 2,  /* Optimized JIT (after 100 executions) */
    JIT_TIER_FULL    = 3,  /* Fully optimized (after 1000 executions) */
} jit_tier_t;

#define JIT_TIER1_THRESHOLD   10
#define JIT_TIER2_THRESHOLD   100
#define JIT_TIER3_THRESHOLD   1000

typedef struct {
    void        *code;          /* JIT-compiled machine code */
    uint64_t     code_size;
    jit_tier_t   tier;
    uint64_t     exec_count;
    uint64_t     compile_time_ns;
} jit_compiled_t;

/* =============================================================================
 * Runtime Value (used during interpretation)
 * =============================================================================*/

typedef enum {
    VAL_NONE,
    VAL_INT,
    VAL_FLOAT,
    VAL_BOOL,
    VAL_STRING,
    VAL_TENSOR,
    VAL_MODEL,
    VAL_FUNCTION,
} value_type_t;

typedef struct {
    value_type_t type;
    union {
        int64_t     int_val;
        double      float_val;
        bool        bool_val;
        char       *string_val;
        tensor_desc_t *tensor_val;
        void       *model_val;
        void       *func_val;
    };
} runtime_value_t;

/* =============================================================================
 * Pseudocode Runtime Environment
 * =============================================================================*/

#define PSEUDO_MAX_VARS      256
#define PSEUDO_MAX_FUNCTIONS 128
#define PSEUDO_MAX_MODELS    32
#define PSEUDO_STACK_SIZE    1024

typedef struct {
    char            name[64];
    runtime_value_t value;
} variable_t;

typedef struct {
    char            name[64];
    ast_node_t     *body;
    uint32_t        param_count;
    char            params[8][64];
    jit_compiled_t *compiled;
} function_def_t;

typedef struct {
    /* Variable store */
    variable_t      vars[PSEUDO_MAX_VARS];
    uint32_t        var_count;

    /* Function definitions */
    function_def_t  functions[PSEUDO_MAX_FUNCTIONS];
    uint32_t        func_count;

    /* Execution stack */
    runtime_value_t stack[PSEUDO_STACK_SIZE];
    uint32_t        stack_top;

    /* Current MEU context */
    model_exec_unit_t *current_meu;

    /* Tensor IR program (for JIT) */
    tir_program_t   ir_program;

    /* Statistics */
    uint64_t        ops_executed;
    uint64_t        tensors_allocated;
    uint64_t        jit_compilations;
} pseudo_runtime_t;

/* =============================================================================
 * API
 * =============================================================================*/

/* Initialization */
int  pseudocode_jit_init(void);
pseudo_runtime_t *pseudo_runtime_create(void);
void pseudo_runtime_destroy(pseudo_runtime_t *rt);

/* Lexer */
int  pseudo_lex(const char *source, token_t *tokens, uint32_t max_tokens,
                 uint32_t *token_count);

/* Parser */
ast_node_t *pseudo_parse(const token_t *tokens, uint32_t token_count);
void         pseudo_ast_free(ast_node_t *node);

/* Type checker and shape inference */
int  pseudo_typecheck(ast_node_t *ast);

/* IR generation */
int  pseudo_lower_to_ir(ast_node_t *ast, tir_program_t *program);

/* Optimization passes */
int  tir_optimize_fuse_ops(tir_program_t *program);
int  tir_optimize_layout(tir_program_t *program);
int  tir_optimize_precision(tir_program_t *program);

/* JIT compilation */
jit_compiled_t *jit_compile(tir_program_t *program, jit_tier_t tier);
int  jit_execute(jit_compiled_t *compiled, runtime_value_t *args,
                  uint32_t arg_count, runtime_value_t *result);

/* Interpreter (fallback) */
int  pseudo_interpret(pseudo_runtime_t *rt, ast_node_t *ast,
                       runtime_value_t *result);

/* High-level execution */
int  pseudo_exec_file(pseudo_runtime_t *rt, const char *path);
int  pseudo_exec_string(pseudo_runtime_t *rt, const char *source);

/* REPL */
int  pseudo_repl(pseudo_runtime_t *rt);

#endif /* TENSOROS_PSEUDOCODE_JIT_H */
