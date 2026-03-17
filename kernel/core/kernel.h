/* =============================================================================
 * TensorOS Kernel - Core Header
 * Type definitions, kernel state, and core API declarations
 * =============================================================================*/

#ifndef TENSOROS_KERNEL_H
#define TENSOROS_KERNEL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* =============================================================================
 * Core Type Definitions
 * =============================================================================*/

/* Tensor data types - native to the kernel */
typedef enum {
    TENSOR_DTYPE_F32   = 0,   /* 32-bit float */
    TENSOR_DTYPE_F16   = 1,   /* 16-bit float (half precision) */
    TENSOR_DTYPE_BF16  = 2,   /* bfloat16 */
    TENSOR_DTYPE_F8    = 3,   /* 8-bit float (FP8) */
    TENSOR_DTYPE_INT8  = 4,   /* 8-bit quantized */
    TENSOR_DTYPE_INT4  = 5,   /* 4-bit quantized */
    TENSOR_DTYPE_F64   = 6,   /* 64-bit double */
    TENSOR_DTYPE_BOOL  = 7,   /* Boolean */
} tensor_dtype_t;

/* Tensor descriptor - kernel-level representation */
typedef struct tensor_desc {
    uint32_t    ndim;              /* Number of dimensions */
    uint64_t    shape[8];          /* Shape (max 8 dimensions) */
    uint64_t    strides[8];        /* Strides in bytes */
    tensor_dtype_t dtype;          /* Data type */
    uint64_t    data_phys;         /* Physical address of data */
    uint64_t    data_virt;         /* Virtual address of data */
    uint64_t    size_bytes;        /* Total size in bytes */
    uint32_t    device_id;         /* Device: 0=CPU, 1+=GPU/TPU */
    uint32_t    flags;             /* TENSOR_FLAG_* */
} tensor_desc_t;

/* Tensor flags */
#define TENSOR_FLAG_CONTIGUOUS  (1 << 0)
#define TENSOR_FLAG_PINNED      (1 << 1)   /* Pinned in memory, no swap */
#define TENSOR_FLAG_READONLY    (1 << 2)
#define TENSOR_FLAG_SHARED      (1 << 3)   /* Shared between models */
#define TENSOR_FLAG_GRADIENT    (1 << 4)   /* This is a gradient tensor */
#define TENSOR_FLAG_DEVICE_MEM  (1 << 5)   /* Resides in GPU/TPU memory */
#define TENSOR_FLAG_CACHED      (1 << 6)   /* Cached model weight */

/* =============================================================================
 * Model Execution Unit (replaces POSIX process for AI workloads)
 * =============================================================================*/

typedef enum {
    MEU_STATE_CREATED    = 0,
    MEU_STATE_LOADING    = 1,  /* Loading model weights */
    MEU_STATE_READY      = 2,  /* Ready for inference/training */
    MEU_STATE_RUNNING    = 3,  /* Actively computing */
    MEU_STATE_WAITING    = 4,  /* Waiting for tensor data */
    MEU_STATE_SUSPENDED  = 5,  /* Suspended by scheduler */
    MEU_STATE_COMPLETED  = 6,  /* Task complete */
    MEU_STATE_ERROR      = 7,
} meu_state_t;

typedef enum {
    MEU_TYPE_INFERENCE  = 0,
    MEU_TYPE_TRAINING   = 1,
    MEU_TYPE_FINETUNE   = 2,
    MEU_TYPE_PIPELINE   = 3,   /* Multi-model pipeline */
    MEU_TYPE_SYSTEM     = 4,   /* System service */
} meu_type_t;

/* Priority levels for AI workload scheduling */
typedef enum {
    MEU_PRIO_REALTIME   = 0,   /* Real-time inference */
    MEU_PRIO_HIGH       = 1,   /* Interactive inference */
    MEU_PRIO_NORMAL     = 2,   /* Batch inference */
    MEU_PRIO_LOW        = 3,   /* Training jobs */
    MEU_PRIO_BACKGROUND = 4,   /* Background optimization */
    MEU_PRIO_IDLE       = 5,   /* Cache warming, etc. */
} meu_priority_t;

typedef struct model_exec_unit {
    uint64_t        meu_id;         /* Unique identifier */
    char            name[64];       /* Human-readable name */
    meu_state_t     state;
    meu_type_t      type;
    meu_priority_t  priority;

    /* Model information */
    uint64_t        model_hash;     /* SHA-256 of model weights */
    uint64_t        param_count;    /* Number of parameters */
    tensor_dtype_t  compute_dtype;  /* Preferred compute precision */

    /* Resource allocation */
    uint32_t        cpu_affinity;   /* CPU core mask */
    uint32_t        gpu_id;         /* Assigned GPU (-1 = any) */
    uint64_t        mem_budget;     /* Memory budget in bytes */
    uint64_t        mem_used;       /* Current memory usage */
    uint64_t        vram_budget;    /* GPU memory budget */
    uint64_t        vram_used;

    /* Execution statistics */
    uint64_t        tensor_ops;     /* Total tensor operations */
    uint64_t        flops;          /* Floating point operations */
    uint64_t        start_tick;     /* Start time */
    uint64_t        cpu_ticks;      /* CPU time consumed */
    uint64_t        gpu_ticks;      /* GPU time consumed */
    uint64_t        inferences;     /* Inference count */

    /* Sandbox context */
    uint64_t        sandbox_id;     /* Security sandbox */
    uint32_t        permissions;    /* MEU_PERM_* flags */

    /* Execution context */
    int           (*exec_fn)(struct model_exec_unit *meu, void *arg);
    void           *exec_arg;       /* Opaque argument for exec_fn */

    /* Linked list for scheduler queues */
    struct model_exec_unit *next;
    struct model_exec_unit *prev;
} model_exec_unit_t;

/* MEU permissions */
#define MEU_PERM_GPU_ACCESS   (1 << 0)
#define MEU_PERM_NETWORK      (1 << 1)
#define MEU_PERM_FILESYSTEM   (1 << 2)
#define MEU_PERM_IPC          (1 << 3)
#define MEU_PERM_MODEL_LOAD   (1 << 4)
#define MEU_PERM_TRAINING     (1 << 5)

/* =============================================================================
 * Kernel State
 * =============================================================================*/

typedef enum {
    KSTATE_BOOT    = 0,
    KSTATE_INIT    = 1,
    KSTATE_RUNNING = 2,
    KSTATE_PANIC   = 3,
} kernel_phase_t;

#define MAX_MEUS 256

struct kernel_state {
    kernel_phase_t  phase;
    uint32_t        cpu_count;
    uint32_t        gpu_count;
    uint32_t        tpu_count;
    uint64_t        tensor_ops_total;
    uint32_t        models_loaded;
    uint64_t        uptime_ticks;
    uint64_t        memory_used_bytes;
    uint64_t        memory_total_bytes;
    uint32_t        meu_count;
    model_exec_unit_t meus[MAX_MEUS];
};

extern struct kernel_state kstate;

/* =============================================================================
 * Linker-provided symbols
 * =============================================================================*/
extern char __text_start[];
extern char __text_end[];
extern char __rodata_start[];
extern char __rodata_end[];
extern char __data_start[];
extern char __data_end[];
extern char __bss_start[];
extern char __bss_end[];
extern char __tensor_heap_start[];
extern char __tensor_heap_end[];
extern char __model_cache_start[];
extern char __model_cache_end[];
extern char __git_objects_start[];
extern char __git_objects_end[];
extern char __kernel_end[];

/* =============================================================================
 * Core Kernel API
 * =============================================================================*/

/* Console output */
void vga_init(void);
int  kprintf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
int  kprintf_debug(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void kpanic(const char *msg) __attribute__((noreturn));

/* Interrupts */
void idt_init(void);
void idt_set_gate(int num, uint64_t handler);
void pic_init(void);
void timer_init(uint32_t freq_hz);

/* CPU Exception Handlers (vectors 0-31) */
void exception_install_handlers(void);

/* Watchdog Timer */
void watchdog_install(void);
void watchdog_set(uint64_t timeout_ms);
void watchdog_kick(void);
void watchdog_disable(void);
uint64_t watchdog_uptime_ms(void);
extern volatile uint64_t watchdog_ticks;

/* Keyboard */
char keyboard_getchar(void);   /* Blocking: waits for keypress */
int  keyboard_has_key(void);   /* Non-blocking: returns 1 if key available */

#if defined(__aarch64__)
#include "kernel/arch/arm64/arm64_hal.h"
static inline void cli(void) { arm_disable_irq(); }
static inline void sti(void) { arm_enable_irq(); }
#else
static inline void cli(void) { __asm__ volatile ("cli"); }
static inline void sti(void) { __asm__ volatile ("sti"); }
#endif

/* CPU */
int  cpu_detect_and_init(void);
void pci_enumerate(void);

/* CPU Feature Detection */
void cpu_detect_features(void);
void cpu_enable_avx(void);
void cpu_print_features(void);

/* Self-Test Suite */
void selftest_run_all(void);

/* AVX2 GEMM */
void avx2_gemm_benchmark(void);

/* Networking (stub and real) */
void net_init(void);

/* kprintf_to_buf (snprintf-like) */
int kprintf_to_buf(char *buf, int buflen, const char *fmt, ...);

/* Utility */
void *kmemset(void *s, int c, size_t n);
void *kmemcpy(void *dest, const void *src, size_t n);
int   kstrcmp(const char *s1, const char *s2);
int   kstrncmp(const char *s1, const char *s2, size_t n);
size_t kstrlen(const char *s);
char *kstrcpy(char *dest, const char *src);
char *kstrncpy(char *dest, const char *src, size_t n);

#if defined(__aarch64__)
/* ARM64: No port I/O — everything is MMIO via arm64_hal.h */
/* Stub out x86 port I/O as no-ops so shared code compiles */
static inline void outb(uint16_t port, uint8_t val)  { (void)port; (void)val; }
static inline uint8_t inb(uint16_t port)             { (void)port; return 0; }
static inline void outw(uint16_t port, uint16_t val) { (void)port; (void)val; }
static inline uint16_t inw(uint16_t port)            { (void)port; return 0; }
static inline void outl(uint16_t port, uint32_t val) { (void)port; (void)val; }
static inline uint32_t inl(uint16_t port)            { (void)port; return 0; }
#else
/* x86 Inline port I/O */
static inline void outw(uint16_t port, uint16_t val) {
    __asm__ volatile ("outw %0, %1" : : "a"(val), "Nd"(port));
}
static inline uint16_t inw(uint16_t port) {
    uint16_t ret;
    __asm__ volatile ("inw %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}
static inline void outb(uint16_t port, uint8_t val) {
    __asm__ volatile ("outb %0, %1" : : "a"(val), "Nd"(port));
}
static inline uint8_t inb(uint16_t port) {
    uint8_t ret;
    __asm__ volatile ("inb %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}
static inline void outl(uint16_t port, uint32_t val) {
    __asm__ volatile ("outl %0, %1" : : "a"(val), "Nd"(port));
}
static inline uint32_t inl(uint16_t port) {
    uint32_t ret;
    __asm__ volatile ("inl %1, %0" : "=a"(ret) : "Nd"(port));
    return ret;
}
#endif

/* Bluetooth SPP console (ARM64 real HW / x86 stubs) */
int  bt_init(void);
void bt_poll(void);
void bt_putchar(char c);
int  bt_has_data(void);
char bt_getchar(void);
int  bt_connected(void);

/* OTA firmware update (ARM64 real / x86 stubs) */
int  ota_receive_and_chainload(void);
int  ota_receive_and_flash(void);

/* SD card driver (ARM64 real / x86 stubs) */
int  sd_init(void);
int  sd_read_sector(uint32_t lba, void *buf);
int  sd_write_sector(uint32_t lba, const void *buf);
int  sd_read_sectors(uint32_t lba, uint32_t count, void *buf);
int  sd_write_sectors(uint32_t lba, uint32_t count, const void *buf);

/* SD card boot logger (ARM64 only — writes BOOTLOG.TXT) */
#if defined(__aarch64__)
#include "kernel/drivers/blk/sdlog.h"
#endif

#endif /* TENSOROS_KERNEL_H */
