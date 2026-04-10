/*
 * HyperTensor — Hardware Abstraction Layer
 *
 * Maps bare-metal kernel APIs to host OS equivalents (Windows/Linux/macOS).
 * Include this instead of kernel headers when building in hosted mode.
 */
#ifndef HYPERTENSOR_HAL_H
#define HYPERTENSOR_HAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Memory ────────────────────────────────────────────────────────────── */

void *kmalloc(uint64_t size);
void  kfree(void *ptr);
void *kmemcpy(void *dest, const void *src, size_t n);
void *kmemset(void *s, int c, size_t n);
int   kmemcmp(const void *s1, const void *s2, size_t n);

/* Tensor-specific allocators (all map to aligned malloc in hosted mode) */
void *tensor_alloc(uint64_t size);
void *tensor_alloc_pinned(uint64_t size);
void *tensor_alloc_dma(uint64_t size);
void *tensor_alloc_shared(uint64_t size);
void  tensor_free(void *ptr);
void *tensor_mm_realloc(void *ptr, uint64_t new_size);

/* Tensor memory pool info (hosted: returns system memory estimates) */
uint64_t tensor_mm_free_bytes(void);
void    *tensor_mm_model_cache_base(void);
uint64_t tensor_mm_model_cache_max(void);

/* ── Console I/O ───────────────────────────────────────────────────────── */

int   kprintf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
int   kprintf_debug(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
void  kpanic(const char *msg) __attribute__((noreturn));

/* ── Strings ───────────────────────────────────────────────────────────── */

size_t kstrlen(const char *s);
char  *kstrcpy(char *dest, const char *src);
int    kstrcmp(const char *a, const char *b);
int    kstrncmp(const char *a, const char *b, size_t n);
char  *kstrncpy(char *dest, const char *src, size_t n);
char  *kstrstr(const char *haystack, const char *needle);
size_t kstrlcpy(char *dest, const char *src, size_t size);

/* ── SMP / Threading ───────────────────────────────────────────────────── */

#define MAX_CPUS 64

typedef void (*smp_work_fn_t)(void *arg);

typedef struct {
    uint32_t id;
    volatile uint32_t state;
} smp_cpu_t;

typedef struct {
    uint64_t           lapic_base;
    uint32_t           bsp_id;
    uint32_t           cpu_count;
    smp_cpu_t          cpus[MAX_CPUS];
    volatile uint32_t  ap_started;
} smp_state_t;

extern smp_state_t smp;

int   smp_init_hosted(void);
int   smp_dispatch(uint32_t cpu_id, smp_work_fn_t fn, void *arg);
void  smp_wait_all(void);
void  smp_shutdown(void);

/* ── CPU Features ──────────────────────────────────────────────────────── */

typedef struct {
    bool has_sse2;
    bool has_sse3, has_ssse3, has_sse41, has_sse42;
    bool has_avx, has_avx2, has_fma, has_avx512f;
    bool has_xsave;
    bool avx2_usable;
} cpu_features_t;

extern cpu_features_t cpu_features;

void cpu_features_detect(void);

/* ── Virtual Memory / JIT support ──────────────────────────────────────── */

int  vmm_mark_rx(void *addr, uint32_t size);
int  vmm_mark_rw(void *addr, uint32_t size);
extern int vmm_hypervisor_active;

/* ── Performance Counters ──────────────────────────────────────────────── */

uint64_t hal_rdtsc(void);
uint64_t hal_timer_us(void);   /* Microsecond wall-clock timer */
uint64_t perf_cycles_to_us(uint64_t cycles);

/* ── Cryptographic RNG ─────────────────────────────────────────────────── */

void crypto_random(void *buf, uint32_t len);

/* ── File I/O (hosted only) ────────────────────────────────────────────── */

typedef struct {
    void    *data;
    uint64_t size;
} hal_mmap_t;

hal_mmap_t hal_mmap_file(const char *path);
void       hal_munmap(hal_mmap_t *m);

/* ── Init / Shutdown ───────────────────────────────────────────────────── */

void hal_init(void);
void hal_shutdown(void);

/* ── Crypto ────────────────────────────────────────────────────────────── */

void crypto_random(void *buf, uint32_t len);

#ifdef __cplusplus
}
#endif

#endif /* HYPERTENSOR_HAL_H */
