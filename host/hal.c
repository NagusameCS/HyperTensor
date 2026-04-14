/*
 * HyperTensor — Hosted HAL Implementation
 *
 * Maps kernel APIs to Windows/Linux standard library calls.
 * Provides threading via Win32 threads / pthreads, JIT via VirtualAlloc/mmap.
 */
#define _CRT_SECURE_NO_WARNINGS
#include "hal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <intrin.h>
#else
  #include <pthread.h>
  #include <sys/mman.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <sys/stat.h>
  #ifdef __x86_64__
    #include <x86intrin.h>
  #endif
#endif

#ifdef __x86_64__
  #include <cpuid.h>
#endif

/* ════════════════════════════════════════════════════════════════════════
 * Globals
 * ════════════════════════════════════════════════════════════════════════ */

smp_state_t    smp;
cpu_features_t cpu_features;
int            vmm_hypervisor_active = 0;

/* ════════════════════════════════════════════════════════════════════════
 * Memory
 * ════════════════════════════════════════════════════════════════════════ */

void *kmalloc(uint64_t size)  { return malloc((size_t)size); }
void  kfree(void *ptr)        { free(ptr); }

void *kmemcpy(void *d, const void *s, size_t n) { return memcpy(d, s, n); }
void *kmemset(void *s, int c, size_t n)          { return memset(s, c, n); }
int   kmemcmp(const void *a, const void *b, size_t n) { return memcmp(a, b, n); }

static void *aligned_alloc_impl(uint64_t size) {
    /* 64-byte aligned for AVX-512 compatibility */
    size_t sz = (size_t)size;
    if (sz == 0) sz = 1;
#ifdef _WIN32
    return _aligned_malloc(sz, 64);
#else
    void *p = NULL;
    posix_memalign(&p, 64, sz);
    return p;
#endif
}

static void aligned_free_impl(void *ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void *tensor_alloc(uint64_t size)        { return aligned_alloc_impl(size); }
void *tensor_alloc_pinned(uint64_t size) { return aligned_alloc_impl(size); }
void *tensor_alloc_dma(uint64_t size)    { return aligned_alloc_impl(size); }
void *tensor_alloc_shared(uint64_t size) { return aligned_alloc_impl(size); }
void  tensor_free(void *ptr)             { aligned_free_impl(ptr); }

void *tensor_mm_realloc(void *ptr, uint64_t new_size) {
#ifdef _WIN32
    return _aligned_realloc(ptr, (size_t)new_size, 64);
#else
    /* No posix_memalign realloc — allocate new, copy, free */
    void *np = NULL;
    posix_memalign(&np, 64, (size_t)new_size);
    if (np && ptr) {
        /* Copy conservatively — caller must know old size */
        memcpy(np, ptr, (size_t)new_size);
        free(ptr);
    }
    return np;
#endif
}

uint64_t tensor_mm_free_bytes(void) {
    /* In hosted mode, report a generous amount for allocation decisions */
    return 8ULL * 1024 * 1024 * 1024; /* 8 GB */
}

void *tensor_mm_model_cache_base(void) {
    /* Not used in hosted mode (llm_load_from_buffer sets data_buf directly) */
    return NULL;
}

uint64_t tensor_mm_model_cache_max(void) {
    return 8ULL * 1024 * 1024 * 1024; /* 8 GB */
}

/* ════════════════════════════════════════════════════════════════════════
 * Console I/O
 * ════════════════════════════════════════════════════════════════════════ */

int kprintf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
    return r;
}

int kprintf_debug(const char *fmt, ...) {
    if (g_log_level < LOG_DEBUG) { (void)fmt; return 0; }
    va_list ap;
    va_start(ap, fmt);
    int r = vprintf(fmt, ap);
    va_end(ap);
    fflush(stdout);
    return r;
}

void kpanic(const char *msg) {
    fprintf(stderr, "\n[PANIC] %s\n", msg);
    fflush(stderr);
    abort();
}

/* ── Structured Logging ─────────────────────────────────────────────────── */

volatile int g_log_level = LOG_INFO;

static const char *log_level_prefix[] = {
    "\033[31mERR ",   /* red */
    "\033[33mWARN",   /* yellow */
    "\033[0mINFO",   /* default */
    "\033[2mDBG ",   /* dim */
    "\033[2mTRC ",   /* dim */
};

void klog(log_level_t level, const char *tag, const char *fmt, ...) {
    if ((int)level > g_log_level) return;
    printf("%s\033[0m [%s] ", log_level_prefix[level], tag);
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
    printf("\n");
    fflush(stdout);
}

void klog_set_level(log_level_t level) {
    g_log_level = (int)level;
}

/* ════════════════════════════════════════════════════════════════════════
 * Strings
 * ════════════════════════════════════════════════════════════════════════ */

size_t kstrlen(const char *s)                          { return strlen(s); }
char  *kstrcpy(char *d, const char *s)                 { return strcpy(d, s); }
int    kstrcmp(const char *a, const char *b)           { return strcmp(a, b); }
int    kstrncmp(const char *a, const char *b, size_t n){ return strncmp(a, b, n); }
char  *kstrncpy(char *d, const char *s, size_t n)      { return strncpy(d, s, n); }
char  *kstrstr(const char *h, const char *n)           { return (char *)strstr(h, n); }

size_t kstrlcpy(char *dest, const char *src, size_t size) {
    size_t slen = strlen(src);
    if (size > 0) {
        size_t n = (slen >= size) ? size - 1 : slen;
        memcpy(dest, src, n);
        dest[n] = '\0';
    }
    return slen;
}

/* ════════════════════════════════════════════════════════════════════════
 * SMP / Threading
 * ════════════════════════════════════════════════════════════════════════ */

typedef struct {
    smp_work_fn_t fn;
    void         *arg;
    volatile int  ready;
    volatile int  done;
    volatile int  quit;
#ifdef _WIN32
    HANDLE        thread;
    HANDLE        wake_event;
#else
    pthread_t     thread;
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
#endif
} worker_t;

static worker_t workers[MAX_CPUS];
static uint32_t n_workers = 0;

#ifdef _WIN32

static DWORD WINAPI worker_thread(LPVOID arg) {
    worker_t *w = (worker_t *)arg;
    for (;;) {
        WaitForSingleObject(w->wake_event, INFINITE);
        if (w->quit) break;
        if (w->fn) w->fn(w->arg);
        w->done = 1;
    }
    return 0;
}

#else

static void *worker_thread(void *arg) {
    worker_t *w = (worker_t *)arg;
    pthread_mutex_lock(&w->mutex);
    for (;;) {
        while (!w->ready && !w->quit)
            pthread_cond_wait(&w->cond, &w->mutex);
        if (w->quit) break;
        w->ready = 0;
        pthread_mutex_unlock(&w->mutex);
        if (w->fn) w->fn(w->arg);
        pthread_mutex_lock(&w->mutex);
        w->done = 1;
    }
    pthread_mutex_unlock(&w->mutex);
    return NULL;
}

#endif

int smp_init_hosted(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    uint32_t ncpu = si.dwNumberOfProcessors;
#else
    uint32_t ncpu = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
#endif

    if (ncpu > MAX_CPUS) ncpu = MAX_CPUS;
    n_workers = ncpu - 1;  /* BSP = main thread */

    smp.cpu_count  = ncpu;
    smp.ap_started = n_workers;
    smp.bsp_id     = 0;

    for (uint32_t i = 0; i < n_workers; i++) {
        worker_t *w = &workers[i];
        memset(w, 0, sizeof(*w));

#ifdef _WIN32
        w->wake_event = CreateEvent(NULL, FALSE, FALSE, NULL);
        w->thread = CreateThread(NULL, 0, worker_thread, w, 0, NULL);
#else
        pthread_mutex_init(&w->mutex, NULL);
        pthread_cond_init(&w->cond, NULL);
        pthread_create(&w->thread, NULL, worker_thread, w);
#endif

        smp.cpus[i + 1].id = i + 1;
        smp.cpus[i + 1].state = 1;  /* running */
    }

    kprintf("[SMP] %u CPUs online (%u workers + BSP)\n", ncpu, n_workers);
    return (int)ncpu;
}

int smp_dispatch(uint32_t cpu_id, smp_work_fn_t fn, void *arg) {
    if (cpu_id == 0 || cpu_id > n_workers) return -1;
    worker_t *w = &workers[cpu_id - 1];
    w->fn   = fn;
    w->arg  = arg;
    w->done = 0;

#ifdef _WIN32
    SetEvent(w->wake_event);
#else
    pthread_mutex_lock(&w->mutex);
    w->ready = 1;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mutex);
#endif

    return 0;
}

void smp_wait_all(void) {
    for (uint32_t i = 0; i < n_workers; i++) {
        while (!workers[i].done) {
#ifdef _WIN32
            _mm_pause();
#else
            __asm__ volatile("pause" ::: "memory");
#endif
        }
    }
}

void smp_shutdown(void) {
    for (uint32_t i = 0; i < n_workers; i++) {
        workers[i].quit = 1;
#ifdef _WIN32
        SetEvent(workers[i].wake_event);
        WaitForSingleObject(workers[i].thread, INFINITE);
        CloseHandle(workers[i].thread);
        CloseHandle(workers[i].wake_event);
#else
        pthread_mutex_lock(&workers[i].mutex);
        pthread_cond_signal(&workers[i].cond);
        pthread_mutex_unlock(&workers[i].mutex);
        pthread_join(workers[i].thread, NULL);
        pthread_mutex_destroy(&workers[i].mutex);
        pthread_cond_destroy(&workers[i].cond);
#endif
    }
    n_workers = 0;
    smp.ap_started = 0;
}

/* ════════════════════════════════════════════════════════════════════════
 * CPU Feature Detection
 * ════════════════════════════════════════════════════════════════════════ */

void cpu_features_detect(void) {
    memset(&cpu_features, 0, sizeof(cpu_features));

#if defined(__x86_64__) || defined(_M_X64)
    uint32_t eax, ebx, ecx, edx;

  #ifdef _WIN32
    int info[4];
    __cpuid(info, 1);
    eax = info[0]; ebx = info[1]; ecx = info[2]; edx = info[3];
  #else
    __cpuid(1, eax, ebx, ecx, edx);
  #endif

    cpu_features.has_sse2   = !!(edx & (1 << 26));
    cpu_features.has_sse3   = !!(ecx & (1 <<  0));
    cpu_features.has_ssse3  = !!(ecx & (1 <<  9));
    cpu_features.has_sse41  = !!(ecx & (1 << 19));
    cpu_features.has_sse42  = !!(ecx & (1 << 20));
    cpu_features.has_avx    = !!(ecx & (1 << 28));
    cpu_features.has_fma    = !!(ecx & (1 << 12));
    cpu_features.has_xsave  = !!(ecx & (1 << 26));

  #ifdef _WIN32
    __cpuidex(info, 7, 0);
    ebx = info[1];
  #else
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
  #endif

    cpu_features.has_avx2    = !!(ebx & (1 << 5));
    cpu_features.has_avx512f = !!(ebx & (1 << 16));
    cpu_features.avx2_usable = cpu_features.has_avx2 && cpu_features.has_fma;

#elif defined(__aarch64__)
    /* ARM64 — NEON is mandatory */
    cpu_features.has_sse2 = false;
    cpu_features.avx2_usable = false;
#endif

    kprintf("[CPU] SSE2=%d AVX2=%d FMA=%d AVX512=%d\n",
            cpu_features.has_sse2, cpu_features.has_avx2,
            cpu_features.has_fma, cpu_features.has_avx512f);
}

/* ════════════════════════════════════════════════════════════════════════
 * Virtual Memory / JIT W^X
 * ════════════════════════════════════════════════════════════════════════ */

int vmm_mark_rx(void *addr, uint32_t size) {
#ifdef _WIN32
    DWORD old;
    return VirtualProtect(addr, size, PAGE_EXECUTE_READ, &old) ? 0 : -1;
#else
    /* Align down to page boundary */
    uintptr_t page = (uintptr_t)addr & ~(uintptr_t)0xFFF;
    size_t len = size + ((uintptr_t)addr - page);
    return mprotect((void *)page, len, PROT_READ | PROT_EXEC);
#endif
}

int vmm_mark_rw(void *addr, uint32_t size) {
#ifdef _WIN32
    DWORD old;
    return VirtualProtect(addr, size, PAGE_READWRITE, &old) ? 0 : -1;
#else
    uintptr_t page = (uintptr_t)addr & ~(uintptr_t)0xFFF;
    size_t len = size + ((uintptr_t)addr - page);
    return mprotect((void *)page, len, PROT_READ | PROT_WRITE);
#endif
}

/* ════════════════════════════════════════════════════════════════════════
 * Performance Counters
 * ════════════════════════════════════════════════════════════════════════ */

uint64_t hal_rdtsc(void) {
#if defined(__x86_64__) || defined(_M_X64)
  #ifdef _WIN32
    return __rdtsc();
  #else
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
  #endif
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
#endif
}

uint64_t hal_timer_us(void) {
#ifdef _WIN32
    static LARGE_INTEGER freq = {0};
    if (freq.QuadPart == 0) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    return (uint64_t)(now.QuadPart * 1000000 / freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
#endif
}

/* ════════════════════════════════════════════════════════════════════════
 * Perf: cycle-to-microsecond conversion
 * ════════════════════════════════════════════════════════════════════════ */

static uint64_t tsc_freq_mhz = 0;

static void calibrate_tsc(void) {
    if (tsc_freq_mhz > 0) return;
    uint64_t t0_us = hal_timer_us();
    uint64_t c0    = hal_rdtsc();
    /* Yield for ~2ms via a volatile loop (avoid sleep calls) */
    volatile int spin = 0;
    for (int i = 0; i < 5000000; i++) spin += i;
    (void)spin;
    uint64_t c1    = hal_rdtsc();
    uint64_t t1_us = hal_timer_us();
    uint64_t dt = t1_us - t0_us;
    if (dt > 0)
        tsc_freq_mhz = (c1 - c0) / dt;
    if (tsc_freq_mhz == 0) tsc_freq_mhz = 3000; /* fallback 3 GHz */
}

uint64_t perf_cycles_to_us(uint64_t cycles) {
    if (tsc_freq_mhz == 0) calibrate_tsc();
    return cycles / tsc_freq_mhz;
}

/* ════════════════════════════════════════════════════════════════════════
 * Cryptographic RNG
 * ════════════════════════════════════════════════════════════════════════ */

void crypto_random(void *buf, uint32_t len) {
#ifdef _WIN32
    /* Use RtlGenRandom (SystemFunction036) */
    extern BOOLEAN NTAPI SystemFunction036(PVOID, ULONG);
    SystemFunction036(buf, len);
#else
    /* Read from /dev/urandom */
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        size_t done = 0;
        while (done < len) {
            ssize_t r = read(fd, (char *)buf + done, len - done);
            if (r <= 0) break;
            done += (size_t)r;
        }
        close(fd);
    }
#endif
}

/* ════════════════════════════════════════════════════════════════════════
 * File I/O
 * ════════════════════════════════════════════════════════════════════════ */

hal_mmap_t hal_mmap_file(const char *path) {
    hal_mmap_t m = {NULL, 0};

#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        kprintf("[HAL] Cannot open: %s\n", path);
        return m;
    }
    LARGE_INTEGER sz;
    GetFileSizeEx(hFile, &sz);
    m.size = (uint64_t)sz.QuadPart;

    HANDLE hMap = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMap) {
        CloseHandle(hFile);
        kprintf("[HAL] Cannot map: %s\n", path);
        return m;
    }
    m.data = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);

    if (!m.data) {
        kprintf("[HAL] MapViewOfFile failed: %s\n", path);
        m.size = 0;
    }
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        kprintf("[HAL] Cannot open: %s\n", path);
        return m;
    }
    struct stat st;
    fstat(fd, &st);
    m.size = (uint64_t)st.st_size;
    m.data = mmap(NULL, m.size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m.data == MAP_FAILED) {
        m.data = NULL;
        m.size = 0;
    }
#endif

    return m;
}

void hal_munmap(hal_mmap_t *m) {
    if (!m || !m->data) return;
#ifdef _WIN32
    UnmapViewOfFile(m->data);
#else
    munmap(m->data, m->size);
#endif
    m->data = NULL;
    m->size = 0;
}

/* ════════════════════════════════════════════════════════════════════════
 * Init / Shutdown
 * ════════════════════════════════════════════════════════════════════════ */

void hal_init(void) {
#ifdef _WIN32
    /* Enable ANSI escape codes on Windows console */
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode;
    GetConsoleMode(hOut, &mode);
    SetConsoleMode(hOut, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#endif

    cpu_features_detect();
    smp_init_hosted();
}

void hal_shutdown(void) {
    smp_shutdown();
}
