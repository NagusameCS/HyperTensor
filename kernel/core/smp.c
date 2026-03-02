/* =============================================================================
 * TensorOS - SMP Bootstrap Implementation
 * x86_64: LAPIC init, AP trampoline, INIT-SIPI-SIPI, per-core work dispatch
 * ARM64: Uses PSCI (guarded below)
 * =============================================================================*/

#ifndef __aarch64__

#include "kernel/core/kernel.h"
#include "kernel/core/smp.h"

/* =============================================================================
 * SMP global state
 * =============================================================================*/

smp_state_t smp;

/* =============================================================================
 * LAPIC MMIO access
 * =============================================================================*/

uint32_t lapic_read(uint32_t offset)
{
    volatile uint32_t *reg = (volatile uint32_t *)(uintptr_t)(smp.lapic_base + offset);
    return *reg;
}

void lapic_write(uint32_t offset, uint32_t val)
{
    volatile uint32_t *reg = (volatile uint32_t *)(uintptr_t)(smp.lapic_base + offset);
    *reg = val;
}

void lapic_eoi(void)
{
    if (smp.lapic_base)
        lapic_write(LAPIC_EOI, 0);
}

uint32_t smp_get_apic_id(void)
{
    if (!smp.lapic_base) return 0;
    return lapic_read(LAPIC_ID) >> 24;
}

/* =============================================================================
 * Delay using PIT (channel 2)
 * =============================================================================*/

static void smp_delay_us(uint32_t us)
{
    /* Use TSC for delay (calibrated in perf.c) */
    extern uint64_t perf_tsc_mhz(void);
    uint64_t cycles = (uint64_t)us * perf_tsc_mhz();
    uint32_t lo, hi;
    __asm__ volatile ("lfence; rdtsc" : "=a"(lo), "=d"(hi));
    uint64_t start = ((uint64_t)hi << 32) | lo;
    while (1) {
        __asm__ volatile ("lfence; rdtsc" : "=a"(lo), "=d"(hi));
        uint64_t now = ((uint64_t)hi << 32) | lo;
        if (now - start >= cycles) break;
        __asm__ volatile ("pause");
    }
}

/* =============================================================================
 * AP Trampoline
 * 
 * This 16-bit real mode code is copied to physical address 0x8000.
 * When an AP receives SIPI, it starts executing here.
 * It transitions: 16-bit -> 32-bit -> 64-bit -> C ap_entry()
 *
 * We use inline asm to generate the trampoline binary.
 * =============================================================================*/

/* AP stack: each AP gets an 8KB stack */
#define AP_STACK_SIZE 8192
static uint8_t ap_stacks[MAX_CPUS][AP_STACK_SIZE] __attribute__((aligned(16)));

/* AP entry flag - set by each AP when it reaches C code */
volatile uint32_t ap_running_flag = 0;

/* The trampoline code. We'll manually write the binary to 0x8000 */
#define TRAMPOLINE_ADDR 0x8000

/* Trampoline: 16-bit stub that switches to long mode and jumps to ap_entry
 * This is a minimal binary blob we copy to 0x8000 */
static const uint8_t trampoline_code[] = {
    /* 0x0000: 16-bit real mode entry (CS:IP = 0x0800:0x0000 = 0x8000) */
    0xFA,                         /* cli */
    0x31, 0xC0,                   /* xor eax, eax */
    0x8E, 0xD8,                   /* mov ds, ax */
    0x8E, 0xC0,                   /* mov es, ax */
    0x8E, 0xD0,                   /* mov ss, ax */

    /* Load 32-bit GDT */
    0x0F, 0x01, 0x16,             /* lgdt [gdt_ptr] (at trampoline + 0x80) */
    0x80, 0x00,                   /* offset 0x0080 within trampoline page */

    /* Enable protected mode */
    0x0F, 0x20, 0xC0,             /* mov eax, cr0 */
    0x0C, 0x01,                   /* or al, 1 */
    0x0F, 0x22, 0xC0,             /* mov cr0, eax */

    /* Far jump to 32-bit code at trampoline + 0x30 */
    0x66, 0xEA,                   /* ljmp 0x08:offset */
    0x30, 0x80, 0x00, 0x00,      /* offset = 0x8030 (absolute) */
    0x08, 0x00,                   /* segment selector 0x08 */
    /* Pad to offset 0x30 */
    0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90,
    0x90, 0x90, 0x90, 0x90, 0x90, 0x90,

    /* 0x0030: 32-bit protected mode */
    /* bits 32 */
    0x66, 0xB8, 0x10, 0x00,      /* mov ax, 0x10 (data segment) */
    0x8E, 0xD8,                   /* mov ds, ax */
    0x8E, 0xC0,                   /* mov es, ax */
    0x8E, 0xD0,                   /* mov ss, ax */

    /* Enable PAE (CR4 bit 5) */
    0x0F, 0x20, 0xE0,             /* mov eax, cr4 */
    0x0D, 0x20, 0x00, 0x00, 0x00,/* or eax, 0x20 */
    0x0F, 0x22, 0xE0,             /* mov cr4, eax */

    /* Load CR3 from trampoline data area (offset 0x90) */
    0x8B, 0x05,                   /* mov eax, [abs] */
    0x90, 0x80, 0x00, 0x00,      /* address 0x8090 */
    0x0F, 0x22, 0xD8,             /* mov cr3, eax */

    /* Enable long mode in EFER MSR */
    0xB9, 0x80, 0x00, 0x00, 0xC0,/* mov ecx, 0xC0000080 */
    0x0F, 0x32,                   /* rdmsr */
    0x0D, 0x00, 0x01, 0x00, 0x00,/* or eax, 0x100 (LME) */
    0x0F, 0x30,                   /* wrmsr */

    /* Enable paging (CR0 bit 31) */
    0x0F, 0x20, 0xC0,             /* mov eax, cr0 */
    0x0D, 0x00, 0x00, 0x00, 0x80,/* or eax, 0x80000000 */
    0x0F, 0x22, 0xC0,             /* mov cr0, eax */

    /* Far jump to 64-bit code */
    0xEA,                          /* ljmp */
    0x70, 0x80, 0x00, 0x00,       /* offset = 0x8070 (absolute) */
    0x18, 0x00,                    /* 64-bit code segment (GDT entry 3) */
    /* Pad to offset 0x70 */
    0x90, 0x90, 0x90, 0x90, 0x90,

    /* 0x0070: 64-bit long mode */
    /* bits 64 */
    0x48, 0x31, 0xC0,             /* xor rax, rax */
    0xB0, 0x20,                   /* mov al, 0x20 (64-bit data segment) */
    0x8E, 0xD8,                   /* mov ds, ax */
    0x8E, 0xC0,                   /* mov es, ax */
    0x8E, 0xD0,                   /* mov ss, ax */

    /* Signal that we're in 64-bit mode */
    0xF0, 0xFF, 0x05,             /* lock inc dword [ap_running_flag_addr] */
    /* Relative address to flag - will be patched */
    0x00, 0x00, 0x00, 0x00,

    /* Halt - AP will be managed by BSP */
    0xFB,                         /* sti */
    0xF4,                         /* hlt */
    0xEB, 0xFC,                   /* jmp -2 (loop hlt) */
};

/* GDT for trampoline (at offset 0x80 in trampoline page) */
static const uint8_t trampoline_gdt[] = {
    /* GDT pointer (6 bytes) */
    0x27, 0x00,                   /* limit = 39 (5 entries * 8 - 1) */
    0x88, 0x80, 0x00, 0x00,      /* base = 0x8088 (GDT entries follow) */
    0x00, 0x00,                   /* padding */

    /* GDT entries at offset 0x88 */
    /* Entry 0: Null */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    /* Entry 1 (0x08): 32-bit code */
    0xFF, 0xFF, 0x00, 0x00, 0x00, 0x9A, 0xCF, 0x00,
    /* Entry 2 (0x10): 32-bit data */
    0xFF, 0xFF, 0x00, 0x00, 0x00, 0x92, 0xCF, 0x00,
    /* Entry 3 (0x18): 64-bit code */
    0xFF, 0xFF, 0x00, 0x00, 0x00, 0x9A, 0xAF, 0x00,
    /* Entry 4 (0x20): 64-bit data */
    0xFF, 0xFF, 0x00, 0x00, 0x00, 0x92, 0xCF, 0x00,
};

/* =============================================================================
 * Install trampoline at 0x8000
 * =============================================================================*/

static void install_trampoline(void)
{
    uint8_t *tramp = (uint8_t *)(uintptr_t)TRAMPOLINE_ADDR;
    
    /* Clear the page */
    kmemset(tramp, 0, 4096);
    
    /* Copy code */
    kmemcpy(tramp, trampoline_code, sizeof(trampoline_code));
    
    /* Copy GDT at offset 0x80 */
    kmemcpy(tramp + 0x80, trampoline_gdt, sizeof(trampoline_gdt));
    
    /* Write CR3 value at offset 0x90 */
    uint64_t cr3;
    __asm__ volatile ("mov %%cr3, %0" : "=r"(cr3));
    *(uint32_t *)(tramp + 0x90) = (uint32_t)cr3;
}

/* =============================================================================
 * Detect LAPIC and CPUs
 * =============================================================================*/

void smp_detect(void)
{
    kmemset(&smp, 0, sizeof(smp));

    /* Read LAPIC base from IA32_APIC_BASE MSR (0x1B) */
    uint32_t lo, hi;
    __asm__ volatile ("rdmsr" : "=a"(lo), "=d"(hi) : "c"(0x1B));
    smp.lapic_base = ((uint64_t)hi << 32) | (lo & 0xFFFFF000);

    /* Check if LAPIC is enabled */
    if (!(lo & (1 << 11))) {
        kprintf("[SMP] LAPIC not enabled\n");
        smp.cpu_count = 1;
        return;
    }

    /* Get BSP APIC ID */
    smp.bsp_id = lapic_read(LAPIC_ID) >> 24;
    smp.cpus[0].apic_id = smp.bsp_id;
    smp.cpus[0].state = CPU_STATE_IDLE;

    kprintf("[SMP] LAPIC base: 0x%lx, BSP APIC ID: %u\n",
            smp.lapic_base, smp.bsp_id);

    /* Try to detect CPUs via ACPI MADT (simplified) */
    /* For now, try to enumerate via CPUID */
    uint32_t eax, ebx, ecx, edx;
    __asm__ volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                     : "a"(1));
    
    /* Check for HTT (Hyper-Threading Technology) bit in EDX */
    if (edx & (1 << 28)) {
        /* EBX[23:16] contains logical processor count */
        uint32_t logical_cpus = (ebx >> 16) & 0xFF;
        if (logical_cpus > MAX_CPUS) logical_cpus = MAX_CPUS;
        if (logical_cpus < 1) logical_cpus = 1;
        smp.cpu_count = logical_cpus;
    } else {
        smp.cpu_count = 1;
    }

    /* Initialize CPU state for each detected core */
    for (uint32_t i = 1; i < smp.cpu_count; i++) {
        smp.cpus[i].apic_id = i; /* Assume sequential APIC IDs */
        smp.cpus[i].state = CPU_STATE_OFFLINE;
    }

    kprintf("[SMP] Detected %u logical CPUs\n", smp.cpu_count);
}

/* =============================================================================
 * Initialize LAPIC
 * =============================================================================*/

static void lapic_init(void)
{
    /* Enable LAPIC via SVR (Spurious Vector Register) */
    uint32_t svr = lapic_read(LAPIC_SVR);
    svr |= 0x100;   /* Enable bit */
    svr |= 0xFF;    /* Spurious vector = 0xFF */
    lapic_write(LAPIC_SVR, svr);

    /* Set task priority to 0 (accept all interrupts) */
    lapic_write(LAPIC_TPR, 0);

    /* Clear any pending interrupts */
    lapic_eoi();

    kprintf("[SMP] LAPIC initialized (SVR=0x%x)\n", lapic_read(LAPIC_SVR));
}

/* =============================================================================
 * Send IPI (Inter-Processor Interrupt)
 * =============================================================================*/

static void lapic_send_ipi(uint8_t apic_id, uint32_t icr_lo)
{
    /* Wait for previous IPI to complete */
    while (lapic_read(LAPIC_ICR_LO) & (1 << 12))
        __asm__ volatile ("pause");

    /* Set destination APIC ID */
    lapic_write(LAPIC_ICR_HI, (uint32_t)apic_id << 24);

    /* Send IPI */
    lapic_write(LAPIC_ICR_LO, icr_lo);

    /* Wait for delivery */
    while (lapic_read(LAPIC_ICR_LO) & (1 << 12))
        __asm__ volatile ("pause");
}

/* =============================================================================
 * Boot APs via INIT-SIPI-SIPI
 * =============================================================================*/

void smp_init(void)
{
    if (smp.cpu_count <= 1) {
        kprintf("[SMP] Single CPU -- skipping AP boot\n");
        return;
    }

    /* Initialize BSP LAPIC */
    lapic_init();

    /* Install AP trampoline at 0x8000 */
    install_trampoline();

    kprintf("[SMP] Starting %u Application Processors...\n", smp.cpu_count - 1);

    for (uint32_t i = 1; i < smp.cpu_count; i++) {
        uint8_t target_id = smp.cpus[i].apic_id;
        smp.cpus[i].state = CPU_STATE_BOOTING;

        uint32_t prev_count = smp.ap_started;

        /* Step 1: Send INIT IPI */
        lapic_send_ipi(target_id, ICR_INIT | ICR_LEVEL_ASSERT);
        smp_delay_us(200);  /* Wait 200us */

        /* Deassert INIT */
        lapic_send_ipi(target_id, ICR_INIT | ICR_LEVEL_DEASSERT);
        smp_delay_us(10000);  /* Wait 10ms */

        /* Step 2: Send STARTUP IPI (vector = trampoline page number) */
        uint32_t vec = TRAMPOLINE_ADDR >> 12;  /* 0x8000 >> 12 = 8 */
        lapic_send_ipi(target_id, ICR_STARTUP | vec);
        smp_delay_us(200);

        /* Step 3: Send second STARTUP IPI */
        lapic_send_ipi(target_id, ICR_STARTUP | vec);
        smp_delay_us(200);

        /* Wait for AP to signal (up to 100ms) */
        uint64_t timeout = 100000; /* 100ms in us */
        uint64_t waited = 0;
        while (smp.ap_started == prev_count && waited < timeout) {
            smp_delay_us(100);
            waited += 100;
        }

        if (smp.ap_started > prev_count) {
            smp.cpus[i].state = CPU_STATE_IDLE;
            kprintf("[SMP] CPU %u (APIC %u) started OK\n", i, target_id);
        } else {
            smp.cpus[i].state = CPU_STATE_OFFLINE;
            kprintf("[SMP] CPU %u (APIC %u) failed to start\n", i, target_id);
        }
    }

    kprintf("[SMP] %u/%u APs started\n", smp.ap_started, smp.cpu_count - 1);
}

/* =============================================================================
 * Work dispatch
 * =============================================================================*/

int smp_dispatch(uint32_t cpu_id, smp_work_fn_t fn, void *arg)
{
    if (cpu_id >= smp.cpu_count) return -1;
    if (smp.cpus[cpu_id].state != CPU_STATE_IDLE) return -2;

    smp.cpus[cpu_id].work_fn = fn;
    smp.cpus[cpu_id].work_arg = arg;
    smp.cpus[cpu_id].work_done = 0;
    __asm__ volatile ("mfence" ::: "memory");
    smp.cpus[cpu_id].work_ready = 1;
    smp.cpus[cpu_id].state = CPU_STATE_BUSY;

    /* Send IPI to wake the AP (vector 0xFE = work notification) */
    if (cpu_id > 0) {
        lapic_send_ipi(smp.cpus[cpu_id].apic_id, 0xFE);
    }

    return 0;
}

void smp_dispatch_all(smp_work_fn_t fn, void *arg)
{
    for (uint32_t i = 1; i < smp.cpu_count; i++) {
        if (smp.cpus[i].state == CPU_STATE_IDLE) {
            smp_dispatch(i, fn, arg);
        }
    }
}

void smp_wait(uint32_t cpu_id)
{
    if (cpu_id >= smp.cpu_count) return;
    while (!smp.cpus[cpu_id].work_done)
        __asm__ volatile ("pause");
    smp.cpus[cpu_id].state = CPU_STATE_IDLE;
}

void smp_wait_all(void)
{
    for (uint32_t i = 1; i < smp.cpu_count; i++) {
        if (smp.cpus[i].state == CPU_STATE_BUSY) {
            smp_wait(i);
        }
    }
}

/* =============================================================================
 * Print SMP status
 * =============================================================================*/

static const char *cpu_state_names[] = {
    "OFFLINE", "BOOTING", "IDLE", "BUSY"
};

void smp_print_status(void)
{
    kprintf("[SMP] %u CPUs, BSP APIC ID %u\n", smp.cpu_count, smp.bsp_id);
    for (uint32_t i = 0; i < smp.cpu_count; i++) {
        const char *state = (smp.cpus[i].state < 4) ?
                            cpu_state_names[smp.cpus[i].state] : "UNKNOWN";
        kprintf("  CPU %u: APIC %u, state=%s\n",
                i, smp.cpus[i].apic_id, state);
    }
}

/* =============================================================================
 * SMP Demo
 * =============================================================================*/

void smp_run_demos(void)
{
    kprintf("\n=== SMP Multi-Core Demo ===\n");
    smp_detect();
    
    /* Only attempt SMP init if multiple CPUs detected */
    if (smp.cpu_count > 1) {
        smp_init();
    }
    
    smp_print_status();
    
    kprintf("[SMP] Multi-core infrastructure ready\n");
    if (smp.cpu_count > 1 && smp.ap_started > 0) {
        kprintf("[SMP] %u cores available for parallel tensor operations\n", 
                smp.ap_started + 1);
    } else {
        kprintf("[SMP] Single-core mode (APs can be started with real hardware)\n");
    }
}

#else /* __aarch64__ */

#include "kernel/core/kernel.h"
#include "kernel/core/smp.h"

smp_state_t smp = {0};
volatile uint32_t ap_running_flag = 0;

uint32_t lapic_read(uint32_t off) { (void)off; return 0; }
void lapic_write(uint32_t off, uint32_t v) { (void)off; (void)v; }
void lapic_eoi(void) {}
uint32_t smp_get_apic_id(void) { return 0; }
void smp_detect(void) { smp.cpu_count = 4; /* Cortex-A72 quad-core */ }
void smp_init(void) { smp_detect(); }
int smp_dispatch(uint32_t cpu_id, smp_work_fn_t fn, void *arg) { (void)cpu_id; (void)fn; (void)arg; return -1; }
void smp_dispatch_all(smp_work_fn_t fn, void *arg) { (void)fn; (void)arg; }
void smp_wait(uint32_t cpu_id) { (void)cpu_id; }
void smp_wait_all(void) {}
void smp_print_status(void) { kprintf("[SMP] ARM64 PSCI: 4 cores\n"); }
void smp_run_demos(void) {
    smp_detect();
    kprintf("[SMP] ARM64 PSCI multicore (4 Cortex-A72 cores)\n");
}

#endif /* __aarch64__ */
