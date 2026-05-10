/* =============================================================================
 * TensorOS ARM64 Boot Stub for Raspberry Pi 4 (BCM2711)
 *
 * ULTRA-MINIMAL — identity-map MMU, caches ON for Normal memory.
 * Peripherals mapped as Device-nGnRnE for MMIO safety.
 *
 * The RPi GPU firmware loads kernel8.img to 0x80000 in EL2 (AArch64).
 * This stub:
 *   1. Parks secondary cores
 *   2. Configures GPIO 42 (ACT LED) immediately
 *   3. Initialises mini UART for serial breadcrumb logging
 *   4. Blinks milestones + sends UART characters at every stage
 *   5. Drops from EL2 to EL1
 *   6. Installs exception vectors (with ESR/ELR/FAR dump on crash)
 *   7. Sets up stack, clears BSS
 *   8. Jumps to kernel_main() in C
 *
 * LED Debug Milestones (500ms on/off, 2s gap):
 *   1 blink  = CPU alive, GPIO + UART ready
 *   2 blinks = EL1 reached, exception vectors installed
 *   tick     = stack set
 *   tick     = BSS clear start
 *   tick     = BSS clear done
 *   3 blinks = entering C
 *
 * On Exception:
 *   UART sends "!!! S BC=02 EC=37 ESR=<hex> ELR=<hex> FAR=<hex>\r\n"
 *   LED: [3s ON]  sync beacon
 *        [1s OFF] start gap
 *        [N blinks] breadcrumb (which C checkpoint reached)
 *        [2s OFF] separator
 *        [6 binary bits] exception class (long=1, short=0)
 *        [3s OFF] end gap, repeat
 *   Exception classes: 33=inst abort, 37=data abort, 38=SP align, etc.
 *
 * UART Breadcrumbs (115200 8N1 on GPIO14):
 *   "TOS 1"   = alive, milestone 1
 *   "E"       = about to drop to EL1
 *   "V 2"     = vectors set, milestone 2
 *   "S"       = stack set
 *   "B"       = BSS clear start
 *   "D"       = BSS clear done
 *   "3 C\r\n" = milestone 3, calling kernel_main
 *
 * Target: BCM2711 Cortex-A72 (ARMv8.0)
 * Peripherals at 0xFE000000, ARM local at 0xFF800000
 * =============================================================================*/

/* ===================== MACROS ===================== */

/*
 * UART_CHAR — send one byte over mini UART
 * Requires x8 = 0xFE215000 (AUX peripheral base), clobbers w17
 * Uses numbered local label 9991 with backward reference (safe in macros)
 */
.macro UART_CHAR ch
9991:
    ldr     w17, [x8, #0x54]       /* AUX_MU_LSR_REG */
    tst     w17, #0x20             /* Transmitter empty? */
    b.eq    9991b
    mov     w17, #\ch
    str     w17, [x8, #0x40]       /* AUX_MU_IO_REG */
.endm

/*
 * LED_TICK — quick 150ms blink (sub-milestone between major milestones)
 * Requires x9 = GPIO base, w10 = GPIO42 mask, clobbers x11-x13
 */
.macro LED_TICK
    str     w10, [x9, #0x20]       /* GPSET1: LED ON */
    mrs     x11, cntpct_el0
    mrs     x12, cntfrq_el0
    lsr     x12, x12, #3           /* freq/8 ~ 125ms */
    add     x11, x11, x12
9992:
    mrs     x13, cntpct_el0
    cmp     x13, x11
    b.lo    9992b
    str     w10, [x9, #0x2C]       /* GPCLR1: LED OFF */
    mrs     x11, cntpct_el0
    add     x11, x11, x12
9993:
    mrs     x13, cntpct_el0
    cmp     x13, x11
    b.lo    9993b
.endm

/* ===================== CODE ===================== */

.section ".text.boot"
.globl _start

_start:
    /* ---------------------------------------------------------------
     * Multi-core parking: only core 0 proceeds, others spin
     * --------------------------------------------------------------- */
    mrs     x0, mpidr_el1
    and     x0, x0, #3             /* Core ID (bits 1:0) */
    cbnz    x0, .Lpark

    /* ---------------------------------------------------------------
     * GPIO 42 (ACT LED) setup — FIRST thing, our only visual debug
     * GPFSEL4 at GPIO_BASE+0x10, GPIO 42 = bits [8:6] = 001 (output)
     * GPSET1  at GPIO_BASE+0x20, GPIO 42 = bit 10
     * GPCLR1  at GPIO_BASE+0x2C, GPIO 42 = bit 10
     * --------------------------------------------------------------- */
    ldr     x9, =0xFE200000        /* GPIO base — kept in x9 throughout */
    ldr     w0, [x9, #0x10]        /* GPFSEL4 */
    bic     w0, w0, #(7 << 6)      /* Clear function for GPIO 42 */
    orr     w0, w0, #(1 << 6)      /* Set to output */
    str     w0, [x9, #0x10]
    mov     w10, #(1 << 10)        /* GPIO 42 bitmask — kept in w10 */

    /* Clear breadcrumb at 0x1000 (C code writes progress here) */
    mov     x0, #0x1000
    str     wzr, [x0]

    /* ---------------------------------------------------------------
     * Mini UART setup — firmware likely did this via enable_uart=1,
     * but we re-initialise to be safe.  115200 8N1, TX only.
     * AUX peripheral base = 0xFE215000
     * --------------------------------------------------------------- */
    ldr     x8, =0xFE215000        /* AUX base — kept in x8 throughout */
    mov     w0, #1
    str     w0, [x8, #0x04]        /* AUX_ENABLES: enable mini UART */
    str     wzr, [x8, #0x60]       /* AUX_MU_CNTL: disable TX/RX during cfg */
    mov     w0, #3
    str     w0, [x8, #0x48]        /* AUX_MU_LCR: 8-bit mode */
    str     wzr, [x8, #0x4C]       /* AUX_MU_MCR: clear */
    str     wzr, [x8, #0x44]       /* AUX_MU_IER: no interrupts */
    movz    w0, #542               /* 500MHz / (8 * 115200) - 1 = 542 */
    str     w0, [x8, #0x68]        /* AUX_MU_BAUD */
    mov     w0, #3
    str     w0, [x8, #0x60]        /* AUX_MU_CNTL: enable TX + RX */

    /* GPIO 14 = ALT5 (mini UART TXD1) */
    ldr     w0, [x9, #0x04]        /* GPFSEL1 */
    bic     w0, w0, #(7 << 12)     /* Clear GPIO 14 function */
    orr     w0, w0, #(2 << 12)     /* ALT5 = 010 */
    str     w0, [x9, #0x04]

    /* First UART sign of life */
    UART_CHAR '\r'
    UART_CHAR '\n'
    UART_CHAR 'T'
    UART_CHAR 'O'
    UART_CHAR 'S'
    UART_CHAR ' '

    /* ===== MILESTONE 1: CPU alive, GPIO + UART ready ===== */
    UART_CHAR '1'

    /* ---------------------------------------------------------------
     * Drop from EL2 to EL1 (RPi firmware starts us in EL2)
     * --------------------------------------------------------------- */
    UART_CHAR 'E'

    mrs     x0, CurrentEL
    lsr     x0, x0, #2
    cmp     x0, #2
    b.ne    .Lin_el1

    /* HCR_EL2: EL1 uses AArch64 */
    mov     x0, #(1 << 31)
    msr     hcr_el2, x0

    /* SCTLR_EL1: MMU off, caches off, but RES1 bits MUST be set.
     * Bits 29,28,23,22,20,11 are RES1 on Cortex-A72 (ARMv8.0).
     * Linux uses 0x30D00800 for these.  Clearing them is
     * CONSTRAINED UNPREDICTABLE per the ARM ARM. */
    movz    x0, #0x0800
    movk    x0, #0x30D0, lsl #16   /* x0 = 0x30D00800 */
    msr     sctlr_el1, x0

    /* Enable FP/NEON before dropping to EL1 */
    mov     x0, #(3 << 20)
    msr     cpacr_el1, x0

    /* Enable timer access from EL1 */
    mrs     x0, cnthctl_el2
    orr     x0, x0, #3             /* EL1PCEN | EL1PCTEN */
    msr     cnthctl_el2, x0
    msr     cntvoff_el2, xzr

    /* Install exception vectors EARLY from EL2 — close the window
     * between eret and the msr in .Lin_el1 where VBAR_EL1 was 0. */
    adr     x0, _exception_vectors
    msr     vbar_el1, x0

    /* SPSR_EL2: Return to EL1h (SP_EL1), all DAIF masked */
    mov     x0, #0x3C5
    msr     spsr_el2, x0

    /* ELR_EL2: resume at .Lin_el1 */
    adr     x0, .Lin_el1
    msr     elr_el2, x0

    isb
    eret

.Lin_el1:
    /* Re-enable FP/NEON at EL1 (belt-and-suspenders) */
    mov     x0, #(3 << 20)
    msr     cpacr_el1, x0
    isb

    /* Reload x8/x9/w10 — be safe after EL transition */
    ldr     x9, =0xFE200000
    mov     w10, #(1 << 10)
    ldr     x8, =0xFE215000

    /* Re-install exception vectors (belt-and-suspenders, also set from EL2) */
    adr     x0, _exception_vectors
    msr     vbar_el1, x0
    isb

    UART_CHAR 'V'
    UART_CHAR ' '

    /* ===== MILESTONE 2: EL1 reached ===== */
    UART_CHAR '2'

    /* ---------------------------------------------------------------
     * MMU Identity Map
     *
     * Without MMU, all memory defaults to Device-nGnRnE on AArch64.
     * The C compiler generates STP/LDP (16-byte stores/loads) which
     * require 16-byte alignment on Device memory — causing alignment
     * faults.  We enable a simple 1GB-block identity map so DRAM is
     * Normal Cacheable (fast, relaxed alignment) and peripherals
     * remain Device-nGnRnE (MMIO-safe).
     *
     * L1 table at physical 0x2000 (4K-aligned, 4 entries used):
     *   [0] 0x00000000-3FFFFFFF  Normal WB (DRAM, kernel, stack, BSS)
     *   [1] 0x40000000-7FFFFFFF  Normal WB (DRAM)
     *   [2] 0x80000000-BFFFFFFF  Normal WB (DRAM)
     *   [3] 0xC0000000-FFFFFFFF  Device-nGnRnE (GPIO, UART, GIC)
     * --------------------------------------------------------------- */
    UART_CHAR 'M'
    /* MAIR_EL1: Attr0 = 0xFF (Normal Inner+Outer WB-RW-Alloc)
     *           Attr1 = 0x00 (Device-nGnRnE) */
    mov     x0, #0xFF
    msr     mair_el1, x0

    /* L1 page table at physical 0x2000 */
    mov     x0, #0x2000

    /* Zero 4K table (stp to 16-byte-aligned Device addrs: safe) */
    mov     x1, x0
    add     x2, x0, #4096
.Lclear_pt:
    stp     xzr, xzr, [x1], #16
    cmp     x1, x2
    b.lo    .Lclear_pt

    /* Block descriptor attributes:
     *  Normal: AF(1<<10) | SH=IS(3<<8) | AttrIndx=0 | Block(01) = 0x701
     *  Device: PXN(1<<53)|UXN(1<<54)|AF(1<<10)|AttrIndx=1(1<<2)|Block(01) */
    mov     x1, #0x701             /* Normal attributes */

    /* Entry 0: 0x00000000 → Normal */
    str     x1, [x0, #0]

    /* Entry 1: 0x40000000 → Normal */
    movz    x2, #0x4000, lsl #16
    orr     x2, x2, x1
    str     x2, [x0, #8]

    /* Entry 2: 0x80000000 → Normal */
    movz    x2, #0x8000, lsl #16
    orr     x2, x2, x1
    str     x2, [x0, #16]

    /* Entry 3: 0xC0000000 → Device PXN+UXN */
    movz    x2, #0x0405
    movk    x2, #0xC000, lsl #16
    movk    x2, #0x0060, lsl #48
    str     x2, [x0, #24]

    /* TTBR0_EL1: point at our L1 table */
    msr     ttbr0_el1, x0

    /* TCR_EL1: T0SZ=25 (39-bit VA), 4KB granule, Inner Shareable walks,
     *          WB-WA cacheable walks, IPS=40-bit, EPD1=1 (no TTBR1) */
    movz    x0, #0x3519
    movk    x0, #0x0080, lsl #16
    movk    x0, #0x0002, lsl #32
    msr     tcr_el1, x0

    /* Invalidate TLBs + I-cache before enabling MMU */
    tlbi    vmalle1
    ic      iallu
    dsb     sy
    isb

    /* Enable MMU + D-cache + I-cache
     * SCTLR_EL1 = RES1(0x30D00800) | M(1) | C(1<<2) | I(1<<12)
     *           = 0x30D01805 */
    movz    x0, #0x1805
    movk    x0, #0x30D0, lsl #16
    msr     sctlr_el1, x0
    isb

    UART_CHAR 'P'

    /* ---------------------------------------------------------------
     * Stack and BSS — identity-mapped, Normal Cacheable DRAM
     * Stack at 0x80000 (grows down), kernel at 0x80000+ (grows up)
     * --------------------------------------------------------------- */
    UART_CHAR 'S'
    mov     x0, #0x80000
    mov     sp, x0

    /* Clear BSS — now on Normal Cacheable memory (caches enabled).
     * __bss_start is 4K-aligned per kernel.ld.  ~48 MB, takes ~20ms. */
    UART_CHAR 'B'
    ldr     x0, =__bss_start
    ldr     x1, =__bss_end
.Lbss_clear:
    cmp     x0, x1
    b.ge    .Lbss_done
    str     xzr, [x0], #8
    b       .Lbss_clear
.Lbss_done:

    UART_CHAR 'D'

    /* ===== MILESTONE 3: entering C ===== */
    UART_CHAR ' '
    UART_CHAR '3'

    /* Jump to C kernel_main */
    UART_CHAR 'C'
    UART_CHAR '\r'
    UART_CHAR '\n'
    bl      kernel_main

    /* kernel_main returned — solid LED ON (should not happen) */
    UART_CHAR 'R'
    str     w10, [x9, #0x20]
.Lhalt:
    wfi
    b       .Lhalt

/* ---------------------------------------------------------------
 * Secondary core parking
 * --------------------------------------------------------------- */
.Lpark:
    wfe
    b       .Lpark

/* =============================================================================
 * LED Blink Helpers — use ARM Generic Timer for accurate timing
 *
 * Registers preserved: x8 (AUX), x9 (GPIO), w10 (mask)
 * Clobbers: x11, x12, x13 only (.Lgap also x14)
 * Uses bl/ret (link register x30), does NOT need stack
 * =============================================================================*/

/* One blink: ~500ms ON, ~500ms OFF (slow enough to count!) */
.Lblink:
    str     w10, [x9, #0x20]       /* GPSET1: LED ON */
    mrs     x11, cntpct_el0
    mrs     x12, cntfrq_el0
    lsr     x12, x12, #1           /* freq/2 ~ 500ms */
    add     x11, x11, x12
.Lb_on:
    mrs     x13, cntpct_el0
    cmp     x13, x11
    b.lo    .Lb_on
    str     w10, [x9, #0x2C]       /* GPCLR1: LED OFF */
    mrs     x11, cntpct_el0
    add     x11, x11, x12          /* another 500ms dark */
.Lb_off:
    mrs     x13, cntpct_el0
    cmp     x13, x11
    b.lo    .Lb_off
    ret

/* Dark gap: ~2 seconds pause between milestones */
.Lgap:
    mrs     x11, cntpct_el0
    mrs     x12, cntfrq_el0
    mov     x14, #2
    mul     x12, x12, x14          /* freq*2 ~ 2 seconds */
    add     x11, x11, x12
.Lg_wait:
    mrs     x13, cntpct_el0
    cmp     x13, x11
    b.lo    .Lg_wait
    ret

/* =============================================================================
 * UART Hex64 Helper — send 64-bit value in x0 as 16-char hex over UART
 * Requires x8 = AUX base.  Clobbers x0, x2, x3, w17.
 * Uses bl/ret.
 * =============================================================================*/
.Luart_hex64:
    mov     x2, x0                  /* save value */
    mov     x3, #60                 /* bit shift (start from MSB) */
.Lhex_loop:
    lsr     x0, x2, x3
    and     x0, x0, #0xF
    cmp     x0, #10
    b.lo    .Lhex_digit
    add     x0, x0, #('A' - 10)
    b       .Lhex_out
.Lhex_digit:
    add     x0, x0, #'0'
.Lhex_out:
9994:
    ldr     w17, [x8, #0x54]
    tst     w17, #0x20
    b.eq    9994b
    str     w0, [x8, #0x40]
    subs    x3, x3, #4
    b.ge    .Lhex_loop
    ret

/* =============================================================================
 * Exception Vector Table — 16 entries x 128 bytes = 2KB, 2KB-aligned
 * Each exception type sets x28 to a type letter before jumping to handler:
 *   'S' = Synchronous    'I' = IRQ    'F' = FIQ    'E' = SError
 * =============================================================================*/

.balign 2048
_exception_vectors:
    /* Current EL, SP_EL0 */
    b       _exc_sync               /* +0x000 Synchronous */
    .balign 128
    b       _exc_irq                /* +0x080 IRQ */
    .balign 128
    b       _exc_fiq                /* +0x100 FIQ */
    .balign 128
    b       _exc_serror             /* +0x180 SError */
    .balign 128

    /* Current EL, SP_ELx */
    b       _exc_sync               /* +0x200 Synchronous */
    .balign 128
    b       _exc_irq                /* +0x280 IRQ */
    .balign 128
    b       _exc_fiq                /* +0x300 FIQ */
    .balign 128
    b       _exc_serror             /* +0x380 SError */
    .balign 128

    /* Lower EL, AArch64 */
    b       _exc_sync               /* +0x400 Synchronous */
    .balign 128
    b       _exc_irq                /* +0x480 IRQ */
    .balign 128
    b       _exc_fiq                /* +0x500 FIQ */
    .balign 128
    b       _exc_serror             /* +0x580 SError */
    .balign 128

    /* Lower EL, AArch32 */
    b       _exc_sync               /* +0x600 Synchronous */
    .balign 128
    b       _exc_irq                /* +0x680 IRQ */
    .balign 128
    b       _exc_fiq                /* +0x700 FIQ */
    .balign 128
    b       _exc_serror             /* +0x780 SError */
    .balign 128

/* Exception type stubs — set x28 to type letter, then common handler */
_exc_sync:
    mov     x28, #'S'
    b       _exception_hang
_exc_irq:
    mov     x28, #'I'
    b       _exception_hang
_exc_fiq:
    mov     x28, #'F'
    b       _exception_hang
_exc_serror:
    mov     x28, #'E'
    b       _exception_hang

/* =============================================================================
 * Exception Handler — comprehensive diagnostic dump
 *
 * UART output:
 *   "!!! S BC=02 EC=37 ESR=<16hex> ELR=<16hex> FAR=<16hex>\r\n"
 *   Type letter: S=Sync I=IRQ F=FIQ E=SError
 *   BC = breadcrumb (last C checkpoint, from address 0x1000)
 *   EC = exception class
 *
 * LED pattern (repeats forever, self-synchronizing):
 *   [3s SOLID ON]          sync beacon (nothing else is 3s solid)
 *   [1s OFF]               start gap
 *   [N slow blinks]        breadcrumb count (500ms each)
 *   [2s OFF]               separator
 *   [6 binary bits]        exception class MSB first:
 *                             bit 1: 750ms ON, 250ms OFF (long flash)
 *                             bit 0: 250ms ON, 750ms OFF (short flash)
 *   [3s OFF]               end gap, then loop
 *
 * Common exception classes:
 *   0 = Unknown/SError    21 = SVC       33 = Inst Abort
 *   37 = Data Abort       38 = SP Align  60 = BRK
 * =============================================================================*/
_exception_hang:
    /* Re-setup GPIO 42 as output */
    ldr     x9, =0xFE200000
    ldr     w0, [x9, #0x10]
    bic     w0, w0, #(7 << 6)
    orr     w0, w0, #(1 << 6)
    str     w0, [x9, #0x10]
    mov     w10, #(1 << 10)

    /* Re-setup mini UART */
    ldr     x8, =0xFE215000
    mov     w0, #1
    str     w0, [x8, #0x04]
    str     wzr, [x8, #0x60]
    mov     w0, #3
    str     w0, [x8, #0x48]
    str     wzr, [x8, #0x4C]
    str     wzr, [x8, #0x44]
    movz    w0, #542
    str     w0, [x8, #0x68]
    mov     w0, #3
    str     w0, [x8, #0x60]
    ldr     w0, [x9, #0x04]
    bic     w0, w0, #(7 << 12)
    orr     w0, w0, #(2 << 12)     /* GPIO14 ALT5 = mini UART */
    str     w0, [x9, #0x04]

    /* Read exception info */
    mrs     x19, esr_el1
    mrs     x20, elr_el1
    mrs     x21, far_el1

    /* Read breadcrumb from 0x1000 */
    mov     x0, #0x1000
    ldr     w27, [x0]              /* x27 = breadcrumb value */

    /* Extract exception class */
    lsr     x22, x19, #26
    and     x22, x22, #0x3F       /* x22 = EC (0-63) */

    /* Precompute timer frequency */
    mrs     x23, cntfrq_el0

    /* ---- UART crash report ---- */
    UART_CHAR '\r'
    UART_CHAR '\n'
    UART_CHAR '!'
    UART_CHAR '!'
    UART_CHAR '!'
    UART_CHAR ' '
    /* Type letter from x28 */
9960:
    ldr     w17, [x8, #0x54]
    tst     w17, #0x20
    b.eq    9960b
    str     w28, [x8, #0x40]
    UART_CHAR ' '
    UART_CHAR 'B'
    UART_CHAR 'C'
    UART_CHAR '='
    /* Breadcrumb as 2-digit decimal */
    mov     x14, #10
    udiv    x15, x27, x14
    msub    x16, x15, x14, x27
    add     x0, x15, #'0'
9961:
    ldr     w17, [x8, #0x54]
    tst     w17, #0x20
    b.eq    9961b
    str     w0, [x8, #0x40]
    add     x0, x16, #'0'
9962:
    ldr     w17, [x8, #0x54]
    tst     w17, #0x20
    b.eq    9962b
    str     w0, [x8, #0x40]
    UART_CHAR ' '
    UART_CHAR 'E'
    UART_CHAR 'C'
    UART_CHAR '='
    /* EC as 2-digit decimal */
    udiv    x15, x22, x14
    msub    x16, x15, x14, x22
    add     x0, x15, #'0'
9963:
    ldr     w17, [x8, #0x54]
    tst     w17, #0x20
    b.eq    9963b
    str     w0, [x8, #0x40]
    add     x0, x16, #'0'
9964:
    ldr     w17, [x8, #0x54]
    tst     w17, #0x20
    b.eq    9964b
    str     w0, [x8, #0x40]
    UART_CHAR ' '
    UART_CHAR 'E'
    UART_CHAR 'S'
    UART_CHAR 'R'
    UART_CHAR '='
    mov     x0, x19
    bl      .Luart_hex64
    UART_CHAR ' '
    UART_CHAR 'E'
    UART_CHAR 'L'
    UART_CHAR 'R'
    UART_CHAR '='
    mov     x0, x20
    bl      .Luart_hex64
    UART_CHAR ' '
    UART_CHAR 'F'
    UART_CHAR 'A'
    UART_CHAR 'R'
    UART_CHAR '='
    mov     x0, x21
    bl      .Luart_hex64
    UART_CHAR '\r'
    UART_CHAR '\n'

    /* ---- Try to flush crash details to BOOTLOG.TXT on SD card ----
     * Call sdlog_crash_c(type, bc, ec, esr, elr, far)
     * x19-x28 are callee-saved so they survive the bl.
     * x8/x9/x10 (caller-saved) will be reloaded afterwards.        */
    mov     x0, x28         /* type letter */
    mov     x1, x27         /* breadcrumb */
    mov     x2, x22         /* EC */
    mov     x3, x19         /* ESR */
    mov     x4, x20         /* ELR */
    mov     x5, x21         /* FAR */
    bl      sdlog_crash_c

    /* Re-load caller-saved registers clobbered by C call */
    ldr     x9, =0xFE200000        /* GPIO base */
    mov     w10, #(1 << 10)        /* GPIO 42 bit mask */
    ldr     x8, =0xFE215000        /* AUX (mini UART) base */

    /* Reload timer freq (bl may have clobbered things, be safe) */
    mrs     x23, cntfrq_el0

    /* ================================================================
     * LED diagnostic loop — self-synchronizing binary pattern
     * ================================================================ */
.Lexc_display_loop:

    /* --- SYNC BEACON: 3 seconds SOLID ON --- */
    str     w10, [x9, #0x20]
    mrs     x11, cntpct_el0
    mov     x14, #3
    mul     x12, x23, x14
    add     x11, x11, x12
9970:
    mrs     x13, cntpct_el0
    cmp     x13, x11
    b.lo    9970b
    str     w10, [x9, #0x2C]

    /* --- 1 second START GAP --- */
    mrs     x11, cntpct_el0
    add     x11, x11, x23
9971:
    mrs     x13, cntpct_el0
    cmp     x13, x11
    b.lo    9971b

    /* --- BREADCRUMB BLINKS: x27 slow blinks (500ms each) --- */
    mov     x26, x27
.Lbc_blink_loop:
    cbz     x26, .Lbc_blink_done
    bl      .Lblink
    sub     x26, x26, #1
    b       .Lbc_blink_loop
.Lbc_blink_done:

    /* --- 2 second SEPARATOR --- */
    mrs     x11, cntpct_el0
    mov     x14, #2
    mul     x12, x23, x14
    add     x11, x11, x12
9972:
    mrs     x13, cntpct_el0
    cmp     x13, x11
    b.lo    9972b

    /* --- 6 BINARY BITS: exception class MSB first --- */
    mov     x24, #5                 /* bit index 5 down to 0 */

.Lbit_loop:
    lsr     x25, x22, x24
    tst     x25, #1
    b.eq    .Lbit_zero

    /* BIT = 1: 750ms ON, 250ms OFF */
    str     w10, [x9, #0x20]
    mrs     x11, cntpct_el0
    mov     x14, #3
    mul     x12, x23, x14
    lsr     x12, x12, #2           /* freq*3/4 = 750ms */
    add     x11, x11, x12
9973:
    mrs     x13, cntpct_el0
    cmp     x13, x11
    b.lo    9973b
    str     w10, [x9, #0x2C]
    mrs     x11, cntpct_el0
    lsr     x12, x23, #2           /* freq/4 = 250ms */
    add     x11, x11, x12
9974:
    mrs     x13, cntpct_el0
    cmp     x13, x11
    b.lo    9974b
    b       .Lbit_next

.Lbit_zero:
    /* BIT = 0: 250ms ON, 750ms OFF */
    str     w10, [x9, #0x20]
    mrs     x11, cntpct_el0
    lsr     x12, x23, #2           /* freq/4 = 250ms */
    add     x11, x11, x12
9975:
    mrs     x13, cntpct_el0
    cmp     x13, x11
    b.lo    9975b
    str     w10, [x9, #0x2C]
    mrs     x11, cntpct_el0
    mov     x14, #3
    mul     x12, x23, x14
    lsr     x12, x12, #2           /* freq*3/4 = 750ms */
    add     x11, x11, x12
9976:
    mrs     x13, cntpct_el0
    cmp     x13, x11
    b.lo    9976b

.Lbit_next:
    subs    x24, x24, #1
    b.ge    .Lbit_loop

    /* --- 3 second END GAP --- */
    mrs     x11, cntpct_el0
    mov     x14, #3
    mul     x12, x23, x14
    add     x11, x11, x12
9977:
    mrs     x13, cntpct_el0
    cmp     x13, x11
    b.lo    9977b

    /* Re-send compact UART info each cycle */
    UART_CHAR '!'
9978:
    ldr     w17, [x8, #0x54]
    tst     w17, #0x20
    b.eq    9978b
    str     w28, [x8, #0x40]       /* type letter */
    UART_CHAR ' '
    UART_CHAR 'B'
    mov     x14, #10
    udiv    x15, x27, x14
    add     x0, x15, #'0'
9979:
    ldr     w17, [x8, #0x54]
    tst     w17, #0x20
    b.eq    9979b
    str     w0, [x8, #0x40]
    msub    x15, x15, x14, x27
    add     x0, x15, #'0'
9950:
    ldr     w17, [x8, #0x54]
    tst     w17, #0x20
    b.eq    9950b
    str     w0, [x8, #0x40]
    UART_CHAR ' '
    UART_CHAR 'E'
    udiv    x15, x22, x14
    add     x0, x15, #'0'
9951:
    ldr     w17, [x8, #0x54]
    tst     w17, #0x20
    b.eq    9951b
    str     w0, [x8, #0x40]
    msub    x15, x15, x14, x22
    add     x0, x15, #'0'
9952:
    ldr     w17, [x8, #0x54]
    tst     w17, #0x20
    b.eq    9952b
    str     w0, [x8, #0x40]
    UART_CHAR '\r'
    UART_CHAR '\n'

    /* Reload timer freq */
    mrs     x23, cntfrq_el0

    b       .Lexc_display_loop
