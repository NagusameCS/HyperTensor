/* =============================================================================
 * TensorOS - ARM64 AArch64 JIT Compiler Implementation
 *
 * NEON SIMD-based JIT code generation for tensor operations on ARM64.
 * Generates native AArch64 machine code at runtime for common operations.
 *
 * Architecture:
 *   1. Low-level emitter: AArch64 instruction encoding (fixed 32-bit words)
 *   2. NEON SIMD emission: 128-bit vector operations (float32x4)
 *   3. High-level compiler: tensor ops → native ARM64 code
 *
 * AArch64 calling convention (AAPCS64):
 *   Args: x0-x7 (integer/pointer), v0-v7 (float/SIMD)
 *   Callee-saved: x19-x28, v8-v15
 *   Return: x0 or v0
 *   LR: x30, FP: x29, SP must be 16-byte aligned
 * =============================================================================*/

#ifdef __aarch64__

#include "runtime/jit/x86_jit.h"  /* Reuse type definitions */
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"

/* =============================================================================
 * AArch64 Register Definitions
 * =============================================================================*/

/* General-purpose registers */
#define X0   0
#define X1   1
#define X2   2
#define X3   3
#define X4   4
#define X5   5
#define X6   6
#define X7   7
#define X8   8
#define X9   9
#define X10 10
#define X11 11
#define X12 12
#define X13 13
#define X14 14
#define X15 15
#define X16 16  /* IP0 - intra-procedure scratch */
#define X17 17  /* IP1 - intra-procedure scratch */
#define X18 18  /* Platform register */
#define X19 19  /* Callee-saved */
#define X20 20
#define X21 21
#define X22 22
#define X23 23
#define X24 24
#define X25 25
#define X26 26
#define X27 27
#define X28 28
#define X29 29  /* Frame pointer */
#define X30 30  /* Link register */
#define XZR 31  /* Zero register */
#define SP  31  /* Stack pointer (context-dependent) */

/* NEON/SIMD registers */
#define V0   0
#define V1   1
#define V2   2
#define V3   3
#define V4   4
#define V5   5
#define V6   6
#define V7   7
#define V8   8   /* Callee-saved */
#define V9   9
#define V10 10
#define V11 11
#define V12 12
#define V13 13
#define V14 14
#define V15 15
#define V16 16  /* Caller-saved */
#define V17 17
#define V18 18
#define V19 19
#define V20 20
#define V21 21
#define V22 22
#define V23 23
#define V24 24
#define V25 25
#define V26 26
#define V27 27
#define V28 28
#define V29 29
#define V30 30
#define V31 31

/* =============================================================================
 * Buffer Management
 * =============================================================================*/

#define A64_JIT_POOL_SIZE  (2 * 1024 * 1024)  /* 2MB for all JIT code */
static uint8_t a64_jit_pool[A64_JIT_POOL_SIZE] __attribute__((aligned(4096)));
static int a64_jit_pool_offset = 0;

#define A64_JIT_MAX_BUFS 64
static jit_buf_t a64_jit_buf_storage[A64_JIT_MAX_BUFS];
static int a64_jit_buf_count = 0;
static bool a64_jit_buf_active[A64_JIT_MAX_BUFS];

jit_buf_t *jit_create(int capacity)
{
    capacity = (capacity + 15) & ~15;

    for (int i = 0; i < a64_jit_buf_count; i++) {
        if (!a64_jit_buf_active[i] && a64_jit_buf_storage[i].cap >= capacity) {
            a64_jit_buf_active[i] = true;
            a64_jit_buf_storage[i].len = 0;
            return &a64_jit_buf_storage[i];
        }
    }

    if (a64_jit_buf_count >= A64_JIT_MAX_BUFS) return NULL;
    if (a64_jit_pool_offset + capacity > A64_JIT_POOL_SIZE) return NULL;

    int idx = a64_jit_buf_count++;
    jit_buf_t *b = &a64_jit_buf_storage[idx];
    b->code = a64_jit_pool + a64_jit_pool_offset;
    b->len = 0;
    b->cap = capacity;
    a64_jit_pool_offset += capacity;
    a64_jit_buf_active[idx] = true;
    return b;
}

void jit_destroy(jit_buf_t *buf)
{
    if (!buf) return;
    for (int i = 0; i < a64_jit_buf_count; i++) {
        if (&a64_jit_buf_storage[i] == buf) {
            a64_jit_buf_active[i] = false;
            buf->len = 0;
            return;
        }
    }
}

void jit_reset(jit_buf_t *buf)
{
    if (buf) buf->len = 0;
}

jit_void_fn jit_get_fn(jit_buf_t *buf)
{
    if (!buf || !buf->code) return NULL;
    /* Clear instruction cache */
    __builtin___clear_cache((char *)buf->code, (char *)buf->code + buf->len);
    return (jit_void_fn)(void *)buf->code;
}

/* =============================================================================
 * Low-Level Instruction Emission
 *
 * AArch64 instructions are always 32-bit (4 bytes).
 * =============================================================================*/

static void a64_emit32(jit_buf_t *b, uint32_t insn)
{
    if (b->len + 4 <= b->cap) {
        b->code[b->len++] = (uint8_t)(insn & 0xFF);
        b->code[b->len++] = (uint8_t)((insn >> 8) & 0xFF);
        b->code[b->len++] = (uint8_t)((insn >> 16) & 0xFF);
        b->code[b->len++] = (uint8_t)((insn >> 24) & 0xFF);
    }
}

/* =============================================================================
 * Data Processing Instructions
 * =============================================================================*/

/* MOV (register): Rd = Rm */
static void a64_mov_reg(jit_buf_t *b, int rd, int rm)
{
    /* ORR Rd, XZR, Rm */
    uint32_t insn = 0xAA000000 | (rm << 16) | (XZR << 5) | rd;
    a64_emit32(b, insn);
}

/* MOV (immediate): Rd = imm16 << (hw * 16) */
static void a64_movz(jit_buf_t *b, int rd, uint16_t imm16, int hw)
{
    uint32_t insn = 0xD2800000 | ((hw & 3) << 21) | ((uint32_t)imm16 << 5) | rd;
    a64_emit32(b, insn);
}

/* MOVK: Rd[hw*16..hw*16+15] = imm16 */
static void a64_movk(jit_buf_t *b, int rd, uint16_t imm16, int hw)
{
    uint32_t insn = 0xF2800000 | ((hw & 3) << 21) | ((uint32_t)imm16 << 5) | rd;
    a64_emit32(b, insn);
}

/* Load 64-bit immediate into register */
static void a64_mov_imm64(jit_buf_t *b, int rd, uint64_t imm)
{
    a64_movz(b, rd, (uint16_t)(imm & 0xFFFF), 0);
    if (imm > 0xFFFF)
        a64_movk(b, rd, (uint16_t)((imm >> 16) & 0xFFFF), 1);
    if (imm > 0xFFFFFFFFULL)
        a64_movk(b, rd, (uint16_t)((imm >> 32) & 0xFFFF), 2);
    if (imm > 0xFFFFFFFFFFFFULL)
        a64_movk(b, rd, (uint16_t)((imm >> 48) & 0xFFFF), 3);
}

/* ADD Rd, Rn, imm12 */
static void a64_add_imm(jit_buf_t *b, int rd, int rn, uint32_t imm12)
{
    uint32_t insn = 0x91000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd;
    a64_emit32(b, insn);
}

/* SUB Rd, Rn, imm12 */
static void a64_sub_imm(jit_buf_t *b, int rd, int rn, uint32_t imm12)
{
    uint32_t insn = 0xD1000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | rd;
    a64_emit32(b, insn);
}

/* ADD Rd, Rn, Rm */
static void a64_add_reg(jit_buf_t *b, int rd, int rn, int rm)
{
    uint32_t insn = 0x8B000000 | (rm << 16) | (rn << 5) | rd;
    a64_emit32(b, insn);
}

/* SUB Rd, Rn, Rm */
static void a64_sub_reg(jit_buf_t *b, int rd, int rn, int rm)
{
    uint32_t insn = 0xCB000000 | (rm << 16) | (rn << 5) | rd;
    a64_emit32(b, insn);
}

/* MUL Rd, Rn, Rm (MADD Rd, Rn, Rm, XZR) */
static void a64_mul(jit_buf_t *b, int rd, int rn, int rm)
{
    uint32_t insn = 0x9B007C00 | (rm << 16) | (rn << 5) | rd;
    a64_emit32(b, insn);
}

/* CMP Rn, imm12 (alias for SUBS XZR, Rn, imm12) */
static void a64_cmp_imm(jit_buf_t *b, int rn, uint32_t imm12)
{
    uint32_t insn = 0xF1000000 | ((imm12 & 0xFFF) << 10) | (rn << 5) | XZR;
    a64_emit32(b, insn);
}

/* CMP Rn, Rm */
static void a64_cmp_reg(jit_buf_t *b, int rn, int rm)
{
    uint32_t insn = 0xEB000000 | (rm << 16) | (rn << 5) | XZR;
    a64_emit32(b, insn);
}

/* =============================================================================
 * Load/Store Instructions
 * =============================================================================*/

/* LDR Xt, [Xn, #imm12*8] */
static void a64_ldr(jit_buf_t *b, int rt, int rn, int32_t offset)
{
    uint32_t uoff = (uint32_t)(offset >> 3) & 0xFFF;
    uint32_t insn = 0xF9400000 | (uoff << 10) | (rn << 5) | rt;
    a64_emit32(b, insn);
}

/* STR Xt, [Xn, #imm12*8] */
static void a64_str(jit_buf_t *b, int rt, int rn, int32_t offset)
{
    uint32_t uoff = (uint32_t)(offset >> 3) & 0xFFF;
    uint32_t insn = 0xF9000000 | (uoff << 10) | (rn << 5) | rt;
    a64_emit32(b, insn);
}

/* LDP Xt1, Xt2, [Xn, #imm7*8] (load pair) */
static void a64_ldp(jit_buf_t *b, int rt1, int rt2, int rn, int32_t offset)
{
    int32_t imm7 = (offset >> 3) & 0x7F;
    uint32_t insn = 0xA9400000 | ((uint32_t)imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1;
    a64_emit32(b, insn);
}

/* STP Xt1, Xt2, [Xn, #imm7*8]! (store pair, pre-index) */
static void a64_stp_pre(jit_buf_t *b, int rt1, int rt2, int rn, int32_t offset)
{
    int32_t imm7 = (offset >> 3) & 0x7F;
    uint32_t insn = 0xA9800000 | ((uint32_t)imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1;
    a64_emit32(b, insn);
}

/* LDP Xt1, Xt2, [Xn], #imm7*8 (load pair, post-index) */
static void a64_ldp_post(jit_buf_t *b, int rt1, int rt2, int rn, int32_t offset)
{
    int32_t imm7 = (offset >> 3) & 0x7F;
    uint32_t insn = 0xA8C00000 | ((uint32_t)imm7 << 15) | (rt2 << 10) | (rn << 5) | rt1;
    a64_emit32(b, insn);
}

/* =============================================================================
 * Branch Instructions
 * =============================================================================*/

/* RET (return to LR) */
static void a64_ret(jit_buf_t *b)
{
    a64_emit32(b, 0xD65F03C0);
}

/* B.cond offset (conditional branch) */
static int a64_bcond_fwd(jit_buf_t *b, int cond)
{
    int patch = b->len;
    uint32_t insn = 0x54000000 | cond;
    a64_emit32(b, insn);
    return patch;
}

/* Patch conditional branch */
static void a64_patch_bcond(jit_buf_t *b, int patch_offset)
{
    int32_t disp = (b->len - patch_offset) >> 2;
    uint32_t *p = (uint32_t *)(b->code + patch_offset);
    *p = (*p & 0xFF00001F) | ((uint32_t)(disp & 0x7FFFF) << 5);
}

/* B offset (unconditional branch) */
static int a64_b_fwd(jit_buf_t *b)
{
    int patch = b->len;
    a64_emit32(b, 0x14000000);
    return patch;
}

static void a64_b_back(jit_buf_t *b, int target)
{
    int32_t disp = (target - b->len) >> 2;
    uint32_t insn = 0x14000000 | ((uint32_t)disp & 0x3FFFFFF);
    a64_emit32(b, insn);
}

static void a64_patch_b(jit_buf_t *b, int patch_offset)
{
    int32_t disp = (b->len - patch_offset) >> 2;
    uint32_t *p = (uint32_t *)(b->code + patch_offset);
    *p = 0x14000000 | ((uint32_t)disp & 0x3FFFFFF);
}

/* Branch and link */
static void a64_bl(jit_buf_t *b, int32_t offset)
{
    int32_t disp = offset >> 2;
    uint32_t insn = 0x94000000 | ((uint32_t)disp & 0x3FFFFFF);
    a64_emit32(b, insn);
}

/* Condition codes for B.cond */
#define A64_COND_EQ  0   /* Z=1 */
#define A64_COND_NE  1   /* Z=0 */
#define A64_COND_LT  11  /* N!=V */
#define A64_COND_GE  10  /* N==V */
#define A64_COND_LE  13  /* Z=1 or N!=V */
#define A64_COND_GT  12  /* Z=0 and N==V */

/* =============================================================================
 * NEON SIMD Instructions (float32x4)
 * =============================================================================*/

/* LD1 {Vt.4S}, [Xn] — load 4 floats */
static void a64_ld1_4s(jit_buf_t *b, int vt, int xn)
{
    uint32_t insn = 0x4C407800 | (xn << 5) | vt;
    a64_emit32(b, insn);
}

/* ST1 {Vt.4S}, [Xn] — store 4 floats */
static void a64_st1_4s(jit_buf_t *b, int vt, int xn)
{
    uint32_t insn = 0x4C007800 | (xn << 5) | vt;
    a64_emit32(b, insn);
}

/* LD1 {Vt.4S}, [Xn], #16 — load 4 floats, post-increment */
static void a64_ld1_4s_post(jit_buf_t *b, int vt, int xn)
{
    uint32_t insn = 0x4CDF7800 | (xn << 5) | vt;
    a64_emit32(b, insn);
}

/* ST1 {Vt.4S}, [Xn], #16 — store 4 floats, post-increment */
static void a64_st1_4s_post(jit_buf_t *b, int vt, int xn)
{
    uint32_t insn = 0x4C9F7800 | (xn << 5) | vt;
    a64_emit32(b, insn);
}

/* FADD Vd.4S, Vn.4S, Vm.4S */
static void a64_fadd_4s(jit_buf_t *b, int vd, int vn, int vm)
{
    uint32_t insn = 0x4E20D400 | (vm << 16) | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FSUB Vd.4S, Vn.4S, Vm.4S */
static void a64_fsub_4s(jit_buf_t *b, int vd, int vn, int vm)
{
    uint32_t insn = 0x4EA0D400 | (vm << 16) | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FMUL Vd.4S, Vn.4S, Vm.4S */
static void a64_fmul_4s(jit_buf_t *b, int vd, int vn, int vm)
{
    uint32_t insn = 0x6E20DC00 | (vm << 16) | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FDIV Vd.4S, Vn.4S, Vm.4S */
static void a64_fdiv_4s(jit_buf_t *b, int vd, int vn, int vm)
{
    uint32_t insn = 0x6E20FC00 | (vm << 16) | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FMAX Vd.4S, Vn.4S, Vm.4S */
static void a64_fmax_4s(jit_buf_t *b, int vd, int vn, int vm)
{
    uint32_t insn = 0x4E20F400 | (vm << 16) | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FMIN Vd.4S, Vn.4S, Vm.4S */
static void a64_fmin_4s(jit_buf_t *b, int vd, int vn, int vm)
{
    uint32_t insn = 0x4EA0F400 | (vm << 16) | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FNEG Vd.4S, Vn.4S */
static void a64_fneg_4s(jit_buf_t *b, int vd, int vn)
{
    uint32_t insn = 0x6EA0F800 | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FABS Vd.4S, Vn.4S */
static void a64_fabs_4s(jit_buf_t *b, int vd, int vn)
{
    uint32_t insn = 0x4EA0F800 | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FSQRT Vd.4S, Vn.4S */
static void a64_fsqrt_4s(jit_buf_t *b, int vd, int vn)
{
    uint32_t insn = 0x6EA1F800 | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FRSQRTE Vd.4S, Vn.4S (reciprocal square root estimate) */
static void a64_frsqrte_4s(jit_buf_t *b, int vd, int vn)
{
    uint32_t insn = 0x6EA1D800 | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FRECPE Vd.4S, Vn.4S (reciprocal estimate) */
static void a64_frecpe_4s(jit_buf_t *b, int vd, int vn)
{
    uint32_t insn = 0x4EA1D800 | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FMLA Vd.4S, Vn.4S, Vm.4S (fused multiply-add: Vd = Vd + Vn * Vm) */
static void a64_fmla_4s(jit_buf_t *b, int vd, int vn, int vm)
{
    uint32_t insn = 0x4E20CC00 | (vm << 16) | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FMLS Vd.4S, Vn.4S, Vm.4S (fused multiply-sub: Vd = Vd - Vn * Vm) */
static void a64_fmls_4s(jit_buf_t *b, int vd, int vn, int vm)
{
    uint32_t insn = 0x4EA0CC00 | (vm << 16) | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* DUP Vd.4S, Vn.S[lane] — duplicate scalar to all lanes */
static void a64_dup_4s_lane(jit_buf_t *b, int vd, int vn, int lane)
{
    uint32_t imm5 = ((lane & 3) << 3) | 4;
    uint32_t insn = 0x4E000400 | (imm5 << 16) | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* MOVI Vd.4S, #0 — zero vector */
static void a64_movi_zero(jit_buf_t *b, int vd)
{
    uint32_t insn = 0x4F000400 | vd;
    a64_emit32(b, insn);
}

/* MOV Vd, Vn (ORR Vd, Vn, Vn) */
static void a64_mov_v(jit_buf_t *b, int vd, int vn)
{
    uint32_t insn = 0x4EA01C00 | (vn << 16) | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FADDP Vd.4S, Vn.4S, Vm.4S — pairwise add */
static void a64_faddp_4s(jit_buf_t *b, int vd, int vn, int vm)
{
    uint32_t insn = 0x6E20D400 | (vm << 16) | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* Additional NEON integer/conversion emitters */

/* FCVTNS Vd.4S, Vn.4S — float to signed int round-to-nearest */
static void a64_fcvtns_4s(jit_buf_t *b, int vd, int vn)
{
    uint32_t insn = 0x4EA1A800 | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* SCVTF Vd.4S, Vn.4S — signed int to float */
static void a64_scvtf_4s(jit_buf_t *b, int vd, int vn)
{
    uint32_t insn = 0x4E21D800 | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* SSHLL Vd.8H, Vn.8B, #0 — sign-extend int8 → int16 (8 elements) */
static void a64_sxtl_8h(jit_buf_t *b, int vd, int vn)
{
    uint32_t insn = 0x0F080400 | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* SSHLL Vd.4S, Vn.4H, #0 — sign-extend int16 → int32 (4 elements) */
static void a64_sxtl_4s(jit_buf_t *b, int vd, int vn)
{
    uint32_t insn = 0x0F100400 | (vn << 5) | vd;
    a64_emit32(b, insn);
}

/* FMOV Sd, Wn — move 32-bit GP reg into FP scalar */
static void a64_fmov_w_to_s(jit_buf_t *b, int sd, int wn)
{
    uint32_t insn = 0x1E270000 | ((uint32_t)wn << 5) | (uint32_t)sd;
    a64_emit32(b, insn);
}

/* LDRH Wt, [Xn] — load unsigned halfword (for f16 scale) */
static void a64_ldrh(jit_buf_t *b, int wt, int xn)
{
    uint32_t insn = 0x79400000 | ((uint32_t)xn << 5) | (uint32_t)wt;
    a64_emit32(b, insn);
}

/* LD1 {Vt.16B}, [Xn], #16 — load 16 bytes with post-increment */
static void a64_ld1_16b_post(jit_buf_t *b, int vt, int xn)
{
    uint32_t insn = 0x4CDF7000 | ((uint32_t)xn << 5) | (uint32_t)vt;
    a64_emit32(b, insn);
}

/* ADD Vd.4S, Vn.4S, Vm.4S — integer add of 32-bit lanes */
static void a64_add_v4s(jit_buf_t *b, int vd, int vn, int vm)
{
    uint32_t insn = 0x4E208400 | ((uint32_t)vm << 16) | ((uint32_t)vn << 5) | (uint32_t)vd;
    a64_emit32(b, insn);
}

/* Broadcast float constant to all 4 lanes of Vd using Xscratch */
static void a64_vld_const_f32(jit_buf_t *b, int vd, int x_scratch, float val)
{
    union { float f; uint32_t u; } cv;
    cv.f = val;
    a64_movz(b, x_scratch, (uint16_t)(cv.u & 0xFFFF), 0);
    if ((cv.u >> 16) & 0xFFFF) a64_movk(b, x_scratch, (uint16_t)(cv.u >> 16), 1);
    a64_fmov_w_to_s(b, vd, x_scratch);
    a64_dup_4s_lane(b, vd, vd, 0);
}

/* Convert f16 bit-pattern in Wn to f32 in Sd:
 * FMOV Hd, Wn + FCVT Sd, Hn */
static void a64_fp16_to_f32(jit_buf_t *b, int sd, int wn)
{
    /* FMOV Hd, Wn: encoding 0001 1110 1110 0111 0000 00 Wn Hd */
    a64_emit32(b, 0x1EE70000 | ((uint32_t)wn << 5) | (uint32_t)sd);
    /* FCVT Sd, Hn: encoding 0001 1110 1110 0100 0000 00 Hn Sd */
    a64_emit32(b, 0x1EE40000 | ((uint32_t)sd << 5) | (uint32_t)sd);
}

/* =============================================================================
 * Function Prologue / Epilogue
 * =============================================================================*/

void jit_prologue(jit_buf_t *b)
{
    a64_stp_pre(b, X29, X30, SP, -16);
    a64_mov_reg(b, X29, SP);
    a64_stp_pre(b, X19, X20, SP, -16);
    a64_stp_pre(b, X21, X22, SP, -16);
    a64_stp_pre(b, X23, X24, SP, -16);
}

void jit_epilogue(jit_buf_t *b)
{
    a64_ldp_post(b, X23, X24, SP, 16);
    a64_ldp_post(b, X21, X22, SP, 16);
    a64_ldp_post(b, X19, X20, SP, 16);
    a64_ldp_post(b, X29, X30, SP, 16);
    a64_ret(b);
}

/* =============================================================================
 * SiLU Kernel
 *
 * void silu_jit(float *x)   — n is a compile-time constant, loop unrolled
 *
 * SiLU(x) = x * sigmoid(x)
 * sigmoid(x) ≈ 0.5 + 0.25·x − (1/48)·x³ + (1/576)·x⁵   [Taylor, accurate |x|<3.5]
 * Clamped to [0,1] via FMAX/FMIN for stability outside that range.
 * =============================================================================*/

jit_silu_fn jit_compile_silu_kernel(int n)
{
    int vecs = n / 4;
    if (vecs < 1) return NULL;

    jit_buf_t *b = jit_create(512 + vecs * 72);
    if (!b) return NULL;

    jit_prologue(b);
    a64_mov_reg(b, X19, X0);   /* X19 = x pointer */

    /* Load polynomial constants into callee-saved NEON regs V16..V21 */
    a64_vld_const_f32(b, V16, X8, 0.5f);
    a64_vld_const_f32(b, V17, X8, 0.25f);
    a64_vld_const_f32(b, V18, X8, -0.020833f);   /* -1/48 */
    a64_vld_const_f32(b, V19, X8, 0.001736f);    /*  1/576 */
    a64_vld_const_f32(b, V20, X8, 0.0f);
    a64_vld_const_f32(b, V21, X8, 1.0f);

    /* Unrolled vectorised body */
    for (int i = 0; i < vecs; i++) {
        a64_ld1_4s(b, V0, X19);            /* V0  = x[i*4 .. i*4+3]  */
        a64_fmul_4s(b, V1, V0, V0);        /* V1  = x²               */
        a64_fmul_4s(b, V2, V1, V0);        /* V2  = x³               */
        a64_fmul_4s(b, V3, V2, V1);        /* V3  = x⁵               */
        a64_mov_v(b, V4, V16);             /* V4  = 0.5               */
        a64_fmla_4s(b, V4, V17, V0);       /* V4 += 0.25·x            */
        a64_fmla_4s(b, V4, V18, V2);       /* V4 -= (1/48)·x³         */
        a64_fmla_4s(b, V4, V19, V3);       /* V4 += (1/576)·x⁵        */
        a64_fmax_4s(b, V4, V4, V20);       /* clamp sigmoid ≥ 0       */
        a64_fmin_4s(b, V4, V4, V21);       /* clamp sigmoid ≤ 1       */
        a64_fmul_4s(b, V0, V0, V4);        /* V0  = x·sigmoid         */
        a64_st1_4s_post(b, V0, X19);       /* store, ptr += 16        */
    }

    /* Scalar tail (n % 4 elements) — reuse the same polynomial */
    for (int t = 0; t < n % 4; t++) {
        /* LDR S0, [X19] */
        a64_emit32(b, 0xBD400000u | ((uint32_t)X19 << 5) | V0);
        a64_fmul_4s(b, V1, V0, V0);
        a64_fmul_4s(b, V2, V1, V0);
        a64_fmul_4s(b, V3, V2, V1);
        a64_mov_v(b, V4, V16);
        a64_fmla_4s(b, V4, V17, V0);
        a64_fmla_4s(b, V4, V18, V2);
        a64_fmla_4s(b, V4, V19, V3);
        a64_fmax_4s(b, V4, V4, V20);
        a64_fmin_4s(b, V4, V4, V21);
        a64_fmul_4s(b, V0, V0, V4);
        /* STR S0, [X19], #4 (post-index) */
        a64_emit32(b, 0xBC000C00u | (4u << 12) | ((uint32_t)X19 << 5) | V0);
    }

    jit_epilogue(b);
#if defined(__GNUC__) || defined(__clang__)
    __builtin___clear_cache((char *)b->code, (char *)b->code + b->len);
#endif
    return (jit_silu_fn)(void *)b->code;
}

/* =============================================================================
 * RMSNorm + Scale Kernel
 *
 * void rmsnorm_scale_jit(float *out, const float *x,
 *                        const float *w, float scale)
 * x0=out, x1=x, x2=w; scale arrives in V0.s[0] (AArch64 float ABI)
 * dim is compile-time constant.
 *
 * out[i] = x[i] * (1/sqrt(mean(x²)+ε)) * scale * w[i]
 * =============================================================================*/

jit_rmsnorm_scale_fn jit_compile_rmsnorm_scale_kernel(int dim)
{
    int vecs = dim / 4;
    if (vecs < 1) return NULL;

    jit_buf_t *b = jit_create(512 + vecs * 96);
    if (!b) return NULL;

    jit_prologue(b);

    a64_mov_reg(b, X19, X0);             /* X19 = out */
    a64_mov_reg(b, X20, X1);             /* X20 = x   */
    a64_mov_reg(b, X21, X2);             /* X21 = w   */
    a64_dup_4s_lane(b, V22, V0, 0);      /* V22 = {scale,scale,scale,scale} */

    /* ── Pass 1: sum of squares ── */
    a64_movi_zero(b, V24);
    a64_mov_reg(b, X9, X20);
    for (int i = 0; i < vecs; i++) {
        a64_ld1_4s_post(b, V0, X9);
        a64_fmla_4s(b, V24, V0, V0);
    }
    for (int t = 0; t < dim % 4; t++) {
        a64_emit32(b, 0xBD400000u | ((uint32_t)X9  << 5) | V0);
        a64_dup_4s_lane(b, V1, V0, 0);
        a64_fmla_4s(b, V24, V1, V1);
        a64_add_imm(b, X9, X9, 4);
    }

    /* Horizontal reduce V24 → S24 */
    a64_faddp_4s(b, V24, V24, V24);
    a64_faddp_4s(b, V24, V24, V24);

    /* sum_sq / dim */
    a64_vld_const_f32(b, V25, X8, (float)dim);
    /* FDIV S24, S24, S25 */
    a64_emit32(b, 0x1E201800u | ((uint32_t)V25 << 16) | ((uint32_t)V24 << 5) | V24);
    /* + epsilon */
    a64_vld_const_f32(b, V25, X8, 1e-6f);
    /* FADD S24, S24, S25 */
    a64_emit32(b, 0x1E202800u | ((uint32_t)V25 << 16) | ((uint32_t)V24 << 5) | V24);

    /* rsqrt: FRSQRTE + 1 Newton step */
    a64_frsqrte_4s(b, V26, V24);
    a64_fmul_4s(b, V27, V26, V26);
    /* FRSQRTS V27.4S, V24.4S, V27.4S → V27 = (3 - V24*V27) / 2 */
    a64_emit32(b, 0x4EA0FC00u | ((uint32_t)V24 << 16) | ((uint32_t)V27 << 5) | V27);
    a64_fmul_4s(b, V26, V26, V27);

    /* Broadcast scalar rsqrt × scale → V25 */
    a64_dup_4s_lane(b, V25, V26, 0);
    a64_fmul_4s(b, V25, V25, V22);

    /* ── Pass 2: out[i] = x[i] * (rsqrt * scale) * w[i] ── */
    a64_mov_reg(b, X9, X20);
    for (int i = 0; i < vecs; i++) {
        a64_ld1_4s_post(b, V0, X9);
        a64_fmul_4s(b, V0, V0, V25);
        a64_ld1_4s_post(b, V1, X21);
        a64_fmul_4s(b, V0, V0, V1);
        a64_st1_4s_post(b, V0, X19);
    }
    for (int t = 0; t < dim % 4; t++) {
        a64_emit32(b, 0xBD400000u | ((uint32_t)X9  << 5) | V0);
        a64_fmul_4s(b, V0, V0, V25);
        a64_emit32(b, 0xBD400000u | ((uint32_t)X21 << 5) | V1);
        a64_fmul_4s(b, V0, V0, V1);
        a64_emit32(b, 0xBC000C00u | (4u << 12) | ((uint32_t)X19 << 5) | V0);
        a64_add_imm(b, X9,  X9,  4);
        a64_add_imm(b, X21, X21, 4);
    }

    jit_epilogue(b);
#if defined(__GNUC__) || defined(__clang__)
    __builtin___clear_cache((char *)b->code, (char *)b->code + b->len);
#endif
    return (jit_rmsnorm_scale_fn)(void *)b->code;
}

/* =============================================================================
 * Q8_0 GEMV Kernel
 *
 * void q8_gemv_jit(float *out, const void *W, const float *x)
 * x0=out, x1=W (Q8_0), x2=x (f32)
 * rows and cols are compile-time constants; cols must be a multiple of 32.
 *
 * Q8_0 block: [2-byte f16 scale][32 × int8 quants] = 34 bytes per 32 elements
 * Per block: dequant w_f32[k] = scale * (int8)q[k],  acc += dot(w_f32, x)
 * =============================================================================*/

jit_gemv_q8_fn jit_compile_q8_gemv(int rows, int cols)
{
    if (rows <= 0 || cols <= 0 || (cols % 32) != 0) return NULL;

    int bpr = cols / 32;   /* blocks per row */
    /* Guard against runaway code size for huge matrices */
    if ((size_t)rows * bpr * 320 > (1u << 20)) return NULL;

    jit_buf_t *b = jit_create(512 + rows * bpr * 320);
    if (!b) return NULL;

    jit_prologue(b);

    a64_mov_reg(b, X19, X0);    /* X19 = out */
    a64_mov_reg(b, X20, X1);    /* X20 = W (advances through rows) */
    a64_mov_reg(b, X21, X2);    /* X21 = x  (reset each row) */

    for (int r = 0; r < rows; r++) {
        a64_movi_zero(b, V24);           /* V24 = row accumulator */
        a64_mov_reg(b, X10, X21);        /* X10 = x walk ptr */

        for (int blk = 0; blk < bpr; blk++) {
            /* Load f16 scale → S26 → broadcast V26.4S */
            a64_ldrh(b, X8, X20);
            a64_add_imm(b, X20, X20, 2);
            a64_fp16_to_f32(b, V26, X8);
            a64_dup_4s_lane(b, V26, V26, 0);

            /* Load 32 int8 quants into V0 (16B) and V1 (16B) */
            a64_ld1_16b_post(b, V0, X20);
            a64_ld1_16b_post(b, V1, X20);

            /* Process V0 lower 8 bytes: int8[0..7] → f32 two groups of 4 */
            a64_sxtl_8h(b, V2, V0);                         /* int8[0..7]  → int16[0..7]  */
            a64_sxtl_4s(b, V3, V2);                         /* int16[0..3] → int32[0..3]  */
            a64_scvtf_4s(b, V3, V3);
            a64_fmul_4s(b, V3, V3, V26);
            a64_ld1_4s_post(b, V4, X10);
            a64_fmla_4s(b, V24, V3, V4);

            /* int16[4..7] → int32 using SSHLL2 (upper half) */
            a64_emit32(b, 0x4F100400u | ((uint32_t)V2 << 5) | V3);  /* SSHLL2 4S, 8H, #0 */
            a64_scvtf_4s(b, V3, V3);
            a64_fmul_4s(b, V3, V3, V26);
            a64_ld1_4s_post(b, V4, X10);
            a64_fmla_4s(b, V24, V3, V4);

            /* V0 upper 8 bytes: int8[8..15] via SSHLL2 */
            a64_emit32(b, 0x4F080400u | ((uint32_t)V0 << 5) | V2);  /* SSHLL2 8H, 16B, #0 */
            a64_sxtl_4s(b, V3, V2);
            a64_scvtf_4s(b, V3, V3);
            a64_fmul_4s(b, V3, V3, V26);
            a64_ld1_4s_post(b, V4, X10);
            a64_fmla_4s(b, V24, V3, V4);

            a64_emit32(b, 0x4F100400u | ((uint32_t)V2 << 5) | V3);
            a64_scvtf_4s(b, V3, V3);
            a64_fmul_4s(b, V3, V3, V26);
            a64_ld1_4s_post(b, V4, X10);
            a64_fmla_4s(b, V24, V3, V4);

            /* Process V1 (int8[16..31]) — identical pattern */
            a64_sxtl_8h(b, V2, V1);
            a64_sxtl_4s(b, V3, V2);
            a64_scvtf_4s(b, V3, V3);
            a64_fmul_4s(b, V3, V3, V26);
            a64_ld1_4s_post(b, V4, X10);
            a64_fmla_4s(b, V24, V3, V4);

            a64_emit32(b, 0x4F100400u | ((uint32_t)V2 << 5) | V3);
            a64_scvtf_4s(b, V3, V3);
            a64_fmul_4s(b, V3, V3, V26);
            a64_ld1_4s_post(b, V4, X10);
            a64_fmla_4s(b, V24, V3, V4);

            a64_emit32(b, 0x4F080400u | ((uint32_t)V1 << 5) | V2);
            a64_sxtl_4s(b, V3, V2);
            a64_scvtf_4s(b, V3, V3);
            a64_fmul_4s(b, V3, V3, V26);
            a64_ld1_4s_post(b, V4, X10);
            a64_fmla_4s(b, V24, V3, V4);

            a64_emit32(b, 0x4F100400u | ((uint32_t)V2 << 5) | V3);
            a64_scvtf_4s(b, V3, V3);
            a64_fmul_4s(b, V3, V3, V26);
            a64_ld1_4s_post(b, V4, X10);
            a64_fmla_4s(b, V24, V3, V4);
        }

        /* Horizontal reduce V24 → S24 */
        a64_faddp_4s(b, V24, V24, V24);
        a64_faddp_4s(b, V24, V24, V24);

        /* STR S24, [X19], #4 */
        a64_emit32(b, 0xBC000C00u | (4u << 12) | ((uint32_t)X19 << 5) | V24);
    }

    jit_epilogue(b);
#if defined(__GNUC__) || defined(__clang__)
    __builtin___clear_cache((char *)b->code, (char *)b->code + b->len);
#endif
    return (jit_gemv_q8_fn)(void *)b->code;
}

/* =============================================================================
 * Runtime Info
 * =============================================================================*/

void jit_init(void) { /* static pool — nothing to do */ }

int jit_selftest(void)
{
    jit_silu_fn fn = jit_compile_silu_kernel(4);
    if (!fn) return -1;
    float x[4] = { 1.0f, -1.0f, 2.0f, -2.0f };
    fn(x);
    /* SiLU(1) ≈ 0.731 > 0, SiLU(-1) ≈ -0.269 < 0 */
    if (x[0] <= 0.0f || x[1] >= 0.0f) return -1;
    return 0;
}

int jit_kernel_count(void) { return a64_jit_buf_count; }
int jit_code_bytes(void)   { return a64_jit_pool_offset; }

#endif /* __aarch64__ */
