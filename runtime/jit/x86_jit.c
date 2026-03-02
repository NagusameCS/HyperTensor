/* =============================================================================
 * TensorOS - x86_64 JIT Compiler Implementation
 *
 * x86_64-only: ARM64 would need an AArch64 JIT backend.
 * =============================================================================*/

#ifndef __aarch64__
/* =============================================================================
 * Generates native machine code at runtime for tensor operations.
 * This is the core differentiator: an OS that JIT-compiles AI workloads
 * directly to hardware instructions with zero overhead.
 *
 * Architecture:
 *   1. Low-level emitter: byte-accurate x86_64 instruction encoding
 *   2. SSE2 SIMD emission: packed float operations
 *   3. High-level compiler: tensor ops → native code
 *
 * All generated code uses System V AMD64 ABI:
 *   Args: rdi, rsi, rdx, rcx, r8, r9  |  XMM0-7 for floats
 *   Callee-saved: rbx, rbp, r12-r15   |  Caller-saved: rax,rcx,rdx,rsi,rdi,r8-r11
 * =============================================================================*/

#include "runtime/jit/x86_jit.h"
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"

/* =============================================================================
 * Buffer Management
 *
 * Uses a static pool instead of kmalloc to avoid allocator issues
 * and ensure JIT code is always in executable kernel memory.
 * Pool lives in BSS — zero-initialized, within the kernel's identity map.
 * =============================================================================*/

#define JIT_POOL_SIZE  (1024 * 1024)  /* 1MB for all JIT code */
static uint8_t jit_pool[JIT_POOL_SIZE] __attribute__((aligned(4096)));
static int jit_pool_offset = 0;

/* Statically allocate jit_buf_t structs (max 32 concurrent buffers) */
#define JIT_MAX_BUFS 32
static jit_buf_t jit_buf_storage[JIT_MAX_BUFS];
static int jit_buf_count = 0;

jit_buf_t *jit_create(int capacity)
{
    if (jit_buf_count >= JIT_MAX_BUFS) return NULL;

    /* Align capacity to 16 bytes */
    capacity = (capacity + 15) & ~15;
    if (jit_pool_offset + capacity > JIT_POOL_SIZE) return NULL;

    jit_buf_t *b = &jit_buf_storage[jit_buf_count++];
    b->code = jit_pool + jit_pool_offset;
    b->len = 0;
    b->cap = capacity;
    jit_pool_offset += capacity;
    return b;
}

void jit_destroy(jit_buf_t *buf)
{
    /* Static pool — no deallocation needed */
    (void)buf;
}

void jit_reset(jit_buf_t *buf)
{
    if (buf) buf->len = 0;
}

jit_void_fn jit_get_fn(jit_buf_t *buf)
{
    if (!buf || !buf->code) return NULL;
    return (jit_void_fn)(void *)buf->code;
}

/* =============================================================================
 * Low-Level Byte Emission
 * =============================================================================*/

void jit_emit8(jit_buf_t *b, uint8_t v)
{
    if (b->len < b->cap)
        b->code[b->len++] = v;
}

void jit_emit16(jit_buf_t *b, uint16_t v)
{
    jit_emit8(b, v & 0xFF);
    jit_emit8(b, (v >> 8) & 0xFF);
}

void jit_emit32(jit_buf_t *b, uint32_t v)
{
    jit_emit8(b, v & 0xFF);
    jit_emit8(b, (v >> 8) & 0xFF);
    jit_emit8(b, (v >> 16) & 0xFF);
    jit_emit8(b, (v >> 24) & 0xFF);
}

void jit_emit64(jit_buf_t *b, uint64_t v)
{
    jit_emit32(b, (uint32_t)v);
    jit_emit32(b, (uint32_t)(v >> 32));
}

/* =============================================================================
 * x86_64 Encoding Helpers
 * =============================================================================*/

/* REX prefix: 0100WRXB */
static void emit_rex(jit_buf_t *b, int w, int r, int x, int bv)
{
    uint8_t rex = 0x40;
    if (w) rex |= 0x08;
    if (r) rex |= 0x04;
    if (x) rex |= 0x02;
    if (bv) rex |= 0x01;
    jit_emit8(b, rex);
}

/* REX.W for 64-bit operation with potential extended registers */
static void emit_rex_w(jit_buf_t *b, int reg, int rm)
{
    emit_rex(b, 1, reg >= 8, 0, rm >= 8);
}

/* ModR/M byte */
static void emit_modrm(jit_buf_t *b, int mod, int reg, int rm)
{
    jit_emit8(b, (uint8_t)((mod << 6) | ((reg & 7) << 3) | (rm & 7)));
}

/* ModR/M with displacement (handles 0, int8, int32 cases) */
static void emit_modrm_disp(jit_buf_t *b, int reg, int base, int32_t disp)
{
    if (base == RSP || base == R12) {
        /* RSP/R12 needs SIB byte */
        if (disp == 0 && base != RBP && base != R13) {
            emit_modrm(b, 0, reg, 4);
            jit_emit8(b, 0x24); /* SIB: scale=0, index=RSP(none), base=RSP */
        } else if (disp >= -128 && disp <= 127) {
            emit_modrm(b, 1, reg, 4);
            jit_emit8(b, 0x24);
            jit_emit8(b, (uint8_t)(int8_t)disp);
        } else {
            emit_modrm(b, 2, reg, 4);
            jit_emit8(b, 0x24);
            jit_emit32(b, (uint32_t)disp);
        }
    } else if (disp == 0 && base != RBP && base != R13) {
        emit_modrm(b, 0, reg, base);
    } else if (disp >= -128 && disp <= 127) {
        emit_modrm(b, 1, reg, base);
        jit_emit8(b, (uint8_t)(int8_t)disp);
    } else {
        emit_modrm(b, 2, reg, base);
        jit_emit32(b, (uint32_t)disp);
    }
}

/* =============================================================================
 * GP Register Instructions
 * =============================================================================*/

void jit_push(jit_buf_t *b, int reg)
{
    if (reg >= 8) jit_emit8(b, 0x41); /* REX.B for r8-r15 */
    jit_emit8(b, 0x50 + (reg & 7));
}

void jit_pop(jit_buf_t *b, int reg)
{
    if (reg >= 8) jit_emit8(b, 0x41);
    jit_emit8(b, 0x58 + (reg & 7));
}

void jit_ret(jit_buf_t *b)
{
    jit_emit8(b, 0xC3);
}

/* mov r64, r64 */
void jit_mov_reg_reg(jit_buf_t *b, int dst, int src)
{
    emit_rex_w(b, src, dst);
    jit_emit8(b, 0x89);
    emit_modrm(b, 3, src, dst);
}

/* mov r64, imm64 */
void jit_mov_reg_imm64(jit_buf_t *b, int reg, uint64_t imm)
{
    emit_rex(b, 1, 0, 0, reg >= 8);
    jit_emit8(b, 0xB8 + (reg & 7));
    jit_emit64(b, imm);
}

/* mov r64, [base + disp] */
void jit_mov_reg_mem(jit_buf_t *b, int dst, int base, int32_t disp)
{
    emit_rex_w(b, dst, base);
    jit_emit8(b, 0x8B);
    emit_modrm_disp(b, dst, base, disp);
}

/* mov [base + disp], r64 */
void jit_mov_mem_reg(jit_buf_t *b, int base, int32_t disp, int src)
{
    emit_rex_w(b, src, base);
    jit_emit8(b, 0x89);
    emit_modrm_disp(b, src, base, disp);
}

/* lea r64, [base + disp] */
void jit_lea(jit_buf_t *b, int dst, int base, int32_t disp)
{
    emit_rex_w(b, dst, base);
    jit_emit8(b, 0x8D);
    emit_modrm_disp(b, dst, base, disp);
}

/* add r64, r64 */
void jit_add_reg_reg(jit_buf_t *b, int dst, int src)
{
    emit_rex_w(b, src, dst);
    jit_emit8(b, 0x01);
    emit_modrm(b, 3, src, dst);
}

/* add r64, imm32 (sign-extended) */
void jit_add_reg_imm32(jit_buf_t *b, int dst, int32_t imm)
{
    emit_rex_w(b, 0, dst);
    if (imm >= -128 && imm <= 127) {
        jit_emit8(b, 0x83);
        emit_modrm(b, 3, 0, dst);
        jit_emit8(b, (uint8_t)(int8_t)imm);
    } else {
        jit_emit8(b, 0x81);
        emit_modrm(b, 3, 0, dst);
        jit_emit32(b, (uint32_t)imm);
    }
}

/* sub r64, imm32 */
void jit_sub_reg_imm32(jit_buf_t *b, int dst, int32_t imm)
{
    emit_rex_w(b, 0, dst);
    if (imm >= -128 && imm <= 127) {
        jit_emit8(b, 0x83);
        emit_modrm(b, 3, 5, dst);  /* /5 = SUB */
        jit_emit8(b, (uint8_t)(int8_t)imm);
    } else {
        jit_emit8(b, 0x81);
        emit_modrm(b, 3, 5, dst);
        jit_emit32(b, (uint32_t)imm);
    }
}

/* imul r64, r64 */
void jit_imul_reg_reg(jit_buf_t *b, int dst, int src)
{
    emit_rex_w(b, dst, src);
    jit_emit8(b, 0x0F);
    jit_emit8(b, 0xAF);
    emit_modrm(b, 3, dst, src);
}

/* cmp r64, r64 */
void jit_cmp_reg_reg(jit_buf_t *b, int a, int reg_b)
{
    emit_rex_w(b, reg_b, a);
    jit_emit8(b, 0x39);
    emit_modrm(b, 3, reg_b, a);
}

/* cmp r64, imm32 */
void jit_cmp_reg_imm32(jit_buf_t *b, int reg, int32_t imm)
{
    emit_rex_w(b, 0, reg);
    if (imm >= -128 && imm <= 127) {
        jit_emit8(b, 0x83);
        emit_modrm(b, 3, 7, reg);  /* /7 = CMP */
        jit_emit8(b, (uint8_t)(int8_t)imm);
    } else {
        jit_emit8(b, 0x81);
        emit_modrm(b, 3, 7, reg);
        jit_emit32(b, (uint32_t)imm);
    }
}

/* xor r64, r64 (used to zero a register) */
void jit_xor_reg_reg(jit_buf_t *b, int dst, int src)
{
    emit_rex_w(b, src, dst);
    jit_emit8(b, 0x31);
    emit_modrm(b, 3, src, dst);
}

/* inc r64 */
void jit_inc_reg(jit_buf_t *b, int reg)
{
    emit_rex_w(b, 0, reg);
    jit_emit8(b, 0xFF);
    emit_modrm(b, 3, 0, reg);  /* /0 = INC */
}

/* =============================================================================
 * Control Flow
 * =============================================================================*/

/* call via rax (mov rax, addr; call rax) */
void jit_call_abs(jit_buf_t *b, uint64_t addr)
{
    jit_mov_reg_imm64(b, RAX, addr);
    /* call rax: FF D0 */
    jit_emit8(b, 0xFF);
    emit_modrm(b, 3, 2, RAX); /* /2 = CALL, mod=11 for register */
}

/* Forward jump (placeholder — returns patch offset) */
int jit_jmp_fwd(jit_buf_t *b)
{
    jit_emit8(b, 0xE9); /* JMP rel32 */
    int patch = b->len;
    jit_emit32(b, 0); /* placeholder */
    return patch;
}

int jit_jl_fwd(jit_buf_t *b)
{
    jit_emit8(b, 0x0F);
    jit_emit8(b, 0x8C); /* JL rel32 */
    int patch = b->len;
    jit_emit32(b, 0);
    return patch;
}

int jit_jge_fwd(jit_buf_t *b)
{
    jit_emit8(b, 0x0F);
    jit_emit8(b, 0x8D); /* JGE rel32 */
    int patch = b->len;
    jit_emit32(b, 0);
    return patch;
}

/* Patch a forward jump to land at current position */
void jit_patch_jump(jit_buf_t *b, int patch_offset)
{
    int32_t rel = b->len - (patch_offset + 4);
    b->code[patch_offset + 0] = (uint8_t)(rel & 0xFF);
    b->code[patch_offset + 1] = (uint8_t)((rel >> 8) & 0xFF);
    b->code[patch_offset + 2] = (uint8_t)((rel >> 16) & 0xFF);
    b->code[patch_offset + 3] = (uint8_t)((rel >> 24) & 0xFF);
}

/* Backward jump: JMP to a known target */
void jit_jmp_back(jit_buf_t *b, int target_offset)
{
    jit_emit8(b, 0xE9);
    int32_t rel = target_offset - (b->len + 4);
    jit_emit32(b, (uint32_t)rel);
}

void jit_jl_back(jit_buf_t *b, int target_offset)
{
    jit_emit8(b, 0x0F);
    jit_emit8(b, 0x8C);
    int32_t rel = target_offset - (b->len + 4);
    jit_emit32(b, (uint32_t)rel);
}

void jit_jge_back(jit_buf_t *b, int target_offset)
{
    jit_emit8(b, 0x0F);
    jit_emit8(b, 0x8D);
    int32_t rel = target_offset - (b->len + 4);
    jit_emit32(b, (uint32_t)rel);
}

/* =============================================================================
 * SSE2 Packed Float Instructions
 * =============================================================================*/

/* Encode SSE prefix + opcode with REX for extended regs */
static void emit_sse_op(jit_buf_t *b, uint8_t op, int dst_xmm, int src_or_base,
                         int is_mem, int32_t disp)
{
    /* REX prefix if either register is xmm8+ or base is r8+ */
    int need_rex = (dst_xmm >= 8 || src_or_base >= 8);
    if (need_rex) {
        emit_rex(b, 0, dst_xmm >= 8, 0, src_or_base >= 8);
    }
    jit_emit8(b, 0x0F);
    jit_emit8(b, op);
    if (is_mem) {
        emit_modrm_disp(b, dst_xmm, src_or_base, disp);
    } else {
        emit_modrm(b, 3, dst_xmm, src_or_base);
    }
}

/* movups xmm, [base+disp] — unaligned load */
void jit_movups_load(jit_buf_t *b, int xmm, int base, int32_t disp)
{
    emit_sse_op(b, 0x10, xmm, base, 1, disp);
}

/* movups [base+disp], xmm — unaligned store */
void jit_movups_store(jit_buf_t *b, int base, int32_t disp, int xmm)
{
    emit_sse_op(b, 0x11, xmm, base, 1, disp);
}

/* movaps xmm, [base+disp] — aligned load (data MUST be 16-byte aligned) */
void jit_movaps_load(jit_buf_t *b, int xmm, int base, int32_t disp)
{
    emit_sse_op(b, 0x28, xmm, base, 1, disp);
}

/* movaps [base+disp], xmm — aligned store (data MUST be 16-byte aligned) */
void jit_movaps_store(jit_buf_t *b, int base, int32_t disp, int xmm)
{
    emit_sse_op(b, 0x29, xmm, base, 1, disp);
}

/* addps xmm, xmm */
void jit_addps(jit_buf_t *b, int dst_xmm, int src_xmm)
{
    emit_sse_op(b, 0x58, dst_xmm, src_xmm, 0, 0);
}

/* mulps xmm, xmm */
void jit_mulps(jit_buf_t *b, int dst_xmm, int src_xmm)
{
    emit_sse_op(b, 0x59, dst_xmm, src_xmm, 0, 0);
}

/* xorps xmm, xmm (used to zero a register) */
void jit_xorps(jit_buf_t *b, int dst_xmm, int src_xmm)
{
    emit_sse_op(b, 0x57, dst_xmm, src_xmm, 0, 0);
}

/* maxps xmm, xmm */
void jit_maxps(jit_buf_t *b, int dst_xmm, int src_xmm)
{
    emit_sse_op(b, 0x5F, dst_xmm, src_xmm, 0, 0);
}

/* shufps xmm, xmm, imm8 */
void jit_shufps(jit_buf_t *b, int dst_xmm, int src_xmm, uint8_t imm)
{
    emit_sse_op(b, 0xC6, dst_xmm, src_xmm, 0, 0);
    jit_emit8(b, imm);
}

/* movss xmm, [base+disp] — load single float */
void jit_movss_load(jit_buf_t *b, int xmm, int base, int32_t disp)
{
    jit_emit8(b, 0xF3);
    emit_sse_op(b, 0x10, xmm, base, 1, disp);
}

/* movss [base+disp], xmm — store single float */
void jit_movss_store(jit_buf_t *b, int base, int32_t disp, int xmm)
{
    jit_emit8(b, 0xF3);
    emit_sse_op(b, 0x11, xmm, base, 1, disp);
}

/* movaps xmm, xmm — register copy (aligned) */
void jit_movaps_reg(jit_buf_t *b, int dst_xmm, int src_xmm)
{
    emit_sse_op(b, 0x28, dst_xmm, src_xmm, 0, 0);
}

/* Scalar SSE2: addss, mulss, maxss */
void jit_addss(jit_buf_t *b, int d, int s) { jit_emit8(b, 0xF3); emit_sse_op(b, 0x58, d, s, 0, 0); }
void jit_mulss(jit_buf_t *b, int d, int s) { jit_emit8(b, 0xF3); emit_sse_op(b, 0x59, d, s, 0, 0); }
void jit_maxss(jit_buf_t *b, int d, int s) { jit_emit8(b, 0xF3); emit_sse_op(b, 0x5F, d, s, 0, 0); }
void jit_addss_mem(jit_buf_t *b, int xmm, int base, int32_t disp)
{
    jit_emit8(b, 0xF3);
    emit_sse_op(b, 0x58, xmm, base, 1, disp);
}

/* =============================================================================
 * Function Prologue / Epilogue
 * =============================================================================*/

void jit_prologue(jit_buf_t *b)
{
    jit_push(b, RBP);
    jit_mov_reg_reg(b, RBP, RSP);
    jit_push(b, RBX);
    jit_push(b, R12);
    jit_push(b, R13);
    jit_push(b, R14);
    jit_push(b, R15);
    /* Align stack to 16 bytes (5 pushes + return addr = 6*8 = 48, need 16-align) */
    jit_sub_reg_imm32(b, RSP, 8);
}

void jit_epilogue(jit_buf_t *b)
{
    jit_add_reg_imm32(b, RSP, 8);
    jit_pop(b, R15);
    jit_pop(b, R14);
    jit_pop(b, R13);
    jit_pop(b, R12);
    jit_pop(b, RBX);
    jit_pop(b, RBP);
    jit_ret(b);
}

/* =============================================================================
 * High-Level JIT: Compile MatMul Kernel
 *
 * Generates: void matmul(float *C, float *A, float *B, int M, int N, int K)
 * System V ABI: C=rdi, A=rsi, B=rdx  (M, N, K are compile-time constants)
 *
 * Strategy: i-k-j order with SSE2 4-wide inner j loop.
 * All address computation uses simple pointer arithmetic (no SIB).
 *
 * Register allocation:
 *   R12 = C_base,  R13 = A_base,  R14 = B_base    (callee-saved, persist)
 *   R15 = i_counter,  RBX = k_counter              (callee-saved loops)
 *   RAX = A_row pointer (A + i*K*4)
 *   RCX = B_row pointer (B + k*N*4)
 *   RDX = C_row pointer (C + i*N*4)
 *   RDI = j_byte offset (inner loop)
 * =============================================================================*/

/* Cached compiled kernels */
#define JIT_MAX_KERNELS 16
static struct {
    int M, N, K;
    jit_matmul_fn fn;
    int code_size;
} jit_kernel_cache[JIT_MAX_KERNELS];
static int jit_num_kernels = 0;
static int jit_total_bytes = 0;

/* Helper: emit shl reg, imm8 */
static void jit_shl_imm(jit_buf_t *b, int reg, uint8_t imm)
{
    emit_rex_w(b, 0, reg);
    jit_emit8(b, 0xC1);
    emit_modrm(b, 3, 4, reg);  /* /4 = SHL */
    jit_emit8(b, imm);
}

/* Helper: emit imul dst, src, imm32 */
static void jit_imul_imm32(jit_buf_t *b, int dst, int src, int32_t imm)
{
    emit_rex_w(b, dst, src);
    jit_emit8(b, 0x69);
    emit_modrm(b, 3, dst, src);
    jit_emit32(b, (uint32_t)imm);
}

jit_matmul_fn jit_compile_matmul_kernel(int M, int N, int K)
{
    /* Check cache first */
    for (int i = 0; i < jit_num_kernels; i++) {
        if (jit_kernel_cache[i].M == M && jit_kernel_cache[i].N == N &&
            jit_kernel_cache[i].K == K)
            return jit_kernel_cache[i].fn;
    }

    int j_vecs = N / 4;
    if (j_vecs < 1) return NULL; /* N must be >= 4 and multiple of 4 */

    jit_buf_t *b = jit_create(4096);
    if (!b) return NULL;

    jit_prologue(b);

    /* Save base pointers to callee-saved regs */
    jit_mov_reg_reg(b, R12, RDI); /* C */
    jit_mov_reg_reg(b, R13, RSI); /* A */
    jit_mov_reg_reg(b, R14, RDX); /* B */

    /* ---- Step 1: Zero C[M*N] using xorps + movups ---- */
    jit_xorps(b, XMM7, XMM7);
    jit_mov_reg_reg(b, RAX, R12);   /* RAX = walk pointer through C */
    jit_xor_reg_reg(b, RCX, RCX);   /* RCX = counter */
    int ztop = b->len;
    jit_movups_store(b, RAX, 0, XMM7);
    jit_add_reg_imm32(b, RAX, 16);
    jit_inc_reg(b, RCX);
    jit_cmp_reg_imm32(b, RCX, (M * N + 3) / 4);
    jit_jl_back(b, ztop);

    /* ---- Step 2: Triple loop i-k-j ---- */
    /* for (i = 0; i < M; i++) */
    jit_xor_reg_reg(b, R15, R15);   /* i = 0 */
    int i_top = b->len;

    /*   Compute C_row = C + i * N * 4  (DX = C_row pointer, persists across k) */
    jit_mov_reg_reg(b, RDX, R15);
    jit_imul_imm32(b, RDX, RDX, N * 4);
    jit_add_reg_reg(b, RDX, R12);    /* RDX = &C[i * N] */

    /*   Compute A_row_base = A + i * K * 4 */
    jit_mov_reg_reg(b, RAX, R15);
    jit_imul_imm32(b, RAX, RAX, K * 4);
    jit_add_reg_reg(b, RAX, R13);    /* RAX = &A[i * K] */
    /* Save A_row to stack since we need RAX for scratch */
    jit_push(b, RAX);

    /*   for (k = 0; k < K; k++) */
    jit_xor_reg_reg(b, RBX, RBX);   /* k = 0 */
    int k_top = b->len;

    /*     Load A[i*K + k]: address = A_row + k*4 */
    /*     RAX = stack-saved A_row */
    jit_mov_reg_mem(b, RAX, RSP, 0); /* reload A_row from stack */
    /*     movss xmm0, [RAX + RBX*4]  --- but avoid SIB encoding.
     *     Instead: compute temp = A_row + k*4 in RSI */
    jit_mov_reg_reg(b, RSI, RBX);   /* RSI = k */
    jit_shl_imm(b, RSI, 2);         /* RSI = k * 4 */
    jit_add_reg_reg(b, RSI, RAX);   /* RSI = &A[i*K + k] */
    jit_movss_load(b, XMM0, RSI, 0); /* xmm0 = A[i*K+k] (scalar) */

    /*     Broadcast xmm0 to all 4 lanes */
    jit_shufps(b, XMM0, XMM0, 0x00);

    /*     Compute B_row = B + k * N * 4 */
    jit_mov_reg_reg(b, RCX, RBX);
    jit_imul_imm32(b, RCX, RCX, N * 4);
    jit_add_reg_reg(b, RCX, R14);   /* RCX = &B[k * N] */

    /*     Inner j loop using pointer increment (no SIB needed) */
    /*     RDI = j_byte_offset, starts at 0, increments by 16 */
    jit_xor_reg_reg(b, RDI, RDI);
    int j_top = b->len;

    /*       Load B[k*N + j..j+3]: movups xmm1, [RCX + RDI] */
    /*       Use: add RCX, RDI → load [RCX] → sub RCX, RDI
     *       Actually simpler: compute temp address in RSI */
    jit_mov_reg_reg(b, RSI, RCX);
    jit_add_reg_reg(b, RSI, RDI);   /* RSI = &B[k*N] + j_bytes */
    jit_movups_load(b, XMM1, RSI, 0);

    /*       Load C[i*N + j..j+3]: movups xmm2, [RDX + RDI] */
    jit_mov_reg_reg(b, RSI, RDX);
    jit_add_reg_reg(b, RSI, RDI);   /* RSI = &C[i*N] + j_bytes */
    jit_movups_load(b, XMM2, RSI, 0);

    /*       xmm2 += xmm0 * xmm1 */
    jit_mulps(b, XMM1, XMM0);
    jit_addps(b, XMM2, XMM1);

    /*       Store C[i*N + j..j+3] */
    jit_movups_store(b, RSI, 0, XMM2);

    /*       j_byte += 16 */
    jit_add_reg_imm32(b, RDI, 16);
    jit_cmp_reg_imm32(b, RDI, j_vecs * 16);
    jit_jl_back(b, j_top);

    /*   k++ */
    jit_inc_reg(b, RBX);
    jit_cmp_reg_imm32(b, RBX, K);
    jit_jl_back(b, k_top);

    /* Pop saved A_row */
    jit_pop(b, RAX);

    /* i++ */
    jit_inc_reg(b, R15);
    jit_cmp_reg_imm32(b, R15, M);
    jit_jl_back(b, i_top);

    jit_epilogue(b);

    /* Cache the compiled kernel */
    jit_matmul_fn fn = (jit_matmul_fn)(void *)b->code;
    if (jit_num_kernels < JIT_MAX_KERNELS) {
        jit_kernel_cache[jit_num_kernels].M = M;
        jit_kernel_cache[jit_num_kernels].N = N;
        jit_kernel_cache[jit_num_kernels].K = K;
        jit_kernel_cache[jit_num_kernels].fn = fn;
        jit_kernel_cache[jit_num_kernels].code_size = b->len;
        jit_total_bytes += b->len;
        jit_num_kernels++;
    }

    /* Don't destroy buf — the code must stay alive! */
    return fn;
}

/* =============================================================================
 * High-Level JIT: Compile ReLU Kernel
 * void relu(float *out, float *in, int n)
 * Args: out=rdi, in=rsi, n=edx
 * =============================================================================*/

jit_unary_fn jit_compile_relu_kernel(int n)
{
    jit_buf_t *b = jit_create(1024);
    if (!b) return NULL;

    jit_prologue(b);

    /* R12 = out, R13 = in */
    jit_mov_reg_reg(b, R12, RDI);
    jit_mov_reg_reg(b, R13, RSI);

    /* xmm7 = {0,0,0,0} for max(0, x) */
    jit_xorps(b, XMM7, XMM7);

    int vecs = n / 4;
    jit_xor_reg_reg(b, RBX, RBX);  /* byte offset = 0 */

    if (vecs > 0) {
        int loop_top = b->len;

        /* Load 4 floats from in */
        jit_movups_load(b, XMM0, R13, 0);
        /* max(xmm0, xmm7) */
        jit_maxps(b, XMM0, XMM7);
        /* Store to out */
        jit_movups_store(b, R12, 0, XMM0);

        jit_add_reg_imm32(b, R12, 16);
        jit_add_reg_imm32(b, R13, 16);
        jit_inc_reg(b, RBX);
        jit_cmp_reg_imm32(b, RBX, vecs);
        jit_jl_back(b, loop_top);
    }

    jit_epilogue(b);

    jit_unary_fn fn = (jit_unary_fn)(void *)b->code;
    jit_total_bytes += b->len;
    jit_num_kernels++;
    return fn;
}

/* =============================================================================
 * Fused MatMul + ReLU (eliminates intermediate memory write)
 * =============================================================================*/

jit_matmul_fn jit_compile_fused_matmul_relu(int M, int N, int K)
{
    /* First compile a regular matmul */
    jit_matmul_fn base = jit_compile_matmul_kernel(M, N, K);
    if (!base) return NULL;
    /* For now, return the base matmul — a full fused kernel would interleave
     * the relu into the innermost loop. TODO: implement true fusion. */
    return base;
}

/* =============================================================================
 * JIT Subsystem Init & Stats
 * =============================================================================*/

void jit_init(void)
{
    jit_num_kernels = 0;
    jit_total_bytes = 0;
    kprintf_debug("[JIT] x86_64 JIT compiler initialized\n");
}

int jit_kernel_count(void)  { return jit_num_kernels; }
int jit_code_bytes(void)    { return jit_total_bytes; }

/* =============================================================================
 * Self-Test: Compile a small matmul, execute, verify
 * =============================================================================*/

int jit_selftest(void)
{
    /* Test: 4×4 identity × B = B  (same test as tensor_cpu_selftest) */
    static float A[16] __attribute__((aligned(16)));
    static float B[16] __attribute__((aligned(16)));
    static float C[16] __attribute__((aligned(16)));

    /* Identity matrix */
    kmemset(A, 0, sizeof(A));
    A[0] = 1.0f; A[5] = 1.0f; A[10] = 1.0f; A[15] = 1.0f;

    /* Sequential values */
    for (int i = 0; i < 16; i++)
        B[i] = (float)(i + 1);
    kmemset(C, 0, sizeof(C));

    /* JIT compile a 4×4 matmul */
    jit_matmul_fn fn = jit_compile_matmul_kernel(4, 4, 4);
    if (!fn) return -1;

    /* Execute JIT-compiled code */
    fn(C, A, B, 4, 4, 4);

    /* Verify: C should equal B */
    for (int i = 0; i < 16; i++) {
        float diff = C[i] - B[i];
        if (diff > 0.01f || diff < -0.01f)
            return -(i + 2);  /* Return negative index for debugging */
    }

    return 0;  /* Success */
}

#else /* __aarch64__ */

/* ARM64: JIT stubs — JIT not available, runtime uses interpreter path */
#include "runtime/jit/x86_jit.h"
#include "kernel/core/kernel.h"
#include <stddef.h>

jit_buf_t *jit_create(int cap) { (void)cap; return NULL; }
void jit_destroy(jit_buf_t *b) { (void)b; }
void jit_prologue(jit_buf_t *b) { (void)b; }
void jit_epilogue(jit_buf_t *b) { (void)b; }
void jit_init(void) { kprintf("[JIT] ARM64 mode: using NEON interpreter\n"); }
int  jit_selftest(void) { return 0; }

#endif /* __aarch64__ */
