/* =============================================================================
 * TensorOS - Paged Attention Implementation (vLLM-style)
 *
 * Key idea from Kwon et al. (2023): instead of pre-allocating a contiguous
 * KV cache per sequence, manage KV memory in fixed-size blocks (like virtual
 * memory pages).  A block table per sequence maps logical positions to
 * physical blocks.
 *
 * Benefits:
 *   - Near-zero memory waste (only last block may be partially filled)
 *   - Copy-on-write block sharing for beam search
 *   - Dynamic sequence expansion without reallocation
 *   - Preemption via block swapping (GPU ↔ CPU)
 *
 * Memory layout per block:
 *   K: [PA_BLOCK_SIZE × n_kv_heads × head_dim] floats
 *   V: [PA_BLOCK_SIZE × n_kv_heads × head_dim] floats
 * =============================================================================*/

#include "runtime/nn/paged_attn.h"
#include "kernel/mm/tensor_mm.h"
#include "kernel/core/perf.h"

static paged_attn_engine_t pa_engine;
static bool pa_initialized = false;

/* Per-block data size in bytes */
static uint64_t block_data_size(void)
{
    return (uint64_t)PA_BLOCK_SIZE * pa_engine.n_kv_heads * pa_engine.head_dim * sizeof(float);
}

/* =============================================================================
 * Initialization
 * =============================================================================*/

int paged_attn_init(int head_dim, int n_kv_heads, int n_heads)
{
    kmemset(&pa_engine, 0, sizeof(pa_engine));
    pa_engine.head_dim = head_dim;
    pa_engine.n_kv_heads = n_kv_heads;
    pa_engine.n_heads = n_heads;
    pa_engine.scale = 1.0f;
    /* Approximate 1/sqrt(head_dim) */
    {
        float x = (float)head_dim;
        /* Newton-Raphson rsqrt */
        union { float f; uint32_t i; } u = { .f = x };
        u.i = 0x5f3759df - (u.i >> 1);
        pa_engine.scale = u.f * (1.5f - 0.5f * x * u.f * u.f);
    }

    uint64_t bds = block_data_size();
    uint64_t total_mem = PA_MAX_BLOCKS * bds * 2; /* K + V per block */
    kprintf("[PAGED-ATTN] Allocating block pool: %d blocks × %lu bytes = %lu MB\n",
            PA_MAX_BLOCKS, (unsigned long)(bds * 2),
            (unsigned long)(total_mem / (1024 * 1024)));

    uint8_t *pool = (uint8_t *)tensor_alloc(total_mem);
    if (!pool) {
        kprintf("[PAGED-ATTN] ERROR: Cannot allocate %lu MB for block pool\n",
                (unsigned long)(total_mem / (1024 * 1024)));
        return -1;
    }
    kmemset(pool, 0, total_mem);

    /* Assign data pointers to each block */
    for (int i = 0; i < PA_MAX_BLOCKS; i++) {
        pa_engine.blocks[i].block_id = i;
        pa_engine.blocks[i].status = PA_BLOCK_FREE;
        pa_engine.blocks[i].ref_count = 0;
        pa_engine.blocks[i].n_filled = 0;
        pa_engine.blocks[i].k_data = (float *)(pool + i * bds * 2);
        pa_engine.blocks[i].v_data = (float *)(pool + i * bds * 2 + bds);
    }

    pa_initialized = true;
    kprintf("[PAGED-ATTN] Ready: %d blocks, %d tokens/block, head_dim=%d, kv_heads=%d\n",
            PA_MAX_BLOCKS, PA_BLOCK_SIZE, head_dim, n_kv_heads);
    return 0;
}

void paged_attn_destroy(void)
{
    /* Pool memory is managed by tensor_alloc, freed on arena reset */
    pa_initialized = false;
}

/* =============================================================================
 * Block Allocator
 * =============================================================================*/

static int alloc_block(void)
{
    for (int i = 0; i < PA_MAX_BLOCKS; i++) {
        if (pa_engine.blocks[i].status == PA_BLOCK_FREE) {
            pa_engine.blocks[i].status = PA_BLOCK_ALLOCATED;
            pa_engine.blocks[i].ref_count = 1;
            pa_engine.blocks[i].n_filled = 0;
            pa_engine.n_blocks_used++;
            pa_engine.blocks_allocated++;
            return i;
        }
    }
    return -1; /* Out of blocks */
}

static void free_block(int block_id)
{
    if (block_id < 0 || block_id >= PA_MAX_BLOCKS) return;
    pa_block_t *blk = &pa_engine.blocks[block_id];
    if (blk->ref_count > 1) {
        blk->ref_count--;
        return;
    }
    blk->status = PA_BLOCK_FREE;
    blk->ref_count = 0;
    blk->n_filled = 0;
    pa_engine.n_blocks_used--;
    pa_engine.blocks_freed++;
}

/* Copy-on-write: duplicate a block if shared */
static int cow_block(int block_id)
{
    pa_block_t *src = &pa_engine.blocks[block_id];
    if (src->ref_count <= 1) return block_id; /* Already exclusive */

    int new_id = alloc_block();
    if (new_id < 0) return -1;

    pa_block_t *dst = &pa_engine.blocks[new_id];
    uint64_t bds = block_data_size();
    kmemcpy(dst->k_data, src->k_data, bds);
    kmemcpy(dst->v_data, src->v_data, bds);
    dst->n_filled = src->n_filled;

    src->ref_count--;
    pa_engine.cow_copies++;
    return new_id;
}

/* =============================================================================
 * Sequence Management
 * =============================================================================*/

int pa_sequence_create(void)
{
    for (int i = 0; i < PA_MAX_SEQUENCES; i++) {
        if (!pa_engine.sequences[i].active) {
            pa_sequence_t *seq = &pa_engine.sequences[i];
            kmemset(seq, 0, sizeof(*seq));
            seq->seq_id = i;
            seq->active = true;
            pa_engine.n_sequences_active++;
            return i;
        }
    }
    return -1;
}

void pa_sequence_destroy(int seq_id)
{
    if (seq_id < 0 || seq_id >= PA_MAX_SEQUENCES) return;
    pa_sequence_t *seq = &pa_engine.sequences[seq_id];
    if (!seq->active) return;

    /* Free all blocks */
    for (int i = 0; i < seq->n_blocks; i++)
        free_block(seq->block_table[i]);

    seq->active = false;
    pa_engine.n_sequences_active--;
}

/* =============================================================================
 * Append KV Token
 * =============================================================================*/

int pa_append_kv(int seq_id, const float *k, const float *v)
{
    if (seq_id < 0 || seq_id >= PA_MAX_SEQUENCES) return -1;
    pa_sequence_t *seq = &pa_engine.sequences[seq_id];
    if (!seq->active) return -1;

    int hd = pa_engine.head_dim;
    int nkv = pa_engine.n_kv_heads;
    int kv_size = nkv * hd; /* floats per token in KV cache */

    /* Determine which block and position within it */
    int logical_block = seq->n_tokens / PA_BLOCK_SIZE;
    int pos_in_block = seq->n_tokens % PA_BLOCK_SIZE;

    /* Allocate new block if needed */
    if (pos_in_block == 0) {
        if (seq->n_blocks >= PA_MAX_BLOCKS_PER_SEQ) return -1;
        int blk_id = alloc_block();
        if (blk_id < 0) return -1;
        seq->block_table[seq->n_blocks++] = blk_id;
    }

    /* Get physical block */
    uint32_t phys_block = seq->block_table[logical_block];
    pa_block_t *blk = &pa_engine.blocks[phys_block];

    /* Copy-on-write if block is shared */
    if (blk->ref_count > 1) {
        int new_id = cow_block(phys_block);
        if (new_id < 0) return -1;
        seq->block_table[logical_block] = new_id;
        blk = &pa_engine.blocks[new_id];
    }

    /* Write K and V data for this token */
    kmemcpy(blk->k_data + pos_in_block * kv_size, k, kv_size * sizeof(float));
    kmemcpy(blk->v_data + pos_in_block * kv_size, v, kv_size * sizeof(float));
    blk->n_filled = pos_in_block + 1;
    seq->n_tokens++;

    return 0;
}

/* =============================================================================
 * Paged Attention Forward
 *
 * For each attention head, compute:
 *   scores[i] = dot(query[h], K_cache[i]) * scale    for all cached tokens
 *   probs = softmax(scores)
 *   output[h] = sum(probs[i] * V_cache[i])
 *
 * The key difference from standard attention: K/V are scattered across blocks,
 * so we iterate through the block table.
 * =============================================================================*/

int pa_attention_forward(float *output, const float *query, int seq_id)
{
    if (!pa_initialized || !output || !query) return -1;
    if (seq_id < 0 || seq_id >= PA_MAX_SEQUENCES) return -1;
    pa_sequence_t *seq = &pa_engine.sequences[seq_id];
    if (!seq->active || seq->n_tokens == 0) return -1;

    int hd = pa_engine.head_dim;
    int nh = pa_engine.n_heads;
    int nkv = pa_engine.n_kv_heads;
    int kv_rep = nh / nkv;
    float scale = pa_engine.scale;
    int kv_size = nkv * hd;
    int n_tok = seq->n_tokens;

    /* Temporary score buffer */
    static float scores[16384]; /* Max sequence length */
    if (n_tok > 16384) return -1;

    for (int h = 0; h < nh; h++) {
        int kv_h = h / kv_rep;
        const float *qh = query + h * hd;
        float *oh = output + h * hd;

        /* Phase 1: Compute attention scores across all blocks */
        float max_score = -1e30f;
        int tok_idx = 0;
        for (int b = 0; b < seq->n_blocks; b++) {
            pa_block_t *blk = &pa_engine.blocks[seq->block_table[b]];
            int block_tokens = (b == seq->n_blocks - 1) ? blk->n_filled : PA_BLOCK_SIZE;

            for (int t = 0; t < block_tokens; t++) {
                const float *k_tok = blk->k_data + t * kv_size + kv_h * hd;
                float s = 0;
                for (int d = 0; d < hd; d++)
                    s += qh[d] * k_tok[d];
                s *= scale;
                scores[tok_idx] = s;
                if (s > max_score) max_score = s;
                tok_idx++;
            }
        }

        /* Phase 2: Softmax */
        float sum_exp = 0;
        for (int i = 0; i < n_tok; i++) {
            /* Use fast exp approximation */
            float x = scores[i] - max_score;
            if (x < -88.0f) { scores[i] = 0.0f; continue; }
            union { float f; int32_t iv; } u;
            u.iv = (int32_t)(12102203.0f * x + 1065353216.0f);
            scores[i] = u.f;
            sum_exp += u.f;
        }
        if (sum_exp > 0) {
            float inv_sum = 1.0f / sum_exp;
            for (int i = 0; i < n_tok; i++)
                scores[i] *= inv_sum;
        }

        /* Phase 3: Weighted sum of V across blocks */
        for (int d = 0; d < hd; d++) oh[d] = 0.0f;

        tok_idx = 0;
        for (int b = 0; b < seq->n_blocks; b++) {
            pa_block_t *blk = &pa_engine.blocks[seq->block_table[b]];
            int block_tokens = (b == seq->n_blocks - 1) ? blk->n_filled : PA_BLOCK_SIZE;

            for (int t = 0; t < block_tokens; t++) {
                float w = scores[tok_idx];
                if (w > 1e-8f) {
                    const float *v_tok = blk->v_data + t * kv_size + kv_h * hd;
                    for (int d = 0; d < hd; d++)
                        oh[d] += w * v_tok[d];
                }
                tok_idx++;
            }
        }

        pa_engine.cache_hits++;
    }

    return 0;
}

/* =============================================================================
 * Sequence Fork (copy-on-write for beam search)
 * =============================================================================*/

int pa_sequence_fork(int parent_seq_id)
{
    if (parent_seq_id < 0 || parent_seq_id >= PA_MAX_SEQUENCES) return -1;
    pa_sequence_t *parent = &pa_engine.sequences[parent_seq_id];
    if (!parent->active) return -1;

    int child_id = pa_sequence_create();
    if (child_id < 0) return -1;

    pa_sequence_t *child = &pa_engine.sequences[child_id];
    child->n_tokens = parent->n_tokens;
    child->n_blocks = parent->n_blocks;

    /* Share blocks via reference counting (copy-on-write) */
    for (int i = 0; i < parent->n_blocks; i++) {
        child->block_table[i] = parent->block_table[i];
        pa_engine.blocks[parent->block_table[i]].ref_count++;
        pa_engine.blocks[parent->block_table[i]].status = PA_BLOCK_SHARED;
    }

    kprintf_debug("[PAGED-ATTN] Forked seq %d → %d (%d blocks shared)\n",
                  parent_seq_id, child_id, parent->n_blocks);
    return child_id;
}

/* =============================================================================
 * Block Swapping (preemption)
 * =============================================================================*/

/* CPU-side block storage for swapped-out sequences */
static struct {
    float *data;      /* Concatenated K+V data */
    int n_blocks;
    bool valid;
} swap_store[PA_MAX_SEQUENCES];

int pa_sequence_swap_out(int seq_id)
{
    if (seq_id < 0 || seq_id >= PA_MAX_SEQUENCES) return -1;
    pa_sequence_t *seq = &pa_engine.sequences[seq_id];
    if (!seq->active) return -1;

    uint64_t bds = block_data_size();
    uint64_t total = seq->n_blocks * bds * 2;
    float *swap = (float *)kmalloc(total);
    if (!swap) return -1;

    /* Copy all blocks to CPU memory */
    for (int i = 0; i < seq->n_blocks; i++) {
        pa_block_t *blk = &pa_engine.blocks[seq->block_table[i]];
        kmemcpy((uint8_t *)swap + i * bds * 2, blk->k_data, bds);
        kmemcpy((uint8_t *)swap + i * bds * 2 + bds, blk->v_data, bds);
        free_block(seq->block_table[i]);
    }

    swap_store[seq_id].data = swap;
    swap_store[seq_id].n_blocks = seq->n_blocks;
    swap_store[seq_id].valid = true;
    seq->n_blocks = 0;

    kprintf_debug("[PAGED-ATTN] Swapped out seq %d (%lu KB)\n",
                  seq_id, (unsigned long)(total / 1024));
    return 0;
}

int pa_sequence_swap_in(int seq_id)
{
    if (seq_id < 0 || seq_id >= PA_MAX_SEQUENCES) return -1;
    if (!swap_store[seq_id].valid) return -1;

    pa_sequence_t *seq = &pa_engine.sequences[seq_id];
    uint64_t bds = block_data_size();
    int n_blocks = swap_store[seq_id].n_blocks;
    float *swap = swap_store[seq_id].data;

    for (int i = 0; i < n_blocks; i++) {
        int blk_id = alloc_block();
        if (blk_id < 0) {
            /* Failed to swap in — free what we allocated */
            for (int j = 0; j < i; j++)
                free_block(seq->block_table[j]);
            seq->n_blocks = 0;
            return -1;
        }
        seq->block_table[i] = blk_id;
        pa_block_t *blk = &pa_engine.blocks[blk_id];
        kmemcpy(blk->k_data, (uint8_t *)swap + i * bds * 2, bds);
        kmemcpy(blk->v_data, (uint8_t *)swap + i * bds * 2 + bds, bds);
        blk->n_filled = PA_BLOCK_SIZE; /* Restored blocks are fully filled */
    }
    seq->n_blocks = n_blocks;

    kfree(swap);
    swap_store[seq_id].valid = false;
    swap_store[seq_id].data = NULL;

    kprintf_debug("[PAGED-ATTN] Swapped in seq %d (%d blocks)\n", seq_id, n_blocks);
    return 0;
}

/* =============================================================================
 * Statistics & Self-Test
 * =============================================================================*/

void pa_print_stats(void)
{
    kprintf("[PAGED-ATTN] Stats:\n");
    kprintf("  Blocks: %u/%d used (%.0f%% utilization)\n",
            pa_engine.n_blocks_used, PA_MAX_BLOCKS,
            100.0f * pa_engine.n_blocks_used / PA_MAX_BLOCKS);
    kprintf("  Sequences: %u active\n", pa_engine.n_sequences_active);
    kprintf("  Alloc/Free: %lu / %lu\n",
            (unsigned long)pa_engine.blocks_allocated,
            (unsigned long)pa_engine.blocks_freed);
    kprintf("  COW copies: %lu\n", (unsigned long)pa_engine.cow_copies);
    kprintf("  Cache hits: %lu\n", (unsigned long)pa_engine.cache_hits);
}

void pa_selftest(void)
{
    kprintf("[PAGED-ATTN] Running self-test...\n");

    int hd = 64, nkv = 4, nh = 4;

    if (paged_attn_init(hd, nkv, nh) != 0) {
        kprintf("  [FAIL] Init failed\n");
        return;
    }

    /* Create sequence and append tokens */
    int seq = pa_sequence_create();
    if (seq < 0) { kprintf("  [FAIL] seq create\n"); return; }

    float *k_tok = (float *)kmalloc(nkv * hd * sizeof(float));
    float *v_tok = (float *)kmalloc(nkv * hd * sizeof(float));
    float *query = (float *)kmalloc(nh * hd * sizeof(float));
    float *out   = (float *)kmalloc(nh * hd * sizeof(float));

    if (!k_tok || !v_tok || !query || !out) {
        kprintf("  [FAIL] alloc\n");
        return;
    }

    /* Fill 48 tokens (3 blocks) */
    for (int t = 0; t < 48; t++) {
        for (int i = 0; i < nkv * hd; i++) {
            k_tok[i] = 0.01f * (float)((t * 7 + i) % 100);
            v_tok[i] = 0.01f * (float)((t * 13 + i) % 100);
        }
        pa_append_kv(seq, k_tok, v_tok);
    }

    /* Query */
    for (int i = 0; i < nh * hd; i++)
        query[i] = 0.01f * (float)(i % 64);

    uint64_t t0 = rdtsc_fenced();
    int rc = pa_attention_forward(out, query, seq);
    uint64_t t1 = rdtsc_fenced();

    if (rc == 0) {
        float sum = 0;
        for (int i = 0; i < nh * hd; i++) sum += out[i];
        kprintf("  [OK] PagedAttention: 48 tokens, 3 blocks, %lu us, output_sum=%.2f\n",
                (unsigned long)perf_cycles_to_us(t1 - t0), (double)sum);
    } else {
        kprintf("  [FAIL] forward returned %d\n", rc);
    }

    /* Test fork (beam search) */
    int fork_id = pa_sequence_fork(seq);
    if (fork_id >= 0) {
        /* Append to fork — should trigger CoW */
        pa_append_kv(fork_id, k_tok, v_tok);
        kprintf("  [OK] Fork seq %d → %d (CoW), %lu cow copies\n",
                seq, fork_id, (unsigned long)pa_engine.cow_copies);
        pa_sequence_destroy(fork_id);
    }

    /* Test swap out/in */
    if (pa_sequence_swap_out(seq) == 0) {
        if (pa_sequence_swap_in(seq) == 0) {
            /* Verify still works */
            rc = pa_attention_forward(out, query, seq);
            kprintf("  [OK] Swap out/in: %s\n", rc == 0 ? "PASS" : "FAIL");
        }
    }

    pa_sequence_destroy(seq);
    pa_print_stats();

    kfree(k_tok);
    kfree(v_tok);
    kfree(query);
    kfree(out);
}
