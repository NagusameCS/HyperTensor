/*
 * HyperTensor Token-Space Communication — Implementation
 *
 * Enables direct token/logit/distributional exchange between LLMs
 * without converting to text. This is the "distributional structure"
 * that allows two models to communicate in their native representation.
 */

#include "runtime/nn/token_comm.h"

#ifdef HYPERTENSOR_HOSTED
#include "hal.h"
#include <math.h>
#else
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Initialization
 * ════════════════════════════════════════════════════════════════════════ */

int token_comm_init(token_channel_t *ch, token_comm_mode_t mode,
                    int vocab_a, int vocab_b) {
    if (!ch || vocab_a <= 0 || vocab_b <= 0) return -1;

    kmemset(ch, 0, sizeof(*ch));
    ch->mode = mode;
    ch->vocab_a_size = vocab_a;
    ch->vocab_b_size = vocab_b;

    /* Allocate shared logit buffer for soft communication */
    if (mode != TOKEN_COMM_HARD) {
        int max_vocab = vocab_a > vocab_b ? vocab_a : vocab_b;
        ch->shared_logits = (float *)tensor_alloc(max_vocab * sizeof(float));
        if (!ch->shared_logits) return -1;
        kmemset(ch->shared_logits, 0, max_vocab * sizeof(float));
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Vocabulary Mapping
 *
 * When two models use different tokenizers, we build a mapping between
 * their vocabularies by exact string match. Tokens that exist in both
 * vocabularies get mapped directly; others get a -1 (unmapped).
 * ════════════════════════════════════════════════════════════════════════ */

/* Simple hash for vocab lookup */
static uint32_t vocab_hash(const char *s, int len) {
    uint32_t h = 5381;
    for (int i = 0; i < len; i++)
        h = ((h << 5) + h) + (uint8_t)s[i];
    return h;
}

int token_comm_build_vocab_map(token_channel_t *ch,
                               const char **vocab_a_strs, const int *vocab_a_lens,
                               int n_vocab_a,
                               const char **vocab_b_strs, const int *vocab_b_lens,
                               int n_vocab_b) {
    if (!ch || !vocab_a_strs || !vocab_b_strs) return -1;

    /* Allocate mapping arrays */
    ch->vocab_map_a_to_b = (int32_t *)tensor_alloc(n_vocab_a * sizeof(int32_t));
    ch->vocab_map_b_to_a = (int32_t *)tensor_alloc(n_vocab_b * sizeof(int32_t));
    if (!ch->vocab_map_a_to_b || !ch->vocab_map_b_to_a) return -1;

    /* Initialize all to -1 (unmapped) */
    for (int i = 0; i < n_vocab_a; i++) ch->vocab_map_a_to_b[i] = -1;
    for (int i = 0; i < n_vocab_b; i++) ch->vocab_map_b_to_a[i] = -1;

    /* Build hash table of vocab B for O(V) mapping instead of O(V²) */
    int ht_size = 1;
    while (ht_size < n_vocab_b * 2) ht_size <<= 1;
    int ht_mask = ht_size - 1;

    int32_t *ht_keys = (int32_t *)tensor_alloc(ht_size * sizeof(int32_t));
    int32_t *ht_vals = (int32_t *)tensor_alloc(ht_size * sizeof(int32_t));
    if (!ht_keys || !ht_vals) {
        if (ht_keys) tensor_free(ht_keys);
        if (ht_vals) tensor_free(ht_vals);
        return -1;
    }
    for (int i = 0; i < ht_size; i++) ht_keys[i] = -1;

    /* Insert vocab B into hash table */
    for (int b = 0; b < n_vocab_b; b++) {
        uint32_t h = vocab_hash(vocab_b_strs[b], vocab_b_lens[b]) & ht_mask;
        while (ht_keys[h] != -1) h = (h + 1) & ht_mask;
        ht_keys[h] = b;
    }

    /* Look up each vocab A token in vocab B */
    int mapped = 0;
    for (int a = 0; a < n_vocab_a; a++) {
        uint32_t h = vocab_hash(vocab_a_strs[a], vocab_a_lens[a]) & ht_mask;
        while (ht_keys[h] != -1) {
            int b = ht_keys[h];
            if (vocab_a_lens[a] == vocab_b_lens[b] &&
                kmemcmp(vocab_a_strs[a], vocab_b_strs[b], vocab_a_lens[a]) == 0) {
                ch->vocab_map_a_to_b[a] = b;
                ch->vocab_map_b_to_a[b] = a;
                mapped++;
                break;
            }
            h = (h + 1) & ht_mask;
        }
    }

    tensor_free(ht_keys);
    tensor_free(ht_vals);

    ch->has_vocab_map = 1;
    ch->vocab_a_size = n_vocab_a;
    ch->vocab_b_size = n_vocab_b;

    return mapped;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Entropy / Uncertainty
 * ════════════════════════════════════════════════════════════════════════ */

static float compute_entropy(const float *logits, int n) {
    /* Softmax + entropy in one pass */
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(logits[i] - mx);

    float entropy = 0.0f;
    for (int i = 0; i < n; i++) {
        float p = expf(logits[i] - mx) / sum;
        if (p > 1e-10f) entropy -= p * logf(p);
    }
    return entropy;
}

static float find_max_prob(const float *logits, int n) {
    float mx = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > mx) mx = logits[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += expf(logits[i] - mx);
    return expf(mx - mx) / sum; /* = 1/sum for max element */
}

/* ═══════════════════════════════════════════════════════════════════════
 * Ring Buffer Operations
 * ════════════════════════════════════════════════════════════════════════ */

static void ring_push(token_channel_t *ch, const token_message_t *msg) {
    ch->ring[ch->ring_head] = *msg;
    /* Don't copy the logits pointer — it's in shared_logits */
    ch->ring[ch->ring_head].logits = (void *)0;

    ch->ring_head = (ch->ring_head + 1) % TOKEN_COMM_RING_SIZE;
    if (ch->ring_count < TOKEN_COMM_RING_SIZE)
        ch->ring_count++;
    else
        ch->ring_tail = (ch->ring_tail + 1) % TOKEN_COMM_RING_SIZE;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Send Operations
 * ════════════════════════════════════════════════════════════════════════ */

int token_comm_send_token(token_channel_t *ch, int32_t token,
                          int position, int model_id) {
    if (!ch) return -1;

    token_message_t msg;
    kmemset(&msg, 0, sizeof(msg));
    msg.token = token;
    msg.position = position;
    msg.model_id = model_id;
    msg.seq_id = ch->messages_sent;
    msg.timestamp_us = hal_timer_us();

    ring_push(ch, &msg);
    ch->messages_sent++;
    ch->hard_exchanges++;
    return 0;
}

int token_comm_send_logits(token_channel_t *ch, const float *logits,
                           int vocab_size, int position, int model_id) {
    if (!ch || !logits || vocab_size <= 0) return -1;
    if (ch->mode == TOKEN_COMM_HARD) return -1;

    /* Copy logits to shared buffer */
    int max_vocab = ch->vocab_a_size > ch->vocab_b_size ?
                    ch->vocab_a_size : ch->vocab_b_size;
    int copy_size = vocab_size < max_vocab ? vocab_size : max_vocab;

    if (ch->shared_logits) {
        kmemcpy(ch->shared_logits, logits, copy_size * sizeof(float));
        /* Zero-pad if needed */
        if (copy_size < max_vocab)
            kmemset(ch->shared_logits + copy_size, 0,
                    (max_vocab - copy_size) * sizeof(float));
    }

    /* Find argmax for the hard token */
    int32_t best = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }

    token_message_t msg;
    kmemset(&msg, 0, sizeof(msg));
    msg.token = best;
    msg.position = position;
    msg.model_id = model_id;
    msg.vocab_size = vocab_size;
    msg.entropy = compute_entropy(logits, vocab_size);
    msg.max_prob = find_max_prob(logits, vocab_size);
    msg.perplexity = expf(msg.entropy);
    msg.seq_id = ch->messages_sent;
    msg.timestamp_us = hal_timer_us();

    ring_push(ch, &msg);
    ch->messages_sent++;
    ch->soft_exchanges++;

    /* Update running average entropy */
    float alpha = 0.01f;
    ch->avg_entropy = ch->avg_entropy * (1.0f - alpha) + msg.entropy * alpha;

    return 0;
}

int token_comm_send_topk(token_channel_t *ch,
                         const topk_entry_t *topk, int k,
                         int32_t selected_token, int position, int model_id) {
    if (!ch || !topk || k <= 0) return -1;

    token_message_t msg;
    kmemset(&msg, 0, sizeof(msg));
    msg.token = selected_token;
    msg.position = position;
    msg.model_id = model_id;
    msg.seq_id = ch->messages_sent;
    msg.timestamp_us = hal_timer_us();

    int copy_k = k < TOKEN_COMM_MAX_TOPK ? k : TOKEN_COMM_MAX_TOPK;
    kmemcpy(msg.topk, topk, copy_k * sizeof(topk_entry_t));
    msg.n_topk = copy_k;

    /* Compute entropy from top-K (approximate) */
    float ent = 0.0f;
    for (int i = 0; i < copy_k; i++) {
        if (topk[i].prob > 1e-10f)
            ent -= topk[i].prob * logf(topk[i].prob);
    }
    msg.entropy = ent;
    msg.max_prob = copy_k > 0 ? topk[0].prob : 0.0f;

    ring_push(ch, &msg);
    ch->messages_sent++;
    ch->soft_exchanges++;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Receive Operations
 * ════════════════════════════════════════════════════════════════════════ */

int token_comm_receive(token_channel_t *ch, token_message_t *out) {
    if (!ch || !out || ch->ring_count == 0) return 0;

    *out = ch->ring[ch->ring_tail];
    ch->ring_tail = (ch->ring_tail + 1) % TOKEN_COMM_RING_SIZE;
    ch->ring_count--;
    ch->messages_received++;

    /* Attach shared logits if in soft mode */
    if (ch->mode != TOKEN_COMM_HARD && ch->shared_logits) {
        out->logits = ch->shared_logits;
        out->vocab_size = ch->vocab_a_size > ch->vocab_b_size ?
                          ch->vocab_a_size : ch->vocab_b_size;
    }

    return 1;
}

int token_comm_peek(const token_channel_t *ch, token_message_t *out) {
    if (!ch || !out || ch->ring_count == 0) return 0;
    *out = ch->ring[ch->ring_tail];
    return 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Logit Remapping
 *
 * Translates a logit distribution from one model's vocabulary to another's.
 * Tokens that exist in both vocabularies get their logits transferred.
 * Unmapped tokens get -infinity (will have zero probability after softmax).
 * ════════════════════════════════════════════════════════════════════════ */

int token_comm_remap_logits(const token_channel_t *ch,
                            float *out_logits, int out_vocab,
                            const float *in_logits, int in_vocab,
                            int direction) {
    if (!ch || !out_logits || !in_logits) return -1;

    /* Initialize output to -infinity (zero probability) */
    for (int i = 0; i < out_vocab; i++)
        out_logits[i] = -1e30f;

    if (!ch->has_vocab_map) {
        /* No mapping: direct 1:1 for common range */
        int common = in_vocab < out_vocab ? in_vocab : out_vocab;
        kmemcpy(out_logits, in_logits, common * sizeof(float));
        return common;
    }

    int mapped = 0;
    if (direction == 0) {
        /* A → B */
        const int32_t *map = ch->vocab_map_a_to_b;
        for (int a = 0; a < in_vocab; a++) {
            if (map[a] >= 0 && map[a] < out_vocab) {
                out_logits[map[a]] = in_logits[a];
                mapped++;
            }
        }
    } else {
        /* B → A */
        const int32_t *map = ch->vocab_map_b_to_a;
        for (int b = 0; b < in_vocab; b++) {
            if (map[b] >= 0 && map[b] < out_vocab) {
                out_logits[map[b]] = in_logits[b];
                mapped++;
            }
        }
    }

    return mapped;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Attention Feedback
 * ════════════════════════════════════════════════════════════════════════ */

int token_comm_send_feedback(token_channel_t *ch,
                             const float *attention_weights, int seq_len) {
    if (!ch || !attention_weights || seq_len <= 0) return -1;

    if (!ch->attention_feedback) {
        ch->attention_feedback = (float *)tensor_alloc(
            TOKEN_COMM_RING_SIZE * sizeof(float));
        if (!ch->attention_feedback) return -1;
    }

    int copy_len = seq_len < TOKEN_COMM_RING_SIZE ? seq_len : TOKEN_COMM_RING_SIZE;
    kmemcpy(ch->attention_feedback, attention_weights, copy_len * sizeof(float));
    ch->feedback_len = copy_len;
    return 0;
}

int token_comm_get_feedback(const token_channel_t *ch,
                            float *out_weights, int max_len) {
    if (!ch || !out_weights || !ch->attention_feedback) return 0;

    int copy_len = ch->feedback_len < max_len ? ch->feedback_len : max_len;
    kmemcpy(out_weights, ch->attention_feedback, copy_len * sizeof(float));
    return copy_len;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Agreement Analysis
 * ════════════════════════════════════════════════════════════════════════ */

float token_comm_agreement(const token_channel_t *ch) {
    if (!ch || ch->ring_count < 2) return 0.0f;

    /* Compare consecutive messages from different models */
    int agreements = 0;
    int comparisons = 0;

    for (int i = 0; i < ch->ring_count - 1; i++) {
        int idx_a = (ch->ring_tail + i) % TOKEN_COMM_RING_SIZE;
        int idx_b = (ch->ring_tail + i + 1) % TOKEN_COMM_RING_SIZE;

        if (ch->ring[idx_a].model_id != ch->ring[idx_b].model_id &&
            ch->ring[idx_a].position == ch->ring[idx_b].position) {
            comparisons++;
            if (ch->ring[idx_a].token == ch->ring[idx_b].token)
                agreements++;
        }
    }

    return comparisons > 0 ? (float)agreements / (float)comparisons : 0.0f;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Statistics
 * ════════════════════════════════════════════════════════════════════════ */

void token_comm_stats(const token_channel_t *ch,
                      uint64_t *sent, uint64_t *received,
                      float *avg_ent, float *agreement) {
    if (!ch) return;
    if (sent)      *sent = ch->messages_sent;
    if (received)  *received = ch->messages_received;
    if (avg_ent)   *avg_ent = ch->avg_entropy;
    if (agreement) *agreement = token_comm_agreement(ch);
}

void token_comm_free(token_channel_t *ch) {
    if (!ch) return;
    if (ch->vocab_map_a_to_b) tensor_free(ch->vocab_map_a_to_b);
    if (ch->vocab_map_b_to_a) tensor_free(ch->vocab_map_b_to_a);
    if (ch->shared_logits)    tensor_free(ch->shared_logits);
    if (ch->attention_feedback) tensor_free(ch->attention_feedback);
    kmemset(ch, 0, sizeof(*ch));
}
