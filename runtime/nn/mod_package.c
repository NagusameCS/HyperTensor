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

/*
 * Geodessical Modification Packaging — Implementation
 *
 * Packages original token sequences with edit operations so that
 * multi-model pipelines can refine output directly in token space
 * without the lossy detokenize → re-tokenize round-trip.
 */

#include "runtime/nn/mod_package.h"

#ifdef GEODESSICAL_HOSTED
#include "hal.h"
#else
#include "kernel/core/kernel.h"
#include "kernel/mm/tensor_mm.h"
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * Initialization
 * ════════════════════════════════════════════════════════════════════════ */

void mod_package_init(mod_package_t *pkg) {
    kmemset(pkg, 0, sizeof(*pkg));
    pkg->magic   = MOD_PKG_MAGIC;
    pkg->version = MOD_PKG_VERSION;
}

int mod_package_set_original(mod_package_t *pkg,
                             const int32_t *tokens, int n_tokens) {
    if (!pkg || !tokens || n_tokens <= 0) return -1;
    if (n_tokens > MOD_MAX_TOKENS) n_tokens = MOD_MAX_TOKENS;

    kmemcpy(pkg->original_tokens, tokens, n_tokens * sizeof(int32_t));
    pkg->n_original = n_tokens;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Edit Operations
 * ════════════════════════════════════════════════════════════════════════ */

static int add_edit(mod_package_t *pkg, mod_edit_type_t type,
                    int pos, int count,
                    const int32_t *new_tokens, int n_new,
                    float confidence) {
    if (!pkg || pkg->n_edits >= MOD_MAX_EDITS) return -1;
    if (pos < 0 || pos > pkg->n_original) return -1;
    if (n_new > MOD_MAX_EDIT_TOKENS) return -1;

    mod_edit_t *e = &pkg->edits[pkg->n_edits];
    e->type = type;
    e->pos  = pos;
    e->count = count;
    e->n_new = 0;
    e->confidence = confidence;
    e->source_model = pkg->editor_model_id;

    if (new_tokens && n_new > 0) {
        kmemcpy(e->new_tokens, new_tokens, n_new * sizeof(int32_t));
        e->n_new = n_new;
    }

    return pkg->n_edits++;
}

int mod_package_add_insert(mod_package_t *pkg, int pos,
                           const int32_t *tokens, int n_tokens,
                           float confidence) {
    if (!tokens || n_tokens <= 0) return -1;
    return add_edit(pkg, MOD_EDIT_INSERT, pos, 0, tokens, n_tokens, confidence);
}

int mod_package_add_delete(mod_package_t *pkg, int pos, int count,
                           float confidence) {
    if (count <= 0) return -1;
    return add_edit(pkg, MOD_EDIT_DELETE, pos, count, (void*)0, 0, confidence);
}

int mod_package_add_replace(mod_package_t *pkg, int pos, int count,
                            const int32_t *new_tokens, int n_new,
                            float confidence) {
    if (count <= 0 || !new_tokens || n_new <= 0) return -1;
    return add_edit(pkg, MOD_EDIT_REPLACE, pos, count, new_tokens, n_new,
                    confidence);
}

/* ═══════════════════════════════════════════════════════════════════════
 * Apply Edits
 *
 * Applies all edit operations to the original token sequence in order
 * of position. Overlapping edits are resolved by confidence score —
 * higher confidence wins if two edits touch the same region.
 * ════════════════════════════════════════════════════════════════════════ */

/* Sort edits by position (insertion sort — small N) */
static void sort_edits_by_pos(mod_edit_t *edits, int n) {
    for (int i = 1; i < n; i++) {
        mod_edit_t tmp = edits[i];
        int j = i - 1;
        while (j >= 0 && (edits[j].pos > tmp.pos ||
               (edits[j].pos == tmp.pos && edits[j].confidence < tmp.confidence))) {
            edits[j + 1] = edits[j];
            j--;
        }
        edits[j + 1] = tmp;
    }
}

int mod_package_apply(const mod_package_t *pkg,
                      int32_t *out_tokens, int max_out) {
    if (!pkg || !out_tokens || max_out <= 0) return -1;

    /* Copy edits for sorting without modifying original */
    mod_edit_t sorted[MOD_MAX_EDITS];
    int n_edits = pkg->n_edits;
    if (n_edits > MOD_MAX_EDITS) n_edits = MOD_MAX_EDITS;
    kmemcpy(sorted, pkg->edits, n_edits * sizeof(mod_edit_t));
    sort_edits_by_pos(sorted, n_edits);

    int out_pos = 0;
    int src_pos = 0;
    int edit_idx = 0;

    while (src_pos < pkg->n_original || edit_idx < n_edits) {
        /* Check if there's an edit at the current position */
        if (edit_idx < n_edits && sorted[edit_idx].pos <= src_pos) {
            mod_edit_t *e = &sorted[edit_idx];
            edit_idx++;

            switch (e->type) {
            case MOD_EDIT_INSERT:
                /* Insert new tokens before current position */
                for (int i = 0; i < e->n_new && out_pos < max_out; i++)
                    out_tokens[out_pos++] = e->new_tokens[i];
                break;

            case MOD_EDIT_DELETE:
                /* Skip 'count' tokens from original */
                src_pos += e->count;
                break;

            case MOD_EDIT_REPLACE:
                /* Skip old tokens, emit new ones */
                src_pos += e->count;
                for (int i = 0; i < e->n_new && out_pos < max_out; i++)
                    out_tokens[out_pos++] = e->new_tokens[i];
                break;
            }
        } else if (src_pos < pkg->n_original) {
            /* Copy original token */
            if (out_pos < max_out)
                out_tokens[out_pos++] = pkg->original_tokens[src_pos];
            src_pos++;
        } else {
            break;
        }
    }

    return out_pos;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Hidden State Attachment
 * ════════════════════════════════════════════════════════════════════════ */

int mod_package_attach_hidden(mod_package_t *pkg,
                              const float *hidden, int dim, int layer) {
    if (!pkg || !hidden || dim <= 0) return -1;

    if (pkg->hidden_state) tensor_free(pkg->hidden_state);
    pkg->hidden_state = (float *)tensor_alloc(dim * sizeof(float));
    if (!pkg->hidden_state) return -1;

    kmemcpy(pkg->hidden_state, hidden, dim * sizeof(float));
    pkg->hidden_dim   = dim;
    pkg->hidden_layer = layer;
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Serialization
 *
 * Binary format:
 *   [4B magic][4B version]
 *   [4B n_original][n_original × 4B tokens]
 *   [4B n_edits][n_edits × edit_record]
 *   [4B hidden_dim][hidden_dim × 4B floats]  (0 if no hidden state)
 *   [4B source_model_id][4B editor_model_id]
 *   [4B original_ppl_bits][4B modified_ppl_bits]
 * ════════════════════════════════════════════════════════════════════════ */

int mod_package_serialize(const mod_package_t *pkg,
                          void *buf, int buf_size) {
    if (!pkg || !buf) return -1;

    uint8_t *p = (uint8_t *)buf;
    uint8_t *end = p + buf_size;

    /* Header */
    if (p + 8 > end) return -1;
    kmemcpy(p, &pkg->magic, 4); p += 4;
    kmemcpy(p, &pkg->version, 4); p += 4;

    /* Original tokens */
    if (p + 4 + pkg->n_original * 4 > end) return -1;
    kmemcpy(p, &pkg->n_original, 4); p += 4;
    kmemcpy(p, pkg->original_tokens, pkg->n_original * 4);
    p += pkg->n_original * 4;

    /* Edit operations */
    if (p + 4 > end) return -1;
    kmemcpy(p, &pkg->n_edits, 4); p += 4;

    for (int i = 0; i < pkg->n_edits; i++) {
        const mod_edit_t *e = &pkg->edits[i];
        int rec_size = 4 + 4 + 4 + 4 + e->n_new * 4 + 4 + 4;
        if (p + rec_size > end) return -1;

        int32_t type = (int32_t)e->type;
        kmemcpy(p, &type, 4); p += 4;
        kmemcpy(p, &e->pos, 4); p += 4;
        kmemcpy(p, &e->count, 4); p += 4;
        kmemcpy(p, &e->n_new, 4); p += 4;
        if (e->n_new > 0) {
            kmemcpy(p, e->new_tokens, e->n_new * 4);
            p += e->n_new * 4;
        }
        kmemcpy(p, &e->confidence, 4); p += 4;
        kmemcpy(p, &e->source_model, 4); p += 4;
    }

    /* Hidden state */
    if (p + 4 > end) return -1;
    kmemcpy(p, &pkg->hidden_dim, 4); p += 4;
    if (pkg->hidden_dim > 0 && pkg->hidden_state) {
        int hs_bytes = pkg->hidden_dim * 4;
        if (p + hs_bytes > end) return -1;
        kmemcpy(p, pkg->hidden_state, hs_bytes);
        p += hs_bytes;
    }

    /* Metadata */
    if (p + 16 > end) return -1;
    kmemcpy(p, &pkg->source_model_id, 4); p += 4;
    kmemcpy(p, &pkg->editor_model_id, 4); p += 4;
    kmemcpy(p, &pkg->original_perplexity, 4); p += 4;
    kmemcpy(p, &pkg->modified_perplexity, 4); p += 4;

    return (int)(p - (uint8_t *)buf);
}

int mod_package_deserialize(mod_package_t *pkg,
                            const void *buf, int buf_size) {
    if (!pkg || !buf || buf_size < 8) return -1;

    const uint8_t *p = (const uint8_t *)buf;
    const uint8_t *end = p + buf_size;

    mod_package_init(pkg);

    /* Header */
    kmemcpy(&pkg->magic, p, 4); p += 4;
    kmemcpy(&pkg->version, p, 4); p += 4;
    if (pkg->magic != MOD_PKG_MAGIC) return -1;
    if (pkg->version > MOD_PKG_VERSION) return -1;

    /* Original tokens */
    if (p + 4 > end) return -1;
    kmemcpy(&pkg->n_original, p, 4); p += 4;
    if (pkg->n_original < 0 || pkg->n_original > MOD_MAX_TOKENS) return -1;
    if (p + pkg->n_original * 4 > end) return -1;
    kmemcpy(pkg->original_tokens, p, pkg->n_original * 4);
    p += pkg->n_original * 4;

    /* Edits */
    if (p + 4 > end) return -1;
    kmemcpy(&pkg->n_edits, p, 4); p += 4;
    if (pkg->n_edits < 0 || pkg->n_edits > MOD_MAX_EDITS) return -1;

    for (int i = 0; i < pkg->n_edits; i++) {
        mod_edit_t *e = &pkg->edits[i];
        if (p + 16 > end) return -1;

        int32_t type;
        kmemcpy(&type, p, 4); p += 4;
        e->type = (mod_edit_type_t)type;
        kmemcpy(&e->pos, p, 4); p += 4;
        kmemcpy(&e->count, p, 4); p += 4;
        kmemcpy(&e->n_new, p, 4); p += 4;

        if (e->n_new < 0 || e->n_new > MOD_MAX_EDIT_TOKENS) return -1;
        if (e->n_new > 0) {
            if (p + e->n_new * 4 > end) return -1;
            kmemcpy(e->new_tokens, p, e->n_new * 4);
            p += e->n_new * 4;
        }

        if (p + 8 > end) return -1;
        kmemcpy(&e->confidence, p, 4); p += 4;
        kmemcpy(&e->source_model, p, 4); p += 4;
    }

    /* Hidden state */
    if (p + 4 > end) return -1;
    kmemcpy(&pkg->hidden_dim, p, 4); p += 4;
    if (pkg->hidden_dim > 0) {
        int hs_bytes = pkg->hidden_dim * 4;
        if (p + hs_bytes > end) return -1;
        pkg->hidden_state = (float *)tensor_alloc(hs_bytes);
        if (pkg->hidden_state) {
            kmemcpy(pkg->hidden_state, p, hs_bytes);
        }
        p += hs_bytes;
    }

    /* Metadata */
    if (p + 16 <= end) {
        kmemcpy(&pkg->source_model_id, p, 4); p += 4;
        kmemcpy(&pkg->editor_model_id, p, 4); p += 4;
        kmemcpy(&pkg->original_perplexity, p, 4); p += 4;
        kmemcpy(&pkg->modified_perplexity, p, 4); p += 4;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Merge
 * ════════════════════════════════════════════════════════════════════════ */

int mod_package_merge(mod_package_t *dst, const mod_package_t *src) {
    if (!dst || !src) return -1;

    for (int i = 0; i < src->n_edits; i++) {
        if (dst->n_edits >= MOD_MAX_EDITS) return -1;

        const mod_edit_t *se = &src->edits[i];

        /* Check for overlap with existing edits — higher confidence wins */
        int conflict = 0;
        for (int j = 0; j < dst->n_edits; j++) {
            mod_edit_t *de = &dst->edits[j];
            /* Overlapping region check */
            int se_end = se->pos + (se->type == MOD_EDIT_INSERT ? 0 : se->count);
            int de_end = de->pos + (de->type == MOD_EDIT_INSERT ? 0 : de->count);

            if (se->pos < de_end && se_end > de->pos) {
                /* Overlapping — keep higher confidence */
                if (se->confidence > de->confidence) {
                    *de = *se; /* Replace with higher-confidence edit */
                }
                conflict = 1;
                break;
            }
        }

        if (!conflict) {
            dst->edits[dst->n_edits++] = *se;
        }
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════
 * Statistics
 * ════════════════════════════════════════════════════════════════════════ */

void mod_package_stats(const mod_package_t *pkg,
                       int *n_inserts, int *n_deletes, int *n_replaces,
                       int *tokens_added, int *tokens_removed) {
    int ins = 0, del = 0, rep = 0, added = 0, removed = 0;

    if (pkg) {
        for (int i = 0; i < pkg->n_edits; i++) {
            const mod_edit_t *e = &pkg->edits[i];
            switch (e->type) {
            case MOD_EDIT_INSERT:
                ins++;
                added += e->n_new;
                break;
            case MOD_EDIT_DELETE:
                del++;
                removed += e->count;
                break;
            case MOD_EDIT_REPLACE:
                rep++;
                added += e->n_new;
                removed += e->count;
                break;
            }
        }
    }

    if (n_inserts)      *n_inserts = ins;
    if (n_deletes)      *n_deletes = del;
    if (n_replaces)     *n_replaces = rep;
    if (tokens_added)   *tokens_added = added;
    if (tokens_removed) *tokens_removed = removed;
}

void mod_package_free(mod_package_t *pkg) {
    if (!pkg) return;
    if (pkg->hidden_state) {
        tensor_free(pkg->hidden_state);
        pkg->hidden_state = (void *)0;
    }
    pkg->hidden_dim = 0;
}
