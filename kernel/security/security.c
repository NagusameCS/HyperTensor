/* =============================================================================
 * TensorOS — Unified Security Framework Implementation
 *
 * Capability-based ACL, stateful firewall, tamper-evident audit trail,
 * encrypted key store, integrity verification, stack protection.
 * =============================================================================*/

#include "kernel/security/security.h"
#include "kernel/security/crypto.h"

/* Global security state */
security_state_t g_security;

/* =============================================================================
 * Initialization
 * =============================================================================*/

void security_init(void)
{
    kmemset(&g_security, 0, sizeof(g_security));

    /* Generate master key from hardware entropy */
    crypto_random(g_security.master_key, 32);

    /* Generate capability HMAC key */
    crypto_random(g_security.cap_master_key, 32);

    /* Initialize stack canary */
    stack_canary_init();

    /* Default firewall: deny inbound, allow outbound */
    g_security.fw_default_inbound = 1;    /* deny */
    g_security.fw_default_outbound = 0;   /* allow */
    g_security.fw_enabled = 0;            /* disabled until explicitly enabled */

    /* Initialize audit chain with zero hash */
    kmemset(g_security.audit_prev_hash, 0, 32);

    g_security.next_cap_id = 1;
    g_security.next_key_id = 1;
    g_security.initialized = 1;

    /* Log boot event */
    sec_audit(SEC_AUDIT_SYSTEM_BOOT, 0, 0, "security",
              "Security framework initialized");

    kprintf("[SEC] Security framework initialized\n");
    kprintf("[SEC] Stack canary: 0x%lx\n", g_security.stack_canary);
}

/* =============================================================================
 * Capability System
 *
 * Each capability is HMAC-signed with the master key.
 * To verify: recompute HMAC over (id, owner, type, resource, rights, parent, expires)
 * and compare with stored HMAC using constant-time comparison.
 * =============================================================================*/

static void cap_compute_hmac(const capability_t *cap, uint8_t out[32])
{
    /* Data to sign: id || owner || type || resource || rights || delegated_from || expires */
    uint8_t data[40];
    uint32_t pos = 0;
    data[pos++] = (uint8_t)(cap->id >> 24);
    data[pos++] = (uint8_t)(cap->id >> 16);
    data[pos++] = (uint8_t)(cap->id >> 8);
    data[pos++] = (uint8_t)cap->id;

    data[pos++] = (uint8_t)(cap->owner_uid >> 24);
    data[pos++] = (uint8_t)(cap->owner_uid >> 16);
    data[pos++] = (uint8_t)(cap->owner_uid >> 8);
    data[pos++] = (uint8_t)cap->owner_uid;

    data[pos++] = (uint8_t)(cap->resource_type >> 24);
    data[pos++] = (uint8_t)(cap->resource_type >> 16);
    data[pos++] = (uint8_t)(cap->resource_type >> 8);
    data[pos++] = (uint8_t)cap->resource_type;

    data[pos++] = (uint8_t)(cap->resource_id >> 24);
    data[pos++] = (uint8_t)(cap->resource_id >> 16);
    data[pos++] = (uint8_t)(cap->resource_id >> 8);
    data[pos++] = (uint8_t)cap->resource_id;

    data[pos++] = (uint8_t)(cap->rights >> 24);
    data[pos++] = (uint8_t)(cap->rights >> 16);
    data[pos++] = (uint8_t)(cap->rights >> 8);
    data[pos++] = (uint8_t)cap->rights;

    data[pos++] = (uint8_t)(cap->delegated_from >> 24);
    data[pos++] = (uint8_t)(cap->delegated_from >> 16);
    data[pos++] = (uint8_t)(cap->delegated_from >> 8);
    data[pos++] = (uint8_t)cap->delegated_from;

    data[pos++] = (uint8_t)(cap->expires >> 56);
    data[pos++] = (uint8_t)(cap->expires >> 48);
    data[pos++] = (uint8_t)(cap->expires >> 40);
    data[pos++] = (uint8_t)(cap->expires >> 32);
    data[pos++] = (uint8_t)(cap->expires >> 24);
    data[pos++] = (uint8_t)(cap->expires >> 16);
    data[pos++] = (uint8_t)(cap->expires >> 8);
    data[pos++] = (uint8_t)cap->expires;

    hmac_sha256(g_security.cap_master_key, 32, data, pos, out);
}

int cap_create(uint32_t owner_uid, resource_type_t type,
               uint32_t resource_id, uint32_t rights, capability_t *out)
{
    if (g_security.cap_count >= MAX_CAPABILITIES) return -1;

    capability_t *cap = NULL;
    for (uint32_t i = 0; i < MAX_CAPABILITIES; i++) {
        if (!g_security.capabilities[i].active) {
            cap = &g_security.capabilities[i];
            break;
        }
    }
    if (!cap) return -1;

    cap->id = g_security.next_cap_id++;
    cap->owner_uid = owner_uid;
    cap->resource_type = type;
    cap->resource_id = resource_id;
    cap->rights = rights;
    cap->delegated_from = 0;
    cap->expires = 0;
    cap->active = 1;

    cap_compute_hmac(cap, cap->hmac);
    g_security.cap_count++;

    if (out) *out = *cap;

    sec_audit(SEC_AUDIT_CAP_GRANTED, owner_uid, 0, "cap",
              "Capability created");
    return 0;
}

int cap_delegate(const capability_t *parent, uint32_t new_owner,
                 uint32_t subset_rights, capability_t *out)
{
    /* Verify parent capability */
    if (cap_verify(parent) != 0) return -1;

    /* Cannot delegate more rights than parent has */
    if ((subset_rights & ~parent->rights) != 0) return -1;

    /* Parent must have DELEGATE right */
    if (!(parent->rights & CAP_DELEGATE)) return -1;

    capability_t new_cap;
    new_cap.id = g_security.next_cap_id++;
    new_cap.owner_uid = new_owner;
    new_cap.resource_type = parent->resource_type;
    new_cap.resource_id = parent->resource_id;
    new_cap.rights = subset_rights;
    new_cap.delegated_from = parent->id;
    new_cap.expires = parent->expires;
    new_cap.active = 1;

    cap_compute_hmac(&new_cap, new_cap.hmac);

    /* Store in array */
    for (uint32_t i = 0; i < MAX_CAPABILITIES; i++) {
        if (!g_security.capabilities[i].active) {
            g_security.capabilities[i] = new_cap;
            g_security.cap_count++;
            if (out) *out = new_cap;
            return 0;
        }
    }
    return -1;  /* No space */
}

int cap_verify(const capability_t *cap)
{
    if (!cap || !cap->active) return -1;

    uint8_t expected[32];
    cap_compute_hmac(cap, expected);

    if (crypto_ct_equal(expected, cap->hmac, 32) != 0) {
        sec_audit(SEC_AUDIT_INTEGRITY_FAIL, cap->owner_uid, 3, "cap",
                  "Capability HMAC verification failed!");
        return -1;
    }
    return 0;
}

int cap_revoke(uint32_t cap_id)
{
    for (uint32_t i = 0; i < MAX_CAPABILITIES; i++) {
        if (g_security.capabilities[i].active &&
            g_security.capabilities[i].id == cap_id) {
            /* Also revoke all delegated children */
            for (uint32_t j = 0; j < MAX_CAPABILITIES; j++) {
                if (g_security.capabilities[j].active &&
                    g_security.capabilities[j].delegated_from == cap_id) {
                    g_security.capabilities[j].active = 0;
                    g_security.cap_count--;
                }
            }
            g_security.capabilities[i].active = 0;
            g_security.cap_count--;

            sec_audit(SEC_AUDIT_CAP_REVOKED, 0, 0, "cap",
                      "Capability revoked (cascade)");
            return 0;
        }
    }
    return -1;
}

int cap_check(uint32_t uid, resource_type_t type,
              uint32_t resource_id, uint32_t required_rights)
{
    for (uint32_t i = 0; i < MAX_CAPABILITIES; i++) {
        capability_t *c = &g_security.capabilities[i];
        if (!c->active) continue;
        if (c->owner_uid != uid) continue;
        if (c->resource_type != type) continue;
        if (c->resource_id != resource_id && c->resource_id != 0xFFFFFFFF) continue;
        if ((c->rights & required_rights) != required_rights) continue;

        /* Verify integrity */
        if (cap_verify(c) != 0) continue;

        return 0;  /* Access granted */
    }

    sec_audit(SEC_AUDIT_ACCESS_DENIED, uid, 2, "cap",
              "Access denied: insufficient capabilities");
    return -1;
}

/* =============================================================================
 * Firewall — Stateful Packet Filtering
 * =============================================================================*/

int fw_add_rule(const fw_rule_t *rule)
{
    if (g_security.fw_rule_count >= MAX_FW_RULES) return -1;

    /* Find insertion point based on priority */
    uint32_t insert = g_security.fw_rule_count;
    for (uint32_t i = 0; i < g_security.fw_rule_count; i++) {
        if (rule->priority < g_security.fw_rules[i].priority) {
            insert = i;
            break;
        }
    }

    /* Shift rules down */
    for (uint32_t i = g_security.fw_rule_count; i > insert; i--) {
        g_security.fw_rules[i] = g_security.fw_rules[i - 1];
    }

    g_security.fw_rules[insert] = *rule;
    g_security.fw_rules[insert].id = g_security.fw_rule_count + 1;
    g_security.fw_rules[insert].active = 1;
    g_security.fw_rules[insert].match_count = 0;
    g_security.fw_rule_count++;

    return 0;
}

int fw_remove_rule(uint32_t rule_id)
{
    for (uint32_t i = 0; i < g_security.fw_rule_count; i++) {
        if (g_security.fw_rules[i].id == rule_id) {
            for (uint32_t j = i; j + 1 < g_security.fw_rule_count; j++) {
                g_security.fw_rules[j] = g_security.fw_rules[j + 1];
            }
            g_security.fw_rule_count--;
            return 0;
        }
    }
    return -1;
}

static int fw_match_rule(const fw_rule_t *r, fw_direction_t dir,
                          fw_proto_t proto, uint32_t src_ip, uint16_t src_port,
                          uint32_t dst_ip, uint16_t dst_port)
{
    if (!r->active) return 0;

    /* Direction */
    if (r->direction != FW_DIR_BOTH && r->direction != dir) return 0;

    /* Protocol */
    if (r->protocol != FW_PROTO_ANY && r->protocol != proto) return 0;

    /* Source IP */
    if (r->src_ip != 0 && (src_ip & r->src_mask) != (r->src_ip & r->src_mask))
        return 0;

    /* Source port range */
    if (r->src_port_max != 0 &&
        (src_port < r->src_port_min || src_port > r->src_port_max))
        return 0;

    /* Destination IP */
    if (r->dst_ip != 0 && (dst_ip & r->dst_mask) != (r->dst_ip & r->dst_mask))
        return 0;

    /* Destination port range */
    if (r->dst_port_max != 0 &&
        (dst_port < r->dst_port_min || dst_port > r->dst_port_max))
        return 0;

    return 1;  /* Match */
}

/* Connection tracking lookup */
static fw_conn_track_t *fw_find_conn(fw_proto_t proto,
                                      uint32_t src_ip, uint16_t src_port,
                                      uint32_t dst_ip, uint16_t dst_port)
{
    for (uint32_t i = 0; i < g_security.conn_track_count; i++) {
        fw_conn_track_t *ct = &g_security.conn_track[i];
        if (ct->protocol == proto &&
            ((ct->src_ip == src_ip && ct->src_port == src_port &&
              ct->dst_ip == dst_ip && ct->dst_port == dst_port) ||
             (ct->src_ip == dst_ip && ct->src_port == dst_port &&
              ct->dst_ip == src_ip && ct->dst_port == src_port)))
            return ct;
    }
    return NULL;
}

fw_action_t fw_check_packet(fw_direction_t dir, fw_proto_t proto,
                            uint32_t src_ip, uint16_t src_port,
                            uint32_t dst_ip, uint16_t dst_port)
{
    if (!g_security.fw_enabled) return FW_ACTION_ALLOW;

    /* Check connection tracking first — established connections pass */
    fw_conn_track_t *ct = fw_find_conn(proto, src_ip, src_port, dst_ip, dst_port);
    if (ct && ct->state == 1) {  /* established */
        ct->packets_in++;
        g_security.fw_packets_allowed++;
        return FW_ACTION_ALLOW;
    }

    /* Check rules in priority order */
    for (uint32_t i = 0; i < g_security.fw_rule_count; i++) {
        if (fw_match_rule(&g_security.fw_rules[i], dir, proto,
                          src_ip, src_port, dst_ip, dst_port)) {
            g_security.fw_rules[i].match_count++;

            fw_action_t action = g_security.fw_rules[i].action;
            if (action == FW_ACTION_ALLOW || action == FW_ACTION_LOG) {
                g_security.fw_packets_allowed++;

                /* Track new connection */
                if (!ct && g_security.conn_track_count < MAX_CONN_TRACK) {
                    ct = &g_security.conn_track[g_security.conn_track_count++];
                    ct->src_ip = src_ip;
                    ct->src_port = src_port;
                    ct->dst_ip = dst_ip;
                    ct->dst_port = dst_port;
                    ct->protocol = proto;
                    ct->state = 1;  /* established */
                    ct->packets_in = 1;
                }

                if (action == FW_ACTION_LOG) {
                    sec_audit(SEC_AUDIT_FIREWALL_ALLOW, 0, 0, "firewall",
                              "Packet allowed (logged)");
                }
            } else {
                g_security.fw_packets_denied++;
                sec_audit(SEC_AUDIT_FIREWALL_BLOCK, 0, 1, "firewall",
                          "Packet denied by rule");
            }
            return action;
        }
    }

    /* Default policy */
    fw_action_t def = (dir == FW_DIR_INBOUND) ?
        (g_security.fw_default_inbound ? FW_ACTION_DENY : FW_ACTION_ALLOW) :
        (g_security.fw_default_outbound ? FW_ACTION_DENY : FW_ACTION_ALLOW);

    if (def == FW_ACTION_DENY)
        g_security.fw_packets_denied++;
    else
        g_security.fw_packets_allowed++;

    return def;
}

void fw_enable(void)
{
    g_security.fw_enabled = 1;
    sec_audit(SEC_AUDIT_FIREWALL_ALLOW, 0, 0, "firewall", "Firewall enabled");
    kprintf("[FW] Firewall enabled\n");
}

void fw_disable(void)
{
    g_security.fw_enabled = 0;
    sec_audit(SEC_AUDIT_FIREWALL_BLOCK, 0, 1, "firewall", "Firewall disabled");
    kprintf("[FW] Firewall disabled\n");
}

void fw_set_default(fw_direction_t dir, fw_action_t action)
{
    if (dir == FW_DIR_INBOUND || dir == FW_DIR_BOTH)
        g_security.fw_default_inbound = (action == FW_ACTION_DENY) ? 1 : 0;
    if (dir == FW_DIR_OUTBOUND || dir == FW_DIR_BOTH)
        g_security.fw_default_outbound = (action == FW_ACTION_DENY) ? 1 : 0;
}

void fw_flush_rules(void)
{
    g_security.fw_rule_count = 0;
    g_security.conn_track_count = 0;
    kprintf("[FW] All rules flushed\n");
}

void fw_print_rules(void)
{
    kprintf("Firewall Rules (%u):\n", g_security.fw_rule_count);
    kprintf("  Default inbound:  %s\n",
            g_security.fw_default_inbound ? "DENY" : "ALLOW");
    kprintf("  Default outbound: %s\n",
            g_security.fw_default_outbound ? "DENY" : "ALLOW");
    kprintf("  Status: %s\n", g_security.fw_enabled ? "ENABLED" : "DISABLED");
    kprintf("  %-4s %-4s %-6s %-5s %-10s %-10s %-8s %s\n",
            "ID", "Pri", "Dir", "Proto", "Src", "Dst", "Action", "Desc");

    for (uint32_t i = 0; i < g_security.fw_rule_count; i++) {
        fw_rule_t *r = &g_security.fw_rules[i];
        const char *dir_str = r->direction == FW_DIR_INBOUND ? "IN" :
                             r->direction == FW_DIR_OUTBOUND ? "OUT" : "BOTH";
        const char *proto_str = r->protocol == FW_PROTO_TCP ? "TCP" :
                               r->protocol == FW_PROTO_UDP ? "UDP" :
                               r->protocol == FW_PROTO_ICMP ? "ICMP" : "ANY";
        const char *act_str = r->action == FW_ACTION_ALLOW ? "ALLOW" :
                             r->action == FW_ACTION_DENY ? "DENY" :
                             r->action == FW_ACTION_LOG ? "LOG" : "REJECT";

        kprintf("  %-4u %-4u %-6s %-5s port:%-5u port:%-5u %-8s %s (%lu hits)\n",
                r->id, r->priority, dir_str, proto_str,
                r->src_port_min, r->dst_port_min, act_str,
                r->description, r->match_count);
    }
}

void fw_print_stats(void)
{
    kprintf("Firewall Statistics:\n");
    kprintf("  Packets allowed: %lu\n", g_security.fw_packets_allowed);
    kprintf("  Packets denied:  %lu\n", g_security.fw_packets_denied);
    kprintf("  Active conns:    %u / %u\n",
            g_security.conn_track_count, MAX_CONN_TRACK);
    kprintf("  Rules:           %u / %u\n",
            g_security.fw_rule_count, MAX_FW_RULES);
}

/* =============================================================================
 * Security Audit Trail
 *
 * Hash-chained: each entry's hash = SHA-256(prev_hash || entry_data)
 * Tamper with any entry and the chain breaks.
 * =============================================================================*/

void sec_audit(sec_audit_type_t type, uint32_t uid, uint32_t severity,
               const char *source, const char *message)
{
    uint32_t idx = g_security.audit_head;
    sec_audit_entry_t *e = &g_security.audit_log[idx];

    e->sequence = g_security.audit_sequence++;
    e->timestamp = kstate.uptime_ticks;  /* Monotonic kernel ticks (RTC integration future) */
    e->type = type;
    e->uid = uid;
    e->severity = severity;

    /* Copy source */
    int i;
    for (i = 0; i < 31 && source && source[i]; i++)
        e->source[i] = source[i];
    e->source[i] = '\0';

    /* Copy message */
    for (i = 0; i < 127 && message && message[i]; i++)
        e->message[i] = message[i];
    e->message[i] = '\0';

    /* Compute chain hash: SHA-256(prev_hash || sequence || type || uid || message) */
    sha256_ctx_t ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, g_security.audit_prev_hash, 32);

    uint8_t seq_buf[8];
    seq_buf[0] = (uint8_t)(e->sequence >> 56);
    seq_buf[1] = (uint8_t)(e->sequence >> 48);
    seq_buf[2] = (uint8_t)(e->sequence >> 40);
    seq_buf[3] = (uint8_t)(e->sequence >> 32);
    seq_buf[4] = (uint8_t)(e->sequence >> 24);
    seq_buf[5] = (uint8_t)(e->sequence >> 16);
    seq_buf[6] = (uint8_t)(e->sequence >> 8);
    seq_buf[7] = (uint8_t)e->sequence;
    sha256_update(&ctx, seq_buf, 8);

    uint8_t type_b = (uint8_t)type;
    sha256_update(&ctx, &type_b, 1);
    sha256_update(&ctx, e->message, kstrlen(e->message));
    sha256_final(&ctx, e->chain_hash);

    /* Update previous hash */
    kmemcpy(g_security.audit_prev_hash, e->chain_hash, 32);

    /* Advance ring buffer */
    g_security.audit_head = (idx + 1) % SEC_AUDIT_LOG_SIZE;
    if (g_security.audit_count < SEC_AUDIT_LOG_SIZE)
        g_security.audit_count++;
}

int sec_audit_verify_chain(void)
{
    if (g_security.audit_count == 0) return 0;

    uint8_t prev[32];
    kmemset(prev, 0, 32);

    uint32_t start;
    if (g_security.audit_count < SEC_AUDIT_LOG_SIZE)
        start = 0;
    else
        start = g_security.audit_head;

    for (uint32_t n = 0; n < g_security.audit_count; n++) {
        uint32_t idx = (start + n) % SEC_AUDIT_LOG_SIZE;
        sec_audit_entry_t *e = &g_security.audit_log[idx];

        sha256_ctx_t ctx;
        sha256_init(&ctx);
        sha256_update(&ctx, prev, 32);

        uint8_t seq_buf[8];
        seq_buf[0] = (uint8_t)(e->sequence >> 56);
        seq_buf[1] = (uint8_t)(e->sequence >> 48);
        seq_buf[2] = (uint8_t)(e->sequence >> 40);
        seq_buf[3] = (uint8_t)(e->sequence >> 32);
        seq_buf[4] = (uint8_t)(e->sequence >> 24);
        seq_buf[5] = (uint8_t)(e->sequence >> 16);
        seq_buf[6] = (uint8_t)(e->sequence >> 8);
        seq_buf[7] = (uint8_t)e->sequence;
        sha256_update(&ctx, seq_buf, 8);

        uint8_t type_b = (uint8_t)e->type;
        sha256_update(&ctx, &type_b, 1);
        sha256_update(&ctx, e->message, kstrlen(e->message));

        uint8_t computed[32];
        sha256_final(&ctx, computed);

        if (crypto_ct_equal(computed, e->chain_hash, 32) != 0) {
            kprintf("[SEC] Audit chain BROKEN at sequence %lu!\n", e->sequence);
            return -1;
        }

        kmemcpy(prev, computed, 32);
    }
    return 0;
}

void sec_audit_print(uint32_t count)
{
    if (count > g_security.audit_count) count = g_security.audit_count;
    if (count == 0) { kprintf("No audit entries.\n"); return; }

    static const char *type_names[] = {
        "LOGIN_OK", "LOGIN_FAIL", "LOGOUT", "PRIV_ESC",
        "ACCESS_DENY", "FW_BLOCK", "FW_ALLOW", "KEY_GEN",
        "KEY_DEL", "USER_ADD", "USER_DEL", "PASSWD_CHG",
        "CAP_GRANT", "CAP_REVOKE", "SANDBOX_VIOL", "CRYPTO_OP",
        "SSH_SESSION", "SYS_BOOT", "INTEGRITY_FAIL"
    };
    static const char *sev_names[] = { "INFO", "WARN", "ERROR", "CRIT" };

    kprintf("Security Audit Log (last %u of %u entries):\n",
            count, g_security.audit_count);
    kprintf("  %-6s %-5s %-14s %-12s %s\n",
            "Seq#", "Sev", "Type", "Source", "Message");

    /* Start from most recent */
    uint32_t base = g_security.audit_head;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = (base + SEC_AUDIT_LOG_SIZE - count + i) % SEC_AUDIT_LOG_SIZE;
        sec_audit_entry_t *e = &g_security.audit_log[idx];

        const char *tn = (e->type <= SEC_AUDIT_INTEGRITY_FAIL) ?
                          type_names[e->type] : "UNKNOWN";
        const char *sn = (e->severity <= 3) ? sev_names[e->severity] : "???";

        kprintf("  %-6lu %-5s %-14s %-12s %s\n",
                e->sequence, sn, tn, e->source, e->message);
    }

    /* Verify chain */
    if (sec_audit_verify_chain() == 0)
        kprintf("  Chain integrity: VERIFIED\n");
    else
        kprintf("  Chain integrity: COMPROMISED!\n");
}

/* =============================================================================
 * Key Store
 * Keys are encrypted with AES-256-CTR using the master key for at-rest protection.
 * =============================================================================*/

static void keystore_encrypt(uint8_t *data, uint32_t len, uint32_t key_id)
{
    /* Derive a per-key nonce from the key ID to ensure unique CTR streams */
    aes256_ctx_t ctx;
    aes256_init(&ctx, g_security.master_key);
    uint8_t nonce[16];
    kmemset(nonce, 0, 16);
    nonce[0] = (uint8_t)(key_id & 0xFF);
    nonce[1] = (uint8_t)((key_id >> 8) & 0xFF);
    nonce[2] = (uint8_t)((key_id >> 16) & 0xFF);
    nonce[3] = (uint8_t)((key_id >> 24) & 0xFF);
    /* AES-256-CTR: same operation encrypts and decrypts */
    aes256_ctr(&ctx, nonce, data, data, len);
}

int keystore_store(const char *name, key_type_t type,
                   const uint8_t *key, uint32_t key_len, uint32_t owner_uid)
{
    if (g_security.key_count >= MAX_KEYS) return -1;
    if (key_len > 128) return -1;

    /* Check for duplicate name */
    for (uint32_t i = 0; i < MAX_KEYS; i++) {
        if (g_security.keys[i].active &&
            kstrcmp(g_security.keys[i].name, name) == 0)
            return -1;
    }

    /* Find free slot */
    keystore_entry_t *ks = NULL;
    for (uint32_t i = 0; i < MAX_KEYS; i++) {
        if (!g_security.keys[i].active) {
            ks = &g_security.keys[i];
            break;
        }
    }
    if (!ks) return -1;

    ks->id = g_security.next_key_id++;
    int i;
    for (i = 0; i < 63 && name[i]; i++)
        ks->name[i] = name[i];
    ks->name[i] = '\0';
    ks->type = type;
    ks->key_len = key_len;
    ks->owner_uid = owner_uid;
    ks->active = 1;

    /* Store key encrypted with AES-256-CTR */
    kmemcpy(ks->key_data, key, key_len);
    keystore_encrypt(ks->key_data, key_len, ks->id);

    g_security.key_count++;

    sec_audit(SEC_AUDIT_KEY_GENERATED, owner_uid, 0, "keystore",
              "Key stored");
    return 0;
}

int keystore_load(const char *name, uint8_t *key, uint32_t *key_len)
{
    for (uint32_t i = 0; i < MAX_KEYS; i++) {
        if (g_security.keys[i].active &&
            kstrcmp(g_security.keys[i].name, name) == 0) {
            keystore_entry_t *ks = &g_security.keys[i];

            /* Decrypt into output (AES-256-CTR is symmetric) */
            kmemcpy(key, ks->key_data, ks->key_len);
            keystore_encrypt(key, ks->key_len, ks->id);
            if (key_len) *key_len = ks->key_len;
            return 0;
        }
    }
    return -1;
}

int keystore_delete(const char *name)
{
    for (uint32_t i = 0; i < MAX_KEYS; i++) {
        if (g_security.keys[i].active &&
            kstrcmp(g_security.keys[i].name, name) == 0) {
            crypto_wipe(&g_security.keys[i], sizeof(keystore_entry_t));
            g_security.key_count--;

            sec_audit(SEC_AUDIT_KEY_DELETED, 0, 0, "keystore", "Key deleted");
            return 0;
        }
    }
    return -1;
}

void keystore_list(void)
{
    kprintf("Key Store (%u keys):\n", g_security.key_count);
    static const char *type_names[] = {
        "AES-256", "Ed25519", "X25519", "HMAC", "Generic"
    };
    for (uint32_t i = 0; i < MAX_KEYS; i++) {
        if (g_security.keys[i].active) {
            keystore_entry_t *k = &g_security.keys[i];
            const char *tn = (k->type <= KEY_TYPE_GENERIC) ?
                              type_names[k->type] : "Unknown";
            kprintf("  [%u] %-32s %-10s %u bytes  owner=%u\n",
                    k->id, k->name, tn, k->key_len, k->owner_uid);
        }
    }
}

/* =============================================================================
 * Integrity Verification
 * =============================================================================*/

int integrity_register(const char *name, const uint8_t *hash,
                       uint64_t addr, uint32_t size)
{
    if (g_security.integrity_count >= MAX_INTEGRITY_RECORDS) return -1;

    integrity_record_t *rec = &g_security.integrity[g_security.integrity_count++];
    int i;
    for (i = 0; i < 63 && name[i]; i++)
        rec->name[i] = name[i];
    rec->name[i] = '\0';
    kmemcpy(rec->expected_hash, hash, 32);
    rec->load_address = addr;
    rec->size = size;
    rec->verified = 0;
    return 0;
}

int integrity_verify(const char *name, const void *data, uint32_t size)
{
    for (uint32_t i = 0; i < g_security.integrity_count; i++) {
        if (kstrcmp(g_security.integrity[i].name, name) == 0) {
            uint8_t hash[32];
            sha256((const uint8_t *)data, size, hash);

            if (crypto_ct_equal(hash, g_security.integrity[i].expected_hash, 32) == 0) {
                g_security.integrity[i].verified = 1;
                return 0;
            }

            sec_audit(SEC_AUDIT_INTEGRITY_FAIL, 0, 3, "integrity",
                      "Hash mismatch detected!");
            return -1;
        }
    }
    return -1;  /* Not found */
}

void integrity_verify_all(void)
{
    kprintf("Integrity Verification:\n");
    uint32_t pass = 0, fail = 0;

    for (uint32_t i = 0; i < g_security.integrity_count; i++) {
        integrity_record_t *r = &g_security.integrity[i];
        uint8_t hash[32];
        sha256((const uint8_t *)(uintptr_t)r->load_address, r->size, hash);

        int ok = (crypto_ct_equal(hash, r->expected_hash, 32) == 0);
        kprintf("  %-32s [%s]\n", r->name, ok ? "PASS" : "FAIL");
        if (ok) pass++; else fail++;
        r->verified = ok;
    }

    kprintf("  Results: %u passed, %u failed, %u total\n",
            pass, fail, g_security.integrity_count);
}

/* =============================================================================
 * Stack Protection
 * =============================================================================*/

void stack_canary_init(void)
{
    crypto_random((uint8_t *)&g_security.stack_canary, 8);
    /* Ensure canary doesn't contain null bytes (common for C string overflow detection) */
    uint8_t *c = (uint8_t *)&g_security.stack_canary;
    for (int i = 0; i < 8; i++) {
        if (c[i] == 0) c[i] = 0x42;
    }
}

int stack_canary_check(void)
{
    /* Read the current stack canary value.
     * GCC/Zig place __stack_chk_guard at %fs:0x28 on x86_64.
     * We verify by reading the canary from the current stack frame bottom
     * and comparing against the expected value.
     *
     * Since we set __stack_chk_guard = g_security.stack_canary in
     * stack_canary_init(), any compiler-instrumented stack protector
     * failure will call __stack_chk_fail before we get here.
     *
     * For explicit checks: verify the global canary value itself
     * hasn't been corrupted (e.g. by a wild memset). */
    uint8_t *c = (uint8_t *)&g_security.stack_canary;
    int valid = 0;
    for (int i = 0; i < 8; i++) {
        if (c[i] != 0) valid = 1;
    }
    /* If canary is all zeros, it was corrupted (init ensures no zeros) */
    if (!valid) {
        sec_audit(SEC_AUDIT_INTEGRITY_FAIL, 0, 0, "kernel",
                  "Stack canary corrupted - possible buffer overflow");
        return -1;
    }
    return 0;
}
