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

/* =============================================================================
 * TensorOS — Unified Security Framework
 *
 * Provides:
 * - Capability-based access control (caps)
 * - Network firewall with stateful packet filtering
 * - Secure audit trail with integrity verification
 * - System user management (integration with SSH)
 * - Stack canaries and memory guard pages
 * - Secure boot chain verification (hash chain)
 * - Key store for persistent cryptographic material
 * =============================================================================*/

#ifndef TENSOROS_SECURITY_H
#define TENSOROS_SECURITY_H

#include "kernel/core/kernel.h"
#include "kernel/security/crypto.h"

/* =============================================================================
 * Capability-Based Access Control
 *
 * Every resource access requires a valid capability token.
 * Capabilities are unforgeable (HMAC-verified), delegatable, revocable.
 * =============================================================================*/

/* Capability rights */
#define CAP_READ        (1 << 0)
#define CAP_WRITE       (1 << 1)
#define CAP_EXECUTE     (1 << 2)
#define CAP_DELETE      (1 << 3)
#define CAP_ADMIN       (1 << 4)
#define CAP_NETWORK     (1 << 5)
#define CAP_DEVICE      (1 << 6)
#define CAP_DELEGATE    (1 << 7)   /* Can create sub-capabilities */
#define CAP_ALL         0xFF

/* Resource types */
typedef enum {
    RES_FILE       = 0,
    RES_DIRECTORY  = 1,
    RES_DEVICE     = 2,
    RES_NETWORK    = 3,
    RES_PROCESS    = 4,
    RES_MEMORY     = 5,
    RES_MODEL      = 6,
    RES_GPU        = 7,
} resource_type_t;

/* Capability token — unforgeable via HMAC */
typedef struct {
    uint32_t        id;
    uint32_t        owner_uid;
    resource_type_t resource_type;
    uint32_t        resource_id;
    uint32_t        rights;          /* Bitmask of CAP_* */
    uint32_t        delegated_from;  /* Parent capability ID (0 = root) */
    uint64_t        expires;         /* Expiry timestamp (0 = never) */
    uint8_t         hmac[32];        /* HMAC-SHA256 integrity tag */
    uint8_t         active;
} capability_t;

#define MAX_CAPABILITIES 256

/* =============================================================================
 * Firewall — Stateful Packet Filtering
 * =============================================================================*/

typedef enum {
    FW_ACTION_ALLOW  = 0,
    FW_ACTION_DENY   = 1,
    FW_ACTION_LOG    = 2,   /* Allow but log */
    FW_ACTION_REJECT = 3,   /* Deny with RST/ICMP */
} fw_action_t;

typedef enum {
    FW_DIR_INBOUND  = 0,
    FW_DIR_OUTBOUND = 1,
    FW_DIR_BOTH     = 2,
} fw_direction_t;

typedef enum {
    FW_PROTO_ANY  = 0,
    FW_PROTO_TCP  = 6,
    FW_PROTO_UDP  = 17,
    FW_PROTO_ICMP = 1,
} fw_proto_t;

typedef struct {
    uint32_t        id;
    uint8_t         active;
    uint8_t         priority;       /* 0 = highest, 255 = lowest */
    fw_direction_t  direction;
    fw_proto_t      protocol;
    uint32_t        src_ip;         /* 0 = any */
    uint32_t        src_mask;
    uint16_t        src_port_min;
    uint16_t        src_port_max;   /* 0 = any */
    uint32_t        dst_ip;         /* 0 = any */
    uint32_t        dst_mask;
    uint16_t        dst_port_min;
    uint16_t        dst_port_max;   /* 0 = any */
    fw_action_t     action;
    char            description[64];
    uint64_t        match_count;    /* Statistics */
} fw_rule_t;

#define MAX_FW_RULES 64

/* Connection tracking for stateful filtering */
typedef struct {
    uint32_t        src_ip;
    uint16_t        src_port;
    uint32_t        dst_ip;
    uint16_t        dst_port;
    fw_proto_t      protocol;
    uint8_t         state;          /* 0=new, 1=established, 2=closing */
    uint64_t        last_seen;
    uint64_t        bytes_in;
    uint64_t        bytes_out;
    uint64_t        packets_in;
    uint64_t        packets_out;
} fw_conn_track_t;

#define MAX_CONN_TRACK 128

/* =============================================================================
 * Security Audit Trail — Tamper-Evident Log
 *
 * Each entry is hash-chained: entry.hash = SHA-256(prev_hash || entry_data)
 * =============================================================================*/

typedef enum {
    SEC_AUDIT_LOGIN_SUCCESS,
    SEC_AUDIT_LOGIN_FAILURE,
    SEC_AUDIT_LOGOUT,
    SEC_AUDIT_PRIV_ESCALATION,
    SEC_AUDIT_ACCESS_DENIED,
    SEC_AUDIT_FIREWALL_BLOCK,
    SEC_AUDIT_FIREWALL_ALLOW,
    SEC_AUDIT_KEY_GENERATED,
    SEC_AUDIT_KEY_DELETED,
    SEC_AUDIT_USER_CREATED,
    SEC_AUDIT_USER_DELETED,
    SEC_AUDIT_PASSWORD_CHANGED,
    SEC_AUDIT_CAP_GRANTED,
    SEC_AUDIT_CAP_REVOKED,
    SEC_AUDIT_SANDBOX_VIOLATION,
    SEC_AUDIT_CRYPTO_OP,
    SEC_AUDIT_SSH_SESSION,
    SEC_AUDIT_SYSTEM_BOOT,
    SEC_AUDIT_INTEGRITY_FAIL,
} sec_audit_type_t;

typedef struct {
    uint64_t         sequence;       /* Monotonic sequence number */
    uint64_t         timestamp;
    sec_audit_type_t type;
    uint32_t         uid;            /* User ID (0 = system) */
    uint32_t         severity;       /* 0=info, 1=warn, 2=error, 3=critical */
    char             source[32];     /* Subsystem name */
    char             message[128];
    uint8_t          chain_hash[32]; /* SHA-256(prev_hash || this_entry_data) */
} sec_audit_entry_t;

#define SEC_AUDIT_LOG_SIZE 1024

/* =============================================================================
 * System Key Store
 * Keys are stored encrypted with a master key derived from hardware entropy
 * =============================================================================*/

typedef enum {
    KEY_TYPE_AES256   = 0,
    KEY_TYPE_ED25519  = 1,
    KEY_TYPE_X25519   = 2,
    KEY_TYPE_HMAC     = 3,
    KEY_TYPE_GENERIC  = 4,
} key_type_t;

typedef struct {
    uint32_t    id;
    char        name[64];
    key_type_t  type;
    uint32_t    key_len;
    uint8_t     key_data[128];   /* Encrypted in memory (XOR with master key) */
    uint32_t    owner_uid;
    uint8_t     active;
    uint64_t    created;
    uint64_t    last_used;
} keystore_entry_t;

#define MAX_KEYS 32

/* =============================================================================
 * Integrity Verification
 * Hash chain for secure boot / module loading
 * =============================================================================*/

typedef struct {
    char        name[64];
    uint8_t     expected_hash[32];   /* SHA-256 */
    uint8_t     verified;
    uint64_t    load_address;
    uint32_t    size;
} integrity_record_t;

#define MAX_INTEGRITY_RECORDS 32

/* =============================================================================
 * Global Security State
 * =============================================================================*/

typedef struct {
    /* Capabilities */
    capability_t    capabilities[MAX_CAPABILITIES];
    uint32_t        cap_count;
    uint32_t        next_cap_id;
    uint8_t         cap_master_key[32]; /* For HMAC verification */

    /* Firewall */
    fw_rule_t       fw_rules[MAX_FW_RULES];
    uint32_t        fw_rule_count;
    fw_conn_track_t conn_track[MAX_CONN_TRACK];
    uint32_t        conn_track_count;
    uint8_t         fw_default_inbound;   /* 0=allow, 1=deny */
    uint8_t         fw_default_outbound;  /* 0=allow, 1=deny */
    uint8_t         fw_enabled;
    uint64_t        fw_packets_allowed;
    uint64_t        fw_packets_denied;

    /* Audit */
    sec_audit_entry_t  audit_log[SEC_AUDIT_LOG_SIZE];
    uint32_t           audit_head;
    uint32_t           audit_count;
    uint64_t           audit_sequence;
    uint8_t            audit_prev_hash[32];

    /* Key Store */
    keystore_entry_t   keys[MAX_KEYS];
    uint32_t           key_count;
    uint32_t           next_key_id;
    uint8_t            master_key[32];     /* Derived from hardware entropy */

    /* Integrity */
    integrity_record_t integrity[MAX_INTEGRITY_RECORDS];
    uint32_t           integrity_count;

    /* Stack canary */
    uint64_t           stack_canary;

    uint8_t            initialized;
} security_state_t;

extern security_state_t g_security;

/* =============================================================================
 * Security API
 * =============================================================================*/

/* Initialization */
void security_init(void);

/* --- Capability System --- */
int  cap_create(uint32_t owner_uid, resource_type_t type,
                uint32_t resource_id, uint32_t rights, capability_t *out);
int  cap_delegate(const capability_t *parent, uint32_t new_owner,
                  uint32_t subset_rights, capability_t *out);
int  cap_verify(const capability_t *cap);
int  cap_revoke(uint32_t cap_id);
int  cap_check(uint32_t uid, resource_type_t type,
               uint32_t resource_id, uint32_t required_rights);

/* --- Firewall --- */
int  fw_add_rule(const fw_rule_t *rule);
int  fw_remove_rule(uint32_t rule_id);
fw_action_t fw_check_packet(fw_direction_t dir, fw_proto_t proto,
                            uint32_t src_ip, uint16_t src_port,
                            uint32_t dst_ip, uint16_t dst_port);
void fw_enable(void);
void fw_disable(void);
void fw_set_default(fw_direction_t dir, fw_action_t action);
void fw_print_rules(void);
void fw_print_stats(void);
void fw_flush_rules(void);

/* --- Security Audit --- */
void sec_audit(sec_audit_type_t type, uint32_t uid, uint32_t severity,
               const char *source, const char *message);
int  sec_audit_verify_chain(void);   /* Returns 0 if chain intact */
void sec_audit_print(uint32_t count); /* Print last N entries */

/* --- Key Store --- */
int  keystore_store(const char *name, key_type_t type,
                    const uint8_t *key, uint32_t key_len, uint32_t owner_uid);
int  keystore_load(const char *name, uint8_t *key, uint32_t *key_len);
int  keystore_delete(const char *name);
void keystore_list(void);

/* --- Integrity --- */
int  integrity_register(const char *name, const uint8_t *hash,
                        uint64_t addr, uint32_t size);
int  integrity_verify(const char *name, const void *data, uint32_t size);
void integrity_verify_all(void);

/* --- Stack Protection --- */
void stack_canary_init(void);
int  stack_canary_check(void);

#endif /* TENSOROS_SECURITY_H */
