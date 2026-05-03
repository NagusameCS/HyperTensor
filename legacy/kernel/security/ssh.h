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
 * TensorOS — SSH-2 Server
 *
 * Implementation of the SSH-2 protocol (RFC 4250–4254) for secure remote
 * access to the TensorOS shell.
 *
 * Supported algorithms:
 *   Key Exchange:    curve25519-sha256
 *   Host Key:        ssh-ed25519
 *   Encryption:      chacha20-poly1305@openssh.com, aes256-ctr
 *   MAC:             hmac-sha2-256 (implicit in chacha20-poly1305)
 *   Compression:     none
 *   Authentication:  password, publickey (ssh-ed25519)
 *
 * Security features:
 *   • Constant-time key comparison
 *   • Brute-force rate limiting
 *   • Session key rotation
 *   • Audit logging
 *   • Secure memory wiping
 * =============================================================================*/

#ifndef TENSOROS_SSH_H
#define TENSOROS_SSH_H

#include "kernel/core/kernel.h"
#include "kernel/security/crypto.h"
#include "kernel/net/netstack.h"

/* =============================================================================
 * SSH Protocol Constants (RFC 4250)
 * =============================================================================*/

#define SSH_PORT                22

/* Transport layer message IDs */
#define SSH_MSG_DISCONNECT              1
#define SSH_MSG_IGNORE                  2
#define SSH_MSG_UNIMPLEMENTED           3
#define SSH_MSG_DEBUG                   4
#define SSH_MSG_SERVICE_REQUEST         5
#define SSH_MSG_SERVICE_ACCEPT          6
#define SSH_MSG_KEXINIT                 20
#define SSH_MSG_NEWKEYS                 21

/* Key exchange (curve25519-sha256) */
#define SSH_MSG_KEX_ECDH_INIT           30
#define SSH_MSG_KEX_ECDH_REPLY          31

/* User authentication */
#define SSH_MSG_USERAUTH_REQUEST        50
#define SSH_MSG_USERAUTH_FAILURE        51
#define SSH_MSG_USERAUTH_SUCCESS        52
#define SSH_MSG_USERAUTH_BANNER         53
#define SSH_MSG_USERAUTH_PK_OK          60

/* Connection layer */
#define SSH_MSG_GLOBAL_REQUEST          80
#define SSH_MSG_REQUEST_SUCCESS         81
#define SSH_MSG_REQUEST_FAILURE         82
#define SSH_MSG_CHANNEL_OPEN            90
#define SSH_MSG_CHANNEL_OPEN_CONFIRM    91
#define SSH_MSG_CHANNEL_OPEN_FAILURE    92
#define SSH_MSG_CHANNEL_WINDOW_ADJUST   93
#define SSH_MSG_CHANNEL_DATA            94
#define SSH_MSG_CHANNEL_EXTENDED_DATA   95
#define SSH_MSG_CHANNEL_EOF             96
#define SSH_MSG_CHANNEL_CLOSE           97
#define SSH_MSG_CHANNEL_REQUEST         98
#define SSH_MSG_CHANNEL_SUCCESS         99
#define SSH_MSG_CHANNEL_FAILURE         100

/* Disconnect reason codes */
#define SSH_DISCONNECT_HOST_NOT_ALLOWED         2
#define SSH_DISCONNECT_PROTOCOL_ERROR           2
#define SSH_DISCONNECT_KEY_EXCHANGE_FAILED       3
#define SSH_DISCONNECT_AUTH_CANCELLED_BY_USER    13
#define SSH_DISCONNECT_NO_MORE_AUTH_METHODS      14
#define SSH_DISCONNECT_BY_APPLICATION            11

/* =============================================================================
 * SSH Session State Machine
 * =============================================================================*/

typedef enum {
    SSH_STATE_NONE = 0,
    SSH_STATE_VERSION_EXCHANGE,
    SSH_STATE_KEXINIT_SENT,
    SSH_STATE_KEX_DH,
    SSH_STATE_NEWKEYS,
    SSH_STATE_AUTHENTICATED,
    SSH_STATE_CHANNEL_OPEN,
    SSH_STATE_INTERACTIVE,
    SSH_STATE_CLOSING,
} ssh_state_t;

/* =============================================================================
 * SSH Channel
 * =============================================================================*/

#define SSH_MAX_CHANNELS        4
#define SSH_CHANNEL_WINDOW      (256 * 1024)  /* 256 KB initial window */
#define SSH_CHANNEL_MAX_PACKET  32768

typedef struct {
    uint32_t local_id;
    uint32_t remote_id;
    uint32_t local_window;
    uint32_t remote_window;
    uint32_t remote_max_packet;
    int      active;
    int      pty_allocated;
    /* PTY dimensions */
    uint32_t term_cols;
    uint32_t term_rows;
    char     term_type[64];
    /* Channel data buffer (for shell I/O) */
    uint8_t  rx_buf[4096];
    uint32_t rx_len;
} ssh_channel_t;

/* =============================================================================
 * SSH User / Authentication
 * =============================================================================*/

#define SSH_MAX_USERS           16
#define SSH_MAX_AUTH_TRIES      6
#define SSH_AUTH_LOCKOUT_TICKS  3000   /* ~30 seconds at 100 Hz tick */

typedef struct {
    char     username[64];
    uint8_t  password_hash[32];     /* SHA-256 of salted password */
    uint8_t  password_salt[16];
    uint8_t  pubkey[32];            /* Ed25519 public key (if configured) */
    int      pubkey_set;
    int      active;
    uint32_t permissions;           /* Bitmask: 0x1=shell, 0x2=sftp, 0x4=admin */
} ssh_user_t;

/* =============================================================================
 * SSH Session
 * =============================================================================*/

#define SSH_MAX_SESSIONS        4
#define SSH_MAX_PACKET_SIZE     35000
#define SSH_ID_STRING           "SSH-2.0-TensorOS_1.0"

typedef struct {
    ssh_state_t  state;
    tcp_conn_t  *conn;

    /* Version exchange */
    char         client_version[256];
    int          version_received;

    /* Key exchange state */
    uint8_t      session_id[32];      /* First exchange hash (H) */
    int          session_id_set;
    uint8_t      kex_hash[32];        /* Current exchange hash */

    /* Client and server KEXINIT payloads (for hash) */
    uint8_t      client_kexinit[2048];
    uint32_t     client_kexinit_len;
    uint8_t      server_kexinit[2048];
    uint32_t     server_kexinit_len;

    /* Ephemeral X25519 keys */
    uint8_t      kex_priv[32];
    uint8_t      kex_pub[32];
    uint8_t      shared_secret[32];

    /* Derived session keys */
    uint8_t      key_c2s_enc[32];     /* Client → Server encryption key */
    uint8_t      key_s2c_enc[32];     /* Server → Client encryption key */
    uint8_t      key_c2s_mac[32];     /* Client → Server MAC key */
    uint8_t      key_s2c_mac[32];     /* Server → Client MAC key */
    uint8_t      iv_c2s[16];          /* Client → Server IV */
    uint8_t      iv_s2c[16];          /* Server → Client IV */

    int          keys_active;          /* Encryption active after NEWKEYS */

    /* Packet sequence numbers */
    uint32_t     seq_c2s;             /* Client → Server */
    uint32_t     seq_s2c;             /* Server → Client */

    /* Encryption contexts */
    aes256_ctx_t aes_c2s;
    aes256_ctx_t aes_s2c;
    int          use_chacha;           /* Using chacha20-poly1305 */

    /* Authentication */
    char         auth_user[64];
    int          auth_tries;
    uint64_t     last_auth_fail;

    /* Channels */
    ssh_channel_t channels[SSH_MAX_CHANNELS];
    uint32_t      next_channel_id;

    /* Packet assembly */
    uint8_t      rx_buf[SSH_MAX_PACKET_SIZE];
    uint32_t     rx_len;
    uint8_t      tx_buf[SSH_MAX_PACKET_SIZE];

    int          active;
} ssh_session_t;

/* =============================================================================
 * SSH Server State
 * =============================================================================*/

typedef struct {
    /* Host key (Ed25519) */
    uint8_t      host_priv[64];
    uint8_t      host_pub[32];
    int          host_key_generated;

    /* User database */
    ssh_user_t   users[SSH_MAX_USERS];
    uint32_t     user_count;

    /* Sessions */
    ssh_session_t sessions[SSH_MAX_SESSIONS];

    /* Server state */
    int          running;
    uint64_t     total_connections;
    uint64_t     total_auth_failures;
    uint64_t     total_packets;

    /* Rate limiting */
    uint32_t     connections_per_minute;
    uint64_t     last_rate_check;
} ssh_server_t;

/* Global SSH server instance */
extern ssh_server_t g_ssh_server;

/* =============================================================================
 * SSH Server API
 * =============================================================================*/

/* Initialize SSH server (generate host key, etc.) */
void ssh_server_init(void);

/* Start listening for SSH connections on port 22 */
int  ssh_server_start(void);

/* Stop SSH server */
void ssh_server_stop(void);

/* Poll for SSH events (call in main loop alongside netstack_poll) */
void ssh_server_poll(void);

/* =============================================================================
 * User Management
 * =============================================================================*/

/* Add a user with password authentication */
int  ssh_add_user(const char *username, const char *password, uint32_t perms);

/* Remove a user */
int  ssh_remove_user(const char *username);

/* Change a user's password */
int  ssh_change_password(const char *username, const char *new_password);

/* Add a public key for a user */
int  ssh_add_user_pubkey(const char *username, const uint8_t pubkey[32]);

/* List users (returns count, fills array) */
int  ssh_list_users(ssh_user_t *users, uint32_t max, uint32_t *count);

/* =============================================================================
 * Host Key Management
 * =============================================================================*/

/* Regenerate host key */
void ssh_regenerate_host_key(void);

/* Get host key fingerprint (SHA-256, writes hex string) */
void ssh_host_key_fingerprint(char *buf, uint32_t buflen);

/* =============================================================================
 * SSH Info / Status
 * =============================================================================*/

/* Print SSH server status */
void ssh_print_status(void);

/* Get number of active sessions */
int  ssh_active_sessions(void);

/* Disconnect a specific session */
int  ssh_disconnect_session(int session_id, const char *reason);

#endif /* TENSOROS_SSH_H */
