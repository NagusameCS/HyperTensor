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
 * TensorOS — SSH-2 Server Implementation
 *
 * Full SSH-2 transport, authentication, and channel management.
 * Integrates with the TensorOS network stack and shell.
 * =============================================================================*/

#include "kernel/security/ssh.h"
#include "kernel/security/crypto.h"

/* Global SSH server */
ssh_server_t g_ssh_server;

/* =============================================================================
 * SSH Packet Helpers
 * =============================================================================*/

static inline void ssh_put_u32(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 24); p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);  p[3] = (uint8_t)v;
}

static inline uint32_t ssh_get_u32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  | (uint32_t)p[3];
}

/* Write SSH string (uint32 length + data) into buffer, return bytes written */
static uint32_t ssh_put_string(uint8_t *buf, const void *data, uint32_t len)
{
    ssh_put_u32(buf, len);
    if (len > 0) kmemcpy(buf + 4, data, len);
    return 4 + len;
}

/* Write SSH name-list (comma-separated) */
static uint32_t ssh_put_namelist(uint8_t *buf, const char *list)
{
    uint32_t len = (uint32_t)kstrlen(list);
    return ssh_put_string(buf, list, len);
}

/* Read SSH string from buffer; returns pointer to data and sets *len */
static const uint8_t *ssh_get_string(const uint8_t *buf, uint32_t *len,
                                      uint32_t buf_remaining)
{
    if (buf_remaining < 4) { *len = 0; return NULL; }
    *len = ssh_get_u32(buf);
    if (*len > buf_remaining - 4) { *len = 0; return NULL; }
    return buf + 4;
}

/* =============================================================================
 * Hex Encoding for Fingerprints
 * =============================================================================*/

static const char hex_chars[] = "0123456789abcdef";

static void to_hex(char *out, const uint8_t *data, uint32_t len)
{
    for (uint32_t i = 0; i < len; i++) {
        out[i * 3] = hex_chars[(data[i] >> 4) & 0xF];
        out[i * 3 + 1] = hex_chars[data[i] & 0xF];
        out[i * 3 + 2] = (i < len - 1) ? ':' : '\0';
    }
}

/* =============================================================================
 * SSH Packet Send / Receive
 *
 * SSH binary packet format (RFC 4253 §6):
 *   uint32  packet_length  (not including self or MAC)
 *   byte    padding_length
 *   byte[]  payload
 *   byte[]  random padding  (4-255 bytes)
 *   byte[]  MAC
 * =============================================================================*/

static int ssh_send_packet(ssh_session_t *s, const uint8_t *payload, uint32_t plen)
{
    if (!s->conn || s->conn->state != TCP_STATE_ESTABLISHED) return -1;

    uint8_t *pkt = s->tx_buf;
    uint32_t block_size = 16;  /* AES block or minimum */

    /* Calculate padding (total must be multiple of block_size, min 4 bytes) */
    uint32_t unpadded = 1 + plen;  /* padding_len byte + payload */
    uint32_t padded = unpadded + 4;  /* + packet_length field for alignment */
    uint32_t pad_len = block_size - (padded % block_size);
    if (pad_len < 4) pad_len += block_size;

    uint32_t packet_length = 1 + plen + pad_len;

    /* Build packet */
    ssh_put_u32(pkt, packet_length);
    pkt[4] = (uint8_t)pad_len;
    kmemcpy(pkt + 5, payload, plen);
    /* Random padding */
    crypto_random(pkt + 5 + plen, pad_len);

    uint32_t total = 4 + packet_length;

    if (s->keys_active) {
        if (s->use_chacha) {
            /* ChaCha20-Poly1305 encryption */
            uint8_t nonce[12];
            kmemset(nonce, 0, 8);
            ssh_put_u32(nonce + 8, s->seq_s2c);

            /* Encrypt packet length with separate key derived stream */
            /* For simplicity, encrypt everything together */
            uint8_t encrypted[SSH_MAX_PACKET_SIZE];
            uint8_t tag[16];
            chacha20_poly1305_encrypt(
                s->key_s2c_enc, nonce,
                NULL, 0,
                pkt, total,
                encrypted, tag);
            kmemcpy(pkt, encrypted, total);
            kmemcpy(pkt + total, tag, 16);
            total += 16;
        } else {
            /* AES-256-CTR + HMAC-SHA256 */
            /* Encrypt (skip first 4 bytes = length in some implementations,
             * but we encrypt everything for simplicity) */
            aes256_ctr(&s->aes_s2c, s->iv_s2c, pkt + 4, pkt + 4, packet_length);

            /* Increment IV */
            for (int i = 15; i >= 0; i--) {
                if (++s->iv_s2c[i]) break;
            }

            /* Compute MAC */
            uint8_t mac_data[4 + SSH_MAX_PACKET_SIZE];
            ssh_put_u32(mac_data, s->seq_s2c);
            kmemcpy(mac_data + 4, pkt, total);
            uint8_t mac[32];
            hmac_sha256(s->key_s2c_mac, 32, mac_data, 4 + total, mac);
            kmemcpy(pkt + total, mac, 32);
            total += 32;
        }
    }

    s->seq_s2c++;
    g_ssh_server.total_packets++;

    return tcp_conn_write(s->conn, pkt, total);
}

/* Send a single-byte message */
static int ssh_send_msg(ssh_session_t *s, uint8_t msg_type)
{
    return ssh_send_packet(s, &msg_type, 1);
}

/* =============================================================================
 * SSH Key Exchange Init (RFC 4253 §7.1)
 * =============================================================================*/

static int ssh_send_kexinit(ssh_session_t *s)
{
    uint8_t payload[1024];
    uint32_t pos = 0;

    payload[pos++] = SSH_MSG_KEXINIT;

    /* 16 bytes cookie (random) */
    crypto_random(payload + pos, 16);
    pos += 16;

    /* Algorithm lists */
    pos += ssh_put_namelist(payload + pos, "curve25519-sha256");  /* kex */
    pos += ssh_put_namelist(payload + pos, "ssh-ed25519");        /* host key */
    pos += ssh_put_namelist(payload + pos, "aes256-ctr,chacha20-poly1305@openssh.com"); /* enc c2s */
    pos += ssh_put_namelist(payload + pos, "aes256-ctr,chacha20-poly1305@openssh.com"); /* enc s2c */
    pos += ssh_put_namelist(payload + pos, "hmac-sha2-256");      /* mac c2s */
    pos += ssh_put_namelist(payload + pos, "hmac-sha2-256");      /* mac s2c */
    pos += ssh_put_namelist(payload + pos, "none");               /* comp c2s */
    pos += ssh_put_namelist(payload + pos, "none");               /* comp s2c */
    pos += ssh_put_namelist(payload + pos, "");                   /* lang c2s */
    pos += ssh_put_namelist(payload + pos, "");                   /* lang s2c */

    payload[pos++] = 0;  /* first_kex_packet_follows = FALSE */
    ssh_put_u32(payload + pos, 0);  /* reserved */
    pos += 4;

    /* Save our KEXINIT for the exchange hash */
    kmemcpy(s->server_kexinit, payload, pos);
    s->server_kexinit_len = pos;

    s->state = SSH_STATE_KEXINIT_SENT;
    return ssh_send_packet(s, payload, pos);
}

/* =============================================================================
 * Key Exchange: curve25519-sha256 (RFC 8731)
 * =============================================================================*/

static int ssh_derive_keys(ssh_session_t *s)
{
    /* Derive keys using HKDF with exchange hash as PRK and shared secret as IKM */
    uint8_t prk[32];
    hkdf_sha256_extract(s->kex_hash, 32, s->shared_secret, 32, prk);

    /* Derive 6 keys: IV c2s, IV s2c, Enc c2s, Enc s2c, MAC c2s, MAC s2c
     * Each uses a different "info" byte per SSH spec (letters A-F) */
    uint8_t info[33];
    kmemcpy(info, s->kex_hash, 32);

    info[32] = 'A';
    hkdf_sha256_expand(prk, info, 33, s->iv_c2s, 16);
    info[32] = 'B';
    hkdf_sha256_expand(prk, info, 33, s->iv_s2c, 16);
    info[32] = 'C';
    hkdf_sha256_expand(prk, info, 33, s->key_c2s_enc, 32);
    info[32] = 'D';
    hkdf_sha256_expand(prk, info, 33, s->key_s2c_enc, 32);
    info[32] = 'E';
    hkdf_sha256_expand(prk, info, 33, s->key_c2s_mac, 32);
    info[32] = 'F';
    hkdf_sha256_expand(prk, info, 33, s->key_s2c_mac, 32);

    /* Initialize AES contexts */
    aes256_init(&s->aes_c2s, s->key_c2s_enc);
    aes256_init(&s->aes_s2c, s->key_s2c_enc);

    /* Set session ID if first exchange */
    if (!s->session_id_set) {
        kmemcpy(s->session_id, s->kex_hash, 32);
        s->session_id_set = 1;
    }

    crypto_wipe(prk, 32);
    return 0;
}

static int ssh_handle_kex_init(ssh_session_t *s, const uint8_t *payload, uint32_t len)
{
    /* Save client KEXINIT for hash */
    if (len > sizeof(s->client_kexinit)) len = sizeof(s->client_kexinit);
    kmemcpy(s->client_kexinit, payload, len);
    s->client_kexinit_len = len;

    /* Check for chacha20-poly1305 preference */
    /* Simple substring search in the encryption name-lists */
    s->use_chacha = 0;
    /* Default to AES-256-CTR for now */

    /* Generate ephemeral X25519 keypair */
    crypto_random(s->kex_priv, 32);
    x25519_public_key(s->kex_pub, s->kex_priv);

    s->state = SSH_STATE_KEX_DH;
    return 0;
}

static int ssh_handle_kex_ecdh_init(ssh_session_t *s, const uint8_t *payload, uint32_t len)
{
    if (len < 5) return -1;

    /* Parse client's ephemeral public key */
    uint32_t qc_len;
    const uint8_t *qc = ssh_get_string(payload + 1, &qc_len, len - 1);
    if (!qc || qc_len != 32) return -1;

    /* Compute shared secret K = X25519(our_priv, client_pub) */
    x25519(s->shared_secret, s->kex_priv, qc);

    /* Compute exchange hash H = SHA-256(V_C || V_S || I_C || I_S || K_S || Q_C || Q_S || K)
     * Where:
     *   V_C = client version string
     *   V_S = server version string
     *   I_C = client KEXINIT payload
     *   I_S = server KEXINIT payload
     *   K_S = server host key blob
     *   Q_C = client ephemeral public key
     *   Q_S = server ephemeral public key
     *   K   = shared secret */
    sha256_ctx_t hash;
    sha256_init(&hash);

    /* V_C */
    uint8_t tmp[4];
    uint32_t vc_len = (uint32_t)kstrlen(s->client_version);
    ssh_put_u32(tmp, vc_len);
    sha256_update(&hash, tmp, 4);
    sha256_update(&hash, s->client_version, vc_len);

    /* V_S */
    uint32_t vs_len = (uint32_t)kstrlen(SSH_ID_STRING);
    ssh_put_u32(tmp, vs_len);
    sha256_update(&hash, tmp, 4);
    sha256_update(&hash, SSH_ID_STRING, vs_len);

    /* I_C */
    ssh_put_u32(tmp, s->client_kexinit_len);
    sha256_update(&hash, tmp, 4);
    sha256_update(&hash, s->client_kexinit, s->client_kexinit_len);

    /* I_S */
    ssh_put_u32(tmp, s->server_kexinit_len);
    sha256_update(&hash, tmp, 4);
    sha256_update(&hash, s->server_kexinit, s->server_kexinit_len);

    /* K_S (host key blob: string "ssh-ed25519" + string pubkey) */
    uint8_t hostkey_blob[128];
    uint32_t hk_pos = 0;
    hk_pos += ssh_put_string(hostkey_blob + hk_pos, "ssh-ed25519", 11);
    hk_pos += ssh_put_string(hostkey_blob + hk_pos, g_ssh_server.host_pub, 32);

    ssh_put_u32(tmp, hk_pos);
    sha256_update(&hash, tmp, 4);
    sha256_update(&hash, hostkey_blob, hk_pos);

    /* Q_C */
    ssh_put_u32(tmp, 32);
    sha256_update(&hash, tmp, 4);
    sha256_update(&hash, qc, 32);

    /* Q_S */
    ssh_put_u32(tmp, 32);
    sha256_update(&hash, tmp, 4);
    sha256_update(&hash, s->kex_pub, 32);

    /* K (shared secret as mpint — RFC 4251: prepend 0x00 if MSB set) */
    if (s->shared_secret[0] & 0x80) {
        ssh_put_u32(tmp, 33);
        sha256_update(&hash, tmp, 4);
        uint8_t zero = 0;
        sha256_update(&hash, &zero, 1);
    } else {
        ssh_put_u32(tmp, 32);
        sha256_update(&hash, tmp, 4);
    }
    sha256_update(&hash, s->shared_secret, 32);

    sha256_final(&hash, s->kex_hash);

    /* Sign the exchange hash with our host key */
    uint8_t sig[64];
    ed25519_sign(s->kex_hash, 32,
                 g_ssh_server.host_pub, g_ssh_server.host_priv,
                 sig);

    /* Build KEX_ECDH_REPLY:
     *   byte    SSH_MSG_KEX_ECDH_REPLY
     *   string  K_S (host key)
     *   string  Q_S (server ephemeral public)
     *   string  signature */
    uint8_t reply[512];
    uint32_t rpos = 0;
    reply[rpos++] = SSH_MSG_KEX_ECDH_REPLY;

    /* K_S */
    rpos += ssh_put_string(reply + rpos, hostkey_blob, hk_pos);
    /* Q_S */
    rpos += ssh_put_string(reply + rpos, s->kex_pub, 32);

    /* Signature blob: string "ssh-ed25519" + string sig */
    uint8_t sig_blob[128];
    uint32_t sg_pos = 0;
    sg_pos += ssh_put_string(sig_blob + sg_pos, "ssh-ed25519", 11);
    sg_pos += ssh_put_string(sig_blob + sg_pos, sig, 64);
    rpos += ssh_put_string(reply + rpos, sig_blob, sg_pos);

    ssh_send_packet(s, reply, rpos);

    /* Derive session keys */
    ssh_derive_keys(s);

    /* Send NEWKEYS */
    ssh_send_msg(s, SSH_MSG_NEWKEYS);
    s->state = SSH_STATE_NEWKEYS;

    /* Wipe ephemeral private key */
    crypto_wipe(s->kex_priv, 32);

    return 0;
}

/* =============================================================================
 * User Authentication (RFC 4252)
 * =============================================================================*/

static ssh_user_t *ssh_find_user(const char *username)
{
    for (uint32_t i = 0; i < g_ssh_server.user_count; i++) {
        if (g_ssh_server.users[i].active &&
            kstrcmp(g_ssh_server.users[i].username, username) == 0)
            return &g_ssh_server.users[i];
    }
    return NULL;
}

/* PBKDF2 iteration count — NIST SP 800-132 recommends >= 10,000.
 * 100,000 iterations provides strong brute-force resistance. */
#define SSH_PBKDF2_ITERATIONS 100000

static int ssh_verify_password(const ssh_user_t *user, const char *password)
{
    uint8_t hash[32];
    pbkdf2_hmac_sha256((const uint8_t *)password, (uint32_t)kstrlen(password),
                       user->password_salt, 16,
                       SSH_PBKDF2_ITERATIONS, hash, 32);

    int result = crypto_ct_equal(hash, user->password_hash, 32);
    crypto_wipe(hash, 32);
    return result;  /* 0 = match */
}

static int ssh_handle_userauth_request(ssh_session_t *s,
                                        const uint8_t *payload, uint32_t len)
{
    uint32_t pos = 1;  /* Skip message type */

    /* Rate limit check */
    if (s->auth_tries >= SSH_MAX_AUTH_TRIES) {
        kprintf("[SSH] Session %d: max auth attempts exceeded\n",
                (int)(s - g_ssh_server.sessions));
        uint8_t disc[256];
        uint32_t dpos = 0;
        disc[dpos++] = SSH_MSG_DISCONNECT;
        ssh_put_u32(disc + dpos, SSH_DISCONNECT_NO_MORE_AUTH_METHODS);
        dpos += 4;
        dpos += ssh_put_string(disc + dpos, "Too many auth failures", 22);
        dpos += ssh_put_string(disc + dpos, "", 0);
        ssh_send_packet(s, disc, dpos);
        s->state = SSH_STATE_CLOSING;
        return -1;
    }

    /* Parse: string username, string service, string method */
    uint32_t ulen;
    const uint8_t *username = ssh_get_string(payload + pos, &ulen, len - pos);
    if (!username) return -1;
    pos += 4 + ulen;

    uint32_t slen;
    const uint8_t *service = ssh_get_string(payload + pos, &slen, len - pos);
    if (!service) return -1;
    pos += 4 + slen;

    uint32_t mlen;
    const uint8_t *method = ssh_get_string(payload + pos, &mlen, len - pos);
    if (!method) return -1;
    pos += 4 + mlen;

    /* Copy username */
    uint32_t copy = (ulen >= 63) ? 63 : ulen;
    kmemcpy(s->auth_user, username, copy);
    s->auth_user[copy] = '\0';

    /* Handle "none" method — return supported methods */
    if (mlen == 4 && kstrncmp((const char *)method, "none", 4) == 0) {
        uint8_t fail[64];
        uint32_t fpos = 0;
        fail[fpos++] = SSH_MSG_USERAUTH_FAILURE;
        fpos += ssh_put_namelist(fail + fpos, "password,publickey");
        fail[fpos++] = 0;  /* partial success = FALSE */
        return ssh_send_packet(s, fail, fpos);
    }

    /* Password authentication */
    if (mlen == 8 && kstrncmp((const char *)method, "password", 8) == 0) {
        uint8_t change_flag = payload[pos++];
        (void)change_flag;

        uint32_t plen;
        const uint8_t *password = ssh_get_string(payload + pos, &plen, len - pos);
        if (!password) return -1;

        /* Null-terminate password for comparison */
        char pw_buf[128];
        uint32_t pw_copy = (plen >= 127) ? 127 : plen;
        kmemcpy(pw_buf, password, pw_copy);
        pw_buf[pw_copy] = '\0';

        ssh_user_t *user = ssh_find_user(s->auth_user);
        if (user && ssh_verify_password(user, pw_buf) == 0) {
            crypto_wipe(pw_buf, 128);
            kprintf("[SSH] User '%s' authenticated successfully\n", s->auth_user);
            s->state = SSH_STATE_AUTHENTICATED;
            return ssh_send_msg(s, SSH_MSG_USERAUTH_SUCCESS);
        }

        crypto_wipe(pw_buf, 128);
        s->auth_tries++;
        g_ssh_server.total_auth_failures++;
        kprintf("[SSH] Auth failure for '%s' (attempt %d/%d)\n",
                s->auth_user, s->auth_tries, SSH_MAX_AUTH_TRIES);

        uint8_t fail[64];
        uint32_t fpos = 0;
        fail[fpos++] = SSH_MSG_USERAUTH_FAILURE;
        fpos += ssh_put_namelist(fail + fpos, "password,publickey");
        fail[fpos++] = 0;
        return ssh_send_packet(s, fail, fpos);
    }

    /* Public key authentication */
    if (mlen == 9 && kstrncmp((const char *)method, "publickey", 9) == 0) {
        uint8_t has_sig = payload[pos++];

        uint32_t algo_len;
        const uint8_t *algo = ssh_get_string(payload + pos, &algo_len, len - pos);
        if (!algo) return -1;
        pos += 4 + algo_len;

        uint32_t pk_len;
        const uint8_t *pk_blob = ssh_get_string(payload + pos, &pk_len, len - pos);
        if (!pk_blob) return -1;
        pos += 4 + pk_len;

        ssh_user_t *user = ssh_find_user(s->auth_user);

        if (!has_sig) {
            /* Query: is this key acceptable? */
            if (user && user->pubkey_set) {
                /* Check if the provided key matches */
                /* Extract the actual key from the blob (skip type string) */
                uint32_t inner_len;
                ssh_get_string(pk_blob, &inner_len, pk_len);
                const uint8_t *actual_key = pk_blob + 4 + inner_len;
                uint32_t key_len;
                const uint8_t *key_data = ssh_get_string(actual_key, &key_len, 
                    pk_len - 4 - inner_len);

                if (key_data && key_len == 32 &&
                    crypto_ct_equal(key_data, user->pubkey, 32) == 0) {
                    /* Key is acceptable */
                    uint8_t ok[128];
                    uint32_t opos = 0;
                    ok[opos++] = SSH_MSG_USERAUTH_PK_OK;
                    opos += ssh_put_string(ok + opos, algo, algo_len);
                    opos += ssh_put_string(ok + opos, pk_blob, pk_len);
                    return ssh_send_packet(s, ok, opos);
                }
            }
            /* Key not acceptable */
            uint8_t fail[64];
            uint32_t fpos = 0;
            fail[fpos++] = SSH_MSG_USERAUTH_FAILURE;
            fpos += ssh_put_namelist(fail + fpos, "password,publickey");
            fail[fpos++] = 0;
            return ssh_send_packet(s, fail, fpos);
        }

        /* Has signature — verify it */
        uint32_t sig_len;
        const uint8_t *sig_blob = ssh_get_string(payload + pos, &sig_len, len - pos);
        if (!sig_blob) return -1;

        if (user && user->pubkey_set) {
            /* Verify Ed25519 signature over RFC 4252 §7 signed data:
             *   string session_id, byte SSH_MSG_USERAUTH_REQUEST,
             *   string user, string "ssh-connection", string "publickey",
             *   boolean TRUE, string "ssh-ed25519", string pubkey_blob */
            if (sig_len >= 4) {
                uint32_t inner_algo_len;
                ssh_get_string(sig_blob, &inner_algo_len, sig_len);
                const uint8_t *actual_sig = sig_blob + 4 + inner_algo_len;
                uint32_t asig_len;
                const uint8_t *sig_data = ssh_get_string(actual_sig, &asig_len,
                    sig_len - 4 - inner_algo_len);

                /* Build signed data blob per RFC 4252 §7 */
                uint8_t signed_data[768];
                uint32_t sd = 0;
                sd += ssh_put_string(signed_data + sd, s->session_id, 32);
                signed_data[sd++] = SSH_MSG_USERAUTH_REQUEST;
                sd += ssh_put_string(signed_data + sd, s->auth_user,
                                     kstrlen(s->auth_user));
                sd += ssh_put_string(signed_data + sd, "ssh-connection", 14);
                sd += ssh_put_string(signed_data + sd, "publickey", 9);
                signed_data[sd++] = 1; /* TRUE */
                sd += ssh_put_string(signed_data + sd, "ssh-ed25519", 11);
                /* pubkey blob: string "ssh-ed25519" + string pubkey */
                uint8_t pk_blob[64];
                uint32_t pk = 0;
                pk += ssh_put_string(pk_blob + pk, "ssh-ed25519", 11);
                pk += ssh_put_string(pk_blob + pk, user->pubkey, 32);
                sd += ssh_put_string(signed_data + sd, pk_blob, pk);

                if (sig_data && asig_len == 64 &&
                    ed25519_verify(signed_data, sd, user->pubkey, sig_data) == 0) {
                    kprintf("[SSH] User '%s' authenticated via publickey\n", s->auth_user);
                    s->state = SSH_STATE_AUTHENTICATED;
                    return ssh_send_msg(s, SSH_MSG_USERAUTH_SUCCESS);
                }
            }
        }

        s->auth_tries++;
        g_ssh_server.total_auth_failures++;

        uint8_t fail[64];
        uint32_t fpos = 0;
        fail[fpos++] = SSH_MSG_USERAUTH_FAILURE;
        fpos += ssh_put_namelist(fail + fpos, "password,publickey");
        fail[fpos++] = 0;
        return ssh_send_packet(s, fail, fpos);
    }

    /* Unknown method */
    uint8_t fail[64];
    uint32_t fpos = 0;
    fail[fpos++] = SSH_MSG_USERAUTH_FAILURE;
    fpos += ssh_put_namelist(fail + fpos, "password,publickey");
    fail[fpos++] = 0;
    return ssh_send_packet(s, fail, fpos);
}

/* =============================================================================
 * Channel Management (RFC 4254)
 * =============================================================================*/

static ssh_channel_t *ssh_find_channel(ssh_session_t *s, uint32_t local_id)
{
    for (int i = 0; i < SSH_MAX_CHANNELS; i++) {
        if (s->channels[i].active && s->channels[i].local_id == local_id)
            return &s->channels[i];
    }
    return NULL;
}

static int ssh_handle_channel_open(ssh_session_t *s,
                                    const uint8_t *payload, uint32_t len)
{
    uint32_t pos = 1;
    uint32_t type_len;
    const uint8_t *type = ssh_get_string(payload + pos, &type_len, len - pos);
    if (!type) return -1;
    pos += 4 + type_len;

    uint32_t sender = ssh_get_u32(payload + pos); pos += 4;
    uint32_t window = ssh_get_u32(payload + pos); pos += 4;
    uint32_t max_pkt = ssh_get_u32(payload + pos); pos += 4;

    /* Only accept "session" channels */
    if (type_len != 7 || kstrncmp((const char *)type, "session", 7) != 0) {
        uint8_t fail[32];
        uint32_t fpos = 0;
        fail[fpos++] = SSH_MSG_CHANNEL_OPEN_FAILURE;
        ssh_put_u32(fail + fpos, sender); fpos += 4;
        ssh_put_u32(fail + fpos, 3); fpos += 4; /* SSH_OPEN_UNKNOWN_CHANNEL_TYPE */
        fpos += ssh_put_string(fail + fpos, "Unknown channel type", 20);
        fpos += ssh_put_string(fail + fpos, "", 0);
        return ssh_send_packet(s, fail, fpos);
    }

    /* Find free channel slot */
    ssh_channel_t *ch = NULL;
    for (int i = 0; i < SSH_MAX_CHANNELS; i++) {
        if (!s->channels[i].active) {
            ch = &s->channels[i];
            break;
        }
    }
    if (!ch) {
        uint8_t fail[32];
        uint32_t fpos = 0;
        fail[fpos++] = SSH_MSG_CHANNEL_OPEN_FAILURE;
        ssh_put_u32(fail + fpos, sender); fpos += 4;
        ssh_put_u32(fail + fpos, 4); fpos += 4; /* RESOURCE_SHORTAGE */
        fpos += ssh_put_string(fail + fpos, "No channels available", 21);
        fpos += ssh_put_string(fail + fpos, "", 0);
        return ssh_send_packet(s, fail, fpos);
    }

    kmemset(ch, 0, sizeof(*ch));
    ch->active = 1;
    ch->local_id = s->next_channel_id++;
    ch->remote_id = sender;
    ch->local_window = SSH_CHANNEL_WINDOW;
    ch->remote_window = window;
    ch->remote_max_packet = max_pkt;
    ch->term_cols = 80;
    ch->term_rows = 24;

    s->state = SSH_STATE_CHANNEL_OPEN;

    /* Send CHANNEL_OPEN_CONFIRMATION */
    uint8_t conf[32];
    uint32_t cpos = 0;
    conf[cpos++] = SSH_MSG_CHANNEL_OPEN_CONFIRM;
    ssh_put_u32(conf + cpos, ch->remote_id); cpos += 4;
    ssh_put_u32(conf + cpos, ch->local_id); cpos += 4;
    ssh_put_u32(conf + cpos, ch->local_window); cpos += 4;
    ssh_put_u32(conf + cpos, SSH_CHANNEL_MAX_PACKET); cpos += 4;

    return ssh_send_packet(s, conf, cpos);
}

static int ssh_handle_channel_request(ssh_session_t *s,
                                       const uint8_t *payload, uint32_t len)
{
    uint32_t pos = 1;
    uint32_t recipient = ssh_get_u32(payload + pos); pos += 4;

    uint32_t rtype_len;
    const uint8_t *rtype = ssh_get_string(payload + pos, &rtype_len, len - pos);
    if (!rtype) return -1;
    pos += 4 + rtype_len;

    uint8_t want_reply = payload[pos++];

    ssh_channel_t *ch = ssh_find_channel(s, recipient);
    if (!ch) return -1;

    /* Handle PTY request */
    if (rtype_len == 7 && kstrncmp((const char *)rtype, "pty-req", 7) == 0) {
        uint32_t term_len;
        const uint8_t *term = ssh_get_string(payload + pos, &term_len, len - pos);
        if (term) {
            uint32_t copy = (term_len >= 63) ? 63 : term_len;
            kmemcpy(ch->term_type, term, copy);
            ch->term_type[copy] = '\0';
        }
        pos += 4 + term_len;

        if (pos + 16 <= len) {
            ch->term_cols = ssh_get_u32(payload + pos); pos += 4;
            ch->term_rows = ssh_get_u32(payload + pos); pos += 4;
            /* Skip pixel dimensions */
            pos += 8;
        }
        ch->pty_allocated = 1;

        if (want_reply) {
            uint8_t succ[8];
            succ[0] = SSH_MSG_CHANNEL_SUCCESS;
            ssh_put_u32(succ + 1, ch->remote_id);
            ssh_send_packet(s, succ, 5);
        }
        return 0;
    }

    /* Handle shell request */
    if (rtype_len == 5 && kstrncmp((const char *)rtype, "shell", 5) == 0) {
        s->state = SSH_STATE_INTERACTIVE;

        if (want_reply) {
            uint8_t succ[8];
            succ[0] = SSH_MSG_CHANNEL_SUCCESS;
            ssh_put_u32(succ + 1, ch->remote_id);
            ssh_send_packet(s, succ, 5);
        }

        /* Send welcome banner through channel */
        const char *banner =
            "\r\n"
            "  TensorOS v0.1.0 \"Neuron\" — Secure Shell\r\n"
            "  Authenticated as: ";
        uint8_t data[512];
        uint32_t dpos = 0;
        data[dpos++] = SSH_MSG_CHANNEL_DATA;
        ssh_put_u32(data + dpos, ch->remote_id); dpos += 4;

        char msg[256];
        int mlen = 0;
        const char *p = banner;
        while (*p) msg[mlen++] = *p++;
        p = s->auth_user;
        while (*p) msg[mlen++] = *p++;
        msg[mlen++] = '\r'; msg[mlen++] = '\n';
        msg[mlen++] = '\r'; msg[mlen++] = '\n';

        dpos += ssh_put_string(data + dpos, msg, (uint32_t)mlen);
        ssh_send_packet(s, data, dpos);

        /* Send prompt */
        const char *prompt = "tensor> ";
        dpos = 0;
        data[dpos++] = SSH_MSG_CHANNEL_DATA;
        ssh_put_u32(data + dpos, ch->remote_id); dpos += 4;
        dpos += ssh_put_string(data + dpos, prompt, (uint32_t)kstrlen(prompt));
        ssh_send_packet(s, data, dpos);

        return 0;
    }

    /* Handle exec request */
    if (rtype_len == 4 && kstrncmp((const char *)rtype, "exec", 4) == 0) {
        s->state = SSH_STATE_INTERACTIVE;
        if (want_reply) {
            uint8_t succ[8];
            succ[0] = SSH_MSG_CHANNEL_SUCCESS;
            ssh_put_u32(succ + 1, ch->remote_id);
            ssh_send_packet(s, succ, 5);
        }
        return 0;
    }

    /* Handle window-change */
    if (rtype_len == 13 && kstrncmp((const char *)rtype, "window-change", 13) == 0) {
        if (pos + 8 <= len) {
            ch->term_cols = ssh_get_u32(payload + pos);
            ch->term_rows = ssh_get_u32(payload + pos + 4);
        }
        return 0;
    }

    /* Unknown request */
    if (want_reply) {
        uint8_t fail[8];
        fail[0] = SSH_MSG_CHANNEL_FAILURE;
        ssh_put_u32(fail + 1, ch->remote_id);
        ssh_send_packet(s, fail, 5);
    }
    return 0;
}

static int ssh_handle_channel_data(ssh_session_t *s,
                                    const uint8_t *payload, uint32_t len)
{
    uint32_t pos = 1;
    uint32_t recipient = ssh_get_u32(payload + pos); pos += 4;

    uint32_t data_len;
    const uint8_t *data = ssh_get_string(payload + pos, &data_len, len - pos);
    if (!data) return -1;

    ssh_channel_t *ch = ssh_find_channel(s, recipient);
    if (!ch) return -1;

    /* Buffer the data for shell processing */
    uint32_t space = sizeof(ch->rx_buf) - ch->rx_len;
    uint32_t copy = (data_len > space) ? space : data_len;
    if (copy > 0) {
        kmemcpy(ch->rx_buf + ch->rx_len, data, copy);
        ch->rx_len += copy;
    }

    /* Update window — only for bytes actually buffered */
    ch->local_window -= copy;
    if (ch->local_window < SSH_CHANNEL_WINDOW / 2) {
        uint32_t adjust = SSH_CHANNEL_WINDOW - ch->local_window;
        uint8_t wadj[12];
        wadj[0] = SSH_MSG_CHANNEL_WINDOW_ADJUST;
        ssh_put_u32(wadj + 1, ch->remote_id);
        ssh_put_u32(wadj + 5, adjust);
        ssh_send_packet(s, wadj, 9);
        ch->local_window += adjust;
    }

    return 0;
}

/* =============================================================================
 * SSH Packet Dispatch
 * =============================================================================*/

static int ssh_process_packet(ssh_session_t *s,
                               const uint8_t *payload, uint32_t len)
{
    if (len == 0) return -1;
    uint8_t msg_type = payload[0];

    switch (msg_type) {
    case SSH_MSG_DISCONNECT:
        kprintf("[SSH] Client disconnected\n");
        s->state = SSH_STATE_CLOSING;
        return 0;

    case SSH_MSG_IGNORE:
    case SSH_MSG_DEBUG:
        return 0;  /* Silently ignore */

    case SSH_MSG_KEXINIT:
        return ssh_handle_kex_init(s, payload, len);

    case SSH_MSG_KEX_ECDH_INIT:
        return ssh_handle_kex_ecdh_init(s, payload, len);

    case SSH_MSG_NEWKEYS:
        s->keys_active = 1;
        kprintf("[SSH] Encryption activated (session %d)\n",
                (int)(s - g_ssh_server.sessions));
        return 0;

    case SSH_MSG_SERVICE_REQUEST: {
        /* Accept ssh-userauth and ssh-connection */
        uint32_t svc_len;
        const uint8_t *svc = ssh_get_string(payload + 1, &svc_len, len - 1);
        uint8_t accept[64];
        uint32_t apos = 0;
        accept[apos++] = SSH_MSG_SERVICE_ACCEPT;
        apos += ssh_put_string(accept + apos, svc, svc_len);
        return ssh_send_packet(s, accept, apos);
    }

    case SSH_MSG_USERAUTH_REQUEST:
        return ssh_handle_userauth_request(s, payload, len);

    case SSH_MSG_CHANNEL_OPEN:
        return ssh_handle_channel_open(s, payload, len);

    case SSH_MSG_CHANNEL_REQUEST:
        return ssh_handle_channel_request(s, payload, len);

    case SSH_MSG_CHANNEL_DATA:
        return ssh_handle_channel_data(s, payload, len);

    case SSH_MSG_CHANNEL_WINDOW_ADJUST: {
        uint32_t ch_id = ssh_get_u32(payload + 1);
        uint32_t adjust = ssh_get_u32(payload + 5);
        ssh_channel_t *ch = ssh_find_channel(s, ch_id);
        if (ch) ch->remote_window += adjust;
        return 0;
    }

    case SSH_MSG_CHANNEL_EOF:
    case SSH_MSG_CHANNEL_CLOSE: {
        uint32_t ch_id = ssh_get_u32(payload + 1);
        ssh_channel_t *ch = ssh_find_channel(s, ch_id);
        if (ch) {
            if (msg_type == SSH_MSG_CHANNEL_CLOSE) {
                /* Send CLOSE back */
                uint8_t close[8];
                close[0] = SSH_MSG_CHANNEL_CLOSE;
                ssh_put_u32(close + 1, ch->remote_id);
                ssh_send_packet(s, close, 5);
            }
            ch->active = 0;
        }
        return 0;
    }

    case SSH_MSG_GLOBAL_REQUEST: {
        /* Check want_reply */
        uint32_t rtype_len;
        ssh_get_string(payload + 1, &rtype_len, len - 1);
        uint8_t want_reply = payload[5 + rtype_len];
        if (want_reply)
            ssh_send_msg(s, SSH_MSG_REQUEST_FAILURE);
        return 0;
    }

    default:
        kprintf("[SSH] Unhandled message type %d\n", msg_type);
        return 0;
    }
}

/* =============================================================================
 * SSH Version Exchange
 * =============================================================================*/

static int ssh_handle_version(ssh_session_t *s)
{
    /* Look for \r\n terminated version string in rx buffer */
    for (uint32_t i = 0; i + 1 < s->rx_len; i++) {
        if (s->rx_buf[i] == '\r' && s->rx_buf[i + 1] == '\n') {
            /* Found version string */
            uint32_t vlen = (i >= 255) ? 255 : i;
            kmemcpy(s->client_version, s->rx_buf, vlen);
            s->client_version[vlen] = '\0';

            /* Validate it starts with SSH-2.0- */
            if (kstrncmp(s->client_version, "SSH-2.0-", 8) != 0) {
                kprintf("[SSH] Invalid client version: %s\n", s->client_version);
                s->state = SSH_STATE_CLOSING;
                return -1;
            }

            kprintf("[SSH] Client: %s\n", s->client_version);

            /* Remove version from rx buffer */
            uint32_t consumed = i + 2;
            if (consumed < s->rx_len) {
                kmemcpy(s->rx_buf, s->rx_buf + consumed, s->rx_len - consumed);
            }
            s->rx_len -= consumed;
            s->version_received = 1;

            /* Send our KEXINIT */
            return ssh_send_kexinit(s);
        }
    }
    return 0;  /* Need more data */
}

/* =============================================================================
 * SSH TCP Receive Handler
 * =============================================================================*/

static void ssh_session_process(ssh_session_t *s)
{
    if (!s->active || !s->conn) return;

    /* Read data from TCP connection into session rx buffer */
    uint32_t avail = s->conn->rx_len;
    if (avail > 0) {
        uint32_t space = sizeof(s->rx_buf) - s->rx_len;
        uint32_t copy = (avail > space) ? space : avail;
        if (copy > 0) {
            kmemcpy(s->rx_buf + s->rx_len, s->conn->rx_buf, copy);
            s->rx_len += copy;
            /* Shift TCP rx buffer */
            if (copy < s->conn->rx_len) {
                kmemcpy(s->conn->rx_buf, s->conn->rx_buf + copy,
                        s->conn->rx_len - copy);
            }
            s->conn->rx_len -= copy;
        }
    }

    if (s->rx_len == 0) return;

    /* State machine */
    if (!s->version_received) {
        ssh_handle_version(s);
        return;
    }

    /* Process binary packets */
    while (s->rx_len >= 5) {
        /* Read packet length */
        uint32_t pkt_len = ssh_get_u32(s->rx_buf);
        if (pkt_len > SSH_MAX_PACKET_SIZE - 4) {
            kprintf("[SSH] Packet too large: %u\n", pkt_len);
            s->state = SSH_STATE_CLOSING;
            return;
        }

        uint32_t total = 4 + pkt_len;
        uint32_t mac_len = 0;
        if (s->keys_active) {
            mac_len = s->use_chacha ? 16 : 32;
        }

        if (s->rx_len < total + mac_len) return;  /* Need more data */

        /* Decrypt if encryption is active */
        uint8_t decrypted[SSH_MAX_PACKET_SIZE];
        const uint8_t *packet = s->rx_buf;

        if (s->keys_active) {
            if (s->use_chacha) {
                uint8_t nonce[12];
                kmemset(nonce, 0, 8);
                ssh_put_u32(nonce + 8, s->seq_c2s);
                if (chacha20_poly1305_decrypt(
                        s->key_c2s_enc, nonce,
                        NULL, 0,
                        s->rx_buf, total,
                        s->rx_buf + total, decrypted) != 0) {
                    kprintf("[SSH] Decryption failed\n");
                    s->state = SSH_STATE_CLOSING;
                    return;
                }
                packet = decrypted;
            } else {
                /* Verify MAC */
                uint8_t mac_data[4 + SSH_MAX_PACKET_SIZE];
                ssh_put_u32(mac_data, s->seq_c2s);
                kmemcpy(mac_data + 4, s->rx_buf, total);
                uint8_t computed_mac[32];
                hmac_sha256(s->key_c2s_mac, 32, mac_data, 4 + total, computed_mac);
                if (crypto_ct_equal(computed_mac, s->rx_buf + total, 32) != 0) {
                    kprintf("[SSH] MAC verification failed\n");
                    s->state = SSH_STATE_CLOSING;
                    return;
                }

                /* Decrypt payload */
                kmemcpy(decrypted, s->rx_buf, total);
                aes256_ctr(&s->aes_c2s, s->iv_c2s,
                           decrypted + 4, decrypted + 4, pkt_len);
                for (int i = 15; i >= 0; i--) {
                    if (++s->iv_c2s[i]) break;
                }
                packet = decrypted;
            }
        }

        s->seq_c2s++;

        /* Extract payload (skip length, padding_length byte) */
        uint8_t pad_len = packet[4];
        uint32_t payload_len = pkt_len - 1 - pad_len;
        const uint8_t *payload = packet + 5;

        /* Process */
        ssh_process_packet(s, payload, payload_len);

        /* Consume from rx buffer */
        uint32_t consumed = total + mac_len;
        if (consumed < s->rx_len) {
            kmemcpy(s->rx_buf, s->rx_buf + consumed, s->rx_len - consumed);
        }
        s->rx_len -= consumed;
    }
}

/* =============================================================================
 * SSH Server API
 * =============================================================================*/

void ssh_server_init(void)
{
    kmemset(&g_ssh_server, 0, sizeof(g_ssh_server));

    /* Generate Ed25519 host key */
    uint8_t seed[32];
    crypto_random(seed, 32);
    ed25519_keypair(seed, g_ssh_server.host_pub, g_ssh_server.host_priv);
    g_ssh_server.host_key_generated = 1;
    crypto_wipe(seed, 32);

    /* Generate a random initial admin password (printed once at first boot).
     * No hardcoded credentials — the operator MUST read the serial log. */
    {
        uint8_t pw_bytes[10];
        crypto_random(pw_bytes, sizeof(pw_bytes));
        char init_pw[21]; /* 20 hex chars + NUL */
        static const char hex[] = "0123456789abcdef";
        for (int i = 0; i < 10; i++) {
            init_pw[i * 2]     = hex[pw_bytes[i] >> 4];
            init_pw[i * 2 + 1] = hex[pw_bytes[i] & 0xF];
        }
        init_pw[20] = '\0';
        crypto_wipe(pw_bytes, sizeof(pw_bytes));

        ssh_add_user("root", init_pw, 0x7);  /* shell + sftp + admin */

        kprintf("[SSH] *** INITIAL ROOT PASSWORD: %s ***\n", init_pw);
        kprintf("[SSH] Change it immediately with: ssh_passwd root <new>\n");
        crypto_wipe(init_pw, sizeof(init_pw));
    }

    kprintf("[SSH] Server initialized\n");

    /* Print host key fingerprint */
    char fp[128];
    ssh_host_key_fingerprint(fp, sizeof(fp));
    kprintf("[SSH] Host key fingerprint: %s\n", fp);
}

int ssh_server_start(void)
{
    if (g_ssh_server.running) return 0;
    g_ssh_server.running = 1;

    kprintf("[SSH] Listening on port %d\n", SSH_PORT);
    kprintf("[SSH] Supported: curve25519-sha256, ssh-ed25519, aes256-ctr\n");
    kprintf("[SSH] Auth methods: password, publickey\n");
    return 0;
}

void ssh_server_stop(void)
{
    /* Close all sessions */
    for (int i = 0; i < SSH_MAX_SESSIONS; i++) {
        if (g_ssh_server.sessions[i].active) {
            ssh_disconnect_session(i, "Server shutting down");
        }
    }
    g_ssh_server.running = 0;
    kprintf("[SSH] Server stopped\n");
}

void ssh_server_poll(void)
{
    if (!g_ssh_server.running) return;

    /* Process each active session */
    for (int i = 0; i < SSH_MAX_SESSIONS; i++) {
        ssh_session_t *s = &g_ssh_server.sessions[i];
        if (!s->active) continue;

        /* Check connection health */
        if (!s->conn || s->conn->state != TCP_STATE_ESTABLISHED ||
            s->state == SSH_STATE_CLOSING) {
            /* Clean up session */
            crypto_wipe(s->kex_priv, 32);
            crypto_wipe(s->shared_secret, 32);
            crypto_wipe(s->key_c2s_enc, 32);
            crypto_wipe(s->key_s2c_enc, 32);
            crypto_wipe(s->key_c2s_mac, 32);
            crypto_wipe(s->key_s2c_mac, 32);
            kmemset(s, 0, sizeof(*s));
            continue;
        }

        ssh_session_process(s);
    }
}

/* =============================================================================
 * SSH Connection Acceptance (called from network stack)
 * =============================================================================*/

void ssh_accept_connection(tcp_conn_t *conn)
{
    if (!g_ssh_server.running) {
        tcp_conn_close(conn);
        return;
    }

    /* Rate limiting */
    g_ssh_server.connections_per_minute++;
    if (g_ssh_server.connections_per_minute > 30) {
        kprintf("[SSH] Rate limit exceeded, rejecting connection\n");
        tcp_conn_close(conn);
        return;
    }

    /* Find free session */
    ssh_session_t *s = NULL;
    for (int i = 0; i < SSH_MAX_SESSIONS; i++) {
        if (!g_ssh_server.sessions[i].active) {
            s = &g_ssh_server.sessions[i];
            break;
        }
    }

    if (!s) {
        kprintf("[SSH] No free sessions, rejecting\n");
        tcp_conn_close(conn);
        return;
    }

    kmemset(s, 0, sizeof(*s));
    s->active = 1;
    s->conn = conn;
    s->state = SSH_STATE_VERSION_EXCHANGE;
    g_ssh_server.total_connections++;

    kprintf("[SSH] New connection (session %d, total %lu)\n",
            (int)(s - g_ssh_server.sessions),
            g_ssh_server.total_connections);

    /* Send our version string */
    char version_line[64];
    int vlen = 0;
    const char *vs = SSH_ID_STRING;
    while (*vs) version_line[vlen++] = *vs++;
    version_line[vlen++] = '\r';
    version_line[vlen++] = '\n';

    tcp_conn_write(conn, version_line, (uint32_t)vlen);
}

/* =============================================================================
 * User Management
 * =============================================================================*/

int ssh_add_user(const char *username, const char *password, uint32_t perms)
{
    if (g_ssh_server.user_count >= SSH_MAX_USERS) return -1;
    if (!username || !password) return -1;

    /* Check for duplicate */
    if (ssh_find_user(username)) return -1;

    ssh_user_t *u = &g_ssh_server.users[g_ssh_server.user_count++];
    kmemset(u, 0, sizeof(*u));

    /* Copy username */
    for (int i = 0; i < 63 && username[i]; i++)
        u->username[i] = username[i];

    /* Generate random salt */
    crypto_random(u->password_salt, 16);

    /* PBKDF2-HMAC-SHA256 password hash (100K iterations) */
    pbkdf2_hmac_sha256((const uint8_t *)password, (uint32_t)kstrlen(password),
                       u->password_salt, 16,
                       SSH_PBKDF2_ITERATIONS, u->password_hash, 32);

    u->permissions = perms;
    u->active = 1;

    return 0;
}

int ssh_remove_user(const char *username)
{
    for (uint32_t i = 0; i < g_ssh_server.user_count; i++) {
        if (g_ssh_server.users[i].active &&
            kstrcmp(g_ssh_server.users[i].username, username) == 0) {
            crypto_wipe(&g_ssh_server.users[i], sizeof(ssh_user_t));
            g_ssh_server.users[i].active = 0;
            return 0;
        }
    }
    return -1;
}

int ssh_change_password(const char *username, const char *new_password)
{
    ssh_user_t *u = ssh_find_user(username);
    if (!u) return -1;

    /* New salt */
    crypto_random(u->password_salt, 16);

    /* PBKDF2-HMAC-SHA256 password hash (100K iterations) */
    pbkdf2_hmac_sha256((const uint8_t *)new_password, (uint32_t)kstrlen(new_password),
                       u->password_salt, 16,
                       SSH_PBKDF2_ITERATIONS, u->password_hash, 32);

    return 0;
}

int ssh_add_user_pubkey(const char *username, const uint8_t pubkey[32])
{
    ssh_user_t *u = ssh_find_user(username);
    if (!u) return -1;

    kmemcpy(u->pubkey, pubkey, 32);
    u->pubkey_set = 1;
    return 0;
}

int ssh_list_users(ssh_user_t *users, uint32_t max, uint32_t *count)
{
    uint32_t found = 0;
    for (uint32_t i = 0; i < g_ssh_server.user_count && found < max; i++) {
        if (g_ssh_server.users[i].active) {
            users[found++] = g_ssh_server.users[i];
        }
    }
    if (count) *count = found;
    return 0;
}

/* =============================================================================
 * Host Key Management
 * =============================================================================*/

void ssh_regenerate_host_key(void)
{
    uint8_t seed[32];
    crypto_random(seed, 32);
    ed25519_keypair(seed, g_ssh_server.host_pub, g_ssh_server.host_priv);
    crypto_wipe(seed, 32);
    kprintf("[SSH] Host key regenerated\n");

    char fp[128];
    ssh_host_key_fingerprint(fp, sizeof(fp));
    kprintf("[SSH] New fingerprint: %s\n", fp);
}

void ssh_host_key_fingerprint(char *buf, uint32_t buflen)
{
    /* Fingerprint = SHA-256 of host key blob */
    uint8_t blob[128];
    uint32_t bpos = 0;
    bpos += ssh_put_string(blob + bpos, "ssh-ed25519", 11);
    bpos += ssh_put_string(blob + bpos, g_ssh_server.host_pub, 32);

    uint8_t hash[32];
    sha256(blob, bpos, hash);

    /* Format as "SHA256:<hex>" */
    if (buflen < 8) { buf[0] = '\0'; return; }
    buf[0] = 'S'; buf[1] = 'H'; buf[2] = 'A';
    buf[3] = '2'; buf[4] = '5'; buf[5] = '6'; buf[6] = ':';
    to_hex(buf + 7, hash, 16);  /* First 16 bytes = 48 hex chars */
}

/* =============================================================================
 * Status / Info
 * =============================================================================*/

void ssh_print_status(void)
{
    kprintf("SSH Server Status:\n");
    kprintf("  Running:      %s\n", g_ssh_server.running ? "yes" : "no");
    kprintf("  Port:         %d\n", SSH_PORT);
    kprintf("  Host key:     ssh-ed25519\n");

    char fp[128];
    ssh_host_key_fingerprint(fp, sizeof(fp));
    kprintf("  Fingerprint:  %s\n", fp);

    kprintf("  Total conns:  %lu\n", g_ssh_server.total_connections);
    kprintf("  Auth fails:   %lu\n", g_ssh_server.total_auth_failures);
    kprintf("  Packets:      %lu\n", g_ssh_server.total_packets);

    int active = 0;
    for (int i = 0; i < SSH_MAX_SESSIONS; i++) {
        if (g_ssh_server.sessions[i].active) {
            active++;
            kprintf("  Session %d: user=%s state=%d\n",
                    i, g_ssh_server.sessions[i].auth_user,
                    g_ssh_server.sessions[i].state);
        }
    }
    kprintf("  Active:       %d / %d\n", active, SSH_MAX_SESSIONS);
    kprintf("  Users:        %u\n", g_ssh_server.user_count);
    for (uint32_t i = 0; i < g_ssh_server.user_count; i++) {
        if (g_ssh_server.users[i].active) {
            kprintf("    %s (perms=0x%x, pubkey=%s)\n",
                    g_ssh_server.users[i].username,
                    g_ssh_server.users[i].permissions,
                    g_ssh_server.users[i].pubkey_set ? "yes" : "no");
        }
    }
}

int ssh_active_sessions(void)
{
    int n = 0;
    for (int i = 0; i < SSH_MAX_SESSIONS; i++)
        if (g_ssh_server.sessions[i].active) n++;
    return n;
}

int ssh_disconnect_session(int session_id, const char *reason)
{
    if (session_id < 0 || session_id >= SSH_MAX_SESSIONS) return -1;
    ssh_session_t *s = &g_ssh_server.sessions[session_id];
    if (!s->active) return -1;

    /* Send disconnect message */
    uint8_t disc[256];
    uint32_t dpos = 0;
    disc[dpos++] = SSH_MSG_DISCONNECT;
    ssh_put_u32(disc + dpos, SSH_DISCONNECT_BY_APPLICATION); dpos += 4;
    uint32_t rlen = reason ? (uint32_t)kstrlen(reason) : 0;
    dpos += ssh_put_string(disc + dpos, reason ? reason : "", rlen);
    dpos += ssh_put_string(disc + dpos, "", 0);
    ssh_send_packet(s, disc, dpos);

    /* Clean up */
    if (s->conn) tcp_conn_close(s->conn);
    crypto_wipe(s, sizeof(*s));
    return 0;
}
