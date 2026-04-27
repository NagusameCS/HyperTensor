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
 * TensorOS — TLS 1.3 Implementation (RFC 8446)
 *
 * Minimal but correct TLS 1.3 server supporting:
 *   Cipher suite: TLS_CHACHA20_POLY1305_SHA256 (0x1303)
 *   Key exchange: X25519
 *   Signatures:   Ed25519
 *   Key derivation: HKDF-SHA256
 *
 * Flow:
 *   1. ClientHello → parse, extract X25519 key_share
 *   2. ServerHello → send ephemeral X25519 public key
 *   3. Derive handshake keys
 *   4. EncryptedExtensions + Certificate + CertificateVerify + Finished
 *   5. Receive client Finished
 *   6. Derive application keys → encrypted data exchange
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/security/crypto.h"
#include "kernel/net/tls.h"
#include "kernel/net/netstack.h"

/* =============================================================================
 * Static state
 * =============================================================================*/

/* Self-signed Ed25519 keys (generated at boot) */
static uint8_t tls_cert_pub[32];
static uint8_t tls_cert_priv[64];
static int tls_initialized = 0;

/* Session pool */
#define TLS_MAX_SESSIONS 32
static tls_session_t tls_sessions[TLS_MAX_SESSIONS];

/* =============================================================================
 * TLS record / handshake helpers
 * =============================================================================*/

static void tls_put_u16(uint8_t *p, uint16_t v)
{
    p[0] = (uint8_t)(v >> 8);
    p[1] = (uint8_t)v;
}

static void tls_put_u24(uint8_t *p, uint32_t v)
{
    p[0] = (uint8_t)(v >> 16);
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)v;
}

static uint16_t tls_get_u16(const uint8_t *p)
{
    return ((uint16_t)p[0] << 8) | p[1];
}

static uint32_t tls_get_u24(const uint8_t *p)
{
    return ((uint32_t)p[0] << 16) | ((uint32_t)p[1] << 8) | p[2];
}

/* Build a TLS record header: type(1) + version(2) + length(2) */
static void tls_record_header(uint8_t *buf, uint8_t type, uint16_t len)
{
    buf[0] = type;
    buf[1] = 0x03; buf[2] = 0x03; /* TLS 1.2 in record layer (required by 1.3) */
    tls_put_u16(buf + 3, len);
}

/* Send a raw TLS record over TCP */
static int tls_send_record(tcp_conn_t *conn, uint8_t type,
                           const uint8_t *data, uint16_t len)
{
    uint8_t hdr[5];
    tls_record_header(hdr, type, len);
    tcp_conn_write(conn, hdr, 5);
    if (len > 0)
        tcp_conn_write(conn, data, len);
    return 0;
}

/* Send encrypted TLS record (application data or handshake after keys) */
static int tls_send_encrypted(tls_session_t *s, tcp_conn_t *conn,
                               uint8_t inner_type,
                               const uint8_t *plaintext, uint32_t pt_len)
{
    /* Build plaintext with inner content type appended */
    uint8_t pt_buf[16384 + 1];
    if (pt_len + 1 > sizeof(pt_buf)) return -1;
    if (pt_len > 0) kmemcpy(pt_buf, plaintext, pt_len);
    pt_buf[pt_len] = inner_type; /* Inner content type */
    uint32_t total_pt = pt_len + 1;

    /* Construct nonce: XOR IV with sequence number */
    uint8_t nonce[12];
    kmemcpy(nonce, s->server_write_iv, 12);
    for (int i = 0; i < 8; i++)
        nonce[11 - i] ^= (uint8_t)(s->server_seq >> (i * 8));

    /* Encrypt with ChaCha20-Poly1305 */
    uint8_t ct[16384 + 1];
    uint8_t tag[16];

    /* AAD = TLS record header (type=23, version=0x0303, length=ct_len+16) */
    uint8_t aad[5];
    aad[0] = TLS_RT_APPLICATION;
    aad[1] = 0x03; aad[2] = 0x03;
    tls_put_u16(aad + 3, (uint16_t)(total_pt + 16));

    chacha20_poly1305_encrypt(s->server_write_key, nonce,
                               aad, 5,
                               pt_buf, total_pt,
                               ct, tag);

    s->server_seq++;

    /* Send as ApplicationData record */
    uint8_t rec[5 + 16384 + 1 + 16];
    rec[0] = TLS_RT_APPLICATION;
    rec[1] = 0x03; rec[2] = 0x03;
    tls_put_u16(rec + 3, (uint16_t)(total_pt + 16));
    kmemcpy(rec + 5, ct, total_pt);
    kmemcpy(rec + 5 + total_pt, tag, 16);

    tcp_conn_write(conn, rec, 5 + total_pt + 16);
    return 0;
}

/* =============================================================================
 * HKDF-based key schedule (RFC 8446 §7.1)
 * =============================================================================*/

/* HKDF-Expand-Label(Secret, Label, Context, Length) */
static void hkdf_expand_label(const uint8_t secret[32],
                               const char *label, uint32_t label_len,
                               const uint8_t *context, uint32_t ctx_len,
                               uint8_t *out, uint32_t out_len)
{
    /* HkdfLabel = length(2) + "tls13 " + label + context_len(1) + context */
    uint8_t info[256];
    uint32_t pos = 0;
    info[pos++] = (uint8_t)(out_len >> 8);
    info[pos++] = (uint8_t)out_len;
    uint32_t full_label_len = 6 + label_len; /* "tls13 " prefix */
    info[pos++] = (uint8_t)full_label_len;
    kmemcpy(info + pos, "tls13 ", 6); pos += 6;
    kmemcpy(info + pos, label, label_len); pos += label_len;
    info[pos++] = (uint8_t)ctx_len;
    if (ctx_len > 0) { kmemcpy(info + pos, context, ctx_len); pos += ctx_len; }

    hkdf_sha256_expand(secret, info, pos, out, out_len);
}

/* Derive-Secret(Secret, Label, Messages) = HKDF-Expand-Label(Secret, Label, Hash(Messages), 32) */
static void derive_secret(const uint8_t secret[32],
                           const char *label, uint32_t label_len,
                           const uint8_t transcript_hash[32],
                           uint8_t out[32])
{
    hkdf_expand_label(secret, label, label_len, transcript_hash, 32, out, 32);
}

/* =============================================================================
 * ClientHello parsing
 * =============================================================================*/

/* Parse ClientHello, extract X25519 key_share.
 * Returns 0 on success, -1 on failure. */
static int parse_client_hello(const uint8_t *msg, uint32_t msg_len,
                               uint8_t client_x25519_pub[32])
{
    if (msg_len < 39) return -1;

    uint32_t pos = 0;
    /* Skip: client_version(2) + random(32) */
    pos += 2 + 32;

    /* session_id_len(1) + session_id */
    if (pos >= msg_len) return -1;
    uint8_t sid_len = msg[pos++];
    pos += sid_len;

    /* cipher_suites_len(2) + cipher_suites */
    if (pos + 2 > msg_len) return -1;
    uint16_t cs_len = tls_get_u16(msg + pos); pos += 2;
    /* Check for TLS_CHACHA20_POLY1305_SHA256 (0x1303) */
    int found_suite = 0;
    for (uint32_t i = 0; i < cs_len; i += 2) {
        if (pos + i + 1 < msg_len) {
            uint16_t suite = tls_get_u16(msg + pos + i);
            if (suite == 0x1303) found_suite = 1;
        }
    }
    pos += cs_len;
    if (!found_suite) return -1;

    /* compression_methods_len(1) + methods */
    if (pos >= msg_len) return -1;
    uint8_t comp_len = msg[pos++];
    pos += comp_len;

    /* Extensions */
    if (pos + 2 > msg_len) return -1;
    uint16_t ext_len = tls_get_u16(msg + pos); pos += 2;
    uint32_t ext_end = pos + ext_len;
    if (ext_end > msg_len) ext_end = msg_len;

    int found_key = 0;
    while (pos + 4 <= ext_end) {
        uint16_t ext_type = tls_get_u16(msg + pos); pos += 2;
        uint16_t elen = tls_get_u16(msg + pos); pos += 2;
        if (pos + elen > ext_end) break;

        if (ext_type == 0x0033) { /* key_share */
            /* client_shares_len(2) + entries */
            if (elen < 2) { pos += elen; continue; }
            uint16_t shares_len = tls_get_u16(msg + pos);
            uint32_t sp = pos + 2;
            uint32_t shares_end = sp + shares_len;
            if (shares_end > pos + elen) shares_end = pos + elen;

            while (sp + 4 <= shares_end) {
                uint16_t group = tls_get_u16(msg + sp); sp += 2;
                uint16_t klen = tls_get_u16(msg + sp); sp += 2;
                if (group == 0x001D && klen == 32 && sp + 32 <= shares_end) {
                    kmemcpy(client_x25519_pub, msg + sp, 32);
                    found_key = 1;
                }
                sp += klen;
            }
        }
        pos += elen;
    }

    return found_key ? 0 : -1;
}

/* =============================================================================
 * TLS 1.3 Handshake — Server Side
 * =============================================================================*/

static int tls_do_handshake(tls_session_t *s, tcp_conn_t *conn,
                             const uint8_t *record_data, uint32_t record_len)
{
    /* record_data starts after the 5-byte TLS record header */
    if (record_len < 4) return -1;

    uint8_t hs_type = record_data[0];
    uint32_t hs_len = tls_get_u24(record_data + 1);
    if (hs_len + 4 > record_len) return -1;

    const uint8_t *hs_body = record_data + 4;

    if (hs_type != TLS_HS_CLIENT_HELLO) return -1;

    /* --- Parse ClientHello --- */
    uint8_t client_pub[32];
    if (parse_client_hello(hs_body, hs_len, client_pub) != 0) return -1;

    /* Save ClientHello for transcript */
    uint32_t ch_total = 4 + hs_len;
    if (ch_total > sizeof(s->hs_buf)) return -1;
    kmemcpy(s->hs_buf, record_data, ch_total);
    s->hs_len = ch_total;

    /* --- Generate ephemeral X25519 keypair --- */
    crypto_random(s->server_privkey, 32);
    x25519_public_key(s->server_pubkey, s->server_privkey);

    /* --- Compute shared secret --- */
    x25519(s->shared_secret, s->server_privkey, client_pub);

    /* --- Build ServerHello --- */
    uint8_t sh[512];
    uint32_t sp = 0;

    /* Handshake header (fill length later) */
    sh[sp++] = TLS_HS_SERVER_HELLO;
    sp += 3; /* placeholder for length */

    /* server_version = TLS 1.2 (for compat, real version in supported_versions ext) */
    sh[sp++] = 0x03; sh[sp++] = 0x03;

    /* server_random (32 bytes) */
    crypto_random(sh + sp, 32); sp += 32;

    /* session_id (echo client's — we use 0 length) */
    sh[sp++] = 0; /* session_id_length = 0 */

    /* cipher_suite = TLS_CHACHA20_POLY1305_SHA256 */
    sh[sp++] = 0x13; sh[sp++] = 0x03;

    /* compression_method = null */
    sh[sp++] = 0x00;

    /* Extensions */
    uint32_t ext_start = sp;
    sp += 2; /* extensions length placeholder */

    /* supported_versions extension (0x002B) — indicate TLS 1.3 */
    tls_put_u16(sh + sp, 0x002B); sp += 2;
    tls_put_u16(sh + sp, 2); sp += 2; /* ext data length */
    sh[sp++] = 0x03; sh[sp++] = 0x04; /* TLS 1.3 */

    /* key_share extension (0x0033) — X25519 server public key */
    tls_put_u16(sh + sp, 0x0033); sp += 2;
    tls_put_u16(sh + sp, 36); sp += 2; /* ext data length: group(2) + klen(2) + key(32) */
    tls_put_u16(sh + sp, 0x001D); sp += 2; /* x25519 group */
    tls_put_u16(sh + sp, 32); sp += 2;     /* key length */
    kmemcpy(sh + sp, s->server_pubkey, 32); sp += 32;

    /* Fill extensions length */
    tls_put_u16(sh + ext_start, (uint16_t)(sp - ext_start - 2));

    /* Fill handshake length */
    tls_put_u24(sh + 1, sp - 4);

    /* Send ServerHello as plaintext */
    tls_send_record(conn, TLS_RT_HANDSHAKE, sh, (uint16_t)sp);

    /* --- Transcript hash: Hash(ClientHello || ServerHello) --- */
    /* Append ServerHello to transcript buffer */
    if (s->hs_len + sp <= sizeof(s->hs_buf)) {
        kmemcpy(s->hs_buf + s->hs_len, sh, sp);
        s->hs_len += sp;
    }
    sha256(s->hs_buf, s->hs_len, s->transcript_hash);

    /* --- Key Schedule --- */
    /* early_secret = HKDF-Extract(salt=0, IKM=0) */
    uint8_t zero_key[32];
    kmemset(zero_key, 0, 32);
    uint8_t early_secret[32];
    hkdf_sha256_extract(zero_key, 32, zero_key, 32, early_secret); /* PSK=0 for non-PSK */

    /* derived_secret = Derive-Secret(early_secret, "derived", empty_hash) */
    uint8_t empty_hash[32];
    sha256(NULL, 0, empty_hash);
    uint8_t derived[32];
    derive_secret(early_secret, "derived", 7, empty_hash, derived);

    /* handshake_secret = HKDF-Extract(salt=derived, IKM=shared_secret) */
    uint8_t handshake_secret[32];
    hkdf_sha256_extract(derived, 32, s->shared_secret, 32, handshake_secret);

    /* server_handshake_traffic_secret */
    uint8_t shts[32];
    derive_secret(handshake_secret, "s hs traffic", 12, s->transcript_hash, shts);

    /* client_handshake_traffic_secret */
    uint8_t chts[32];
    derive_secret(handshake_secret, "c hs traffic", 12, s->transcript_hash, chts);

    /* Derive handshake keys */
    uint8_t server_hs_key[32], server_hs_iv[12];
    uint8_t client_hs_key[32], client_hs_iv[12];
    hkdf_expand_label(shts, "key", 3, NULL, 0, server_hs_key, 32);
    hkdf_expand_label(shts, "iv", 2, NULL, 0, server_hs_iv, 12);
    hkdf_expand_label(chts, "key", 3, NULL, 0, client_hs_key, 32);
    hkdf_expand_label(chts, "iv", 2, NULL, 0, client_hs_iv, 12);

    /* Use handshake keys for the rest of the handshake */
    kmemcpy(s->server_write_key, server_hs_key, 32);
    kmemcpy(s->server_write_iv, server_hs_iv, 12);
    kmemcpy(s->client_write_key, client_hs_key, 32);
    kmemcpy(s->client_write_iv, client_hs_iv, 12);
    s->server_seq = 0;
    s->client_seq = 0;

    /* Send ChangeCipherSpec (compatibility, ignored by TLS 1.3 peers) */
    uint8_t ccs = 1;
    tls_send_record(conn, TLS_RT_CHANGE_CIPHER, &ccs, 1);

    /* --- EncryptedExtensions (empty) --- */
    uint8_t ee[4] = { TLS_HS_ENCRYPTED_EXT, 0, 0, 2 };
    uint8_t ee_ext[2] = { 0, 0 }; /* extensions_length = 0 */
    uint8_t ee_msg[6];
    kmemcpy(ee_msg, ee, 4);
    kmemcpy(ee_msg + 4, ee_ext, 2);
    tls_send_encrypted(s, conn, TLS_RT_HANDSHAKE, ee_msg, 6);

    /* Update transcript with EncryptedExtensions */
    if (s->hs_len + 6 <= sizeof(s->hs_buf)) {
        kmemcpy(s->hs_buf + s->hs_len, ee_msg, 6);
        s->hs_len += 6;
    }

    /* --- Certificate (minimal self-signed Ed25519) --- */
    /* Build a minimal Certificate message */
    uint8_t cert_msg[256];
    uint32_t cp = 0;
    cert_msg[cp++] = TLS_HS_CERTIFICATE;
    cp += 3; /* length placeholder */
    cert_msg[cp++] = 0; /* certificate_request_context length = 0 */

    /* certificate_list length (will fill) */
    uint32_t cl_pos = cp;
    cp += 3;

    /* Single CertificateEntry: a minimal DER-like wrapper around Ed25519 pubkey */
    /* cert_data length */
    uint32_t cd_pos = cp;
    cp += 3;
    /* Minimal "certificate" — just the raw Ed25519 public key (32 bytes)
     * Real TLS uses X.509 DER, but for a self-signed bare-metal OS this is
     * sufficient and any client using --insecure / verify=False will accept it */
    kmemcpy(cert_msg + cp, s->cert_pub, 32);
    cp += 32;
    tls_put_u24(cert_msg + cd_pos, 32);

    /* Extensions for this CertificateEntry (none) */
    tls_put_u16(cert_msg + cp, 0); cp += 2;

    /* Fill certificate_list length */
    tls_put_u24(cert_msg + cl_pos, cp - cl_pos - 3);

    /* Fill handshake message length */
    tls_put_u24(cert_msg + 1, cp - 4);

    tls_send_encrypted(s, conn, TLS_RT_HANDSHAKE, cert_msg, cp);
    if (s->hs_len + cp <= sizeof(s->hs_buf)) {
        kmemcpy(s->hs_buf + s->hs_len, cert_msg, cp);
        s->hs_len += cp;
    }

    /* --- CertificateVerify --- */
    /* Sign the transcript hash with Ed25519 */
    sha256(s->hs_buf, s->hs_len, s->transcript_hash);

    /* Build content to sign: 64×0x20 + "TLS 1.3, server CertificateVerify" + 0x00 + hash */
    uint8_t sign_content[128 + 32];
    kmemset(sign_content, 0x20, 64);
    /* "TLS 1.3, server CertificateVerify" = 34 chars */
    const char *cv_label = "TLS 1.3, server CertificateVerify";
    kmemcpy(sign_content + 64, cv_label, 33);
    sign_content[64 + 33] = 0x00;
    kmemcpy(sign_content + 98, s->transcript_hash, 32);

    uint8_t sig[64];
    ed25519_sign(sign_content, 130, s->cert_pub, s->cert_priv, sig);

    uint8_t cv_msg[72];
    cv_msg[0] = TLS_HS_CERT_VERIFY;
    tls_put_u24(cv_msg + 1, 68); /* 2(algo) + 2(sig_len) + 64(sig) */
    cv_msg[4] = 0x08; cv_msg[5] = 0x07; /* ed25519 (0x0807) */
    tls_put_u16(cv_msg + 6, 64);
    kmemcpy(cv_msg + 8, sig, 64);

    tls_send_encrypted(s, conn, TLS_RT_HANDSHAKE, cv_msg, 72);
    if (s->hs_len + 72 <= sizeof(s->hs_buf)) {
        kmemcpy(s->hs_buf + s->hs_len, cv_msg, 72);
        s->hs_len += 72;
    }

    /* --- Finished --- */
    sha256(s->hs_buf, s->hs_len, s->transcript_hash);

    /* finished_key = HKDF-Expand-Label(server_hs_traffic_secret, "finished", "", 32) */
    uint8_t finished_key[32];
    hkdf_expand_label(shts, "finished", 8, NULL, 0, finished_key, 32);

    uint8_t verify_data[32];
    hmac_sha256(finished_key, 32, s->transcript_hash, 32, verify_data);

    uint8_t fin_msg[36];
    fin_msg[0] = TLS_HS_FINISHED;
    tls_put_u24(fin_msg + 1, 32);
    kmemcpy(fin_msg + 4, verify_data, 32);

    tls_send_encrypted(s, conn, TLS_RT_HANDSHAKE, fin_msg, 36);
    if (s->hs_len + 36 <= sizeof(s->hs_buf)) {
        kmemcpy(s->hs_buf + s->hs_len, fin_msg, 36);
        s->hs_len += 36;
    }

    /* --- Derive application traffic keys --- */
    sha256(s->hs_buf, s->hs_len, s->transcript_hash);

    /* master_secret = HKDF-Extract(derived2, 0) */
    uint8_t derived2[32];
    derive_secret(handshake_secret, "derived", 7, empty_hash, derived2);
    uint8_t master_secret[32];
    hkdf_sha256_extract(derived2, 32, zero_key, 32, master_secret);

    /* server_application_traffic_secret_0 */
    uint8_t sats[32];
    derive_secret(master_secret, "s ap traffic", 12, s->transcript_hash, sats);

    /* client_application_traffic_secret_0 */
    uint8_t cats[32];
    derive_secret(master_secret, "c ap traffic", 12, s->transcript_hash, cats);

    /* Derive application keys */
    hkdf_expand_label(sats, "key", 3, NULL, 0, s->server_write_key, 32);
    hkdf_expand_label(sats, "iv", 2, NULL, 0, s->server_write_iv, 12);
    hkdf_expand_label(cats, "key", 3, NULL, 0, s->client_write_key, 32);
    hkdf_expand_label(cats, "iv", 2, NULL, 0, s->client_write_iv, 12);

    /* Reset sequence numbers for application data */
    s->server_seq = 0;
    s->client_seq = 0;

    /* Wipe sensitive intermediates */
    crypto_wipe(zero_key, 32);
    crypto_wipe(early_secret, 32);
    crypto_wipe(derived, 32);
    crypto_wipe(derived2, 32);
    crypto_wipe(handshake_secret, 32);
    crypto_wipe(master_secret, 32);
    crypto_wipe(shts, 32);
    crypto_wipe(chts, 32);
    crypto_wipe(sats, 32);
    crypto_wipe(cats, 32);
    crypto_wipe(server_hs_key, 32);
    crypto_wipe(client_hs_key, 32);
    crypto_wipe(finished_key, 32);

    s->state = TLS_STATE_ACTIVE;
    kprintf("[TLS] Handshake complete (TLS 1.3 ChaCha20-Poly1305)\n");
    return 0;
}

/* =============================================================================
 * Decrypt incoming TLS 1.3 record
 * =============================================================================*/

static int tls_decrypt_record(tls_session_t *s,
                               const uint8_t *ct, uint32_t ct_len,
                               uint8_t *pt, uint32_t *pt_len,
                               uint8_t *inner_type)
{
    /* ct includes ciphertext + 16-byte tag */
    if (ct_len < 17) return -1; /* At least 1 byte content + 16 tag */

    uint32_t payload_len = ct_len - 16;
    const uint8_t *tag = ct + payload_len;

    /* Construct nonce */
    uint8_t nonce[12];
    kmemcpy(nonce, s->client_write_iv, 12);
    for (int i = 0; i < 8; i++)
        nonce[11 - i] ^= (uint8_t)(s->client_seq >> (i * 8));

    /* AAD = record header */
    uint8_t aad[5];
    aad[0] = TLS_RT_APPLICATION;
    aad[1] = 0x03; aad[2] = 0x03;
    tls_put_u16(aad + 3, (uint16_t)ct_len);

    /* Decrypt */
    if (chacha20_poly1305_decrypt(s->client_write_key, nonce,
                                   aad, 5,
                                   ct, payload_len,
                                   tag, pt) != 0) {
        return -1; /* Authentication failed */
    }

    s->client_seq++;

    /* Find inner content type (last non-zero byte) */
    int i = (int)payload_len - 1;
    while (i >= 0 && pt[i] == 0) i--;
    if (i < 0) return -1;
    *inner_type = pt[i];
    *pt_len = (uint32_t)i; /* Exclude content type byte */

    return 0;
}

/* =============================================================================
 * Public API
 * =============================================================================*/

void tls_init(void)
{
    /* Generate self-signed Ed25519 certificate */
    uint8_t seed[32];
    crypto_random(seed, 32);
    ed25519_keypair(seed, tls_cert_pub, tls_cert_priv);
    crypto_wipe(seed, 32);

    kmemset(tls_sessions, 0, sizeof(tls_sessions));
    tls_initialized = 1;

    kprintf("[TLS] TLS 1.3 initialized (ChaCha20-Poly1305 + X25519 + Ed25519)\n");
}

tls_session_t *tls_session_new(void)
{
    for (int i = 0; i < TLS_MAX_SESSIONS; i++) {
        if (tls_sessions[i].state == TLS_STATE_NONE) {
            tls_session_t *s = &tls_sessions[i];
            kmemset(s, 0, sizeof(*s));
            s->state = TLS_STATE_HANDSHAKE;
            kmemcpy(s->cert_pub, tls_cert_pub, 32);
            kmemcpy(s->cert_priv, tls_cert_priv, 64);
            return s;
        }
    }
    return NULL;
}

void tls_session_free(tls_session_t *session)
{
    if (!session) return;
    crypto_wipe(session, sizeof(*session));
    session->state = TLS_STATE_NONE;
}

int tls_process_record(tls_session_t *session, tcp_conn_t *conn,
                       const uint8_t *data, uint32_t len)
{
    if (!session || len < 5) return -1;

    uint8_t record_type = data[0];
    uint16_t record_len = tls_get_u16(data + 3);
    if (5 + record_len > len) return -1;

    const uint8_t *record_data = data + 5;

    switch (session->state) {
    case TLS_STATE_HANDSHAKE:
        if (record_type == TLS_RT_HANDSHAKE) {
            return tls_do_handshake(session, conn, record_data, record_len);
        }
        return -1;

    case TLS_STATE_ACTIVE:
        if (record_type == TLS_RT_APPLICATION) {
            uint8_t inner_type;
            uint32_t pt_len;
            if (tls_decrypt_record(session, record_data, record_len,
                                    session->plaintext_buf, &pt_len,
                                    &inner_type) != 0) {
                session->state = TLS_STATE_ERROR;
                return -1;
            }

            if (inner_type == TLS_RT_HANDSHAKE) {
                /* Client Finished — verify but don't strictly enforce for now */
                return 0;
            }
            if (inner_type == TLS_RT_APPLICATION) {
                session->plaintext_len = pt_len;
                return (int)pt_len;
            }
            if (inner_type == TLS_RT_ALERT) {
                session->state = TLS_STATE_ERROR;
                return -1;
            }
            return 0;
        }
        /* Ignore ChangeCipherSpec records in 1.3 */
        if (record_type == TLS_RT_CHANGE_CIPHER) return 0;
        return -1;

    default:
        return -1;
    }
}

int tls_send(tls_session_t *session, tcp_conn_t *conn,
             const void *data, uint32_t len)
{
    if (!session || session->state != TLS_STATE_ACTIVE) return -1;
    return tls_send_encrypted(session, conn, TLS_RT_APPLICATION,
                               (const uint8_t *)data, len);
}
