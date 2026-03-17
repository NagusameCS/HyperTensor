/* =============================================================================
 * TensorOS - Network Stack Implementation
 * ARP responder, IPv4, ICMP echo, UDP, TCP, and OpenAI-compatible HTTP API
 *
 * TCP implementation: simplified 3-way handshake, in-order delivery,
 * segmented send, FIN teardown. Sufficient for HTTP request/response.
 *
 * HTTP API: OpenAI-compatible REST endpoints for LLM inference.
 * Any device on the network can connect with curl, Python, or the OpenAI SDK.
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/core/perf.h"
#include "kernel/drivers/net/virtio_net.h"
#include "kernel/net/netstack.h"
#include "kernel/net/tls.h"
#include "kernel/security/crypto.h"
#include "runtime/nn/llm.h"

/* =============================================================================
 * State
 * =============================================================================*/

static net_config_t net_cfg;
static arp_entry_t arp_cache[ARP_CACHE_SIZE];

/* Statistics */
static uint64_t stat_rx_frames;
static uint64_t stat_tx_frames;
static uint64_t stat_arp_req;
static uint64_t stat_arp_rep;
static uint64_t stat_ip_rx;
static uint64_t stat_udp_rx;
static uint64_t stat_icmp_rx;
static uint64_t stat_tcp_rx;
static uint64_t stat_http_req;
static uint64_t stat_http_infer;

/* API authentication — set via net_set_api_key(); empty = no auth required */
static char api_key[128];

void net_set_api_key(const char *key)
{
    if (!key) { api_key[0] = '\0'; return; }
    kstrlcpy(api_key, key, sizeof(api_key));
}

/* UDP port handlers */
#define MAX_UDP_HANDLERS 16
static struct {
    uint16_t     port;
    udp_handler_t handler;
} udp_handlers[MAX_UDP_HANDLERS];
static int n_udp_handlers;

/* TCP connections */
static tcp_conn_t tcp_conns[TCP_MAX_CONNS];
static uint16_t tcp_listen_port;
static uint32_t tcp_isn_counter = 0x10000; /* Initial sequence number counter */
static uint32_t tcp_isn_secret; /* Randomized at init */

/* FNV-1a hash for unpredictable ISN generation (RFC 6528) */
static uint32_t isn_hash(const uint8_t src[4], uint16_t sport,
                         const uint8_t dst[4], uint16_t dport,
                         uint32_t counter)
{
    uint32_t h = 0x811c9dc5; /* FNV-1a offset basis */
    h = (h ^ tcp_isn_secret) * 0x01000193;
    for (int i = 0; i < 4; i++) h = (h ^ src[i]) * 0x01000193;
    for (int i = 0; i < 4; i++) h = (h ^ dst[i]) * 0x01000193;
    h = (h ^ (sport >> 8)) * 0x01000193;
    h = (h ^ (sport & 0xFF)) * 0x01000193;
    h = (h ^ (dport >> 8)) * 0x01000193;
    h = (h ^ (dport & 0xFF)) * 0x01000193;
    for (int i = 0; i < 4; i++) {
        h = (h ^ ((counter >> (i * 8)) & 0xFF)) * 0x01000193;
    }
    return h;
}

/* TX frame buffer */
static uint8_t tx_frame[2048] __attribute__((aligned(16)));

/* IP identification counter */
static uint16_t ip_id_counter = 1;

/* =============================================================================
 * IP checksum
 * =============================================================================*/

static uint16_t ip_checksum(const void *data, uint32_t len)
{
    const uint16_t *p = (const uint16_t *)data;
    uint32_t sum = 0;
    while (len > 1) {
        sum += *p++;
        len -= 2;
    }
    if (len == 1) sum += *(const uint8_t *)p;
    sum = (sum >> 16) + (sum & 0xFFFF);
    sum += (sum >> 16);
    return (uint16_t)(~sum);
}

/* TCP pseudo-header checksum */
static uint16_t tcp_checksum(const uint8_t src[4], const uint8_t dst[4],
                             const void *tcp_segment, uint32_t tcp_len)
{
    uint32_t sum = 0;
    /* Pseudo-header: src IP, dst IP, zero, protocol, TCP length */
    sum += ((uint16_t)src[0] << 8) | src[1];
    sum += ((uint16_t)src[2] << 8) | src[3];
    sum += ((uint16_t)dst[0] << 8) | dst[1];
    sum += ((uint16_t)dst[2] << 8) | dst[3];
    sum += IP_PROTO_TCP;
    sum += tcp_len;

    /* TCP segment data */
    const uint16_t *p = (const uint16_t *)tcp_segment;
    uint32_t remaining = tcp_len;
    while (remaining > 1) {
        sum += *p++;
        remaining -= 2;
    }
    if (remaining == 1) sum += *(const uint8_t *)p;

    sum = (sum >> 16) + (sum & 0xFFFF);
    sum += (sum >> 16);
    return (uint16_t)(~sum);
}

/* =============================================================================
 * Byte comparison helpers
 * =============================================================================*/

static int ip_eq(const uint8_t a[4], const uint8_t b[4])
{
    return a[0]==b[0] && a[1]==b[1] && a[2]==b[2] && a[3]==b[3];
}

__attribute__((unused))
static int mac_eq(const uint8_t a[6], const uint8_t b[6])
{
    return a[0]==b[0] && a[1]==b[1] && a[2]==b[2] &&
           a[3]==b[3] && a[4]==b[4] && a[5]==b[5];
}

static const uint8_t broadcast_mac[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
__attribute__((unused))
static const uint8_t zero_mac[6] = {0,0,0,0,0,0};

/* =============================================================================
 * String helpers for HTTP parsing
 * =============================================================================*/

static int str_starts_with(const char *str, const char *prefix)
{
    while (*prefix) {
        if (*str != *prefix) return 0;
        str++; prefix++;
    }
    return 1;
}

static int str_find(const char *haystack, int haystack_len,
                    const char *needle)
{
    int nlen = 0;
    while (needle[nlen]) nlen++;
    for (int i = 0; i <= haystack_len - nlen; i++) {
        int match = 1;
        for (int j = 0; j < nlen; j++) {
            if (haystack[i + j] != needle[j]) { match = 0; break; }
        }
        if (match) return i;
    }
    return -1;
}

/* Find Content-Length in HTTP headers */
static int http_content_length(const char *headers, int len)
{
    int pos = str_find(headers, len, "Content-Length:");
    if (pos < 0) pos = str_find(headers, len, "content-length:");
    if (pos < 0) return -1;
    pos += 15; /* skip "Content-Length:" */
    while (pos < len && headers[pos] == ' ') pos++;
    int val = 0;
    while (pos < len && headers[pos] >= '0' && headers[pos] <= '9') {
        if (val > (2147483647 - 9) / 10) return -1; /* overflow → reject */
        val = val * 10 + (headers[pos] - '0');
        pos++;
    }
    return val;
}

/* Extract JSON string value for a key: "key": "value" */
static int json_get_string(const char *json, int json_len,
                           const char *key, char *out, int out_max)
{
    int klen = 0;
    while (key[klen]) klen++;

    for (int i = 0; i < json_len - klen - 4; i++) {
        if (json[i] == '"') {
            int match = 1;
            for (int j = 0; j < klen; j++) {
                if (json[i + 1 + j] != key[j]) { match = 0; break; }
            }
            if (match && json[i + 1 + klen] == '"') {
                int vi = i + 1 + klen + 1;
                while (vi < json_len && (json[vi] == ':' || json[vi] == ' '))
                    vi++;
                if (vi < json_len && json[vi] == '"') {
                    vi++;
                    int oi = 0;
                    while (vi < json_len && json[vi] != '"' && oi < out_max - 1) {
                        if (json[vi] == '\\' && vi + 1 < json_len) {
                            vi++;
                            if (json[vi] == 'n') out[oi++] = '\n';
                            else out[oi++] = json[vi];
                        } else {
                            out[oi++] = json[vi];
                        }
                        vi++;
                    }
                    out[oi] = '\0';
                    return oi;
                }
            }
        }
    }
    out[0] = '\0';
    return 0;
}

/* Extract "messages" content from chat completion request */
static int json_extract_chat_prompt(const char *json, int json_len,
                                    char *out, int out_max)
{
    int oi = 0;
    int pos = str_find(json, json_len, "\"messages\"");
    if (pos < 0) return 0;

    for (int i = pos; i < json_len - 12 && oi < out_max - 2; i++) {
        if (json[i] == 'c' && str_starts_with(json + i, "content")) {
            int ci = i + 7;
            while (ci < json_len && (json[ci] == '"' || json[ci] == ':' ||
                   json[ci] == ' ')) ci++;
            if (ci < json_len && json[ci] == '"') {
                ci++;
                if (oi > 0 && oi < out_max - 1) out[oi++] = '\n';
                while (ci < json_len && json[ci] != '"' && oi < out_max - 1) {
                    if (json[ci] == '\\' && ci + 1 < json_len) {
                        ci++;
                        if (json[ci] == 'n') out[oi++] = '\n';
                        else out[oi++] = json[ci];
                    } else {
                        out[oi++] = json[ci];
                    }
                    ci++;
                }
                i = ci;
            }
        }
    }
    out[oi] = '\0';
    return oi;
}

/* Extract integer value for JSON key */
static int json_get_int(const char *json, int json_len,
                        const char *key, int default_val)
{
    int klen = 0;
    while (key[klen]) klen++;

    for (int i = 0; i < json_len - klen - 3; i++) {
        if (json[i] == '"') {
            int match = 1;
            for (int j = 0; j < klen; j++) {
                if (json[i + 1 + j] != key[j]) { match = 0; break; }
            }
            if (match && json[i + 1 + klen] == '"') {
                int vi = i + 1 + klen + 1;
                while (vi < json_len && (json[vi] == ':' || json[vi] == ' '))
                    vi++;
                int val = 0;
                int neg = 0;
                if (vi < json_len && json[vi] == '-') { neg = 1; vi++; }
                int found = 0;
                while (vi < json_len && json[vi] >= '0' && json[vi] <= '9') {
                    if (val > (2147483647 - 9) / 10) { val = 2147483647; vi++; found = 1; continue; }
                    val = val * 10 + (json[vi] - '0');
                    vi++; found = 1;
                }
                if (found) return neg ? -val : val;
            }
        }
    }
    return default_val;
}

/* =============================================================================
 * ARP cache
 * =============================================================================*/

static void arp_cache_add(const uint8_t ip[4], const uint8_t mac[6])
{
    for (int i = 0; i < ARP_CACHE_SIZE; i++) {
        if (arp_cache[i].valid && ip_eq(arp_cache[i].ip, ip)) {
            kmemcpy(arp_cache[i].mac, mac, 6);
            return;
        }
    }
    for (int i = 0; i < ARP_CACHE_SIZE; i++) {
        if (!arp_cache[i].valid) {
            kmemcpy(arp_cache[i].ip, ip, 4);
            kmemcpy(arp_cache[i].mac, mac, 6);
            arp_cache[i].valid = 1;
            return;
        }
    }
    kmemcpy(arp_cache[0].ip, ip, 4);
    kmemcpy(arp_cache[0].mac, mac, 6);
}

static const uint8_t *arp_cache_lookup(const uint8_t ip[4])
{
    for (int i = 0; i < ARP_CACHE_SIZE; i++) {
        if (arp_cache[i].valid && ip_eq(arp_cache[i].ip, ip))
            return arp_cache[i].mac;
    }
    return NULL;
}

/* =============================================================================
 * Send Ethernet frame
 * =============================================================================*/

static int send_eth(const uint8_t dst_mac[6], uint16_t ethertype,
                    const void *payload, uint32_t payload_len)
{
    if (payload_len + 14 > sizeof(tx_frame)) return -1;

    struct eth_hdr *eth = (struct eth_hdr *)tx_frame;
    kmemcpy(eth->dst, dst_mac, 6);
    kmemcpy(eth->src, net_cfg.mac, 6);
    eth->ethertype = htons(ethertype);
    kmemcpy(tx_frame + 14, payload, payload_len);

    stat_tx_frames++;
    return virtio_net_send(tx_frame, 14 + payload_len);
}

/* =============================================================================
 * ARP handler
 * =============================================================================*/

static void handle_arp(const uint8_t *frame, uint32_t len)
{
    if (len < 14 + sizeof(struct arp_hdr)) return;
    const struct arp_hdr *arp = (const struct arp_hdr *)(frame + 14);

    uint16_t op = ntohs(arp->opcode);
    arp_cache_add(arp->sender_ip, arp->sender_mac);

    if (op == 1) {
        stat_arp_req++;
        if (!ip_eq(arp->target_ip, net_cfg.ip)) return;

        struct arp_hdr reply;
        reply.hw_type = htons(1);
        reply.proto_type = htons(0x0800);
        reply.hw_len = 6;
        reply.proto_len = 4;
        reply.opcode = htons(2);
        kmemcpy(reply.sender_mac, net_cfg.mac, 6);
        kmemcpy(reply.sender_ip, net_cfg.ip, 4);
        kmemcpy(reply.target_mac, arp->sender_mac, 6);
        kmemcpy(reply.target_ip, arp->sender_ip, 4);
        send_eth(arp->sender_mac, ETH_TYPE_ARP, &reply, sizeof(reply));
        stat_arp_rep++;
    }
    else if (op == 2) {
        stat_arp_rep++;
    }
}

/* =============================================================================
 * ICMP handler (ping reply)
 * =============================================================================*/

static void handle_icmp(const uint8_t *ip_pkt, uint32_t ip_len,
                        const struct ip_hdr *iph)
{
    uint32_t hdr_len = (iph->version_ihl & 0x0F) * 4;
    if (ip_len < hdr_len + sizeof(struct icmp_hdr)) return;

    const struct icmp_hdr *icmp = (const struct icmp_hdr *)(ip_pkt + hdr_len);
    stat_icmp_rx++;

    if (icmp->type == 8 && icmp->code == 0) {
        uint32_t icmp_total = ip_len - hdr_len;
        uint8_t reply_buf[1500];
        if (icmp_total > sizeof(reply_buf)) return;

        kmemcpy(reply_buf, icmp, icmp_total);
        struct icmp_hdr *rep = (struct icmp_hdr *)reply_buf;
        rep->type = 0;
        rep->checksum = 0;
        rep->checksum = ip_checksum(reply_buf, icmp_total);
        netstack_send_ip(iph->src, IP_PROTO_ICMP, reply_buf, icmp_total);
    }
}

/* =============================================================================
 * UDP handler
 * =============================================================================*/

static void handle_udp(const uint8_t *ip_pkt, uint32_t ip_len,
                       const struct ip_hdr *iph)
{
    uint32_t hdr_len = (iph->version_ihl & 0x0F) * 4;
    if (ip_len < hdr_len + sizeof(struct udp_hdr)) return;

    const struct udp_hdr *udp = (const struct udp_hdr *)(ip_pkt + hdr_len);
    uint16_t dst_port = ntohs(udp->dst_port);
    uint16_t src_port = ntohs(udp->src_port);
    uint32_t udp_data_len = ntohs(udp->length) - sizeof(struct udp_hdr);
    const uint8_t *udp_data = (const uint8_t *)udp + sizeof(struct udp_hdr);

    stat_udp_rx++;

    for (int i = 0; i < n_udp_handlers; i++) {
        if (udp_handlers[i].port == dst_port && udp_handlers[i].handler) {
            udp_handlers[i].handler(iph->src, src_port, udp_data, udp_data_len);
            return;
        }
    }
}

/* =============================================================================
 * TCP Implementation
 * =============================================================================*/

static tcp_conn_t *tcp_find_conn(const uint8_t ip[4], uint16_t local_port,
                                 uint16_t remote_port)
{
    for (int i = 0; i < TCP_MAX_CONNS; i++) {
        if (tcp_conns[i].state != TCP_STATE_CLOSED &&
            ip_eq(tcp_conns[i].remote_ip, ip) &&
            tcp_conns[i].local_port == local_port &&
            tcp_conns[i].remote_port == remote_port)
            return &tcp_conns[i];
    }
    return NULL;
}

static tcp_conn_t *tcp_alloc_conn(void)
{
    for (int i = 0; i < TCP_MAX_CONNS; i++) {
        if (tcp_conns[i].state == TCP_STATE_CLOSED)
            return &tcp_conns[i];
    }
    /* Reclaim TIME_WAIT connections */
    for (int i = 0; i < TCP_MAX_CONNS; i++) {
        if (tcp_conns[i].state == TCP_STATE_TIME_WAIT) {
            kmemset(&tcp_conns[i], 0, sizeof(tcp_conn_t));
            return &tcp_conns[i];
        }
    }
    return NULL;
}

/* Send a TCP segment with given flags and optional data */
int netstack_tcp_send(tcp_conn_t *conn, uint8_t flags,
                      const void *data, uint32_t len)
{
    uint8_t pkt[1500];
    uint32_t tcp_hdr_len = 20;

    if (len + tcp_hdr_len > 1460) len = 1460;

    struct tcp_hdr *tcp = (struct tcp_hdr *)pkt;
    tcp->src_port = htons(conn->local_port);
    tcp->dst_port = htons(conn->remote_port);
    tcp->seq = htonl(conn->snd_nxt);
    tcp->ack = htonl(conn->rcv_nxt);
    tcp->data_off = (uint8_t)((tcp_hdr_len / 4) << 4);
    tcp->flags = flags;
    tcp->window = htons((uint16_t)(TCP_RX_BUF_SIZE - conn->rx_len));
    tcp->checksum = 0;
    tcp->urgent = 0;

    if (data && len > 0)
        kmemcpy(pkt + tcp_hdr_len, data, len);

    tcp->checksum = tcp_checksum(net_cfg.ip, conn->remote_ip,
                                 pkt, tcp_hdr_len + len);

    conn->snd_nxt += len;
    if (flags & (TCP_SYN | TCP_FIN)) conn->snd_nxt++;

    return netstack_send_ip(conn->remote_ip, IP_PROTO_TCP,
                            pkt, tcp_hdr_len + len);
}

/* Send a TCP RST to reject a connection */
static void tcp_send_rst(const uint8_t *ip_pkt, const struct ip_hdr *iph,
                         const struct tcp_hdr *in_tcp)
{
    uint8_t pkt[40];
    struct tcp_hdr *tcp = (struct tcp_hdr *)pkt;
    tcp->src_port = in_tcp->dst_port;
    tcp->dst_port = in_tcp->src_port;
    tcp->seq = in_tcp->ack;
    tcp->ack = htonl(ntohl(in_tcp->seq) + 1);
    tcp->data_off = (20 / 4) << 4;
    tcp->flags = TCP_RST | TCP_ACK;
    tcp->window = 0;
    tcp->checksum = 0;
    tcp->urgent = 0;
    tcp->checksum = tcp_checksum(net_cfg.ip, iph->src, pkt, 20);
    netstack_send_ip(iph->src, IP_PROTO_TCP, pkt, 20);
}

/* Forward declaration */
static void http_handle_request(tcp_conn_t *conn);

static void handle_tcp(const uint8_t *ip_pkt, uint32_t ip_len,
                       const struct ip_hdr *iph)
{
    uint32_t hdr_len = (iph->version_ihl & 0x0F) * 4;
    if (ip_len < hdr_len + sizeof(struct tcp_hdr)) return;

    const struct tcp_hdr *tcp = (const struct tcp_hdr *)(ip_pkt + hdr_len);
    uint16_t dst_port = ntohs(tcp->dst_port);
    uint16_t src_port = ntohs(tcp->src_port);
    uint32_t tcp_hdr_len = ((uint32_t)(tcp->data_off >> 4)) * 4;
    uint32_t payload_len = ip_len - hdr_len - tcp_hdr_len;
    const uint8_t *payload = ip_pkt + hdr_len + tcp_hdr_len;
    uint32_t seq = ntohl(tcp->seq);
    uint32_t ack = ntohl(tcp->ack);

    stat_tcp_rx++;

    /* Find existing connection */
    tcp_conn_t *conn = tcp_find_conn(iph->src, dst_port, src_port);

    /* New SYN on listen port? */
    if (!conn && (tcp->flags & TCP_SYN) && !(tcp->flags & TCP_ACK)) {
        if ((dst_port != tcp_listen_port && dst_port != net_cfg.https_port) ||
            !net_cfg.server_running) {
            tcp_send_rst(ip_pkt, iph, tcp);
            return;
        }

        conn = tcp_alloc_conn();
        if (!conn) {
            tcp_send_rst(ip_pkt, iph, tcp);
            return;
        }

        /* Initialize connection */
        kmemcpy(conn->remote_ip, iph->src, 4);
        conn->local_port = dst_port;
        conn->remote_port = src_port;
        conn->rcv_nxt = seq + 1;
        conn->snd_nxt = isn_hash(net_cfg.ip, dst_port, iph->src, src_port,
                                 tcp_isn_counter);
        tcp_isn_counter += 64000;
        conn->snd_una = conn->snd_nxt;
        conn->remote_win = ntohs(tcp->window);
        conn->rx_len = 0;
        conn->tx_len = 0;
        conn->http_request_complete = 0;
        conn->rto_ms = 1000;       /* RFC 6298: initial RTO = 1s */
        conn->srtt = 0;
        conn->rttvar = 0;
        conn->last_send_tick = watchdog_ticks;
        conn->retransmits = 0;
        conn->dup_acks = 0;
        conn->cwnd = 1460;         /* IW = 1 MSS */
        conn->ssthresh = 65535;
        conn->rtx_len = 0;
        conn->tls_session = NULL;
        conn->state = TCP_STATE_SYN_RCVD;

        /* Allocate TLS session for HTTPS connections */
        if (dst_port == net_cfg.https_port) {
            conn->tls_session = tls_session_new();
        }

        /* Send SYN+ACK */
        netstack_tcp_send(conn, TCP_SYN | TCP_ACK, NULL, 0);
        return;
    }

    if (!conn) {
        if (!(tcp->flags & TCP_RST))
            tcp_send_rst(ip_pkt, iph, tcp);
        return;
    }

    /* Handle RST */
    if (tcp->flags & TCP_RST) {
        conn->state = TCP_STATE_CLOSED;
        return;
    }

    /* State machine */
    switch (conn->state) {
    case TCP_STATE_SYN_RCVD:
        if (tcp->flags & TCP_ACK) {
            conn->snd_una = ack;
            conn->state = TCP_STATE_ESTABLISHED;
        }
        break;

    case TCP_STATE_ESTABLISHED:
        if (tcp->flags & TCP_ACK) {
            if (ack > conn->snd_una) {
                /* New ACK — update RTT estimate (Jacobson/Karn, RFC 6298) */
                if (conn->last_send_tick && conn->retransmits == 0) {
                    uint32_t rtt_ms = (uint32_t)((watchdog_ticks - conn->last_send_tick) * 10);
                    if (rtt_ms == 0) rtt_ms = 1;
                    if (conn->srtt == 0) {
                        /* First measurement */
                        conn->srtt = rtt_ms << 3;       /* srtt = R × 8 */
                        conn->rttvar = (rtt_ms >> 1) << 2; /* rttvar = R/2 × 4 */
                    } else {
                        /* EWMA update */
                        int32_t delta = (int32_t)rtt_ms - (int32_t)(conn->srtt >> 3);
                        int32_t abs_delta = delta < 0 ? -delta : delta;
                        conn->rttvar = conn->rttvar - (conn->rttvar >> 2) + (uint32_t)abs_delta;
                        conn->srtt = conn->srtt - (conn->srtt >> 3) + rtt_ms;
                    }
                    conn->rto_ms = (conn->srtt >> 3) + (conn->rttvar >> 2) * 4;
                    if (conn->rto_ms < 200) conn->rto_ms = 200;   /* Floor */
                    if (conn->rto_ms > 60000) conn->rto_ms = 60000; /* Cap 60s */
                }
                conn->snd_una = ack;
                conn->retransmits = 0;
                conn->dup_acks = 0;
                conn->rtx_len = 0;  /* Acked, clear retransmit buffer */

                /* Congestion control: open window */
                if (conn->cwnd < conn->ssthresh) {
                    /* Slow start: +1 MSS per ACK */
                    conn->cwnd += 1460;
                } else {
                    /* Congestion avoidance: +1 MSS per RTT */
                    conn->cwnd += (1460 * 1460) / conn->cwnd;
                }
            } else if (ack == conn->snd_una && conn->rtx_len > 0) {
                /* Duplicate ACK */
                conn->dup_acks++;
                if (conn->dup_acks == 3) {
                    /* Fast retransmit (RFC 5681) */
                    conn->ssthresh = conn->cwnd / 2;
                    if (conn->ssthresh < 2920) conn->ssthresh = 2920;
                    conn->cwnd = conn->ssthresh + 3 * 1460;

                    /* Retransmit saved segment directly */
                    uint32_t saved_nxt = conn->snd_nxt;
                    conn->snd_nxt = conn->rtx_seq;
                    netstack_tcp_send(conn, TCP_ACK | TCP_PSH,
                                     conn->rtx_buf, conn->rtx_len);
                    conn->snd_nxt = saved_nxt;
                    conn->last_send_tick = watchdog_ticks;
                }
            }
        }

        /* Process incoming data */
        if (payload_len > 0 && seq == conn->rcv_nxt) {
            uint32_t space = TCP_RX_BUF_SIZE - conn->rx_len;
            uint32_t copy = payload_len < space ? payload_len : space;
            if (copy > 0) {
                kmemcpy(conn->rx_buf + conn->rx_len, payload, copy);
                conn->rx_len += copy;
                conn->rcv_nxt += copy;
            }

            /* Send ACK */
            netstack_tcp_send(conn, TCP_ACK, NULL, 0);

            /* TLS connection? Route through TLS processing */
            if (conn->tls_session) {
                tls_session_t *tls = (tls_session_t *)conn->tls_session;
                /* Process all complete TLS records in the buffer */
                while (conn->rx_len >= 5) {
                    uint16_t rec_len = ((uint16_t)conn->rx_buf[3] << 8) | conn->rx_buf[4];
                    uint32_t total = 5 + rec_len;
                    if (total > conn->rx_len) break; /* Incomplete record */

                    int ret = tls_process_record(tls, conn, conn->rx_buf, total);

                    /* Shift remaining data */
                    if (total < conn->rx_len)
                        for (uint32_t i = 0; i < conn->rx_len - total; i++)
                            conn->rx_buf[i] = conn->rx_buf[i + total];
                    conn->rx_len -= total;

                    if (ret > 0) {
                        /* Decrypted HTTP data in tls->plaintext_buf */
                        /* Copy to rx_buf for HTTP parsing */
                        uint32_t pt_len = tls->plaintext_len;
                        if (pt_len <= TCP_RX_BUF_SIZE) {
                            kmemcpy(conn->rx_buf, tls->plaintext_buf, pt_len);
                            conn->rx_len = pt_len;
                            conn->http_request_complete = 1;
                            conn->tls_session = tls; /* Keep for response */
                            http_handle_request(conn);
                            return; /* connection handled */
                        }
                    } else if (ret < 0) {
                        tls_session_free(tls);
                        conn->tls_session = NULL;
                        conn->state = TCP_STATE_CLOSED;
                        return;
                    }
                }
            } else {
                /* Plaintext HTTP processing */
                /* Check if we have a complete HTTP request */
                if (!conn->http_request_complete) {
                    int hdr_end_pos = str_find((const char *)conn->rx_buf,
                                           (int)conn->rx_len, "\r\n\r\n");
                    if (hdr_end_pos >= 0) {
                        int body_start = hdr_end_pos + 4;
                        int clen = http_content_length(
                            (const char *)conn->rx_buf, hdr_end_pos);
                        if (clen <= 0 ||
                            (int)(conn->rx_len - body_start) >= clen) {
                            conn->http_request_complete = 1;
                        }
                    }
                }

                /* Process complete HTTP request */
                if (conn->http_request_complete) {
                    http_handle_request(conn);
                }
            }
        }

        /* Handle FIN */
        if (tcp->flags & TCP_FIN) {
            conn->rcv_nxt++;
            netstack_tcp_send(conn, TCP_ACK, NULL, 0);
            conn->state = TCP_STATE_CLOSE_WAIT;
            netstack_tcp_send(conn, TCP_FIN | TCP_ACK, NULL, 0);
            conn->state = TCP_STATE_LAST_ACK;
        }
        break;

    case TCP_STATE_FIN_WAIT_1:
        if (tcp->flags & TCP_ACK) {
            conn->snd_una = ack;
            if (tcp->flags & TCP_FIN) {
                conn->rcv_nxt++;
                netstack_tcp_send(conn, TCP_ACK, NULL, 0);
                conn->state = TCP_STATE_TIME_WAIT;
            } else {
                conn->state = TCP_STATE_FIN_WAIT_2;
            }
        }
        break;

    case TCP_STATE_FIN_WAIT_2:
        if (tcp->flags & TCP_FIN) {
            conn->rcv_nxt++;
            netstack_tcp_send(conn, TCP_ACK, NULL, 0);
            conn->state = TCP_STATE_TIME_WAIT;
        }
        break;

    case TCP_STATE_LAST_ACK:
        if (tcp->flags & TCP_ACK) {
            conn->state = TCP_STATE_CLOSED;
        }
        break;

    case TCP_STATE_TIME_WAIT:
        conn->state = TCP_STATE_CLOSED;
        break;

    default:
        break;
    }
}

/* Write data to TCP connection and transmit */
int tcp_conn_write(tcp_conn_t *conn, const void *data, uint32_t len)
{
    if (!conn || conn->state != TCP_STATE_ESTABLISHED) return -1;

    const uint8_t *p = (const uint8_t *)data;
    uint32_t sent = 0;

    while (sent < len) {
        uint32_t chunk = len - sent;
        if (chunk > 1400) chunk = 1400;

        /* Save last segment for potential retransmit */
        uint32_t pre_seq = conn->snd_nxt;
        netstack_tcp_send(conn, TCP_ACK | TCP_PSH, p + sent, chunk);
        if (chunk <= sizeof(conn->rtx_buf)) {
            kmemcpy(conn->rtx_buf, p + sent, chunk);
            conn->rtx_len = chunk;
            conn->rtx_seq = pre_seq;
        }
        conn->last_send_tick = watchdog_ticks;
        conn->retransmits = 0;

        sent += chunk;
    }
    return (int)sent;
}

/* Close TCP connection */
void tcp_conn_close(tcp_conn_t *conn)
{
    if (!conn) return;
    /* Free TLS session if present */
    if (conn->tls_session) {
        tls_session_free((tls_session_t *)conn->tls_session);
        conn->tls_session = NULL;
    }
    if (conn->state == TCP_STATE_ESTABLISHED) {
        netstack_tcp_send(conn, TCP_FIN | TCP_ACK, NULL, 0);
        conn->state = TCP_STATE_FIN_WAIT_1;
    } else {
        conn->state = TCP_STATE_CLOSED;
    }
}

/* =============================================================================
 * IP handler (dispatch to ICMP, UDP, TCP)
 * =============================================================================*/

static void handle_ip(const uint8_t *frame, uint32_t len)
{
    if (len < 14 + sizeof(struct ip_hdr)) return;
    const struct ip_hdr *iph = (const struct ip_hdr *)(frame + 14);

    if ((iph->version_ihl >> 4) != 4) return;

    uint8_t bcast[4] = {255,255,255,255};
    if (!ip_eq(iph->dst, net_cfg.ip) && !ip_eq(iph->dst, bcast)) return;

    uint32_t ip_total = ntohs(iph->total_len);
    const uint8_t *ip_pkt = frame + 14;
    stat_ip_rx++;

    switch (iph->proto) {
    case IP_PROTO_ICMP:
        handle_icmp(ip_pkt, ip_total, iph);
        break;
    case IP_PROTO_UDP:
        handle_udp(ip_pkt, ip_total, iph);
        break;
    case IP_PROTO_TCP:
        handle_tcp(ip_pkt, ip_total, iph);
        break;
    default:
        break;
    }
}

/* =============================================================================
 * OpenAI-compatible HTTP API Server
 *
 * Endpoints:
 *   GET  /v1/models            — List available models
 *   POST /v1/completions       — Text completion (OpenAI compat)
 *   POST /v1/chat/completions  — Chat completion (OpenAI compat)
 *   GET  /health               — Health check
 *   GET  /                     — Quick-start guide
 *
 * CORS headers included for browser clients.
 * =============================================================================*/

/* HTTP response buffers — serialized by http_busy lock */
static volatile int http_busy = 0;
static char http_resp[32768];
static char llm_output_buf[8192];

/* =============================================================================
 * Inference Request Queue
 * Queues requests when inference is in-flight so they aren't rejected with 503.
 * Processed in FIFO order by netstack_poll().
 * =============================================================================*/

#define INFER_QUEUE_SIZE 8

typedef struct {
    tcp_conn_t *conn;
    int         is_chat;         /* 1=chat/completions, 0=completions */
    char        body[4096];
    int         body_len;
    void       *tls_session;     /* Saved TLS session for responses */
} infer_request_t;

static infer_request_t infer_queue[INFER_QUEUE_SIZE];
static volatile int infer_queue_head = 0;  /* Next slot to dequeue */
static volatile int infer_queue_tail = 0;  /* Next slot to enqueue */

static int infer_queue_enqueue(tcp_conn_t *conn, int is_chat,
                                const char *body, int body_len)
{
    int next = (infer_queue_tail + 1) % INFER_QUEUE_SIZE;
    if (next == infer_queue_head) return -1; /* Full */

    infer_request_t *r = &infer_queue[infer_queue_tail];
    r->conn = conn;
    r->is_chat = is_chat;
    int copy = body_len < (int)sizeof(r->body) - 1 ? body_len : (int)sizeof(r->body) - 1;
    kmemcpy(r->body, body, copy);
    r->body[copy] = '\0';
    r->body_len = copy;
    r->tls_session = conn->tls_session;
    infer_queue_tail = next;
    return 0;
}

static void http_send_response(tcp_conn_t *conn, int status,
                               const char *status_text,
                               const char *content_type,
                               const char *body, int body_len)
{
    int pos = 0;
    pos += kprintf_to_buf(http_resp + pos, (int)sizeof(http_resp) - pos,
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Server: TensorOS/0.1.0\r\n"
        "\r\n",
        status, status_text, content_type, body_len);

    if (conn->tls_session) {
        tls_session_t *tls = (tls_session_t *)conn->tls_session;
        tls_send(tls, conn, http_resp, pos);
        if (body && body_len > 0)
            tls_send(tls, conn, body, body_len);
    } else {
        tcp_conn_write(conn, http_resp, pos);
        if (body && body_len > 0)
            tcp_conn_write(conn, body, body_len);
    }
}

static void http_send_json(tcp_conn_t *conn, int status,
                           const char *status_text,
                           const char *json, int json_len)
{
    http_send_response(conn, status, status_text,
                       "application/json", json, json_len);
}

/* Escape string for JSON */
static int json_escape(char *out, int out_max, const char *in, int in_len)
{
    int oi = 0;
    for (int i = 0; i < in_len && oi < out_max - 2; i++) {
        switch (in[i]) {
        case '"':  if (oi + 2 < out_max) { out[oi++] = '\\'; out[oi++] = '"'; } break;
        case '\\': if (oi + 2 < out_max) { out[oi++] = '\\'; out[oi++] = '\\'; } break;
        case '\n': if (oi + 2 < out_max) { out[oi++] = '\\'; out[oi++] = 'n'; } break;
        case '\r': if (oi + 2 < out_max) { out[oi++] = '\\'; out[oi++] = 'r'; } break;
        case '\t': if (oi + 2 < out_max) { out[oi++] = '\\'; out[oi++] = 't'; } break;
        default:
            if ((unsigned char)in[i] >= 0x20) out[oi++] = in[i];
            break;
        }
    }
    out[oi] = '\0';
    return oi;
}

/* GET /v1/models */
static void api_list_models(tcp_conn_t *conn)
{
    char body[2048];
    int pos = 0;

    if (llm_is_loaded()) {
        const char *name = llm_model_name();
        char escaped_name[256];
        int nlen = 0;
        while (name[nlen]) nlen++;
        json_escape(escaped_name, (int)sizeof(escaped_name), name, nlen);

        pos += kprintf_to_buf(body + pos, (int)sizeof(body) - pos,
            "{\"object\":\"list\",\"data\":[{"
            "\"id\":\"%s\","
            "\"object\":\"model\","
            "\"created\":1709500800,"
            "\"owned_by\":\"tensoros\","
            "\"permission\":[],"
            "\"root\":\"%s\","
            "\"parent\":null"
            "}]}",
            escaped_name, escaped_name);
    } else {
        pos += kprintf_to_buf(body + pos, (int)sizeof(body) - pos,
            "{\"object\":\"list\",\"data\":[]}");
    }

    http_send_json(conn, 200, "OK", body, pos);
}

/* GET /health */
static void api_health(tcp_conn_t *conn)
{
    char body[512];
    int pos = kprintf_to_buf(body, (int)sizeof(body),
        "{\"status\":\"ok\",\"model_loaded\":%s,"
        "\"model\":\"%s\","
        "\"uptime_ms\":%u,"
        "\"cpus\":%u}",
        llm_is_loaded() ? "true" : "false",
        llm_is_loaded() ? llm_model_name() : "none",
        watchdog_uptime_ms(),
        kstate.cpu_count);
    http_send_json(conn, 200, "OK", body, pos);
}

/* POST /v1/completions */
static void api_completions(tcp_conn_t *conn, const char *body_json, int body_len)
{
    stat_http_infer++;

    if (!llm_is_loaded()) {
        const char *err = "{\"error\":{\"message\":\"No model loaded. "
                          "Attach a GGUF model file to load.\","
                          "\"type\":\"model_error\",\"code\":\"model_not_loaded\"}}";
        int elen = 0; while (err[elen]) elen++;
        http_send_json(conn, 503, "Service Unavailable", err, elen);
        return;
    }

    char prompt[4096];
    int plen = json_get_string(body_json, body_len, "prompt", prompt, (int)sizeof(prompt));
    if (plen == 0) {
        const char *err = "{\"error\":{\"message\":\"Missing 'prompt' field\","
                          "\"type\":\"invalid_request\",\"code\":\"missing_prompt\"}}";
        int elen = 0; while (err[elen]) elen++;
        http_send_json(conn, 400, "Bad Request", err, elen);
        return;
    }

    int max_tokens = json_get_int(body_json, body_len, "max_tokens", 128);
    if (max_tokens > 2048) max_tokens = 2048;
    if (max_tokens < 1) max_tokens = 128;

    kprintf("[API] Completion: prompt=%d chars, max_tokens=%d\n", plen, max_tokens);

    llm_reset_cache();

    uint64_t start = rdtsc_fenced();
    int ntokens = llm_prompt(prompt, llm_output_buf, (int)sizeof(llm_output_buf));
    uint64_t elapsed = rdtsc_fenced() - start;
    uint32_t ms = (uint32_t)(perf_cycles_to_us(elapsed) / 1000);

    if (ntokens < 0) ntokens = 0;
    int olen = 0;
    while (llm_output_buf[olen]) olen++;

    char escaped_output[16384];
    int escaped_len = json_escape(escaped_output, (int)sizeof(escaped_output),
                                  llm_output_buf, olen);

    char resp[24576];
    int rpos = kprintf_to_buf(resp, (int)sizeof(resp),
        "{\"id\":\"cmpl-tensoros-%u\","
        "\"object\":\"text_completion\","
        "\"created\":%u,"
        "\"model\":\"%s\","
        "\"choices\":[{"
        "\"text\":\"%s\","
        "\"index\":0,"
        "\"logprobs\":null,"
        "\"finish_reason\":\"%s\""
        "}],"
        "\"usage\":{"
        "\"prompt_tokens\":%d,"
        "\"completion_tokens\":%d,"
        "\"total_tokens\":%d"
        "},"
        "\"timing\":{"
        "\"total_ms\":%u,"
        "\"tokens_per_sec\":%u"
        "}}",
        (uint32_t)(stat_http_infer & 0xFFFFFFFF),
        watchdog_uptime_ms() / 1000,
        llm_model_name(),
        escaped_output,
        ntokens > 0 ? "stop" : "length",
        plen > 0 ? 1 : 0,
        ntokens,
        (plen > 0 ? 1 : 0) + ntokens,
        ms,
        ms > 0 ? (uint32_t)((uint64_t)ntokens * 1000 / ms) : 0);

    kprintf("[API] Generated %d tokens in %u ms\n", ntokens, ms);
    http_send_json(conn, 200, "OK", resp, rpos);
    (void)escaped_len;
}

/* POST /v1/chat/completions */
static void api_chat_completions(tcp_conn_t *conn, const char *body_json, int body_len)
{
    stat_http_infer++;

    if (!llm_is_loaded()) {
        const char *err = "{\"error\":{\"message\":\"No model loaded.\","
                          "\"type\":\"model_error\",\"code\":\"model_not_loaded\"}}";
        int elen = 0; while (err[elen]) elen++;
        http_send_json(conn, 503, "Service Unavailable", err, elen);
        return;
    }

    char prompt[4096];
    int plen = json_extract_chat_prompt(body_json, body_len, prompt, (int)sizeof(prompt));
    if (plen == 0) {
        plen = json_get_string(body_json, body_len, "prompt", prompt, (int)sizeof(prompt));
    }
    if (plen == 0) {
        const char *err = "{\"error\":{\"message\":\"Missing 'messages' array\","
                          "\"type\":\"invalid_request\",\"code\":\"missing_messages\"}}";
        int elen = 0; while (err[elen]) elen++;
        http_send_json(conn, 400, "Bad Request", err, elen);
        return;
    }

    int max_tokens = json_get_int(body_json, body_len, "max_tokens", 128);
    if (max_tokens > 2048) max_tokens = 2048;
    if (max_tokens < 1) max_tokens = 128;

    kprintf("[API] Chat: prompt=%d chars, max_tokens=%d\n", plen, max_tokens);

    llm_reset_cache();

    uint64_t start = rdtsc_fenced();
    int ntokens = llm_prompt(prompt, llm_output_buf, (int)sizeof(llm_output_buf));
    uint64_t elapsed = rdtsc_fenced() - start;
    uint32_t ms = (uint32_t)(perf_cycles_to_us(elapsed) / 1000);

    if (ntokens < 0) ntokens = 0;
    int olen = 0;
    while (llm_output_buf[olen]) olen++;

    char escaped_output[16384];
    int escaped_len = json_escape(escaped_output, (int)sizeof(escaped_output),
                                  llm_output_buf, olen);

    char resp[24576];
    int rpos = kprintf_to_buf(resp, (int)sizeof(resp),
        "{\"id\":\"chatcmpl-tensoros-%u\","
        "\"object\":\"chat.completion\","
        "\"created\":%u,"
        "\"model\":\"%s\","
        "\"choices\":[{"
        "\"index\":0,"
        "\"message\":{"
        "\"role\":\"assistant\","
        "\"content\":\"%s\""
        "},"
        "\"finish_reason\":\"%s\""
        "}],"
        "\"usage\":{"
        "\"prompt_tokens\":%d,"
        "\"completion_tokens\":%d,"
        "\"total_tokens\":%d"
        "},"
        "\"timing\":{"
        "\"total_ms\":%u,"
        "\"tokens_per_sec\":%u"
        "}}",
        (uint32_t)(stat_http_infer & 0xFFFFFFFF),
        watchdog_uptime_ms() / 1000,
        llm_model_name(),
        escaped_output,
        ntokens > 0 ? "stop" : "length",
        plen > 0 ? 1 : 0,
        ntokens,
        (plen > 0 ? 1 : 0) + ntokens,
        ms,
        ms > 0 ? (uint32_t)((uint64_t)ntokens * 1000 / ms) : 0);

    kprintf("[API] Chat generated %d tokens in %u ms\n", ntokens, ms);
    http_send_json(conn, 200, "OK", resp, rpos);
    (void)escaped_len;
}

/* GET / — Welcome & quick-start guide */
static void api_welcome(tcp_conn_t *conn)
{
    char body[4096];
    int pos = kprintf_to_buf(body, (int)sizeof(body),
        "{\"name\":\"TensorOS LLM API\","
        "\"version\":\"0.1.0\","
        "\"description\":\"OpenAI-compatible bare-metal inference server\","
        "\"model\":\"%s\","
        "\"endpoints\":{"
        "\"/v1/models\":\"GET - List loaded models\","
        "\"/v1/completions\":\"POST - Text completion\","
        "\"/v1/chat/completions\":\"POST - Chat completion (OpenAI format)\","
        "\"/health\":\"GET - Health check\""
        "},"
        "\"quickstart\":{"
        "\"curl\":\"curl http://%u.%u.%u.%u:%u/v1/models\","
        "\"python\":\"from openai import OpenAI; "
        "c = OpenAI(base_url='http://%u.%u.%u.%u:%u/v1', api_key='tensoros'); "
        "r = c.chat.completions.create(model='default', "
        "messages=[{'role':'user','content':'Hello'}])\""
        "}}",
        llm_is_loaded() ? llm_model_name() : "none",
        net_cfg.ip[0], net_cfg.ip[1], net_cfg.ip[2], net_cfg.ip[3],
        net_cfg.http_port,
        net_cfg.ip[0], net_cfg.ip[1], net_cfg.ip[2], net_cfg.ip[3],
        net_cfg.http_port);

    http_send_json(conn, 200, "OK", body, pos);
}

/* CORS preflight */
static void api_options(tcp_conn_t *conn)
{
    http_send_response(conn, 204, "No Content", "text/plain", NULL, 0);
}

/* 404 */
static void api_not_found(tcp_conn_t *conn)
{
    char body[512];
    int pos = kprintf_to_buf(body, (int)sizeof(body),
        "{\"error\":{\"message\":\"Not found. Available: "
        "/v1/models, /v1/completions, /v1/chat/completions, /health\","
        "\"type\":\"not_found\",\"code\":404}}");
    http_send_json(conn, 404, "Not Found", body, pos);
}

/* Parse and route HTTP request */
static void http_handle_request(tcp_conn_t *conn)
{
    /* Serialize HTTP handling — static response buffers not reentrant */
    if (__sync_lock_test_and_set(&http_busy, 1)) {
        /* Busy — try to queue inference requests instead of rejecting */
        const char *rq = (const char *)conn->rx_buf;
        int rqlen = (int)conn->rx_len;
        int is_infer = 0, is_chat = 0;
        if (rqlen > 20 && rq[0] == 'P') { /* POST */
            if (str_find(rq, rqlen > 128 ? 128 : rqlen, "/v1/chat/completions") >= 0)
                { is_infer = 1; is_chat = 1; }
            else if (str_find(rq, rqlen > 128 ? 128 : rqlen, "/v1/completions") >= 0)
                { is_infer = 1; is_chat = 0; }
        }
        if (is_infer) {
            int hdr = str_find(rq, rqlen, "\r\n\r\n");
            const char *bd = (hdr >= 0) ? rq + hdr + 4 : NULL;
            int bdlen = (hdr >= 0) ? rqlen - (hdr + 4) : 0;
            if (bd && bdlen > 0 && infer_queue_enqueue(conn, is_chat, bd, bdlen) == 0)
                return; /* queued — will be processed by netstack_poll */
        }
        const char *busy = "HTTP/1.1 503 Service Unavailable\r\n"
            "Content-Length: 4\r\nConnection: close\r\n\r\nbusy";
        netstack_tcp_send(conn, TCP_ACK | TCP_PSH, busy, 81);
        tcp_conn_close(conn);
        return;
    }
    stat_http_req++;
    const char *req = (const char *)conn->rx_buf;
    int req_len = (int)conn->rx_len;

    /* Parse method and path */
    int method_end = 0;
    while (method_end < req_len && req[method_end] != ' ') method_end++;
    int path_start = method_end + 1;
    int path_end = path_start;
    while (path_end < req_len && req[path_end] != ' ' && req[path_end] != '?')
        path_end++;
    int path_len = path_end - path_start;

    /* Find body (after \r\n\r\n) */
    int hdr_end = str_find(req, req_len, "\r\n\r\n");
    const char *body = NULL;
    int body_len = 0;
    if (hdr_end >= 0) {
        body = req + hdr_end + 4;
        body_len = req_len - (hdr_end + 4);
    }

    int is_get = (method_end == 3 &&
                  req[0] == 'G' && req[1] == 'E' && req[2] == 'T');
    int is_post = (method_end == 4 &&
                   req[0] == 'P' && req[1] == 'O' && req[2] == 'S' &&
                   req[3] == 'T');
    int is_options = (method_end >= 7 &&
                      req[0] == 'O' && req[1] == 'P' && req[2] == 'T');

    kprintf("[HTTP] %.*s %.*s (%d bytes body)\n",
            method_end, req, path_len, req + path_start, body_len);

    /* API key authentication — skip for OPTIONS, GET /, GET /health */
    int needs_auth = api_key[0] && !is_options;
    if (needs_auth && is_get &&
        (path_len == 1 || (path_len >= 7 && str_starts_with(req + path_start, "/health"))))
        needs_auth = 0;
    if (needs_auth) {
        /* Look for "Authorization: Bearer <key>" header */
        int auth_pos = str_find(req, hdr_end >= 0 ? hdr_end : req_len, "Authorization: Bearer ");
        int auth_ok = 0;
        if (auth_pos >= 0) {
            const char *tok = req + auth_pos + 22; /* skip "Authorization: Bearer " */
            int tok_len = 0;
            while (tok[tok_len] && tok[tok_len] != '\r' && tok[tok_len] != '\n') tok_len++;
            int key_len = kstrlen(api_key);
            if (tok_len == key_len) {
                auth_ok = (crypto_ct_equal(tok, api_key, key_len) == 0) ? 1 : 0;
            }
        }
        if (!auth_ok) {
            char err[256];
            int epos = kprintf_to_buf(err, (int)sizeof(err),
                "{\"error\":{\"message\":\"Invalid API key\",\"type\":\"authentication_error\",\"code\":401}}");
            http_send_json(conn, 401, "Unauthorized", err, epos);
            tcp_conn_close(conn);
            return;
        }
    }

    /* Route requests */
    if (is_options) {
        api_options(conn);
    }
    else if (is_get && path_len == 1 && req[path_start] == '/') {
        api_welcome(conn);
    }
    else if (is_get && path_len >= 7 &&
             str_starts_with(req + path_start, "/health")) {
        api_health(conn);
    }
    else if (is_get && path_len >= 10 &&
             str_starts_with(req + path_start, "/v1/models")) {
        api_list_models(conn);
    }
    else if (is_post && path_len >= 20 &&
             str_starts_with(req + path_start, "/v1/chat/completions")) {
        api_chat_completions(conn, body, body_len);
    }
    else if (is_post && path_len >= 15 &&
             str_starts_with(req + path_start, "/v1/completions")) {
        api_completions(conn, body, body_len);
    }
    else {
        api_not_found(conn);
    }

    /* Close connection after response (HTTP/1.0 style) */
    tcp_conn_close(conn);
    (void)is_post;
    __sync_lock_release(&http_busy);
}

/* =============================================================================
 * Legacy UDP handler (backward compat)
 * =============================================================================*/

static void inference_udp_handler(const uint8_t src_ip[4], uint16_t src_port,
                                  const uint8_t *data, uint32_t len)
{
    stat_http_req++;

    if (len >= 4 && data[0] == 'P' && data[1] == 'I' &&
        data[2] == 'N' && data[3] == 'G') {
        const char *resp = "PONG TensorOS v0.1.0 Neuron\n";
        netstack_send_udp(src_ip, net_cfg.http_port, src_port,
                          resp, kstrlen(resp));
        return;
    }

    if (len >= 4 && data[0] == 'I' && data[1] == 'N' &&
        data[2] == 'F' && data[3] == 'O') {
        char info[512];
        int pos = kprintf_to_buf(info, (int)sizeof(info),
            "{\"os\":\"TensorOS\",\"version\":\"0.1.0\","
            "\"model\":\"%s\","
            "\"api\":\"http://%u.%u.%u.%u:%u/v1\"}\n",
            llm_is_loaded() ? llm_model_name() : "none",
            net_cfg.ip[0], net_cfg.ip[1], net_cfg.ip[2], net_cfg.ip[3],
            net_cfg.http_port);
        netstack_send_udp(src_ip, net_cfg.http_port, src_port, info, pos);
        return;
    }

    if (len >= 5 && data[0] == 'I' && data[1] == 'N' && data[2] == 'F' &&
        data[3] == 'E' && data[4] == 'R') {
        if (!llm_is_loaded()) {
            const char *resp = "ERR no model loaded\n";
            netstack_send_udp(src_ip, net_cfg.http_port, src_port,
                              resp, kstrlen(resp));
            return;
        }
        const char *prompt_data = (const char *)data + 6;
        int plen = (int)len - 6;
        if (plen <= 0) {
            const char *resp = "ERR empty prompt\n";
            netstack_send_udp(src_ip, net_cfg.http_port, src_port,
                              resp, kstrlen(resp));
            return;
        }
        char pbuf[2048];
        int lim = plen < (int)sizeof(pbuf) - 1 ? plen : (int)sizeof(pbuf) - 1;
        kmemcpy(pbuf, prompt_data, lim);
        pbuf[lim] = '\0';

        llm_reset_cache();
        int ntokens = llm_prompt(pbuf, llm_output_buf, (int)sizeof(llm_output_buf));
        if (ntokens < 0) {
            const char *resp = "ERR inference failed\n";
            netstack_send_udp(src_ip, net_cfg.http_port, src_port,
                              resp, kstrlen(resp));
        } else {
            char resp[4096];
            int rlen = kprintf_to_buf(resp, (int)sizeof(resp),
                "OK %d tokens\n%s\n", ntokens, llm_output_buf);
            netstack_send_udp(src_ip, net_cfg.http_port, src_port, resp, rlen);
        }
        return;
    }

    const char *resp = "ERR unknown command. Use: PING, INFO, INFER <prompt>\n"
                       "HTTP API: http://<ip>:8080/v1/chat/completions\n";
    netstack_send_udp(src_ip, net_cfg.http_port, src_port,
                      resp, kstrlen(resp));
}

/* =============================================================================
 * TCP Retransmission Timer (call from main loop ~100ms)
 * =============================================================================*/

void netstack_timer_tick(void)
{
    uint64_t now = watchdog_ticks;

    for (int i = 0; i < TCP_MAX_CONNS; i++) {
        tcp_conn_t *c = &tcp_conns[i];

        /* TIME_WAIT cleanup: expire after ~2MSL (60s) */
        if (c->state == TCP_STATE_TIME_WAIT) {
            if (now - c->last_send_tick > 6000) /* 60s at 100Hz */
                c->state = TCP_STATE_CLOSED;
            continue;
        }

        /* Only retransmit for connections with unacked data */
        if (c->state != TCP_STATE_ESTABLISHED &&
            c->state != TCP_STATE_SYN_RCVD)
            continue;
        if (c->rtx_len == 0 || c->snd_una == c->snd_nxt)
            continue;

        /* Check RTO expiry */
        uint64_t elapsed_ticks = now - c->last_send_tick;
        uint32_t elapsed_ms = (uint32_t)(elapsed_ticks * 10); /* 100Hz ticks → ms */
        if (elapsed_ms < c->rto_ms)
            continue;

        /* RTO fired — retransmit (RFC 6298 §5.5-5.7) */
        c->retransmits++;
        if (c->retransmits > 8) {
            /* Too many retransmits — reset connection */
            c->state = TCP_STATE_CLOSED;
            continue;
        }

        /* Exponential backoff */
        c->rto_ms *= 2;
        if (c->rto_ms > 60000) c->rto_ms = 60000;

        /* Congestion collapse: reset to slow start */
        c->ssthresh = c->cwnd / 2;
        if (c->ssthresh < 2920) c->ssthresh = 2920;
        c->cwnd = 1460;  /* 1 MSS */

        /* Retransmit the saved segment */
        uint32_t saved_nxt = c->snd_nxt;
        c->snd_nxt = c->rtx_seq;
        netstack_tcp_send(c, TCP_ACK | TCP_PSH, c->rtx_buf, c->rtx_len);
        c->snd_nxt = saved_nxt;
        c->last_send_tick = now;
    }
}

/* =============================================================================
 * Public API
 * =============================================================================*/

void netstack_init(const uint8_t ip[4], const uint8_t netmask[4],
                   const uint8_t gateway[4])
{
    kmemset(&net_cfg, 0, sizeof(net_cfg));
    kmemcpy(net_cfg.ip, ip, 4);
    kmemcpy(net_cfg.netmask, netmask, 4);
    kmemcpy(net_cfg.gateway, gateway, 4);
    net_cfg.http_port = 8080;
    net_cfg.https_port = 8443;
    net_cfg.server_running = 0;

    virtio_net_dev_t *dev = virtio_net_get_dev();
    if (dev && dev->initialized) {
        kmemcpy(net_cfg.mac, dev->mac, 6);
    }

    kmemset(arp_cache, 0, sizeof(arp_cache));
    kmemset(tcp_conns, 0, sizeof(tcp_conns));
    n_udp_handlers = 0;
    tcp_listen_port = 0;

    net_cfg.configured = 1;

    /* Seed ISN secret from CSPRNG */
    crypto_random(&tcp_isn_secret, sizeof(tcp_isn_secret));

    kprintf("[NET] Stack configured: %u.%u.%u.%u/%u.%u.%u.%u gw %u.%u.%u.%u\n",
            ip[0], ip[1], ip[2], ip[3],
            netmask[0], netmask[1], netmask[2], netmask[3],
            gateway[0], gateway[1], gateway[2], gateway[3]);
}

void netstack_rx(const uint8_t *frame, uint32_t len)
{
    if (len < 14) return;
    stat_rx_frames++;

    const struct eth_hdr *eth = (const struct eth_hdr *)frame;
    uint16_t ethertype = ntohs(eth->ethertype);

    switch (ethertype) {
    case ETH_TYPE_ARP:
        handle_arp(frame, len);
        break;
    case ETH_TYPE_IP:
        handle_ip(frame, len);
        break;
    default:
        break;
    }
}

int netstack_send_ip(const uint8_t dst_ip[4], uint8_t proto,
                     const void *data, uint32_t len)
{
    if (!net_cfg.configured) return -1;
    if (len + sizeof(struct ip_hdr) > 1500) return -2;

    uint8_t pkt[1500];
    struct ip_hdr *iph = (struct ip_hdr *)pkt;
    iph->version_ihl = 0x45;
    iph->tos = 0;
    iph->total_len = htons((uint16_t)(sizeof(struct ip_hdr) + len));
    iph->id = htons(ip_id_counter++);
    iph->flags_frag = htons(0x4000);
    iph->ttl = 64;
    iph->proto = proto;
    iph->checksum = 0;
    kmemcpy(iph->src, net_cfg.ip, 4);
    kmemcpy(iph->dst, dst_ip, 4);
    iph->checksum = ip_checksum(iph, sizeof(struct ip_hdr));

    kmemcpy(pkt + sizeof(struct ip_hdr), data, len);

    const uint8_t *dst_mac = arp_cache_lookup(dst_ip);
    if (!dst_mac) dst_mac = broadcast_mac;

    return send_eth(dst_mac, ETH_TYPE_IP, pkt, (uint32_t)(sizeof(struct ip_hdr) + len));
}

int netstack_send_udp(const uint8_t dst_ip[4], uint16_t src_port,
                      uint16_t dst_port, const void *data, uint32_t len)
{
    if (len + sizeof(struct udp_hdr) > 1400) return -2;

    uint8_t udp_pkt[1500];
    struct udp_hdr *udp = (struct udp_hdr *)udp_pkt;
    udp->src_port = htons(src_port);
    udp->dst_port = htons(dst_port);
    udp->length = htons((uint16_t)(sizeof(struct udp_hdr) + len));
    udp->checksum = 0;

    kmemcpy(udp_pkt + sizeof(struct udp_hdr), data, len);
    return netstack_send_ip(dst_ip, IP_PROTO_UDP, udp_pkt,
                            (uint32_t)(sizeof(struct udp_hdr) + len));
}

void netstack_register_udp(uint16_t port, udp_handler_t handler)
{
    if (n_udp_handlers >= MAX_UDP_HANDLERS) return;
    udp_handlers[n_udp_handlers].port = port;
    udp_handlers[n_udp_handlers].handler = handler;
    n_udp_handlers++;
    kprintf("[NET] Registered UDP handler on port %u\n", port);
}

void netstack_start_http_server(void)
{
    tcp_listen_port = net_cfg.http_port;
    net_cfg.server_running = 1;

    netstack_register_udp(net_cfg.http_port, inference_udp_handler);

    kprintf("\n");
    kprintf("==========================================================\n");
    kprintf("  TensorOS LLM API Server  -  OpenAI Compatible\n");
    kprintf("==========================================================\n");
    kprintf("  Listening:  http://%u.%u.%u.%u:%u\n",
            net_cfg.ip[0], net_cfg.ip[1], net_cfg.ip[2], net_cfg.ip[3],
            net_cfg.http_port);
    kprintf("  Model:      %s\n",
            llm_is_loaded() ? llm_model_name() : "(none)");
    kprintf("\n");
    kprintf("  Endpoints:\n");
    kprintf("    GET  /v1/models            List models\n");
    kprintf("    POST /v1/completions       Text completion\n");
    kprintf("    POST /v1/chat/completions  Chat completion\n");
    kprintf("    GET  /health               Health check\n");
    kprintf("\n");
    kprintf("  Quick start (from any device on the network):\n");
    kprintf("\n");
    kprintf("    # List models\n");
    kprintf("    curl http://%u.%u.%u.%u:%u/v1/models\n",
            net_cfg.ip[0], net_cfg.ip[1], net_cfg.ip[2], net_cfg.ip[3],
            net_cfg.http_port);
    kprintf("\n");
    kprintf("    # Chat completion\n");
    kprintf("    curl -X POST http://%u.%u.%u.%u:%u/v1/chat/completions \\\n",
            net_cfg.ip[0], net_cfg.ip[1], net_cfg.ip[2], net_cfg.ip[3],
            net_cfg.http_port);
    kprintf("      -H 'Content-Type: application/json' \\\n");
    kprintf("      -d '{\"messages\":[{\"role\":\"user\","
            "\"content\":\"Hello\"}]}'\n");
    kprintf("\n");
    kprintf("    # Python (OpenAI SDK)\n");
    kprintf("    from openai import OpenAI\n");
    kprintf("    client = OpenAI(\n");
    kprintf("        base_url=\"http://%u.%u.%u.%u:%u/v1\",\n",
            net_cfg.ip[0], net_cfg.ip[1], net_cfg.ip[2], net_cfg.ip[3],
            net_cfg.http_port);
    kprintf("        api_key=\"tensoros\")\n");
    kprintf("    r = client.chat.completions.create(\n");
    kprintf("        model=\"default\",\n");
    kprintf("        messages=[{\"role\":\"user\",\"content\":\"Hello\"}])\n");
    kprintf("    print(r.choices[0].message.content)\n");
    kprintf("==========================================================\n\n");
}

void netstack_start_https_server(void)
{
    tls_init();
    net_cfg.https_port = 8443;
    kprintf("[HTTPS] TLS 1.3 server on port %u\n", net_cfg.https_port);
    kprintf("[HTTPS] https://%u.%u.%u.%u:%u/v1/models\n",
            net_cfg.ip[0], net_cfg.ip[1], net_cfg.ip[2], net_cfg.ip[3],
            net_cfg.https_port);
}

void netstack_poll(void)
{
    if (!net_cfg.configured) return;
    virtio_net_poll(netstack_rx);

    /* Process queued inference requests */
    while (infer_queue_head != infer_queue_tail) {
        if (__sync_lock_test_and_set(&http_busy, 1)) break; /* Busy */

        infer_request_t *r = &infer_queue[infer_queue_head];
        tcp_conn_t *conn = r->conn;
        if (conn && conn->state == TCP_STATE_ESTABLISHED) {
            conn->tls_session = r->tls_session;
            if (r->is_chat)
                api_chat_completions(conn, r->body, r->body_len);
            else
                api_completions(conn, r->body, r->body_len);
            tcp_conn_close(conn);
        }
        __sync_lock_release(&http_busy);
        infer_queue_head = (infer_queue_head + 1) % INFER_QUEUE_SIZE;
    }
}

int netstack_server_running(void)
{
    return net_cfg.server_running;
}

const net_config_t *netstack_get_config(void)
{
    return &net_cfg;
}

void netstack_print_stats(void)
{
    kprintf("  RX frames:    %lu\n", stat_rx_frames);
    kprintf("  TX frames:    %lu\n", stat_tx_frames);
    kprintf("  ARP req/rep:  %lu / %lu\n", stat_arp_req, stat_arp_rep);
    kprintf("  IP packets:   %lu\n", stat_ip_rx);
    kprintf("  TCP segments: %lu\n", stat_tcp_rx);
    kprintf("  UDP packets:  %lu\n", stat_udp_rx);
    kprintf("  ICMP packets: %lu\n", stat_icmp_rx);
    kprintf("  HTTP reqs:    %lu\n", stat_http_req);
    kprintf("  Inferences:   %lu\n", stat_http_infer);

    if (net_cfg.server_running) {
        kprintf("  Server:       http://%u.%u.%u.%u:%u (running)\n",
                net_cfg.ip[0], net_cfg.ip[1], net_cfg.ip[2], net_cfg.ip[3],
                net_cfg.http_port);
    } else {
        kprintf("  Server:       not started (use 'serve' command)\n");
    }

    int active = 0;
    for (int i = 0; i < TCP_MAX_CONNS; i++) {
        if (tcp_conns[i].state != TCP_STATE_CLOSED) active++;
    }
    kprintf("  TCP conns:    %d / %d\n", active, TCP_MAX_CONNS);
}

/* =============================================================================
 * kprintf_to_buf - snprintf-like formatter (subset: %s, %u, %d, %lu, %ld)
 * =============================================================================*/

int kprintf_to_buf(char *buf, int buflen, const char *fmt, ...)
{
    __builtin_va_list ap;
    __builtin_va_start(ap, fmt);

    int pos = 0;
    while (*fmt && pos < buflen - 1) {
        if (*fmt != '%') {
            buf[pos++] = *fmt++;
            continue;
        }
        fmt++;

        /* Handle format flags */
        int is_long = 0;
        int pad_width = 0;
        int dot_star = 0;
        int dot_width = 0;

        /* Check for width (e.g., %.*s) */
        if (*fmt == '.') {
            fmt++;
            if (*fmt == '*') {
                dot_star = 1;
                dot_width = __builtin_va_arg(ap, int);
                fmt++;
            }
        }

        /* Numeric width */
        while (*fmt >= '0' && *fmt <= '9') {
            pad_width = pad_width * 10 + (*fmt - '0');
            fmt++;
        }

        if (*fmt == 'l') { is_long = 1; fmt++; }

        switch (*fmt) {
        case 's': {
            const char *s = __builtin_va_arg(ap, const char *);
            if (!s) s = "(null)";
            int slen = 0;
            while (s[slen]) slen++;
            int limit = (dot_star && dot_width < slen) ? dot_width : slen;
            for (int i = 0; i < limit && pos < buflen - 1; i++)
                buf[pos++] = s[i];
            break;
        }
        case 'u': {
            uint64_t v;
            if (is_long) v = __builtin_va_arg(ap, uint64_t);
            else v = __builtin_va_arg(ap, uint32_t);
            char tmp[20];
            int ti = 0;
            if (v == 0) { tmp[ti++] = '0'; }
            else { while (v > 0) { tmp[ti++] = '0' + (char)(v % 10); v /= 10; } }
            for (int i = ti - 1; i >= 0 && pos < buflen - 1; i--)
                buf[pos++] = tmp[i];
            break;
        }
        case 'd': {
            int64_t v;
            if (is_long) v = __builtin_va_arg(ap, int64_t);
            else v = __builtin_va_arg(ap, int32_t);
            if (v < 0) { if (pos < buflen - 1) buf[pos++] = '-'; v = -v; }
            char tmp[20];
            int ti = 0;
            if (v == 0) { tmp[ti++] = '0'; }
            else { while (v > 0) { tmp[ti++] = '0' + (char)(v % 10); v /= 10; } }
            for (int i = ti - 1; i >= 0 && pos < buflen - 1; i--)
                buf[pos++] = tmp[i];
            break;
        }
        case '%':
            buf[pos++] = '%';
            break;
        default:
            buf[pos++] = '%';
            if (is_long && pos < buflen - 1) buf[pos++] = 'l';
            if (pos < buflen - 1) buf[pos++] = *fmt;
            break;
        }
        fmt++;
    }

    __builtin_va_end(ap);
    buf[pos] = '\0';
    return pos;
}
