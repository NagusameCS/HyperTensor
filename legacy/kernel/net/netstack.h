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
 * TensorOS - Network Stack
 * ARP, IPv4, ICMP, UDP, TCP, and OpenAI-compatible HTTP API for LLM serving
 * =============================================================================*/

#ifndef TENSOROS_NETSTACK_H
#define TENSOROS_NETSTACK_H

#include <stdint.h>

/* Ethernet */
#define ETH_TYPE_ARP  0x0806
#define ETH_TYPE_IP   0x0800

#define ETH_ADDR_LEN 6
#define IP_ADDR_LEN  4

/* IP protocols */
#define IP_PROTO_ICMP 1
#define IP_PROTO_TCP  6
#define IP_PROTO_UDP  17

/* =============================================================================
 * Protocol headers
 * =============================================================================*/

struct eth_hdr {
    uint8_t  dst[6];
    uint8_t  src[6];
    uint16_t ethertype;  /* Big-endian */
} __attribute__((packed));

struct arp_hdr {
    uint16_t hw_type;     /* 1 = Ethernet */
    uint16_t proto_type;  /* 0x0800 = IPv4 */
    uint8_t  hw_len;      /* 6 */
    uint8_t  proto_len;   /* 4 */
    uint16_t opcode;      /* 1=Request, 2=Reply */
    uint8_t  sender_mac[6];
    uint8_t  sender_ip[4];
    uint8_t  target_mac[6];
    uint8_t  target_ip[4];
} __attribute__((packed));

struct ip_hdr {
    uint8_t  version_ihl;  /* version=4, ihl=5 (20 bytes) */
    uint8_t  tos;
    uint16_t total_len;    /* Big-endian */
    uint16_t id;
    uint16_t flags_frag;
    uint8_t  ttl;
    uint8_t  proto;
    uint16_t checksum;
    uint8_t  src[4];
    uint8_t  dst[4];
} __attribute__((packed));

struct tcp_hdr {
    uint16_t src_port;    /* Big-endian */
    uint16_t dst_port;    /* Big-endian */
    uint32_t seq;         /* Sequence number */
    uint32_t ack;         /* Acknowledgment number */
    uint8_t  data_off;    /* Data offset (upper 4 bits) in 32-bit words */
    uint8_t  flags;       /* TCP flags */
    uint16_t window;      /* Window size */
    uint16_t checksum;    /* Checksum */
    uint16_t urgent;      /* Urgent pointer */
} __attribute__((packed));

/* TCP flags */
#define TCP_FIN  0x01
#define TCP_SYN  0x02
#define TCP_RST  0x04
#define TCP_PSH  0x08
#define TCP_ACK  0x10
#define TCP_URG  0x20

/* TCP connection states */
#define TCP_STATE_CLOSED      0
#define TCP_STATE_LISTEN      1
#define TCP_STATE_SYN_RCVD    2
#define TCP_STATE_ESTABLISHED 3
#define TCP_STATE_FIN_WAIT_1  4
#define TCP_STATE_FIN_WAIT_2  5
#define TCP_STATE_CLOSE_WAIT  6
#define TCP_STATE_LAST_ACK    7
#define TCP_STATE_TIME_WAIT   8

/* TCP connection (simplified — single concurrent connection per slot) */
#define TCP_MAX_CONNS     64
#define TCP_RX_BUF_SIZE   16384  /* 16 KB receive buffer per connection */
#define TCP_TX_BUF_SIZE   32768  /* 32 KB send buffer per connection */

typedef struct tcp_conn {
    uint8_t  state;
    uint8_t  remote_ip[4];
    uint16_t local_port;
    uint16_t remote_port;
    uint32_t snd_nxt;       /* Next sequence number to send */
    uint32_t snd_una;       /* Oldest unacknowledged seq */
    uint32_t rcv_nxt;       /* Next expected receive seq */
    uint16_t remote_win;    /* Remote window size */

    /* Retransmission / congestion control */
    uint32_t rto_ms;        /* Retransmit timeout (ms), Jacobson/Karn */
    uint32_t srtt;          /* Smoothed RTT (ms, fixed-point 8) */
    uint32_t rttvar;        /* RTT variance (ms, fixed-point 4) */
    uint64_t last_send_tick;/* Tick when last data segment was sent */
    uint32_t retransmits;   /* Consecutive retransmit count */
    uint32_t dup_acks;      /* Duplicate ACK counter */
    uint32_t cwnd;          /* Congestion window (bytes) */
    uint32_t ssthresh;      /* Slow-start threshold (bytes) */

    /* Retransmit buffer (last unacked segment) */
    uint8_t  rtx_buf[1460]; /* Saved payload for retransmit */
    uint32_t rtx_len;       /* Length of saved payload */
    uint32_t rtx_seq;       /* Sequence number of saved payload */

    /* Receive buffer (reassembled in-order data) */
    uint8_t  rx_buf[TCP_RX_BUF_SIZE];
    uint32_t rx_len;        /* Bytes available in rx_buf */

    /* Send buffer (data queued for transmission) */
    uint8_t  tx_buf[TCP_TX_BUF_SIZE];
    uint32_t tx_len;        /* Bytes queued in tx_buf */

    /* HTTP request complete flag (received full request) */
    int      http_request_complete;

    /* TLS session (NULL for plaintext connections) */
    void    *tls_session;
} tcp_conn_t;

struct udp_hdr {
    uint16_t src_port;    /* Big-endian */
    uint16_t dst_port;    /* Big-endian */
    uint16_t length;
    uint16_t checksum;
} __attribute__((packed));

struct icmp_hdr {
    uint8_t  type;
    uint8_t  code;
    uint16_t checksum;
    uint16_t id;
    uint16_t seq;
} __attribute__((packed));

/* =============================================================================
 * Network configuration
 * =============================================================================*/

typedef struct {
    uint8_t  ip[4];
    uint8_t  netmask[4];
    uint8_t  gateway[4];
    uint8_t  mac[6];
    uint16_t http_port;      /* Port for inference HTTP server (default 8080) */
    uint16_t https_port;     /* Port for HTTPS/TLS server (default 8443) */
    int      configured;
    int      server_running; /* HTTP API server active */
} net_config_t;

/* ARP cache entry */
#define ARP_CACHE_SIZE 32
typedef struct {
    uint8_t  ip[4];
    uint8_t  mac[6];
    uint8_t  valid;
} arp_entry_t;

/* =============================================================================
 * API
 * =============================================================================*/

/**
 * Initialize network stack with config.
 * Must be called after virtio_net_init().
 */
void netstack_init(const uint8_t ip[4], const uint8_t netmask[4],
                   const uint8_t gateway[4]);

/**
 * Process one received Ethernet frame.
 * Called from virtio_net_poll() callback.
 */
void netstack_rx(const uint8_t *frame, uint32_t len);

/**
 * Send a UDP packet.
 */
int netstack_send_udp(const uint8_t dst_ip[4], uint16_t src_port,
                      uint16_t dst_port, const void *data, uint32_t len);

/**
 * Send a raw IP packet.
 */
int netstack_send_ip(const uint8_t dst_ip[4], uint8_t proto,
                     const void *data, uint32_t len);

/**
 * Send a TCP segment on a connection.
 */
int netstack_tcp_send(tcp_conn_t *conn, uint8_t flags,
                      const void *data, uint32_t len);

/**
 * Write data to a TCP connection's send buffer and transmit.
 */
int tcp_conn_write(tcp_conn_t *conn, const void *data, uint32_t len);

/**
 * Close a TCP connection (send FIN).
 */
void tcp_conn_close(tcp_conn_t *conn);

/**
 * Register a UDP handler for a specific port.
 */
typedef void (*udp_handler_t)(const uint8_t src_ip[4], uint16_t src_port,
                               const uint8_t *data, uint32_t len);
void netstack_register_udp(uint16_t port, udp_handler_t handler);

/**
 * Start the OpenAI-compatible HTTP inference API server.
 * Endpoints:
 *   GET  /v1/models          — list loaded models
 *   POST /v1/completions     — text completion
 *   POST /v1/chat/completions — chat completion
 *   GET  /health             — health check
 *
 * Compatible with: curl, Python requests, OpenAI SDK, any HTTP client.
 */
void netstack_start_http_server(void);

/**
 * Start the HTTPS (TLS 1.3) inference API server on port 8443.
 */
void netstack_start_https_server(void);

/**
 * Poll for network events (call in main loop).
 * Processes pending TCP connections and HTTP requests.
 */
void netstack_poll(void);

/**
 * TCP retransmission timer — call periodically (~100ms) from main loop.
 * Handles RTO-based retransmits and TIME_WAIT cleanup.
 */
void netstack_timer_tick(void);

/**
 * Check if HTTP server is running.
 */
int netstack_server_running(void);

/**
 * Get network configuration (for display).
 */
const net_config_t *netstack_get_config(void);

/**
 * Get network stack statistics.
 */
void netstack_print_stats(void);

/**
 * Set API key for HTTP authentication. Empty/NULL = no auth required.
 */
void net_set_api_key(const char *key);

/**
 * Byte-swap helpers (network byte order) */
static inline uint16_t htons(uint16_t v) { return (v >> 8) | (v << 8); }
static inline uint16_t ntohs(uint16_t v) { return htons(v); }
static inline uint32_t htonl(uint32_t v) {
    return ((v >> 24) & 0xFF) | ((v >> 8) & 0xFF00) |
           ((v << 8) & 0xFF0000) | ((v << 24) & 0xFF000000u);
}
static inline uint32_t ntohl(uint32_t v) { return htonl(v); }

#endif /* TENSOROS_NETSTACK_H */
