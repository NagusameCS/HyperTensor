/* =============================================================================
 * TensorOS - Minimal Network Stack
 * ARP, IPv4, UDP, and simple HTTP for AI inference serving
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
    uint16_t http_port;      /* Port for inference HTTP server */
    int      configured;
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
 * Register a UDP handler for a specific port.
 * handler(src_ip, src_port, data, data_len)
 */
typedef void (*udp_handler_t)(const uint8_t src_ip[4], uint16_t src_port,
                               const uint8_t *data, uint32_t len);
void netstack_register_udp(uint16_t port, udp_handler_t handler);

/**
 * Start the HTTP inference server on the configured port.
 * Handles simple GET/POST requests.
 */
void netstack_start_http_server(void);

/**
 * Get network stack statistics.
 */
void netstack_print_stats(void);

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
