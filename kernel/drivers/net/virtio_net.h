/* =============================================================================
 * TensorOS - Virtio Network Driver
 * PCI-based virtio-net for QEMU, supports legacy virtio (0.9.5)
 * Provides raw Ethernet frame send/receive
 * =============================================================================*/

#ifndef TENSOROS_VIRTIO_NET_H
#define TENSOROS_VIRTIO_NET_H

#include <stdint.h>

/* Virtio PCI vendor/device IDs */
#define VIRTIO_PCI_VENDOR      0x1AF4
#define VIRTIO_NET_DEVICE_ID   0x1000  /* Legacy: subsystem determines type */
#define VIRTIO_NET_SUBSYS      0x0001

/* Virtio device status bits */
#define VIRTIO_STATUS_ACK       1
#define VIRTIO_STATUS_DRIVER    2
#define VIRTIO_STATUS_DRIVER_OK 4
#define VIRTIO_STATUS_FEATURES_OK 8
#define VIRTIO_STATUS_FAILED    128

/* Virtio net feature bits */
#define VIRTIO_NET_F_MAC        (1 << 5)
#define VIRTIO_NET_F_STATUS     (1 << 16)
#define VIRTIO_NET_F_MRG_RXBUF (1 << 15)

/* Legacy virtio PCI register offsets (BAR0, I/O space) */
#define VIRTIO_REG_DEVICE_FEATURES  0x00  /* 32-bit read */
#define VIRTIO_REG_GUEST_FEATURES   0x04  /* 32-bit write */
#define VIRTIO_REG_QUEUE_ADDR       0x08  /* 32-bit write (PFN) */
#define VIRTIO_REG_QUEUE_SIZE       0x0C  /* 16-bit read */
#define VIRTIO_REG_QUEUE_SELECT     0x0E  /* 16-bit write */
#define VIRTIO_REG_QUEUE_NOTIFY     0x10  /* 16-bit write */
#define VIRTIO_REG_DEVICE_STATUS    0x12  /* 8-bit r/w */
#define VIRTIO_REG_ISR_STATUS       0x13  /* 8-bit read */
/* MAC address at offset 0x14 for net devices (6 bytes) */
#define VIRTIO_NET_REG_MAC          0x14

/* Virtqueue descriptor flags */
#define VRING_DESC_F_NEXT     1
#define VRING_DESC_F_WRITE    2  /* Device writes (RX) */
#define VRING_DESC_F_INDIRECT 4

/* Virtqueue structures */
#define VIRTQ_SIZE 256  /* Must be power of 2 */

struct vring_desc {
    uint64_t addr;
    uint32_t len;
    uint16_t flags;
    uint16_t next;
} __attribute__((packed));

struct vring_avail {
    uint16_t flags;
    uint16_t idx;
    uint16_t ring[VIRTQ_SIZE];
    uint16_t used_event;
} __attribute__((packed));

struct vring_used_elem {
    uint32_t id;
    uint32_t len;
} __attribute__((packed));

struct vring_used {
    uint16_t flags;
    uint16_t idx;
    struct vring_used_elem ring[VIRTQ_SIZE];
    uint16_t avail_event;
} __attribute__((packed));

/* Virtio net header (prepended to each packet) */
struct virtio_net_hdr {
    uint8_t  flags;
    uint8_t  gso_type;
    uint16_t hdr_len;
    uint16_t gso_size;
    uint16_t csum_start;
    uint16_t csum_offset;
} __attribute__((packed));

/* Per-queue state */
typedef struct {
    struct vring_desc  *desc;
    struct vring_avail *avail;
    struct vring_used  *used;
    uint16_t num_free;
    uint16_t free_head;
    uint16_t last_used_idx;
    uint16_t queue_size;
} virtq_t;

/* Packet buffer */
#define NET_PKT_SIZE 2048
#define NET_RX_BUFS  64
#define NET_TX_BUFS  64

/* Virtio-net device state */
typedef struct {
    uint16_t  io_base;       /* PCI BAR0 I/O port base */
    uint8_t   mac[6];        /* MAC address */
    uint8_t   pci_bus;
    uint8_t   pci_slot;
    uint8_t   pci_func;
    uint8_t   irq;
    int       initialized;

    /* Virtqueues: 0=RX, 1=TX */
    virtq_t   rxq;
    virtq_t   txq;

    /* RX/TX packet buffers */
    uint8_t  *rx_bufs;   /* NET_RX_BUFS * NET_PKT_SIZE */
    uint8_t  *tx_bufs;   /* NET_TX_BUFS * NET_PKT_SIZE */

    /* Statistics */
    uint64_t  rx_packets;
    uint64_t  tx_packets;
    uint64_t  rx_bytes;
    uint64_t  tx_bytes;
} virtio_net_dev_t;

/* =============================================================================
 * API
 * =============================================================================*/

/**
 * Initialize virtio-net driver. Scans PCI for virtio-net device.
 * Returns 0 on success, negative on failure.
 */
int virtio_net_init(void);

/**
 * Get device state.
 */
virtio_net_dev_t *virtio_net_get_dev(void);

/**
 * Send a raw Ethernet frame.
 * data: pointer to frame (starting with dest MAC)
 * len: frame length in bytes
 * Returns 0 on success.
 */
int virtio_net_send(const void *data, uint32_t len);

/**
 * Poll for received packets. Calls handler for each received frame.
 * handler: callback(frame_data, frame_len)
 * Returns number of packets processed.
 */
int virtio_net_poll(void (*handler)(const uint8_t *data, uint32_t len));

/**
 * Get MAC address string.
 */
void virtio_net_get_mac_str(char *buf, int buflen);

#endif /* TENSOROS_VIRTIO_NET_H */
