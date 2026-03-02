/* =============================================================================
 * TensorOS - Virtio-Net Driver Implementation
 * Legacy virtio (PCI BAR0 I/O) for QEMU's default NIC
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/drivers/net/virtio_net.h"

/* Global device instance */
static virtio_net_dev_t vnet_dev;

/* Static buffers for virtqueue structures (must be page-aligned) */
static uint8_t rxq_mem[32768] __attribute__((aligned(4096)));
static uint8_t txq_mem[32768] __attribute__((aligned(4096)));
static uint8_t rx_packet_bufs[NET_RX_BUFS * NET_PKT_SIZE] __attribute__((aligned(4096)));
static uint8_t tx_packet_bufs[NET_TX_BUFS * NET_PKT_SIZE] __attribute__((aligned(4096)));

/* =============================================================================
 * PCI helpers
 * =============================================================================*/

static uint32_t pci_read32(uint8_t bus, uint8_t slot, uint8_t func, uint8_t offset)
{
    uint32_t addr = (1u << 31) | ((uint32_t)bus << 16) | ((uint32_t)slot << 11) |
                    ((uint32_t)func << 8) | (offset & 0xFC);
    outl(0xCF8, addr);
    return inl(0xCFC);
}

static void pci_write32(uint8_t bus, uint8_t slot, uint8_t func, uint8_t offset, uint32_t val)
{
    uint32_t addr = (1u << 31) | ((uint32_t)bus << 16) | ((uint32_t)slot << 11) |
                    ((uint32_t)func << 8) | (offset & 0xFC);
    outl(0xCF8, addr);
    outl(0xCFC, val);
}

static uint16_t pci_read16(uint8_t bus, uint8_t slot, uint8_t func, uint8_t offset)
{
    uint32_t val = pci_read32(bus, slot, func, offset & 0xFC);
    return (uint16_t)(val >> ((offset & 2) * 8));
}

/* =============================================================================
 * Find virtio-net device on PCI bus
 * =============================================================================*/

static int find_virtio_net(uint8_t *out_bus, uint8_t *out_slot, uint8_t *out_func)
{
    for (uint32_t bus = 0; bus < 8; bus++) {
        for (uint32_t slot = 0; slot < 32; slot++) {
            uint32_t id = pci_read32(bus, slot, 0, 0);
            uint16_t vendor = id & 0xFFFF;
            uint16_t device = (id >> 16) & 0xFFFF;

            if (vendor != VIRTIO_PCI_VENDOR) continue;

            /* Legacy virtio: device 0x1000-0x103F, subsystem determines type */
            if (device >= 0x1000 && device <= 0x103F) {
                uint32_t subsys = pci_read32(bus, slot, 0, 0x2C);
                uint16_t subsys_id = (subsys >> 16) & 0xFFFF;
                if (subsys_id == VIRTIO_NET_SUBSYS) {
                    *out_bus = bus;
                    *out_slot = slot;
                    *out_func = 0;
                    return 0;
                }
            }

            /* Modern virtio-net: device 0x1041 */
            if (device == 0x1041) {
                *out_bus = bus;
                *out_slot = slot;
                *out_func = 0;
                return 0;
            }
        }
    }
    return -1;
}

/* =============================================================================
 * Virtqueue setup
 * =============================================================================*/

static void virtq_init(virtq_t *vq, void *mem, uint16_t qsize)
{
    kmemset(mem, 0, 32768);

    vq->queue_size = qsize;

    /* Layout: descs | avail | padding | used */
    vq->desc = (struct vring_desc *)mem;
    vq->avail = (struct vring_avail *)((uint8_t *)mem + qsize * sizeof(struct vring_desc));

    /* Used ring must be page-aligned */
    uint64_t avail_end = (uint64_t)(uintptr_t)vq->avail + sizeof(struct vring_avail) + qsize * sizeof(uint16_t);
    uint64_t used_offset = (avail_end + 4095) & ~4095ULL;
    vq->used = (struct vring_used *)(uintptr_t)used_offset;

    /* Initialize free descriptor chain */
    for (uint16_t i = 0; i < qsize - 1; i++) {
        vq->desc[i].next = i + 1;
        vq->desc[i].flags = VRING_DESC_F_NEXT;
    }
    vq->desc[qsize - 1].next = 0;
    vq->desc[qsize - 1].flags = 0;

    vq->num_free = qsize;
    vq->free_head = 0;
    vq->last_used_idx = 0;
}

static uint16_t virtq_alloc_desc(virtq_t *vq)
{
    if (vq->num_free == 0) return 0xFFFF;
    uint16_t idx = vq->free_head;
    vq->free_head = vq->desc[idx].next;
    vq->num_free--;
    return idx;
}

static void virtq_free_desc(virtq_t *vq, uint16_t idx)
{
    vq->desc[idx].next = vq->free_head;
    vq->desc[idx].flags = VRING_DESC_F_NEXT;
    vq->free_head = idx;
    vq->num_free++;
}

static void virtq_submit(virtq_t *vq, uint16_t desc_idx)
{
    uint16_t avail_idx = vq->avail->idx;
    vq->avail->ring[avail_idx % vq->queue_size] = desc_idx;
#if defined(__aarch64__)
    __asm__ volatile ("dmb sy" ::: "memory");
#else
    __asm__ volatile ("mfence" ::: "memory");
#endif
    vq->avail->idx = avail_idx + 1;
}

/* =============================================================================
 * Initialize device
 * =============================================================================*/

int virtio_net_init(void)
{
    kmemset(&vnet_dev, 0, sizeof(vnet_dev));
    vnet_dev.rx_bufs = rx_packet_bufs;
    vnet_dev.tx_bufs = tx_packet_bufs;

    uint8_t bus, slot, func;
    if (find_virtio_net(&bus, &slot, &func) != 0) {
        kprintf("[VNET] No virtio-net device found on PCI bus\n");
        return -1;
    }

    vnet_dev.pci_bus = bus;
    vnet_dev.pci_slot = slot;
    vnet_dev.pci_func = func;

    kprintf("[VNET] Found virtio-net at PCI %u:%u.%u\n", bus, slot, func);

    /* Enable PCI bus mastering + I/O space */
    uint32_t cmd = pci_read32(bus, slot, func, 0x04);
    cmd |= 0x05;  /* I/O Space + Bus Master */
    pci_write32(bus, slot, func, 0x04, cmd);

    /* Read BAR0 (I/O port base) */
    uint32_t bar0 = pci_read32(bus, slot, func, 0x10);
    if (!(bar0 & 1)) {
        kprintf("[VNET] BAR0 is not I/O space -- unsupported\n");
        return -2;
    }
    vnet_dev.io_base = (uint16_t)(bar0 & 0xFFFC);
    kprintf("[VNET] I/O base: 0x%x\n", vnet_dev.io_base);

    /* Read IRQ */
    uint32_t irq_reg = pci_read32(bus, slot, func, 0x3C);
    vnet_dev.irq = irq_reg & 0xFF;
    kprintf("[VNET] IRQ: %u\n", vnet_dev.irq);

    uint16_t io = vnet_dev.io_base;

    /* Reset device */
    outb(io + VIRTIO_REG_DEVICE_STATUS, 0);

    /* Set ACKNOWLEDGE */
    outb(io + VIRTIO_REG_DEVICE_STATUS, VIRTIO_STATUS_ACK);

    /* Set DRIVER */
    outb(io + VIRTIO_REG_DEVICE_STATUS,
         VIRTIO_STATUS_ACK | VIRTIO_STATUS_DRIVER);

    /* Read device features */
    uint32_t features = inl(io + VIRTIO_REG_DEVICE_FEATURES);
    kprintf("[VNET] Device features: 0x%x\n", features);

    /* Negotiate features: we want MAC, no mergeable rx bufs */
    uint32_t guest_features = 0;
    if (features & VIRTIO_NET_F_MAC) guest_features |= VIRTIO_NET_F_MAC;
    outl(io + VIRTIO_REG_GUEST_FEATURES, guest_features);

    /* Read MAC address */
    if (features & VIRTIO_NET_F_MAC) {
        for (int i = 0; i < 6; i++)
            vnet_dev.mac[i] = inb(io + VIRTIO_NET_REG_MAC + i);
    } else {
        /* Generate random-ish MAC: 52:54:00:xx:xx:xx (QEMU default range) */
        vnet_dev.mac[0] = 0x52; vnet_dev.mac[1] = 0x54; vnet_dev.mac[2] = 0x00;
        vnet_dev.mac[3] = 0x12; vnet_dev.mac[4] = 0x34; vnet_dev.mac[5] = 0x56;
    }

    char mac_str[20];
    virtio_net_get_mac_str(mac_str, sizeof(mac_str));
    kprintf("[VNET] MAC: %s\n", mac_str);

    /* Setup RX queue (queue 0) */
    outw(io + VIRTIO_REG_QUEUE_SELECT, 0);
    uint16_t rxq_size = inw(io + VIRTIO_REG_QUEUE_SIZE);
    if (rxq_size == 0) {
        kprintf("[VNET] RX queue size is 0\n");
        return -3;
    }
    if (rxq_size > VIRTQ_SIZE) rxq_size = VIRTQ_SIZE;
    kprintf("[VNET] RX queue size: %u\n", rxq_size);

    virtq_init(&vnet_dev.rxq, rxq_mem, rxq_size);
    uint64_t rxq_phys = (uint64_t)(uintptr_t)rxq_mem;
    outl(io + VIRTIO_REG_QUEUE_ADDR, (uint32_t)(rxq_phys / 4096));

    /* Setup TX queue (queue 1) */
    outw(io + VIRTIO_REG_QUEUE_SELECT, 1);
    uint16_t txq_size = inw(io + VIRTIO_REG_QUEUE_SIZE);
    if (txq_size == 0) {
        kprintf("[VNET] TX queue size is 0\n");
        return -4;
    }
    if (txq_size > VIRTQ_SIZE) txq_size = VIRTQ_SIZE;
    kprintf("[VNET] TX queue size: %u\n", txq_size);

    virtq_init(&vnet_dev.txq, txq_mem, txq_size);
    uint64_t txq_phys = (uint64_t)(uintptr_t)txq_mem;
    outl(io + VIRTIO_REG_QUEUE_ADDR, (uint32_t)(txq_phys / 4096));

    /* Pre-fill RX queue with buffers */
    uint16_t rx_fill = rxq_size < NET_RX_BUFS ? rxq_size : NET_RX_BUFS;
    for (uint16_t i = 0; i < rx_fill; i++) {
        uint16_t di = virtq_alloc_desc(&vnet_dev.rxq);
        if (di == 0xFFFF) break;

        uint64_t buf_phys = (uint64_t)(uintptr_t)&rx_packet_bufs[i * NET_PKT_SIZE];
        vnet_dev.rxq.desc[di].addr = buf_phys;
        vnet_dev.rxq.desc[di].len = NET_PKT_SIZE;
        vnet_dev.rxq.desc[di].flags = VRING_DESC_F_WRITE; /* Device writes to this */
        vnet_dev.rxq.desc[di].next = 0;

        virtq_submit(&vnet_dev.rxq, di);
    }

    /* Notify device about RX buffers */
    outw(io + VIRTIO_REG_QUEUE_NOTIFY, 0);

    /* Set DRIVER_OK — device is live */
    outb(io + VIRTIO_REG_DEVICE_STATUS,
         VIRTIO_STATUS_ACK | VIRTIO_STATUS_DRIVER | VIRTIO_STATUS_DRIVER_OK);

    vnet_dev.initialized = 1;
    kprintf("[VNET] Virtio-net initialized successfully\n");

    return 0;
}

virtio_net_dev_t *virtio_net_get_dev(void)
{
    return &vnet_dev;
}

/* =============================================================================
 * Send packet
 * =============================================================================*/

int virtio_net_send(const void *data, uint32_t len)
{
    if (!vnet_dev.initialized) return -1;
    if (len > NET_PKT_SIZE - sizeof(struct virtio_net_hdr)) return -2;

    uint16_t di = virtq_alloc_desc(&vnet_dev.txq);
    if (di == 0xFFFF) return -3; /* No free descriptors */

    /* Build packet with virtio-net header */
    uint8_t *buf = &tx_packet_bufs[(di % NET_TX_BUFS) * NET_PKT_SIZE];
    struct virtio_net_hdr *hdr = (struct virtio_net_hdr *)buf;
    kmemset(hdr, 0, sizeof(*hdr));

    kmemcpy(buf + sizeof(struct virtio_net_hdr), data, len);

    uint64_t buf_phys = (uint64_t)(uintptr_t)buf;
    vnet_dev.txq.desc[di].addr = buf_phys;
    vnet_dev.txq.desc[di].len = sizeof(struct virtio_net_hdr) + len;
    vnet_dev.txq.desc[di].flags = 0; /* Device reads from this */
    vnet_dev.txq.desc[di].next = 0;

    virtq_submit(&vnet_dev.txq, di);

    /* Notify device */
    outw(vnet_dev.io_base + VIRTIO_REG_QUEUE_NOTIFY, 1);

    vnet_dev.tx_packets++;
    vnet_dev.tx_bytes += len;

    return 0;
}

/* =============================================================================
 * Poll for received packets
 * =============================================================================*/

int virtio_net_poll(void (*handler)(const uint8_t *data, uint32_t len))
{
    if (!vnet_dev.initialized) return 0;

    int count = 0;
    virtq_t *rxq = &vnet_dev.rxq;

    while (rxq->last_used_idx != rxq->used->idx) {
        uint16_t ui = rxq->last_used_idx % rxq->queue_size;
        struct vring_used_elem *used = &rxq->used->ring[ui];
        uint16_t di = (uint16_t)used->id;
        uint32_t pkt_len = used->len;

        /* Get packet data (skip virtio-net header) */
        uint8_t *buf = (uint8_t *)(uintptr_t)rxq->desc[di].addr;
        uint32_t hdr_sz = sizeof(struct virtio_net_hdr);
        if (pkt_len > hdr_sz && handler) {
            handler(buf + hdr_sz, pkt_len - hdr_sz);
            vnet_dev.rx_packets++;
            vnet_dev.rx_bytes += pkt_len - hdr_sz;
        }

        /* Re-submit buffer to RX queue */
        rxq->desc[di].len = NET_PKT_SIZE;
        rxq->desc[di].flags = VRING_DESC_F_WRITE;
        virtq_submit(rxq, di);

        rxq->last_used_idx++;
        count++;
    }

    /* Notify device about re-submitted buffers */
    if (count > 0)
        outw(vnet_dev.io_base + VIRTIO_REG_QUEUE_NOTIFY, 0);

    /* Also try to reclaim TX descriptors */
    virtq_t *txq = &vnet_dev.txq;
    while (txq->last_used_idx != txq->used->idx) {
        uint16_t ui = txq->last_used_idx % txq->queue_size;
        uint16_t di = (uint16_t)txq->used->ring[ui].id;
        virtq_free_desc(txq, di);
        txq->last_used_idx++;
    }

    return count;
}

/* =============================================================================
 * Utility
 * =============================================================================*/

void virtio_net_get_mac_str(char *buf, int buflen)
{
    if (buflen < 18) { buf[0] = '\0'; return; }
    static const char hex[] = "0123456789abcdef";
    int pos = 0;
    for (int i = 0; i < 6; i++) {
        if (i > 0) buf[pos++] = ':';
        buf[pos++] = hex[(vnet_dev.mac[i] >> 4) & 0xF];
        buf[pos++] = hex[vnet_dev.mac[i] & 0xF];
    }
    buf[pos] = '\0';
}
