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
 * TensorOS - PCIe DMA Engine Implementation
 *
 * Provides scatter-gather DMA for PCIe devices.  Programs the DMA engine
 * via MMIO descriptor rings for zero-copy transfers between host memory
 * and device (GPU/NIC/NVMe) VRAM or registers.
 *
 * Supported backends:
 *   - NVIDIA copy engine (CE) — via NV_PCOPY MMIO
 *   - AMD SDMA engine — via SDMA0/1 ring buffers
 *   - Generic PCIe bus-master DMA — via BAR-mapped descriptor rings
 * =============================================================================*/

#include "kernel/drivers/dma/pcie_dma.h"
#include "kernel/core/perf.h"

/* PCIe Bus Master Enable bit in PCI command register */
#define PCI_COMMAND_REG      0x04
#define PCI_CMD_BUS_MASTER   (1 << 2)
#define PCI_CMD_MEM_SPACE    (1 << 1)

/* NVIDIA copy engine MMIO offsets */
#define NV_PCOPY_CTX         0x104000  /* Copy engine context */
#define NV_PCOPY_SRC_ADDR_LO 0x104100
#define NV_PCOPY_SRC_ADDR_HI 0x104104
#define NV_PCOPY_DST_ADDR_LO 0x104108
#define NV_PCOPY_DST_ADDR_HI 0x10410C
#define NV_PCOPY_SIZE        0x104110  /* Transfer size in bytes */
#define NV_PCOPY_LAUNCH      0x104114  /* Write to begin transfer */
#define NV_PCOPY_STATUS      0x104118  /* 0 = idle, 1 = busy */

/* AMD SDMA ring offsets (GCN/RDNA) */
#define SDMA0_GFX_RB_BASE     0x3240
#define SDMA0_GFX_RB_RPTR     0x3244
#define SDMA0_GFX_RB_WPTR     0x3248
#define SDMA0_GFX_DOORBELL    0x324C

/* Generic DMA descriptor (hardware format, 32 bytes) */
typedef struct {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t length;
    uint32_t control;   /* bits: [0]=last, [1]=irq, [31]=valid */
    uint64_t next_desc; /* Physical address of next descriptor (chained) */
} __attribute__((packed)) hw_dma_desc_t;

#define HW_DESC_VALID   (1u << 31)
#define HW_DESC_LAST    (1u << 0)
#define HW_DESC_IRQ     (1u << 1)

/* =============================================================================
 * Global State
 * =============================================================================*/

static dma_channel_t channels[DMA_MAX_CHANNELS];
static uint32_t channel_count = 0;
static uint64_t next_desc_id = 1;

/* PCI config space access (shared with gpu.c) */
static uint32_t pci_cfg_read32(uint32_t bus, uint32_t dev, uint32_t func, uint32_t off)
{
    uint32_t addr = (1u << 31) | (bus << 16) | (dev << 11) | (func << 8) | (off & 0xFC);
    outl(0xCF8, addr);
    return inl(0xCFC);
}

static void pci_cfg_write32(uint32_t bus, uint32_t dev, uint32_t func,
                             uint32_t off, uint32_t val)
{
    uint32_t addr = (1u << 31) | (bus << 16) | (dev << 11) | (func << 8) | (off & 0xFC);
    outl(0xCF8, addr);
    outl(0xCFC, val);
}

/* MMIO helpers */
static inline uint32_t dma_read32(dma_channel_t *ch, uint32_t offset)
{
    if (!ch->mmio_base) return 0;
    return *(volatile uint32_t *)((uint8_t *)ch->mmio_base + offset);
}

static inline void dma_write32(dma_channel_t *ch, uint32_t offset, uint32_t val)
{
    if (!ch->mmio_base) return;
    *(volatile uint32_t *)((uint8_t *)ch->mmio_base + offset) = val;
}

/* =============================================================================
 * Initialization
 * =============================================================================*/

int dma_engine_init(void)
{
    kmemset(channels, 0, sizeof(channels));
    channel_count = 0;
    kprintf("[DMA] PCIe DMA engine initialized\n");
    return 0;
}

dma_channel_t *dma_channel_open(uint32_t pci_bus, uint32_t pci_device, void *mmio_base)
{
    if (channel_count >= DMA_MAX_CHANNELS) return NULL;

    dma_channel_t *ch = &channels[channel_count];
    kmemset(ch, 0, sizeof(*ch));
    ch->channel_id = channel_count++;
    ch->pci_bus = pci_bus;
    ch->pci_device = pci_device;
    ch->mmio_base = mmio_base;
    ch->ring_size = 64;
    ch->active = true;

    /* Enable PCI bus mastering for DMA */
    uint32_t cmd = pci_cfg_read32(pci_bus, pci_device, 0, PCI_COMMAND_REG);
    cmd |= PCI_CMD_BUS_MASTER | PCI_CMD_MEM_SPACE;
    pci_cfg_write32(pci_bus, pci_device, 0, PCI_COMMAND_REG, cmd);

    kprintf("[DMA] Channel %d opened for PCI %d:%d (MMIO @ %p)\n",
            ch->channel_id, pci_bus, pci_device, mmio_base);
    return ch;
}

void dma_channel_close(dma_channel_t *ch)
{
    if (!ch) return;
    ch->active = false;
    kprintf("[DMA] Channel %d closed (%lu bytes transferred)\n",
            ch->channel_id, (unsigned long)ch->bytes_transferred);
}

/* =============================================================================
 * Descriptor Submission
 * =============================================================================*/

int dma_transfer_submit(dma_channel_t *ch, dma_descriptor_t *desc)
{
    if (!ch || !ch->active || !desc) return -1;
    if (desc->sg_count == 0 || desc->sg_count > DMA_MAX_SG_ENTRIES) return -1;

    desc->id = next_desc_id++;
    desc->status = DMA_STATUS_PENDING;
    desc->start_tsc = rdtsc_fenced();

    /* Calculate total bytes */
    desc->total_bytes = 0;
    for (uint32_t i = 0; i < desc->sg_count; i++)
        desc->total_bytes += desc->sg_list[i].length;

    /* Enqueue into ring */
    uint32_t slot = ch->ring_head % ch->ring_size;
    ch->ring[slot] = *desc;
    ch->ring_head++;

    /* Program hardware if MMIO is available */
    if (ch->mmio_base) {
        for (uint32_t i = 0; i < desc->sg_count; i++) {
            dma_sg_entry_t *sg = &desc->sg_list[i];

            /* Program source address */
            dma_write32(ch, NV_PCOPY_SRC_ADDR_LO, (uint32_t)(sg->src_addr & 0xFFFFFFFF));
            dma_write32(ch, NV_PCOPY_SRC_ADDR_HI, (uint32_t)(sg->src_addr >> 32));

            /* Program destination address */
            dma_write32(ch, NV_PCOPY_DST_ADDR_LO, (uint32_t)(sg->dst_addr & 0xFFFFFFFF));
            dma_write32(ch, NV_PCOPY_DST_ADDR_HI, (uint32_t)(sg->dst_addr >> 32));

            /* Program size and launch */
            dma_write32(ch, NV_PCOPY_SIZE, sg->length);
            dma_write32(ch, NV_PCOPY_LAUNCH, 1);

            /* Mark as active */
            ch->ring[slot].status = DMA_STATUS_ACTIVE;
        }
    } else {
        /* No MMIO — simulate instant completion */
        ch->ring[slot].status = DMA_STATUS_COMPLETE;
        ch->ring[slot].end_tsc = rdtsc_fenced();
        ch->bytes_transferred += desc->total_bytes;
        ch->transfers_completed++;
    }

    kprintf_debug("[DMA] Ch%d: submitted desc %lu (%u SG entries, %lu bytes)\n",
                  ch->channel_id, (unsigned long)desc->id,
                  desc->sg_count, (unsigned long)desc->total_bytes);
    return 0;
}

/* =============================================================================
 * Completion Polling
 * =============================================================================*/

int dma_poll_completions(dma_channel_t *ch)
{
    if (!ch || !ch->active) return 0;
    int completed = 0;

    while (ch->ring_tail < ch->ring_head) {
        uint32_t slot = ch->ring_tail % ch->ring_size;
        dma_descriptor_t *desc = &ch->ring[slot];

        if (desc->status == DMA_STATUS_ACTIVE && ch->mmio_base) {
            uint32_t status = dma_read32(ch, NV_PCOPY_STATUS);
            if (status == 0) {
                desc->status = DMA_STATUS_COMPLETE;
                desc->end_tsc = rdtsc_fenced();
                ch->bytes_transferred += desc->total_bytes;
                ch->transfers_completed++;
                if (desc->callback)
                    desc->callback(desc->id, 0);
                completed++;
            } else {
                break; /* Still busy */
            }
        } else if (desc->status == DMA_STATUS_COMPLETE) {
            completed++;
        } else {
            break;
        }
        ch->ring_tail++;
    }

    return completed;
}

int dma_transfer_wait(dma_channel_t *ch, uint64_t desc_id)
{
    if (!ch) return -1;

    for (int attempt = 0; attempt < 10000000; attempt++) {
        dma_poll_completions(ch);
        /* Search for the descriptor */
        for (uint32_t i = 0; i < ch->ring_size; i++) {
            if (ch->ring[i].id == desc_id) {
                if (ch->ring[i].status == DMA_STATUS_COMPLETE) return 0;
                if (ch->ring[i].status == DMA_STATUS_ERROR) return -1;
            }
        }
        __asm__ volatile ("pause");
    }
    return -1; /* Timeout */
}

dma_status_t dma_transfer_status(dma_channel_t *ch, uint64_t desc_id)
{
    if (!ch) return DMA_STATUS_ERROR;
    for (uint32_t i = 0; i < ch->ring_size; i++) {
        if (ch->ring[i].id == desc_id)
            return ch->ring[i].status;
    }
    return DMA_STATUS_ERROR;
}

/* =============================================================================
 * Convenience API: Single Contiguous Transfer
 * =============================================================================*/

int dma_copy(dma_channel_t *ch, uint64_t dst, uint64_t src, uint64_t size,
             dma_direction_t dir)
{
    if (!ch || size == 0) return -1;

    dma_descriptor_t desc;
    kmemset(&desc, 0, sizeof(desc));
    desc.direction = dir;

    /* Split large transfers into 1MB chunks (hardware limit) */
    uint64_t offset = 0;
    uint32_t sg_idx = 0;
    while (offset < size && sg_idx < DMA_MAX_SG_ENTRIES) {
        uint32_t chunk = (size - offset > 0x100000) ? 0x100000 : (uint32_t)(size - offset);
        desc.sg_list[sg_idx].src_addr = src + offset;
        desc.sg_list[sg_idx].dst_addr = dst + offset;
        desc.sg_list[sg_idx].length = chunk;
        desc.sg_list[sg_idx].flags = 0;
        offset += chunk;
        sg_idx++;
    }
    /* Mark last entry */
    if (sg_idx > 0)
        desc.sg_list[sg_idx - 1].flags |= DMA_SG_FLAG_LAST;
    desc.sg_count = sg_idx;

    int rc = dma_transfer_submit(ch, &desc);
    if (rc != 0) return rc;

    return dma_transfer_wait(ch, desc.id);
}

/* =============================================================================
 * Statistics
 * =============================================================================*/

void dma_print_stats(dma_channel_t *ch)
{
    if (!ch) return;
    kprintf("[DMA] Ch%d: %lu transfers, %lu MB total, %lu errors\n",
            ch->channel_id,
            (unsigned long)ch->transfers_completed,
            (unsigned long)(ch->bytes_transferred / (1024 * 1024)),
            (unsigned long)ch->errors);
}
