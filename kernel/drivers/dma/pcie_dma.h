/* =============================================================================
 * TensorOS - PCIe DMA Engine Interface
 * Scatter-gather DMA for high-throughput PCIe data transfers between
 * host memory and device VRAM.  Supports NVIDIA, AMD, and generic PCIe.
 * =============================================================================*/

#ifndef TENSOROS_PCIE_DMA_H
#define TENSOROS_PCIE_DMA_H

#include "kernel/core/kernel.h"

/* Maximum scatter-gather list entries per transfer */
#define DMA_MAX_SG_ENTRIES  64

/* Maximum DMA channels (one per PCIe device endpoint) */
#define DMA_MAX_CHANNELS    16

/* DMA transfer direction */
typedef enum {
    DMA_DIR_H2D = 0,       /* Host → Device */
    DMA_DIR_D2H = 1,       /* Device → Host */
    DMA_DIR_D2D = 2,       /* Device → Device (P2P) */
} dma_direction_t;

/* DMA transfer status */
typedef enum {
    DMA_STATUS_IDLE = 0,
    DMA_STATUS_PENDING,
    DMA_STATUS_ACTIVE,
    DMA_STATUS_COMPLETE,
    DMA_STATUS_ERROR,
} dma_status_t;

/* Scatter-gather list entry */
typedef struct {
    uint64_t    src_addr;       /* Source physical/VRAM address */
    uint64_t    dst_addr;       /* Destination physical/VRAM address */
    uint32_t    length;         /* Bytes to transfer */
    uint32_t    flags;          /* Entry flags (last, interrupt, etc.) */
} dma_sg_entry_t;

#define DMA_SG_FLAG_LAST    (1 << 0)   /* Last entry in chain */
#define DMA_SG_FLAG_IRQ     (1 << 1)   /* Generate interrupt on completion */

/* DMA descriptor (submitted to hardware ring) */
typedef struct {
    uint64_t        id;
    dma_direction_t direction;
    dma_status_t    status;
    dma_sg_entry_t  sg_list[DMA_MAX_SG_ENTRIES];
    uint32_t        sg_count;
    uint64_t        total_bytes;
    uint64_t        start_tsc;
    uint64_t        end_tsc;
    void           (*callback)(uint64_t id, int status);
} dma_descriptor_t;

/* DMA channel (one per device endpoint) */
typedef struct {
    uint32_t    channel_id;
    uint32_t    pci_bus;
    uint32_t    pci_device;
    void       *mmio_base;          /* DMA engine MMIO base */
    bool        active;

    /* Descriptor ring */
    dma_descriptor_t ring[64];
    uint32_t    ring_head;          /* Next to submit */
    uint32_t    ring_tail;          /* Next to complete */
    uint32_t    ring_size;

    /* Statistics */
    uint64_t    bytes_transferred;
    uint64_t    transfers_completed;
    uint64_t    errors;
} dma_channel_t;

/* =============================================================================
 * DMA Engine API
 * =============================================================================*/

int  dma_engine_init(void);
dma_channel_t *dma_channel_open(uint32_t pci_bus, uint32_t pci_device, void *mmio_base);
void dma_channel_close(dma_channel_t *ch);

int  dma_transfer_submit(dma_channel_t *ch, dma_descriptor_t *desc);
int  dma_transfer_wait(dma_channel_t *ch, uint64_t desc_id);
dma_status_t dma_transfer_status(dma_channel_t *ch, uint64_t desc_id);

/* Convenience: single contiguous transfer (builds SG list internally) */
int  dma_copy(dma_channel_t *ch, uint64_t dst, uint64_t src, uint64_t size,
              dma_direction_t dir);

/* Poll for completions (called from interrupt handler or idle loop) */
int  dma_poll_completions(dma_channel_t *ch);

/* Statistics */
void dma_print_stats(dma_channel_t *ch);

#endif /* TENSOROS_PCIE_DMA_H */
