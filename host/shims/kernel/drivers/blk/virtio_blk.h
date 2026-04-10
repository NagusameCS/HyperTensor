/* Shim: stub virtio_blk for hosted mode (no real block device) */
#ifndef SHIM_VIRTIO_BLK_H
#define SHIM_VIRTIO_BLK_H

#include <stdint.h>

static inline uint64_t virtio_blk_capacity(void) { return 0; }
static inline int virtio_blk_read(uint64_t sector, uint32_t count, void *buf) {
    (void)sector; (void)count; (void)buf;
    return -1;
}

#endif
