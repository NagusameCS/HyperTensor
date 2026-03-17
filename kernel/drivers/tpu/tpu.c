/* =============================================================================
 * TensorOS - TPU Driver Implementation
 * =============================================================================*/

#include "kernel/drivers/tpu/tpu.h"

static struct tpu_info tpus[TPU_MAX_DEVICES];
static uint32_t tpu_count = 0;

int tpu_detect_and_init(void)
{
    /* TPUs are typically connected via PCIe or custom interconnect.
     * Scan PCI bus for known TPU vendor/device IDs.
     * For now, return 0 as TPUs are not common in standard hardware. */
    tpu_count = 0;
    return tpu_count;
}

struct tpu_info *tpu_get_info(uint32_t tpu_id)
{
    if (tpu_id >= tpu_count) return NULL;
    return &tpus[tpu_id];
}

int tpu_tensor_matmul(uint32_t tpu_id, tensor_desc_t *C,
                       const tensor_desc_t *A, const tensor_desc_t *B)
{
    if (tpu_id >= tpu_count) return -1;
    /* TPU systolic array matrix multiply would be dispatched here */
    kprintf_debug("[TPU] matmul: no firmware loaded for TPU %u\n", tpu_id);
    return -2; /* TPU_ENOSYS: stub not implemented */
}

int tpu_tensor_conv2d(uint32_t tpu_id, tensor_desc_t *output,
                       const tensor_desc_t *input, const tensor_desc_t *kernel)
{
    if (tpu_id >= tpu_count) return -1;
    kprintf_debug("[TPU] conv2d: no firmware loaded for TPU %u\n", tpu_id);
    return -2; /* TPU_ENOSYS: stub not implemented */
}
