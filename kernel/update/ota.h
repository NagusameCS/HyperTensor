/* =============================================================================
 * TensorOS — OTA Update Protocol
 *
 * Two modes:
 *   ota   — Receive kernel over serial/BT, chain-load from RAM (instant dev)
 *   flash — Receive kernel over serial/BT, write to SD card (persistent)
 *
 * Protocol (both directions use serial/BT simultaneously):
 *   PC → Pi: "OTA!" (4 bytes magic)
 *             uint32_t  size        (little-endian, kernel binary size)
 *             uint8_t[] data        (raw kernel binary, 'size' bytes)
 *             uint32_t  crc32       (CRC-32 of data)
 *   Pi → PC: "RDY\n"               (ready to receive)
 *            "OK!\n"                (received + verified)   or
 *            "ERR:<msg>\n"          (error)
 *            "BOOT\n"              (about to chain-load / reboot)
 * =============================================================================*/

#ifndef TENSOROS_OTA_H
#define TENSOROS_OTA_H

#include <stdint.h>

/* Run OTA receive + chain-load (no SD write, RAM only).
 * Does NOT return on success — jumps to the new kernel.
 * Returns <0 on failure. */
int ota_receive_and_chainload(void);

/* Run OTA receive + flash to SD + reboot.
 * Does NOT return on success.
 * Returns <0 on failure. */
int ota_receive_and_flash(void);

#endif /* TENSOROS_OTA_H */
