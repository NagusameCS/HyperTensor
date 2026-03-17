/* =============================================================================
 * TensorOS — Intel E1000 NIC Driver Header
 * Supports Intel 82540EM (QEMU -nic e1000) and common real-hardware variants.
 * =============================================================================*/

#ifndef TENSOROS_E1000_H
#define TENSOROS_E1000_H

#include <stdint.h>

/* PCI Vendor/Device IDs for E1000 family */
#define E1000_VENDOR_INTEL      0x8086
#define E1000_DEV_82540EM       0x100E  /* QEMU default */
#define E1000_DEV_82545EM       0x100F
#define E1000_DEV_82574L        0x10D3

/* Register offsets (MMIO) */
#define E1000_CTRL      0x0000  /* Device Control */
#define E1000_STATUS    0x0008  /* Device Status */
#define E1000_EERD      0x0014  /* EEPROM Read */
#define E1000_ICR       0x00C0  /* Interrupt Cause Read */
#define E1000_IMS       0x00D0  /* Interrupt Mask Set */
#define E1000_IMC       0x00D8  /* Interrupt Mask Clear */
#define E1000_RCTL      0x0100  /* Receive Control */
#define E1000_TCTL      0x0400  /* Transmit Control */
#define E1000_RDBAL     0x2800  /* RX Descriptor Base Low */
#define E1000_RDBAH     0x2804  /* RX Descriptor Base High */
#define E1000_RDLEN     0x2808  /* RX Descriptor Length */
#define E1000_RDH       0x2810  /* RX Descriptor Head */
#define E1000_RDT       0x2818  /* RX Descriptor Tail */
#define E1000_TDBAL     0x3800  /* TX Descriptor Base Low */
#define E1000_TDBAH     0x3804  /* TX Descriptor Base High */
#define E1000_TDLEN     0x3808  /* TX Descriptor Length */
#define E1000_TDH       0x3810  /* TX Descriptor Head */
#define E1000_TDT       0x3818  /* TX Descriptor Tail */
#define E1000_RAL0      0x5400  /* Receive Address Low */
#define E1000_RAH0      0x5404  /* Receive Address High */
#define E1000_MTA       0x5200  /* Multicast Table Array */
#define E1000_TIPG      0x0410  /* TX Inter-Packet Gap */

/* CTRL register bits */
#define E1000_CTRL_SLU      (1 << 6)   /* Set Link Up */
#define E1000_CTRL_RST      (1 << 26)  /* Device Reset */

/* RCTL register bits */
#define E1000_RCTL_EN       (1 << 1)   /* Receiver Enable */
#define E1000_RCTL_SBP      (1 << 2)   /* Store Bad Packets */
#define E1000_RCTL_UPE      (1 << 3)   /* Unicast Promiscuous */
#define E1000_RCTL_MPE      (1 << 4)   /* Multicast Promiscuous */
#define E1000_RCTL_BAM      (1 << 15)  /* Broadcast Accept Mode */
#define E1000_RCTL_BSIZE_2K (0 << 16)  /* Buffer Size 2048 */
#define E1000_RCTL_SECRC    (1 << 26)  /* Strip Ethernet CRC */

/* TCTL register bits */
#define E1000_TCTL_EN       (1 << 1)   /* Transmitter Enable */
#define E1000_TCTL_PSP      (1 << 3)   /* Pad Short Packets */

/* RX/TX descriptor status bits */
#define E1000_RXD_STAT_DD   (1 << 0)   /* Descriptor Done */
#define E1000_RXD_STAT_EOP  (1 << 1)   /* End of Packet */
#define E1000_TXD_CMD_EOP   (1 << 0)   /* End of Packet */
#define E1000_TXD_CMD_IFCS  (1 << 1)   /* Insert FCS */
#define E1000_TXD_CMD_RS    (1 << 3)   /* Report Status */
#define E1000_TXD_STAT_DD   (1 << 0)   /* Descriptor Done */

/* Descriptor ring sizes (must be multiple of 8) */
#define E1000_NUM_RX_DESC   32
#define E1000_NUM_TX_DESC   32
#define E1000_RX_BUF_SIZE   2048

/* Receive descriptor */
typedef struct __attribute__((packed)) {
    uint64_t addr;      /* Buffer physical address */
    uint16_t length;    /* Received byte count */
    uint16_t checksum;  /* Packet checksum */
    uint8_t  status;    /* Descriptor status */
    uint8_t  errors;    /* Descriptor errors */
    uint16_t special;   /* VLAN tag */
} e1000_rx_desc_t;

/* Transmit descriptor (legacy) */
typedef struct __attribute__((packed)) {
    uint64_t addr;      /* Buffer physical address */
    uint16_t length;    /* Packet length */
    uint8_t  cso;       /* Checksum offset */
    uint8_t  cmd;       /* Command field */
    uint8_t  status;    /* Status */
    uint8_t  css;       /* Checksum start */
    uint16_t special;   /* VLAN tag */
} e1000_tx_desc_t;

/* Driver API */
int  e1000_init(void);          /* Probe PCI, init if found */
int  e1000_send(const uint8_t *data, uint16_t len);
void e1000_poll(void (*rx_callback)(const uint8_t *data, uint16_t len));
void e1000_get_mac(uint8_t mac[6]);
int  e1000_link_up(void);

#endif /* TENSOROS_E1000_H */
