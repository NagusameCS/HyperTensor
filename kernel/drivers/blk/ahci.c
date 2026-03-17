/* =============================================================================
 * TensorOS — AHCI SATA Disk Driver
 *
 * Standard AHCI 1.0+ controller for real SATA hardware.
 * Supports: IDENTIFY, READ DMA EXT, WRITE DMA EXT.
 * Uses polled I/O (command issue + spin-wait for completion).
 *
 * Reference: Serial ATA AHCI 1.3.1 Specification
 * =============================================================================*/

#include "kernel/drivers/blk/ahci.h"
#include "kernel/core/kernel.h"

/* =============================================================================
 * MMIO access
 * =============================================================================*/

static volatile uint8_t *ahci_abar = NULL;
static int ahci_present = 0;
static uint32_t ahci_ports_impl = 0;
static int ahci_num_ports = 0;

/* Per-port state */
#define AHCI_MAX_PORTS 8
static struct {
    int      active;
    uint64_t sectors;        /* Total sectors (from IDENTIFY) */
    char     model[44];      /* Model string */
} ahci_port_info[AHCI_MAX_PORTS];

/* Static memory for per-port structures (must be aligned) */
static ahci_cmd_header_t cmd_lists[AHCI_MAX_PORTS][32] __attribute__((aligned(1024)));
static ahci_fis_t        fis_areas[AHCI_MAX_PORTS] __attribute__((aligned(256)));
static ahci_cmd_table_t  cmd_tables[AHCI_MAX_PORTS] __attribute__((aligned(128)));
/* Scratch buffer for IDENTIFY */
static uint8_t identify_buf[512] __attribute__((aligned(16)));

static inline uint32_t ahci_read(uint32_t reg)
{
    return *(volatile uint32_t *)(ahci_abar + reg);
}

static inline void ahci_write(uint32_t reg, uint32_t val)
{
    *(volatile uint32_t *)(ahci_abar + reg) = val;
}

static inline uint32_t port_read(uint32_t port, uint32_t reg)
{
    return ahci_read(0x100 + port * 0x80 + reg);
}

static inline void port_write(uint32_t port, uint32_t reg, uint32_t val)
{
    ahci_write(0x100 + port * 0x80 + reg, val);
}

/* =============================================================================
 * PCI access
 * =============================================================================*/

static inline uint32_t pci_read32(uint8_t bus, uint8_t dev, uint8_t func, uint8_t off)
{
    uint32_t addr = 0x80000000 | ((uint32_t)bus << 16) | ((uint32_t)dev << 11) |
                    ((uint32_t)func << 8) | (off & 0xFC);
    outl(0xCF8, addr);
    return inl(0xCFC);
}

static inline void pci_write32(uint8_t bus, uint8_t dev, uint8_t func, uint8_t off, uint32_t val)
{
    uint32_t addr = 0x80000000 | ((uint32_t)bus << 16) | ((uint32_t)dev << 11) |
                    ((uint32_t)func << 8) | (off & 0xFC);
    outl(0xCF8, addr);
    outl(0xCFC, val);
}

/* =============================================================================
 * Stop/start command engine on a port
 * =============================================================================*/

static void port_stop(uint32_t port)
{
    uint32_t cmd = port_read(port, AHCI_PxCMD);
    cmd &= ~(AHCI_PxCMD_ST | AHCI_PxCMD_FRE);
    port_write(port, AHCI_PxCMD, cmd);

    /* Wait for CR and FR to clear */
    int timeout = 500000;
    while (timeout-- > 0) {
        uint32_t c = port_read(port, AHCI_PxCMD);
        if (!(c & AHCI_PxCMD_CR) && !(c & AHCI_PxCMD_FR))
            return;
    }
}

static void port_start(uint32_t port)
{
    /* Wait for CR to clear before starting */
    int timeout = 500000;
    while ((port_read(port, AHCI_PxCMD) & AHCI_PxCMD_CR) && timeout-- > 0) ;

    uint32_t cmd = port_read(port, AHCI_PxCMD);
    cmd |= AHCI_PxCMD_FRE | AHCI_PxCMD_ST;
    port_write(port, AHCI_PxCMD, cmd);
}

/* =============================================================================
 * Issue a command and wait for completion (polled)
 * =============================================================================*/

static int port_issue_cmd(uint32_t port, int slot)
{
    /* Clear any pending errors */
    port_write(port, AHCI_PxSERR, port_read(port, AHCI_PxSERR));

    /* Wait for slot to be free */
    int timeout = 1000000;
    while ((port_read(port, AHCI_PxCI) & (1U << slot)) && timeout-- > 0) ;
    if (timeout <= 0) return -1;

    /* Issue command */
    port_write(port, AHCI_PxCI, 1U << slot);

    /* Wait for completion */
    timeout = 5000000;
    while (timeout-- > 0) {
        uint32_t ci = port_read(port, AHCI_PxCI);
        if (!(ci & (1U << slot))) return 0;  /* Done */

        uint32_t is = port_read(port, AHCI_PxIS);
        if (is & (1 << 30)) {               /* Task File Error */
            kprintf("[AHCI] Port %d command error (TFD=0x%x)\n",
                    port, port_read(port, AHCI_PxTFD));
            return -1;
        }
    }
    return -1; /* Timeout */
}

/* =============================================================================
 * Build a Register H2D FIS for ATA commands
 * =============================================================================*/

static void build_h2d_fis(uint8_t *cfis, uint8_t command, uint64_t lba, uint16_t count)
{
    kmemset(cfis, 0, 20);
    cfis[0] = FIS_TYPE_H2D;
    cfis[1] = 0x80;              /* Command bit set */
    cfis[2] = command;
    cfis[3] = 0;                 /* Features */
    cfis[4] = (uint8_t)(lba & 0xFF);
    cfis[5] = (uint8_t)((lba >> 8) & 0xFF);
    cfis[6] = (uint8_t)((lba >> 16) & 0xFF);
    cfis[7] = 0x40;              /* LBA mode */
    cfis[8] = (uint8_t)((lba >> 24) & 0xFF);
    cfis[9] = (uint8_t)((lba >> 32) & 0xFF);
    cfis[10] = (uint8_t)((lba >> 40) & 0xFF);
    cfis[12] = (uint8_t)(count & 0xFF);
    cfis[13] = (uint8_t)((count >> 8) & 0xFF);
}

/* =============================================================================
 * Initialize a single port
 * =============================================================================*/

static int port_init(uint32_t port)
{
    /* Check if device present */
    uint32_t ssts = port_read(port, AHCI_PxSSTS);
    if ((ssts & AHCI_SSTS_DET_MASK) != AHCI_SSTS_DET_OK)
        return -1;

    /* Check signature — only handle SATA disks */
    uint32_t sig = port_read(port, AHCI_PxSIG);
    if (sig != AHCI_SIG_ATA)
        return -1;

    /* Stop command engine */
    port_stop(port);

    /* Set up command list and FIS receive area */
    uint64_t clb = (uint64_t)(uintptr_t)&cmd_lists[port];
    uint64_t fb  = (uint64_t)(uintptr_t)&fis_areas[port];

    port_write(port, AHCI_PxCLB, (uint32_t)(clb & 0xFFFFFFFF));
    port_write(port, AHCI_PxCLBU, (uint32_t)(clb >> 32));
    port_write(port, AHCI_PxFB, (uint32_t)(fb & 0xFFFFFFFF));
    port_write(port, AHCI_PxFBU, (uint32_t)(fb >> 32));

    kmemset(&cmd_lists[port], 0, sizeof(cmd_lists[port]));
    kmemset(&fis_areas[port], 0, sizeof(ahci_fis_t));

    /* Set up command header slot 0 → cmd_table */
    uint64_t ctba = (uint64_t)(uintptr_t)&cmd_tables[port];
    cmd_lists[port][0].ctba = ctba;
    cmd_lists[port][0].opts = 0;
    cmd_lists[port][0].prdtl = 0;

    /* Start command engine */
    port_start(port);

    /* Clear pending interrupts */
    port_write(port, AHCI_PxIS, 0xFFFFFFFF);

    /* Issue IDENTIFY to get disk geometry */
    kmemset(&cmd_tables[port], 0, sizeof(ahci_cmd_table_t));
    build_h2d_fis(cmd_tables[port].cfis, ATA_CMD_IDENTIFY, 0, 0);

    /* PRDT: one entry pointing to identify_buf */
    cmd_tables[port].prdt[0].dba = (uint64_t)(uintptr_t)identify_buf;
    cmd_tables[port].prdt[0].dbc = 511;  /* 512 bytes - 1 (bit 0 = interrupt on completion) */
    cmd_tables[port].prdt[0].dbc |= 0;   /* No interrupt */

    cmd_lists[port][0].opts = (5 << 0);  /* CFL = 5 DWORDs (H2D FIS is 5 DWORDs) */
    cmd_lists[port][0].prdtl = 1;
    cmd_lists[port][0].prdbc = 0;

    if (port_issue_cmd(port, 0) < 0) {
        kprintf("[AHCI] Port %d IDENTIFY failed\n", port);
        return -1;
    }

    /* Parse IDENTIFY data */
    uint16_t *id = (uint16_t *)identify_buf;

    /* Total sectors (LBA48): words 100-103 */
    uint64_t sectors = (uint64_t)id[100] | ((uint64_t)id[101] << 16) |
                       ((uint64_t)id[102] << 32) | ((uint64_t)id[103] << 48);
    if (sectors == 0) {
        /* Fall back to LBA28: words 60-61 */
        sectors = (uint64_t)id[60] | ((uint64_t)id[61] << 16);
    }

    /* Model string: words 27-46 (byte-swapped) */
    char *model = ahci_port_info[port].model;
    for (int i = 0; i < 20 && i < 43; i++) {
        uint16_t w = id[27 + i];
        model[i * 2] = (char)(w >> 8);
        model[i * 2 + 1] = (char)(w & 0xFF);
    }
    model[40] = '\0';
    /* Trim trailing spaces */
    for (int i = 39; i >= 0 && model[i] == ' '; i--)
        model[i] = '\0';

    ahci_port_info[port].active = 1;
    ahci_port_info[port].sectors = sectors;

    kprintf("[AHCI] Port %d: %s, %lu MB\n", port, model,
            (unsigned long)(sectors / 2048));
    return 0;
}

/* =============================================================================
 * Top-level init: probe PCI for AHCI controller
 * =============================================================================*/

int ahci_init(void)
{
    int found_dev = -1;

    for (int dev = 0; dev < 32; dev++) {
        uint32_t id = pci_read32(0, dev, 0, 0);
        if ((id & 0xFFFF) == 0xFFFF) continue;

        /* Check class code: offset 0x08, bits [31:8] = class/subclass/progif */
        uint32_t class_reg = pci_read32(0, dev, 0, 0x08);
        uint8_t cls    = (class_reg >> 24) & 0xFF;
        uint8_t subcls = (class_reg >> 16) & 0xFF;
        uint8_t progif = (class_reg >> 8) & 0xFF;

        if (cls == PCI_CLASS_STORAGE && subcls == PCI_SUBCLASS_SATA &&
            progif == PCI_PROGIF_AHCI) {
            found_dev = dev;
            kprintf("[AHCI] Found AHCI controller at PCI 0:%d.0\n", dev);
            break;
        }
    }

    if (found_dev < 0) {
        /* No AHCI controller found — not an error on QEMU with virtio */
        return -1;
    }

    /* Read ABAR (BAR5) */
    uint32_t bar5 = pci_read32(0, found_dev, 0, 0x24);
    uint64_t abar = bar5 & ~0xFULL;
    ahci_abar = (volatile uint8_t *)(uintptr_t)abar;

    /* Enable bus mastering + memory space */
    uint32_t cmd = pci_read32(0, found_dev, 0, 0x04);
    cmd |= (1 << 1) | (1 << 2);
    pci_write32(0, found_dev, 0, 0x04, cmd);

    /* Enable AHCI mode */
    ahci_write(AHCI_GHC, ahci_read(AHCI_GHC) | AHCI_GHC_AE);

    /* Get ports implemented */
    ahci_ports_impl = ahci_read(AHCI_PI);

    uint32_t ver = ahci_read(AHCI_VS);
    kprintf("[AHCI] Version %d.%d, ABAR=0x%lx, PI=0x%x\n",
            (ver >> 16) & 0xFFFF, ver & 0xFFFF, abar, ahci_ports_impl);

    /* Initialize each implemented port */
    kmemset(ahci_port_info, 0, sizeof(ahci_port_info));
    for (int p = 0; p < AHCI_MAX_PORTS; p++) {
        if (ahci_ports_impl & (1U << p)) {
            if (port_init(p) == 0)
                ahci_num_ports++;
        }
    }

    if (ahci_num_ports > 0) {
        ahci_present = 1;
        kprintf("[AHCI] %d SATA disk(s) ready\n", ahci_num_ports);
    } else {
        kprintf("[AHCI] No SATA disks found\n");
    }
    return ahci_num_ports > 0 ? 0 : -1;
}

/* =============================================================================
 * Read sectors via DMA
 * =============================================================================*/

int ahci_read_sectors(uint32_t port, uint64_t lba, uint32_t count, void *buf)
{
    if (!ahci_present || port >= AHCI_MAX_PORTS || !ahci_port_info[port].active)
        return -1;
    if (count == 0 || count > 128) return -1;  /* Max 64KB per command */

    kmemset(&cmd_tables[port], 0, sizeof(ahci_cmd_table_t));
    build_h2d_fis(cmd_tables[port].cfis, ATA_CMD_READ_DMA_EX, lba, count);

    /* Set up PRDT */
    uint32_t byte_count = count * 512;
    cmd_tables[port].prdt[0].dba = (uint64_t)(uintptr_t)buf;
    cmd_tables[port].prdt[0].dbc = byte_count - 1;

    /* Command header */
    cmd_lists[port][0].opts = (5 << 0);  /* CFL = 5 */
    cmd_lists[port][0].prdtl = 1;
    cmd_lists[port][0].prdbc = 0;

    return port_issue_cmd(port, 0);
}

/* =============================================================================
 * Write sectors via DMA
 * =============================================================================*/

int ahci_write_sectors(uint32_t port, uint64_t lba, uint32_t count, const void *buf)
{
    if (!ahci_present || port >= AHCI_MAX_PORTS || !ahci_port_info[port].active)
        return -1;
    if (count == 0 || count > 128) return -1;

    kmemset(&cmd_tables[port], 0, sizeof(ahci_cmd_table_t));
    build_h2d_fis(cmd_tables[port].cfis, ATA_CMD_WRITE_DMA_EX, lba, count);

    uint32_t byte_count = count * 512;
    cmd_tables[port].prdt[0].dba = (uint64_t)(uintptr_t)buf;
    cmd_tables[port].prdt[0].dbc = byte_count - 1;

    /* Command header: Write bit set */
    cmd_lists[port][0].opts = (5 << 0) | (1 << 6);  /* CFL=5, W=1 */
    cmd_lists[port][0].prdtl = 1;
    cmd_lists[port][0].prdbc = 0;

    return port_issue_cmd(port, 0);
}

/* =============================================================================
 * Query functions
 * =============================================================================*/

int ahci_port_count(void) { return ahci_num_ports; }

uint64_t ahci_disk_sectors(uint32_t port)
{
    if (port >= AHCI_MAX_PORTS || !ahci_port_info[port].active) return 0;
    return ahci_port_info[port].sectors;
}
