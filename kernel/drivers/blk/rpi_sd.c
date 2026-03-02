/* =============================================================================
 * TensorOS — Minimal BCM2711 EMMC2 SD Card Driver
 *
 * BCM2711 EMMC2 controller at 0xFE340000 (SDHCI-compatible).
 * Supports SD cards in 1-bit SD mode at 25 MHz (good enough for OTA).
 *
 * References:
 *   - SD Host Controller Simplified Specification v3.00
 *   - BCM2711 ARM Peripherals (section on EMMC2)
 *   - Linux bcm2835-mmc / sdhci-iproc drivers
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/drivers/blk/rpi_sd.h"

#if defined(__aarch64__)

/* =============================================================================
 * EMMC2 Register Definitions (SDHCI layout)
 * =============================================================================*/

#define EMMC2_BASE          0xFE340000UL

#define EMMC2_ARG2          (EMMC2_BASE + 0x00)
#define EMMC2_BLKSIZECNT    (EMMC2_BASE + 0x04)
#define EMMC2_ARG1          (EMMC2_BASE + 0x08)
#define EMMC2_CMDTM         (EMMC2_BASE + 0x0C)
#define EMMC2_RESP0         (EMMC2_BASE + 0x10)
#define EMMC2_RESP1         (EMMC2_BASE + 0x14)
#define EMMC2_RESP2         (EMMC2_BASE + 0x18)
#define EMMC2_RESP3         (EMMC2_BASE + 0x1C)
#define EMMC2_DATA          (EMMC2_BASE + 0x20)
#define EMMC2_STATUS        (EMMC2_BASE + 0x24)
#define EMMC2_CONTROL0      (EMMC2_BASE + 0x28)
#define EMMC2_CONTROL1      (EMMC2_BASE + 0x2C)
#define EMMC2_INTERRUPT     (EMMC2_BASE + 0x30)
#define EMMC2_IRPT_MASK     (EMMC2_BASE + 0x34)
#define EMMC2_IRPT_EN       (EMMC2_BASE + 0x38)
#define EMMC2_CONTROL2      (EMMC2_BASE + 0x3C)
#define EMMC2_SLOTISR_VER   (EMMC2_BASE + 0xFC)

/* Status register bits */
#define SR_CMD_INHIBIT      0x00000001
#define SR_DAT_INHIBIT      0x00000002
#define SR_READ_AVAILABLE   0x00000800
#define SR_WRITE_AVAILABLE  0x00000400

/* Interrupt bits */
#define INT_CMD_DONE        0x00000001
#define INT_DATA_DONE       0x00000002
#define INT_WRITE_RDY       0x00000010
#define INT_READ_RDY        0x00000020
#define INT_ERROR           0x00008000
#define INT_ERR_MASK        0xFFFF0000
#define INT_ALL             0xFFFF003F

/* Command flags */
#define CMD_NEED_APP        0x80000000
#define CMD_RSPNS_48        0x00020000
#define CMD_RSPNS_136       0x00010000
#define CMD_RSPNS_48B       0x00030000
#define CMD_ISDATA          0x00200000
#define CMD_DAT_RD          0x00000010
#define CMD_DAT_WR          0x00000000
#define CMD_CRCCHK_EN       0x00080000
#define CMD_IXCHK_EN        0x00100000

/* SD commands */
#define SD_CMD_GO_IDLE          0x00000000
#define SD_CMD_SEND_IF_COND     (0x08000000 | CMD_RSPNS_48 | CMD_CRCCHK_EN | CMD_IXCHK_EN)
#define SD_CMD_SEND_CSD         (0x09000000 | CMD_RSPNS_136)
#define SD_CMD_STOP_TRANS       (0x0C000000 | CMD_RSPNS_48B)
#define SD_CMD_SET_BLOCKCNT     (0x17000000 | CMD_RSPNS_48)
#define SD_CMD_READ_SINGLE      (0x11000000 | CMD_RSPNS_48 | CMD_ISDATA | CMD_DAT_RD | CMD_CRCCHK_EN)
#define SD_CMD_READ_MULTI       (0x12000000 | CMD_RSPNS_48 | CMD_ISDATA | CMD_DAT_RD | CMD_CRCCHK_EN)
#define SD_CMD_WRITE_SINGLE     (0x18000000 | CMD_RSPNS_48 | CMD_ISDATA | CMD_DAT_WR | CMD_CRCCHK_EN)
#define SD_CMD_WRITE_MULTI      (0x19000000 | CMD_RSPNS_48 | CMD_ISDATA | CMD_DAT_WR | CMD_CRCCHK_EN)
#define SD_CMD_APP_CMD          (0x37000000 | CMD_RSPNS_48)
#define SD_CMD_SEND_OP_COND     (0x29000000 | CMD_RSPNS_48 | CMD_NEED_APP)
#define SD_CMD_SEND_REL_ADDR    (0x03000000 | CMD_RSPNS_48)
#define SD_CMD_SELECT_CARD      (0x07000000 | CMD_RSPNS_48B)
#define SD_CMD_SET_BUS_WIDTH    (0x06000000 | CMD_RSPNS_48 | CMD_NEED_APP)
#define SD_CMD_SEND_STATUS      (0x0D000000 | CMD_RSPNS_48)

/* Card state */
uint32_t sd_rca           = 0;   /* non-static: used by main.c diagnostic */
static int      sd_sdhc          = 0;   /* 1 if SDHC/SDXC (block addressing) */
static int      sd_initialized   = 0;

/* ---------- Mailbox helpers for EMMC2 clocks ---------- */

/* Mailbox property tags for clock/power management */
#define MBOX_TAG_SET_POWER_STATE 0x00028001
#define MBOX_TAG_SET_CLOCK_RATE  0x00038002
#define MBOX_TAG_GET_CLOCK_RATE  0x00030002
#define MBOX_DEV_SD_CARD         0x00000000  /* SD card device ID */
#define MBOX_CLK_EMMC2           0x0000000C  /* EMMC2 clock ID */

/* Short UART diagnostic helper (works before kprintf is available) */
static void sd_uart_str(const char *s) {
    while (*s) {
        if (*s == '\n') uart_putchar('\r');
        uart_putchar(*s++);
    }
}
static void sd_uart_hex(uint32_t v) {
    static const char hex[] = "0123456789ABCDEF";
    uart_putchar('0'); uart_putchar('x');
    for (int i = 28; i >= 0; i -= 4)
        uart_putchar(hex[(v >> i) & 0xF]);
}
static void sd_uart_dec(int v) {
    if (v < 0) { uart_putchar('-'); v = -v; }
    char buf[12]; int n = 0;
    if (v == 0) buf[n++] = '0';
    else while (v > 0) { buf[n++] = '0' + (v % 10); v /= 10; }
    for (int i = n - 1; i >= 0; i--) uart_putchar(buf[i]);
}

/*
 * Set EMMC2 clock to 200 MHz via VideoCore mailbox.
 * This is REQUIRED on RPi4 before the ARM can talk to EMMC2.
 * Returns 0 on success.
 */
static int sd_setup_emmc2_clock(void)
{
    /* Power on the SD card device */
    {
        volatile uint32_t __attribute__((aligned(16))) mb[8];
        mb[0] = 8 * 4;               /* buffer size */
        mb[1] = 0;                    /* request */
        mb[2] = MBOX_TAG_SET_POWER_STATE;
        mb[3] = 8;                    /* value buffer size */
        mb[4] = 8;                    /* request: 8 bytes */
        mb[5] = MBOX_DEV_SD_CARD;     /* device: SD card */
        mb[6] = 0x03;                 /* state: on | wait */
        mb[7] = 0;                    /* end tag */
        if (!mbox_call(8, mb)) {
            sd_uart_str("[SD] mbox power-on FAIL\n");
            return -1;
        }
        if (!(mb[6] & 0x01)) {
            sd_uart_str("[SD] SD card device not powered\n");
            return -2;
        }
    }

    /* Set EMMC2 clock to 200 MHz */
    {
        volatile uint32_t __attribute__((aligned(16))) mb[9];
        mb[0] = 9 * 4;
        mb[1] = 0;
        mb[2] = MBOX_TAG_SET_CLOCK_RATE;
        mb[3] = 12;                   /* value buffer: 3 words */
        mb[4] = 12;                   /* request */
        mb[5] = MBOX_CLK_EMMC2;      /* clock: EMMC2 */
        mb[6] = 200000000;            /* rate: 200 MHz */
        mb[7] = 0;                    /* skip turbo */
        mb[8] = 0;                    /* end tag */
        if (!mbox_call(8, mb)) {
            sd_uart_str("[SD] mbox set-clock FAIL\n");
            return -3;
        }
        sd_uart_str("[SD] EMMC2 clock set to ");
        sd_uart_dec(mb[6] / 1000000);
        sd_uart_str(" MHz\n");
    }

    return 0;
}

/* ---------- Low-level helpers ---------- */

static void sd_delay(uint32_t us)
{
    arm_timer_delay_us(us);
}

static int sd_wait_cmd(void)
{
    uint64_t deadline = arm_timer_count() + arm_timer_freq();  /* 1s timeout */
    while (arm_timer_count() < deadline) {
        uint32_t irq = mmio_read(EMMC2_INTERRUPT);
        if (irq & INT_ERROR) {
            mmio_write(EMMC2_INTERRUPT, irq);  /* clear */
            return -1;
        }
        if (irq & INT_CMD_DONE) {
            mmio_write(EMMC2_INTERRUPT, INT_CMD_DONE);
            return 0;
        }
    }
    return -2;  /* timeout */
}

static int sd_wait_data(void)
{
    uint64_t deadline = arm_timer_count() + arm_timer_freq() * 2;
    while (arm_timer_count() < deadline) {
        uint32_t irq = mmio_read(EMMC2_INTERRUPT);
        if (irq & INT_ERROR) {
            mmio_write(EMMC2_INTERRUPT, irq);
            return -1;
        }
        if (irq & INT_DATA_DONE) {
            mmio_write(EMMC2_INTERRUPT, INT_DATA_DONE);
            return 0;
        }
    }
    return -2;
}

static int sd_send_cmd(uint32_t cmd, uint32_t arg)
{
    /* Wait for CMD line free */
    uint64_t deadline = arm_timer_count() + arm_timer_freq();
    while (mmio_read(EMMC2_STATUS) & SR_CMD_INHIBIT) {
        if (arm_timer_count() > deadline) return -1;
    }

    /* If data command, wait for DAT line free */
    if (cmd & CMD_ISDATA) {
        while (mmio_read(EMMC2_STATUS) & SR_DAT_INHIBIT) {
            if (arm_timer_count() > deadline) return -1;
        }
    }

    /* Handle APP commands (CMD55 + actual command) */
    if (cmd & CMD_NEED_APP) {
        uint32_t app_arg = sd_rca ? (sd_rca << 16) : 0;
        mmio_write(EMMC2_INTERRUPT, INT_ALL);
        mmio_write(EMMC2_ARG1, app_arg);
        mmio_write(EMMC2_CMDTM, SD_CMD_APP_CMD & ~CMD_NEED_APP);
        if (sd_wait_cmd() != 0) return -1;
    }

    /* Clear interrupt flags */
    mmio_write(EMMC2_INTERRUPT, INT_ALL);

    /* Send command */
    mmio_write(EMMC2_ARG1, arg);
    mmio_write(EMMC2_CMDTM, cmd & ~CMD_NEED_APP);

    return sd_wait_cmd();
}

/* =============================================================================
 * Initialization — Two-stage approach:
 *   1. Try to reuse the firmware's existing EMMC2 configuration (fast path).
 *      The RPi4 GPU firmware already initialised the SD card to load us, so
 *      the controller+card are in Transfer mode at 25-50 MHz.  We just set
 *      up interrupt masks, assume SDHC, and test-read sector 0.
 *   2. If that fails, fall back to a full SDHCI reset + card enumeration.
 * =============================================================================*/

static int sd_init_full(void);   /* forward decl — full reset path */

int sd_init(void)
{
    sd_uart_str("[SD] sd_init start\n");

    /* Step 0: Ask VideoCore to power on the SD card device and
     *         set the EMMC2 base clock to 200 MHz.  Without this
     *         the EMMC2 registers are dead on BCM2711. */
    int clk_rc = sd_setup_emmc2_clock();
    if (clk_rc != 0) {
        sd_uart_str("[SD] clock setup failed rc=");
        sd_uart_dec(clk_rc);
        sd_uart_str("\n");
        /* Non-fatal: try anyway in case firmware already did it */
    }

    sd_delay(5000);   /* let clock stabilise */

    /* ---- Fast path: reuse firmware state ---- */
    sd_uart_str("[SD] fast path: probe EMMC2 status=");
    sd_uart_hex(mmio_read(EMMC2_STATUS));
    sd_uart_str(" ctrl1=");
    sd_uart_hex(mmio_read(EMMC2_CONTROL1));
    sd_uart_str("\n");

    /* Wait for any in-flight command to finish */
    uint64_t deadline = arm_timer_count() + arm_timer_freq() / 10; /* 100 ms */
    while (mmio_read(EMMC2_STATUS) & (SR_CMD_INHIBIT | SR_DAT_INHIBIT)) {
        if (arm_timer_count() > deadline) break;
    }

    /* Configure interrupt masks for polling */
    mmio_write(EMMC2_IRPT_MASK, INT_ALL);
    mmio_write(EMMC2_IRPT_EN, 0);
    mmio_write(EMMC2_INTERRUPT, INT_ALL);

    /* Block size 512, block count 1 */
    mmio_write(EMMC2_BLKSIZECNT, (1 << 16) | 512);

    /* Assume SDHC */
    sd_sdhc = 1;
    sd_rca  = 0;
    sd_initialized = 1;

    /* Test: read MBR (sector 0) */
    {
        uint8_t mbr[512];
        int rd = sd_read_sector(0, mbr);
        sd_uart_str("[SD] fast read MBR rc=");
        sd_uart_dec(rd);
        if (rd == 0) {
            sd_uart_str(" sig=");
            sd_uart_hex((uint32_t)mbr[510] << 8 | mbr[511]);
        }
        sd_uart_str("\n");
        if (rd == 0 && mbr[510] == 0x55 && mbr[511] == 0xAA) {
            sd_uart_str("[SD] fast path OK\n");
            return 0;
        }
    }

    /* ---- Slow path: full controller reset + card enumeration ---- */
    sd_uart_str("[SD] fast path failed, trying full init\n");
    sd_initialized = 0;
    return sd_init_full();
}

static int sd_init_full(void)
{
    sd_uart_str("[SD] full init: resetting controller\n");

    /* Reset controller */
    uint32_t c1 = mmio_read(EMMC2_CONTROL1);
    c1 |= (1 << 24);  /* SRST_HC — reset host circuit */
    mmio_write(EMMC2_CONTROL1, c1);
    sd_delay(20000);   /* 20ms for reset */

    /* Wait for reset complete */
    uint64_t deadline = arm_timer_count() + arm_timer_freq() * 2;  /* 2s */
    while (mmio_read(EMMC2_CONTROL1) & (1 << 24)) {
        if (arm_timer_count() > deadline) {
            sd_uart_str("[SD] full init FAIL: reset timeout\n");
            return -1;
        }
    }
    sd_uart_str("[SD] controller reset OK\n");

    /* Enable internal clock */
    c1 = mmio_read(EMMC2_CONTROL1);
    c1 |= (1 << 0);   /* CLK_INTLEN — internal clock enable */
    /* Set divider for ~400 kHz identification mode.
     * Base clock is 200 MHz on BCM2711 EMMC2.
     * Divider = 200000000 / (2 * 400000) = 250 → use 0xFA in upper bits */
    c1 &= ~0x0000FFC0;
    c1 |= (0xFA << 8) | (0 << 6);  /* SDCLK divider */
    c1 |= (0xE << 16);  /* data timeout = max */
    mmio_write(EMMC2_CONTROL1, c1);
    sd_delay(10000);

    /* Wait for clock stable */
    deadline = arm_timer_count() + arm_timer_freq();
    while (!(mmio_read(EMMC2_CONTROL1) & (1 << 1))) {
        if (arm_timer_count() > deadline) return -2;
    }

    /* Enable SD clock */
    c1 = mmio_read(EMMC2_CONTROL1);
    c1 |= (1 << 2);  /* CLK_EN */
    mmio_write(EMMC2_CONTROL1, c1);
    sd_delay(10000);

    /* Enable all interrupt flags (masked, we poll) */
    mmio_write(EMMC2_IRPT_MASK, INT_ALL);
    mmio_write(EMMC2_IRPT_EN, 0);  /* We poll, no IRQ delivery */
    mmio_write(EMMC2_INTERRUPT, INT_ALL);

    /* CMD0 — GO_IDLE_STATE */
    if (sd_send_cmd(SD_CMD_GO_IDLE, 0) != 0) return -3;
    sd_delay(5000);

    /* CMD8 — SEND_IF_COND (voltage check, required for SDHC) */
    if (sd_send_cmd(SD_CMD_SEND_IF_COND, 0x000001AA) != 0) return -4;
    uint32_t r8 = mmio_read(EMMC2_RESP0);
    if ((r8 & 0xFFF) != 0x1AA) return -5;  /* Pattern mismatch */

    /* ACMD41 — Send Operating Conditions (loop until card ready) */
    deadline = arm_timer_count() + arm_timer_freq() * 3;  /* 3s timeout */
    while (1) {
        /* HCS=1 (support SDHC), 3.3V */
        if (sd_send_cmd(SD_CMD_SEND_OP_COND, 0x40FF8000) != 0)
            return -6;
        uint32_t ocr = mmio_read(EMMC2_RESP0);
        if (ocr & (1u << 31)) {  /* Card power-up complete */
            sd_sdhc = (ocr & (1u << 30)) ? 1 : 0;
            break;
        }
        if (arm_timer_count() > deadline) return -7;  /* Card didn't initialize */
        sd_delay(10000);
    }

    /* CMD2 — ALL_SEND_CID (enter identification) */
    mmio_write(EMMC2_INTERRUPT, INT_ALL);
    mmio_write(EMMC2_ARG1, 0);
    mmio_write(EMMC2_CMDTM, 0x02000000 | CMD_RSPNS_136);
    sd_wait_cmd();

    /* CMD3 — SEND_RELATIVE_ADDR (get RCA) */
    if (sd_send_cmd(SD_CMD_SEND_REL_ADDR, 0) != 0) return -8;
    sd_rca = (mmio_read(EMMC2_RESP0) >> 16) & 0xFFFF;
    if (sd_rca == 0) return -9;

    /* CMD7 — SELECT_CARD (transfer mode) */
    if (sd_send_cmd(SD_CMD_SELECT_CARD, sd_rca << 16) != 0) return -10;

    /* Set block size to 512 */
    mmio_write(EMMC2_BLKSIZECNT, (1 << 16) | 512);

    /* Speed up clock to ~25 MHz for data transfer.
     * Divider = 200000000 / (2 * 25000000) = 4 */
    c1 = mmio_read(EMMC2_CONTROL1);
    c1 &= ~0x0000FFC0;
    c1 |= (4 << 8);
    mmio_write(EMMC2_CONTROL1, c1);
    sd_delay(10000);

    sd_initialized = 1;
    return 0;
}

/* =============================================================================
 * Read / Write Sectors
 * =============================================================================*/

/* After a write, poll CMD13 until the card exits programming state.
 * Card status bits 12:9 encode the state:
 *   4 = tran (ready), 7 = prg (programming flash)
 * We also wait for DAT_INHIBIT to clear (card releases DAT0). */
static int sd_wait_write_complete(void)
{
    uint64_t deadline = arm_timer_count() + arm_timer_freq() * 5;  /* 5 sec max */
    while (arm_timer_count() < deadline) {
        /* Also check DAT_INHIBIT — card holds DAT0 low while programming */
        if (mmio_read(EMMC2_STATUS) & SR_DAT_INHIBIT) {
            sd_delay(100);
            continue;
        }
        /* Send CMD13 SEND_STATUS with RCA */
        if (sd_send_cmd(SD_CMD_SEND_STATUS, sd_rca << 16) != 0) {
            sd_delay(1000);
            continue;
        }
        uint32_t status = mmio_read(EMMC2_RESP0);
        uint32_t state = (status >> 9) & 0xF;
        if (state == 4) return 0;  /* tran = ready, programming complete */
        if (status & 0xFDF90008) return -1;  /* error bits set */
        sd_delay(100);
    }
    return -2;  /* timeout */
}

int sd_read_sector(uint32_t lba, void *buf)
{
    if (!sd_initialized) return -1;
    uint8_t *bbuf = (uint8_t *)buf;
    uint32_t addr = sd_sdhc ? lba : (lba * 512);

    mmio_write(EMMC2_BLKSIZECNT, (1 << 16) | 512);

    if (sd_send_cmd(SD_CMD_READ_SINGLE, addr) != 0) return -2;

    /* Wait for read ready */
    uint64_t deadline = arm_timer_count() + arm_timer_freq();
    while (!(mmio_read(EMMC2_INTERRUPT) & INT_READ_RDY)) {
        if (mmio_read(EMMC2_INTERRUPT) & INT_ERROR) {
            mmio_write(EMMC2_INTERRUPT, INT_ALL);
            return -3;
        }
        if (arm_timer_count() > deadline) return -4;
    }
    mmio_write(EMMC2_INTERRUPT, INT_READ_RDY);

    /* Read 128 x 32-bit words = 512 bytes */
    uint32_t *d = (uint32_t *)bbuf;
    for (int i = 0; i < 128; i++)
        d[i] = mmio_read(EMMC2_DATA);

    if (sd_wait_data() != 0) return -5;
    return 0;
}

int sd_write_sector(uint32_t lba, const void *buf)
{
    if (!sd_initialized) return -1;
    const uint8_t *bbuf = (const uint8_t *)buf;
    uint32_t addr = sd_sdhc ? lba : (lba * 512);

    mmio_write(EMMC2_BLKSIZECNT, (1 << 16) | 512);

    if (sd_send_cmd(SD_CMD_WRITE_SINGLE, addr) != 0) return -2;

    /* Wait for write ready */
    uint64_t deadline = arm_timer_count() + arm_timer_freq();
    while (!(mmio_read(EMMC2_INTERRUPT) & INT_WRITE_RDY)) {
        if (mmio_read(EMMC2_INTERRUPT) & INT_ERROR) {
            mmio_write(EMMC2_INTERRUPT, INT_ALL);
            return -3;
        }
        if (arm_timer_count() > deadline) return -4;
    }
    mmio_write(EMMC2_INTERRUPT, INT_WRITE_RDY);

    /* Write 128 x 32-bit words = 512 bytes */
    const uint32_t *d = (const uint32_t *)bbuf;
    for (int i = 0; i < 128; i++)
        mmio_write(EMMC2_DATA, d[i]);

    if (sd_wait_data() != 0) return -5;

    /* Wait for card to finish programming flash (critical!) */
    if (sd_wait_write_complete() != 0) return -6;

    return 0;
}

int sd_read_sectors(uint32_t lba, uint32_t count, void *buf)
{
    uint8_t *bbuf = (uint8_t *)buf;
    for (uint32_t i = 0; i < count; i++) {
        int r = sd_read_sector(lba + i, bbuf + i * 512);
        if (r != 0) return r;
    }
    return 0;
}

int sd_write_sectors(uint32_t lba, uint32_t count, const void *buf)
{
    const uint8_t *bbuf = (const uint8_t *)buf;
    for (uint32_t i = 0; i < count; i++) {
        int r = sd_write_sector(lba + i, bbuf + i * 512);
        if (r != 0) return r;
    }
    return 0;
}

#else /* x86 stubs */

int  sd_init(void)                                            { return -1; }
int  sd_read_sector(uint32_t lba, void *buf)                   { (void)lba; (void)buf; return -1; }
int  sd_write_sector(uint32_t lba, const void *buf)            { (void)lba; (void)buf; return -1; }
int  sd_read_sectors(uint32_t lba, uint32_t n, void *buf)      { (void)lba; (void)n; (void)buf; return -1; }
int  sd_write_sectors(uint32_t lba, uint32_t n, const void *buf) { (void)lba; (void)n; (void)buf; return -1; }

#endif
