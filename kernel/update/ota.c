/* =============================================================================
 * TensorOS — OTA Update Protocol Implementation
 *
 * Receives a new kernel binary over the serial/BT link and either:
 *   (a) chain-loads it from RAM (fast dev iteration), or
 *   (b) writes it to the SD card FAT32 boot partition (persistent)
 *
 * The FAT32 writer does the minimal work: parse MBR → find FAT32 partition →
 * search root directory for "KERNEL8 IMG" → overwrite its clusters.
 * =============================================================================*/

#include "kernel/core/kernel.h"
#include "kernel/update/ota.h"
#include "kernel/drivers/blk/rpi_sd.h"

#if defined(__aarch64__)

/* =============================================================================
 * CRC-32 (IEEE 802.3 polynomial 0xEDB88320, reflected)
 * =============================================================================*/

static uint32_t crc32_update(uint32_t crc, const uint8_t *buf, uint32_t len)
{
    crc = ~crc;
    for (uint32_t i = 0; i < len; i++) {
        crc ^= buf[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
    }
    return ~crc;
}

/* =============================================================================
 * Serial/BT IO helpers (use UART + BT simultaneously)
 *
 * Output goes to both UART and BT.  Input comes from whichever has data first.
 * =============================================================================*/

static void ota_puts(const char *s)
{
    uart_puts(s);
    while (*s) bt_putchar(*s++);
    bt_poll();  /* flush BT TX */
}

__attribute__((unused))
static int ota_has_data(void)
{
    if (uart_has_data()) return 1;
    bt_poll();
    if (bt_has_data()) return 1;
    return 0;
}

__attribute__((unused))
static uint8_t ota_getc(void)
{
    while (1) {
        if (uart_has_data()) return (uint8_t)uart_getchar();
        bt_poll();
        if (bt_has_data()) return (uint8_t)bt_getchar();
        __asm__ volatile ("wfi");
    }
}

__attribute__((unused))
static int ota_getc_timeout(uint32_t ms)
{
    uint64_t deadline = arm_timer_count() + (arm_timer_freq() * ms / 1000);
    while (arm_timer_count() < deadline) {
        if (uart_has_data()) return (uint8_t)uart_getchar();
        bt_poll();
        if (bt_has_data()) return (uint8_t)bt_getchar();
    }
    return -1;  /* timeout */
}

/* Read exact number of bytes.  Returns 0 on success, -1 on timeout */
static int ota_read_exact(uint8_t *buf, uint32_t len, uint32_t timeout_ms)
{
    uint64_t deadline = arm_timer_count() +
                        (arm_timer_freq() * timeout_ms / 1000);
    for (uint32_t i = 0; i < len; i++) {
        while (1) {
            if (arm_timer_count() > deadline) return -1;
            if (uart_has_data()) { buf[i] = (uint8_t)uart_getchar(); break; }
            bt_poll();
            if (bt_has_data()) { buf[i] = (uint8_t)bt_getchar(); break; }
        }
    }
    return 0;
}

/* =============================================================================
 * Receive kernel binary
 * Stores into receive_buf (address passed in).  Returns size, or <0 on error.
 * =============================================================================*/

/* We receive into high RAM (256 MB offset) to avoid stomping on ourselves */
#define OTA_RECV_ADDR   0x10000000UL   /* 256 MB */
#define OTA_MAX_SIZE    (64 * 1024 * 1024)  /* 64 MB max kernel */

static int ota_receive_kernel(uint32_t *out_size)
{
    uint8_t *recv_buf = (uint8_t *)OTA_RECV_ADDR;

    ota_puts("RDY\n");
    kprintf("[OTA] Waiting for kernel binary...\n");
    kprintf("[OTA] Protocol: 'OTA!' + uint32 size + data + uint32 crc32\n");

    /* Wait for magic: "OTA!" */
    uint8_t magic[4];
    if (ota_read_exact(magic, 4, 60000) != 0) {  /* 60s timeout */
        ota_puts("ERR:timeout waiting for magic\n");
        return -1;
    }
    if (magic[0] != 'O' || magic[1] != 'T' || magic[2] != 'A' || magic[3] != '!') {
        ota_puts("ERR:bad magic\n");
        return -2;
    }

    /* Read size (4 bytes, little-endian) */
    uint8_t szb[4];
    if (ota_read_exact(szb, 4, 5000) != 0) {
        ota_puts("ERR:timeout reading size\n");
        return -3;
    }
    uint32_t size = szb[0] | ((uint32_t)szb[1] << 8) |
                    ((uint32_t)szb[2] << 16) | ((uint32_t)szb[3] << 24);

    if (size == 0 || size > OTA_MAX_SIZE) {
        ota_puts("ERR:invalid size\n");
        return -4;
    }

    kprintf("[OTA] Receiving %u bytes", size);

    /* Read data with progress */
    uint32_t received = 0;
    uint32_t last_pct = 0;
    uint64_t timeout_per_block = 30000;  /* 30s timeout per 4KB block */

    while (received < size) {
        uint32_t chunk = size - received;
        if (chunk > 4096) chunk = 4096;

        if (ota_read_exact(recv_buf + received, chunk, timeout_per_block) != 0) {
            kprintf("\n");
            ota_puts("ERR:timeout during transfer\n");
            return -5;
        }
        received += chunk;

        /* Progress dots */
        uint32_t pct = (received * 100) / size;
        if (pct / 10 > last_pct / 10) {
            kprintf(".");
            last_pct = pct;
        }
    }
    kprintf(" done\n");

    /* Read CRC32 (4 bytes) */
    uint8_t crcb[4];
    if (ota_read_exact(crcb, 4, 5000) != 0) {
        ota_puts("ERR:timeout reading crc\n");
        return -6;
    }
    uint32_t expected_crc = crcb[0] | ((uint32_t)crcb[1] << 8) |
                            ((uint32_t)crcb[2] << 16) | ((uint32_t)crcb[3] << 24);

    /* Verify CRC */
    uint32_t actual_crc = crc32_update(0, recv_buf, size);
    if (actual_crc != expected_crc) {
        kprintf("[OTA] CRC mismatch: expected 0x%x got 0x%x\n",
                expected_crc, actual_crc);
        ota_puts("ERR:crc mismatch\n");
        return -7;
    }

    kprintf("[OTA] CRC OK (0x%x), %u bytes verified\n", actual_crc, size);
    ota_puts("OK!\n");

    *out_size = size;
    return 0;
}

/* =============================================================================
 * Chain-load: copy received kernel to 0x80000 and jump
 * =============================================================================*/

/* This function attribute prevents inlining — we need to absolutely control
 * the jump sequence.  The function copies the kernel then branches to it. */
static void __attribute__((noinline, noreturn))
ota_chainload(const uint8_t *src, uint32_t size)
{
    /* Disable interrupts */
    __asm__ volatile ("msr daifset, #0xF" ::: "memory");

    /* Copy kernel to 0x80000 */
    uint8_t *dst = (uint8_t *)0x80000;
    for (uint32_t i = 0; i < size; i++)
        dst[i] = src[i];

    /* DSB + ISB to ensure copied data is visible */
    __asm__ volatile ("dsb sy; isb" ::: "memory");

    /* Invalidate instruction cache (we're overwriting code) */
    __asm__ volatile ("ic iallu; dsb sy; isb" ::: "memory");

    /* Jump to new kernel at 0x80000 */
    __asm__ volatile ("br %0" :: "r"(0x80000UL));
    __builtin_unreachable();
}

int ota_receive_and_chainload(void)
{
    uint32_t size;
    int r = ota_receive_kernel(&size);
    if (r != 0) return r;

    kprintf("[OTA] Chain-loading %u bytes to 0x80000...\n", size);
    ota_puts("BOOT\n");

    /* Small delay to let the "BOOT" message flush over BT */
    arm_timer_delay_ms(100);
    bt_poll();

    ota_chainload((const uint8_t *)OTA_RECV_ADDR, size);
    /* Does not return */
}

/* =============================================================================
 * FAT32 Minimal: Find kernel8.img on boot partition and overwrite it
 * =============================================================================*/

/* MBR partition entry */
typedef struct {
    uint8_t  status;
    uint8_t  chs_first[3];
    uint8_t  type;
    uint8_t  chs_last[3];
    uint32_t lba_start;
    uint32_t sectors;
} __attribute__((packed)) mbr_part_t;

/* FAT32 BPB (BIOS Parameter Block) */
typedef struct {
    uint8_t  jmp[3];
    char     oem[8];
    uint16_t bytes_per_sector;
    uint8_t  sectors_per_cluster;
    uint16_t reserved_sectors;
    uint8_t  num_fats;
    uint16_t root_entry_count;  /* 0 for FAT32 */
    uint16_t total_sectors_16;
    uint8_t  media_type;
    uint16_t fat_size_16;       /* 0 for FAT32 */
    uint16_t sectors_per_track;
    uint16_t num_heads;
    uint32_t hidden_sectors;
    uint32_t total_sectors_32;
    /* FAT32 specific */
    uint32_t fat_size_32;
    uint16_t ext_flags;
    uint16_t fs_version;
    uint32_t root_cluster;
} __attribute__((packed)) fat32_bpb_t;

/* FAT32 directory entry (32 bytes) */
typedef struct {
    char     name[11];          /* 8.3 format, padded with spaces */
    uint8_t  attr;
    uint8_t  nt_reserved;
    uint8_t  create_time_10th;
    uint16_t create_time;
    uint16_t create_date;
    uint16_t access_date;
    uint16_t cluster_hi;
    uint16_t mod_time;
    uint16_t mod_date;
    uint16_t cluster_lo;
    uint32_t file_size;
} __attribute__((packed)) fat32_dirent_t;

/* Compare 11 bytes (8.3 filename) */
static int fat32_namecmp(const char *a, const char *b)
{
    for (int i = 0; i < 11; i++)
        if (a[i] != b[i]) return 1;
    return 0;
}

/* Find and overwrite kernel8.img on the first FAT32 partition.
 * Returns 0 on success. */
static int ota_flash_to_sd(const uint8_t *data, uint32_t size)
{
    uint8_t sector[512];

    /* Step 1: Read MBR (sector 0) */
    if (sd_read_sector(0, sector) != 0) {
        kprintf("[OTA] Failed to read MBR\n");
        return -1;
    }

    /* Check MBR signature */
    if (sector[510] != 0x55 || sector[511] != 0xAA) {
        kprintf("[OTA] Invalid MBR signature\n");
        return -2;
    }

    /* Find first FAT32 partition (type 0x0B or 0x0C) */
    mbr_part_t *parts = (mbr_part_t *)(sector + 446);
    uint32_t part_lba = 0;
    for (int i = 0; i < 4; i++) {
        if (parts[i].type == 0x0B || parts[i].type == 0x0C) {
            part_lba = parts[i].lba_start;
            break;
        }
    }
    if (part_lba == 0) {
        kprintf("[OTA] No FAT32 partition found\n");
        return -3;
    }

    /* Step 2: Read FAT32 BPB (first sector of partition) */
    if (sd_read_sector(part_lba, sector) != 0) {
        kprintf("[OTA] Failed to read BPB\n");
        return -4;
    }

    fat32_bpb_t *bpb = (fat32_bpb_t *)sector;
    uint32_t spc = bpb->sectors_per_cluster;
    uint32_t fat_start = part_lba + bpb->reserved_sectors;
    uint32_t fat_size = bpb->fat_size_32;
    uint32_t data_start = fat_start + bpb->num_fats * fat_size;
    uint32_t root_cluster = bpb->root_cluster;

    kprintf("[OTA] FAT32: spc=%u fat_start=%u data_start=%u root_clust=%u\n",
            spc, fat_start, data_start, root_cluster);

    /* Step 3: Search root directory for "KERNEL8 IMG" */
    /* Follow the cluster chain of the root directory */
    uint32_t cluster = root_cluster;
    uint32_t found_cluster_lo = 0, found_cluster_hi = 0;
    uint32_t found_size = 0;
    uint32_t dir_entry_sector = 0;
    uint32_t dir_entry_offset = 0;
    int found = 0;

    /* FAT32 8.3 name for "kernel8.img" = "KERNEL8 IMG" (8+3, space-padded) */
    const char target_name[11] = {'K','E','R','N','E','L','8',' ','I','M','G'};

    for (int chain = 0; chain < 64 && !found; chain++) {  /* max 64 clusters */
        uint32_t cluster_lba = data_start + (cluster - 2) * spc;

        for (uint32_t s = 0; s < spc && !found; s++) {
            if (sd_read_sector(cluster_lba + s, sector) != 0) break;

            fat32_dirent_t *entries = (fat32_dirent_t *)sector;
            for (int e = 0; e < 16; e++) {  /* 512/32 = 16 entries per sector */
                if (entries[e].name[0] == 0x00) goto dir_end;  /* End of directory */
                if ((uint8_t)entries[e].name[0] == 0xE5) continue;  /* Deleted */
                if (entries[e].attr & 0x08) continue;  /* Volume label */
                if (entries[e].attr & 0x0F) continue;  /* LFN entry, skip */

                if (fat32_namecmp(entries[e].name, target_name) == 0) {
                    found_cluster_lo = entries[e].cluster_lo;
                    found_cluster_hi = entries[e].cluster_hi;
                    found_size = entries[e].file_size;
                    dir_entry_sector = cluster_lba + s;
                    dir_entry_offset = e;
                    found = 1;
                    break;
                }
            }
        }

        /* Follow FAT chain */
        uint32_t fat_sector_idx = cluster / 128;  /* 512/4 = 128 entries per sector */
        if (sd_read_sector(fat_start + fat_sector_idx, sector) != 0) break;
        uint32_t *fat = (uint32_t *)sector;
        uint32_t next = fat[cluster % 128] & 0x0FFFFFFF;
        if (next >= 0x0FFFFFF8) break;  /* End of chain */
        cluster = next;
    }
dir_end:

    if (!found) {
        kprintf("[OTA] kernel8.img not found on SD card\n");
        return -5;
    }

    uint32_t file_cluster = found_cluster_lo | ((uint32_t)found_cluster_hi << 16);
    kprintf("[OTA] Found kernel8.img: cluster=%u size=%u\n", file_cluster, found_size);

    /* Step 4: Write new kernel data over the file's clusters.
     * Follow the FAT chain, writing sector by sector.
     * If new kernel > old size, we may need to allocate new clusters.
     * For safety, require new kernel <= old allocated space. */
    uint32_t old_clusters = (found_size + spc * 512 - 1) / (spc * 512);
    uint32_t new_clusters = (size + spc * 512 - 1) / (spc * 512);

    if (new_clusters > old_clusters + 8) {
        /* We could allocate more clusters from the FAT, but for safety just
         * reject if it's way bigger.  +8 clusters (~32KB) slack is fine. */
        kprintf("[OTA] New kernel too large (%u > %u clusters)\n",
                new_clusters, old_clusters);
        return -6;
    }

    /* Write data following the cluster chain */
    cluster = file_cluster;
    uint32_t written = 0;
    uint32_t cluster_bytes = spc * 512;  (void)cluster_bytes;

    while (written < size) {
        uint32_t cluster_lba = data_start + (cluster - 2) * spc;

        for (uint32_t s = 0; s < spc && written < size; s++) {
            /* Prepare a 512-byte sector (zero-padded at the end) */
            uint8_t wbuf[512];
            uint32_t remain = size - written;
            uint32_t copy = remain > 512 ? 512 : remain;
            for (uint32_t i = 0; i < copy; i++)
                wbuf[i] = data[written + i];
            for (uint32_t i = copy; i < 512; i++)
                wbuf[i] = 0;

            if (sd_write_sector(cluster_lba + s, wbuf) != 0) {
                kprintf("[OTA] SD write error at LBA %u\n", cluster_lba + s);
                return -7;
            }
            written += 512;
        }

        /* Progress */
        kprintf("[OTA] Written %u / %u bytes\r", written > size ? size : written, size);

        /* Follow FAT chain to next cluster */
        uint32_t fat_sector_idx = cluster / 128;
        if (sd_read_sector(fat_start + fat_sector_idx, sector) != 0) {
            kprintf("\n[OTA] FAT read error\n");
            return -8;
        }
        uint32_t *fat = (uint32_t *)sector;
        uint32_t next = fat[cluster % 128] & 0x0FFFFFFF;
        if (next >= 0x0FFFFFF8) {
            if (written < size) {
                /* Need to allocate a new cluster — find a free one in this FAT sector */
                int alloc_found = 0;
                for (int i = 0; i < 128; i++) {
                    if ((fat[i] & 0x0FFFFFFF) == 0) {
                        /* Free cluster! Link it */
                        fat[cluster % 128] = (fat_sector_idx * 128 + i) | 0x00000000;
                        fat[i] = 0x0FFFFFF8;  /* End of chain */
                        sd_write_sector(fat_start + fat_sector_idx, sector);
                        next = fat_sector_idx * 128 + i;
                        alloc_found = 1;
                        break;
                    }
                }
                if (!alloc_found) {
                    kprintf("\n[OTA] No free clusters\n");
                    return -9;
                }
            } else {
                break;  /* All written */
            }
        }
        cluster = next;
    }
    kprintf("\n");

    /* Step 5: Update directory entry with new file size */
    if (sd_read_sector(dir_entry_sector, sector) != 0) {
        kprintf("[OTA] Failed to re-read directory sector\n");
        return -10;
    }
    fat32_dirent_t *entries = (fat32_dirent_t *)sector;
    entries[dir_entry_offset].file_size = size;
    if (sd_write_sector(dir_entry_sector, sector) != 0) {
        kprintf("[OTA] Failed to update directory entry\n");
        return -11;
    }

    kprintf("[OTA] kernel8.img updated: %u -> %u bytes\n", found_size, size);
    return 0;
}

int ota_receive_and_flash(void)
{
    uint32_t size;
    int r = ota_receive_kernel(&size);
    if (r != 0) return r;

    kprintf("[OTA] Initializing SD card...\n");
    if (sd_init() != 0) {
        kprintf("[OTA] SD card init failed — falling back to chain-load\n");
        ota_puts("WARN:sd_fail,chainloading\n");
        ota_chainload((const uint8_t *)OTA_RECV_ADDR, size);
    }

    kprintf("[OTA] Writing kernel8.img to SD card...\n");
    r = ota_flash_to_sd((const uint8_t *)OTA_RECV_ADDR, size);
    if (r != 0) {
        kprintf("[OTA] SD flash failed (%d) — falling back to chain-load\n", r);
        ota_puts("WARN:flash_fail,chainloading\n");
        ota_chainload((const uint8_t *)OTA_RECV_ADDR, size);
    }

    kprintf("[OTA] Flash complete! Rebooting...\n");
    ota_puts("BOOT\n");
    arm_timer_delay_ms(200);
    bt_poll();

    /* Reboot via watchdog (PM_RSTC) */
    #define PM_BASE     0xFE100000UL
    #define PM_RSTC     (PM_BASE + 0x1C)
    #define PM_WDOG     (PM_BASE + 0x24)
    #define PM_PASSWORD 0x5A000000
    mmio_write(PM_WDOG, PM_PASSWORD | 1);              /* Watchdog = 1 tick */
    mmio_write(PM_RSTC, PM_PASSWORD | 0x20);            /* Full reset */
    while (1) __asm__ volatile ("wfi");  /* Wait for reset */
}

#else /* x86 — COM1 serial OTA + ATA PIO flash */

/* =============================================================================
 * x86 COM1 serial helpers (38400 8N1, mirroring klib.c serial_init)
 * =============================================================================*/
#define X86_COM1 0x3F8

static void x86_ota_serial_init(void)
{
    outb(X86_COM1 + 1, 0x00); /* Disable interrupts */
    outb(X86_COM1 + 3, 0x80); /* Enable DLAB */
    outb(X86_COM1 + 0, 0x03); /* 38400 baud divisor = 3 */
    outb(X86_COM1 + 1, 0x00);
    outb(X86_COM1 + 3, 0x03); /* 8N1 */
    outb(X86_COM1 + 2, 0xC7); /* Enable FIFO */
    outb(X86_COM1 + 4, 0x0B); /* RTS/DSR */
}

static int x86_ota_has_data(void)
{
    return (inb(X86_COM1 + 5) & 0x01) != 0;
}

static uint8_t x86_ota_getc(void)
{
    while (!x86_ota_has_data())
        ;
    return inb(X86_COM1);
}

static void x86_ota_putc(char c)
{
    while (!(inb(X86_COM1 + 5) & 0x20))
        ;
    outb(X86_COM1, (uint8_t)c);
    outb(0xE9, (uint8_t)c); /* QEMU debug port */
}

static void x86_ota_puts(const char *s)
{
    while (*s) x86_ota_putc(*s++);
}

/* =============================================================================
 * TSC-based timeout helpers
 * =============================================================================*/
static uint64_t x86_rdtsc(void)
{
    uint32_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

/* Return TSC ticks per millisecond.  Uses PIT channel 2 for calibration if
 * possible; falls back to 3 GHz assumption. */
static uint64_t x86_tsc_ticks_per_ms(void)
{
    /* Try PIT calibration: gate channel 2 for ~1 ms (PIT freq = 1193182 Hz,
     * 1 ms ≈ 1193 counts) */
    outb(0x61, (inb(0x61) & 0xFC) | 0x01); /* gate on, speaker off */
    outb(0x43, 0xB0);                       /* ch2, lobyte/hibyte, one-shot */
    outb(0x42, 0xA9); outb(0x42, 0x04);     /* reload = 0x04A9 = 1193 ≈ 1 ms */
    outb(0x61, inb(0x61) | 0x01);           /* start */
    uint64_t t0 = x86_rdtsc();
    while (inb(0x61) & 0x20)               /* wait for OUT2 to go high */
        ;
    uint64_t ticks = x86_rdtsc() - t0;
    outb(0x61, inb(0x61) & 0xFC);          /* gate off */
    /* Sanity: must be between 0.5 GHz and 8 GHz */
    if (ticks < 500000ULL || ticks > 8000000ULL)
        ticks = 3000000ULL; /* fallback: 3 GHz assumed */
    return ticks;
}

/* Read exactly `len` bytes from COM1 with a per-read timeout.
 * `timeout_ms` is the maximum wait for each individual byte. */
static int x86_ota_read_exact(uint8_t *buf, uint32_t len,
                               uint32_t timeout_ms, uint64_t ticks_per_ms)
{
    uint64_t timeout_ticks = (uint64_t)timeout_ms * ticks_per_ms;
    for (uint32_t i = 0; i < len; i++) {
        uint64_t deadline = x86_rdtsc() + timeout_ticks;
        while (!x86_ota_has_data()) {
            if (x86_rdtsc() > deadline) return -1; /* timeout */
        }
        buf[i] = inb(X86_COM1);
    }
    return 0;
}

/* =============================================================================
 * CRC-32 (IEEE 802.3, same polynomial as ARM64 path)
 * =============================================================================*/
static uint32_t x86_crc32(uint32_t crc, const uint8_t *buf, uint32_t len)
{
    crc = ~crc;
    for (uint32_t i = 0; i < len; i++) {
        crc ^= buf[i];
        for (int j = 0; j < 8; j++)
            crc = (crc >> 1) ^ (0xEDB88320u & -(crc & 1));
    }
    return ~crc;
}

/* =============================================================================
 * Receive kernel binary over COM1
 * Protocol: "OTA!" + uint32_le size + data + uint32_le crc32
 * =============================================================================*/
#define X86_OTA_RECV_ADDR  ((uint8_t *)0x4000000UL)  /* 64 MB — above kernel */
#define X86_OTA_MAX_SIZE   (64u * 1024u * 1024u)

static int x86_ota_receive(uint32_t *out_size, uint64_t ticks_per_ms)
{
    x86_ota_puts("RDY\n");
    kprintf("[OTA/x86] Waiting for kernel binary on COM1 (38400 8N1)...\n");

    uint8_t magic[4];
    if (x86_ota_read_exact(magic, 4, 60000, ticks_per_ms) != 0) {
        x86_ota_puts("ERR:timeout waiting for magic\n"); return -1;
    }
    if (magic[0] != 'O' || magic[1] != 'T' || magic[2] != 'A' || magic[3] != '!') {
        x86_ota_puts("ERR:bad magic\n"); return -2;
    }

    uint8_t szb[4];
    if (x86_ota_read_exact(szb, 4, 5000, ticks_per_ms) != 0) {
        x86_ota_puts("ERR:timeout reading size\n"); return -3;
    }
    uint32_t size = szb[0] | ((uint32_t)szb[1] << 8)
                  | ((uint32_t)szb[2] << 16) | ((uint32_t)szb[3] << 24);
    if (size == 0 || size > X86_OTA_MAX_SIZE) {
        x86_ota_puts("ERR:invalid size\n"); return -4;
    }

    kprintf("[OTA/x86] Receiving %u bytes", size);

    uint8_t  *recv = X86_OTA_RECV_ADDR;
    uint32_t  received = 0, last_pct = 0;
    while (received < size) {
        uint32_t chunk = size - received;
        if (chunk > 4096) chunk = 4096;
        if (x86_ota_read_exact(recv + received, chunk, 30000, ticks_per_ms) != 0) {
            kprintf("\n");
            x86_ota_puts("ERR:timeout during transfer\n"); return -5;
        }
        received += chunk;
        uint32_t pct = (received * 100) / size;
        if (pct / 10 > last_pct / 10) { kprintf("."); last_pct = pct; }
    }
    kprintf(" done\n");

    uint8_t crcb[4];
    if (x86_ota_read_exact(crcb, 4, 5000, ticks_per_ms) != 0) {
        x86_ota_puts("ERR:timeout reading crc\n"); return -6;
    }
    uint32_t exp_crc = crcb[0] | ((uint32_t)crcb[1] << 8)
                     | ((uint32_t)crcb[2] << 16) | ((uint32_t)crcb[3] << 24);
    uint32_t act_crc = x86_crc32(0, recv, size);
    if (act_crc != exp_crc) {
        kprintf("[OTA/x86] CRC mismatch: exp=0x%x got=0x%x\n", exp_crc, act_crc);
        x86_ota_puts("ERR:crc mismatch\n"); return -7;
    }

    kprintf("[OTA/x86] CRC OK (0x%x), %u bytes verified\n", act_crc, size);
    x86_ota_puts("OK!\n");
    *out_size = size;
    return 0;
}

/* =============================================================================
 * Chain-load: copy to 2 MB and jump (flat binary, not PE)
 * =============================================================================*/
#define X86_OTA_LOAD_ADDR 0x200000UL

static void __attribute__((noinline, noreturn))
x86_ota_chainload(const uint8_t *src, uint32_t size)
{
    uint8_t *dst = (uint8_t *)X86_OTA_LOAD_ADDR;
    for (uint32_t i = 0; i < size; i++) dst[i] = src[i];
    /* Flush write-back caches, then jump */
    __asm__ volatile (
        "wbinvd\n"
        "cli\n"
        "jmp *%0\n"
        :: "r"((uintptr_t)X86_OTA_LOAD_ADDR) : "memory"
    );
    __builtin_unreachable();
}

/* =============================================================================
 * ATA PIO sector read/write (primary controller, LBA28, master drive)
 * =============================================================================*/
#define ATA_DATA    0x1F0
#define ATA_FEAT    0x1F1
#define ATA_COUNT   0x1F2
#define ATA_LBA_LO  0x1F3
#define ATA_LBA_MID 0x1F4
#define ATA_LBA_HI  0x1F5
#define ATA_DRIVE   0x1F6
#define ATA_CMD     0x1F7
#define ATA_ALT_ST  0x3F6   /* alternate status (no interrupt clear) */
#define ATA_BSY     0x80
#define ATA_DRQ     0x08
#define ATA_ERR     0x01

static int x86_ata_wait_ready(void)
{
    /* 400 ns delay: 4 reads from alternate-status */
    inb(ATA_ALT_ST); inb(ATA_ALT_ST); inb(ATA_ALT_ST); inb(ATA_ALT_ST);
    uint32_t n = 0x7FFFFF;
    while (inb(ATA_CMD) & ATA_BSY) { if (!--n) return -1; }
    return 0;
}

static int x86_ata_read_sector(uint32_t lba, uint8_t *buf)
{
    if (x86_ata_wait_ready() != 0) return -1;
    outb(ATA_DRIVE, 0xE0u | ((lba >> 24) & 0x0Fu)); /* LBA master */
    outb(ATA_FEAT,  0x00);
    outb(ATA_COUNT, 1);
    outb(ATA_LBA_LO,  (uint8_t)(lba));
    outb(ATA_LBA_MID, (uint8_t)(lba >>  8));
    outb(ATA_LBA_HI,  (uint8_t)(lba >> 16));
    outb(ATA_CMD,  0x20);   /* READ SECTORS */
    if (x86_ata_wait_ready() != 0) return -1;
    if (inb(ATA_CMD) & ATA_ERR) return -2;
    uint16_t *words = (uint16_t *)buf;
    for (int i = 0; i < 256; i++) words[i] = inw(ATA_DATA);
    return 0;
}

static int x86_ata_write_sector(uint32_t lba, const uint8_t *buf)
{
    if (x86_ata_wait_ready() != 0) return -1;
    outb(ATA_DRIVE, 0xE0u | ((lba >> 24) & 0x0Fu));
    outb(ATA_FEAT,  0x00);
    outb(ATA_COUNT, 1);
    outb(ATA_LBA_LO,  (uint8_t)(lba));
    outb(ATA_LBA_MID, (uint8_t)(lba >>  8));
    outb(ATA_LBA_HI,  (uint8_t)(lba >> 16));
    outb(ATA_CMD,  0x30);   /* WRITE SECTORS */
    if (x86_ata_wait_ready() != 0) return -1;
    if (inb(ATA_CMD) & ATA_ERR) return -2;
    const uint16_t *words = (const uint16_t *)buf;
    for (int i = 0; i < 256; i++) outw(ATA_DATA, words[i]);
    outb(ATA_CMD, 0xE7);    /* FLUSH CACHE */
    x86_ata_wait_ready();
    return 0;
}

/* =============================================================================
 * FAT32 minimal writer — find "GEODSS  EFI" (EFI\BOOT\geodss.efi) and
 * overwrite its cluster chain with the new binary.
 * =============================================================================*/
typedef struct {
    uint8_t  status;
    uint8_t  chs_first[3];
    uint8_t  type;
    uint8_t  chs_last[3];
    uint32_t lba_start;
    uint32_t sectors;
} __attribute__((packed)) x86_mbr_part_t;

typedef struct {
    uint8_t  jmp[3];
    char     oem[8];
    uint16_t bytes_per_sector;
    uint8_t  sectors_per_cluster;
    uint16_t reserved_sectors;
    uint8_t  num_fats;
    uint16_t root_entry_count;
    uint16_t total_sectors_16;
    uint8_t  media_type;
    uint16_t fat_size_16;
    uint16_t sectors_per_track;
    uint16_t num_heads;
    uint32_t hidden_sectors;
    uint32_t total_sectors_32;
    uint32_t fat_size_32;
    uint16_t ext_flags;
    uint16_t fs_version;
    uint32_t root_cluster;
} __attribute__((packed)) x86_fat32_bpb_t;

typedef struct {
    char     name[11];
    uint8_t  attr;
    uint8_t  nt_reserved;
    uint8_t  create_time_10th;
    uint16_t create_time, create_date, access_date;
    uint16_t cluster_hi;
    uint16_t mod_time, mod_date;
    uint16_t cluster_lo;
    uint32_t file_size;
} __attribute__((packed)) x86_fat32_dirent_t;

static int x86_fat32_namecmp(const char *a, const char *b)
{
    for (int i = 0; i < 11; i++) if (a[i] != b[i]) return 1;
    return 0;
}

static int x86_ota_flash_to_disk(const uint8_t *data, uint32_t size)
{
    uint8_t sector[512];

    if (x86_ata_read_sector(0, sector) != 0) {
        kprintf("[OTA/x86] ATA: failed to read MBR\n"); return -1;
    }
    if (sector[510] != 0x55 || sector[511] != 0xAA) {
        kprintf("[OTA/x86] ATA: invalid MBR signature\n"); return -2;
    }

    /* Find first FAT32 partition (type 0x0B or 0x0C) */
    x86_mbr_part_t *parts = (x86_mbr_part_t *)(sector + 446);
    uint32_t part_lba = 0;
    for (int i = 0; i < 4; i++) {
        if (parts[i].type == 0x0B || parts[i].type == 0x0C) {
            part_lba = parts[i].lba_start; break;
        }
    }
    if (part_lba == 0) { kprintf("[OTA/x86] No FAT32 partition\n"); return -3; }

    if (x86_ata_read_sector(part_lba, sector) != 0) {
        kprintf("[OTA/x86] ATA: failed to read BPB\n"); return -4;
    }
    x86_fat32_bpb_t *bpb = (x86_fat32_bpb_t *)sector;
    uint32_t spc       = bpb->sectors_per_cluster;
    uint32_t fat_start = part_lba + bpb->reserved_sectors;
    uint32_t fat_size  = bpb->fat_size_32;
    uint32_t data_start = fat_start + bpb->num_fats * fat_size;
    uint32_t root_cluster = bpb->root_cluster;

    /* Search root dir for "GEODSS  EFI" (8.3, space-padded) */
    const char target[11] = {'G','E','O','D','S','S',' ',' ','E','F','I'};
    uint32_t cluster = root_cluster;
    int found = 0;
    uint32_t found_cl = 0, found_size = 0, dir_sector = 0;
    int dir_entry_idx = 0;

    for (int chain = 0; chain < 64 && !found; chain++) {
        uint32_t cl_lba = data_start + (cluster - 2) * spc;
        for (uint32_t s = 0; s < spc && !found; s++) {
            if (x86_ata_read_sector(cl_lba + s, sector) != 0) break;
            x86_fat32_dirent_t *ents = (x86_fat32_dirent_t *)sector;
            for (int e = 0; e < 16; e++) {
                if (ents[e].name[0] == 0x00) goto dir_end;
                if ((uint8_t)ents[e].name[0] == 0xE5) continue;
                if (ents[e].attr & 0x08) continue;
                if ((ents[e].attr & 0x0F) == 0x0F) continue; /* LFN */
                if (x86_fat32_namecmp(ents[e].name, target) == 0) {
                    found_cl  = ents[e].cluster_lo | ((uint32_t)ents[e].cluster_hi << 16);
                    found_size = ents[e].file_size;
                    dir_sector = cl_lba + s;
                    dir_entry_idx = e;
                    found = 1; break;
                }
            }
        }
        /* Follow FAT chain */
        uint32_t fi = cluster / 128;
        if (x86_ata_read_sector(fat_start + fi, sector) != 0) break;
        uint32_t next = ((uint32_t *)sector)[cluster % 128] & 0x0FFFFFFFu;
        if (next >= 0x0FFFFFF8u) break;
        cluster = next;
    }
dir_end:
    if (!found) {
        kprintf("[OTA/x86] geodss.efi not found on EFI partition\n"); return -5;
    }
    kprintf("[OTA/x86] Found geodss.efi: cluster=%u size=%u\n", found_cl, found_size);

    /* Write data following the cluster chain */
    cluster = found_cl;
    uint32_t written = 0;
    while (written < size) {
        uint32_t cl_lba = data_start + (cluster - 2) * spc;
        for (uint32_t s = 0; s < spc && written < size; s++) {
            uint8_t wbuf[512];
            uint32_t remain = size - written;
            uint32_t copy = remain > 512 ? 512 : remain;
            for (uint32_t i = 0; i < copy; i++) wbuf[i] = data[written + i];
            for (uint32_t i = copy; i < 512; i++) wbuf[i] = 0;
            if (x86_ata_write_sector(cl_lba + s, wbuf) != 0) {
                kprintf("[OTA/x86] ATA write error at LBA %u\n", cl_lba + s);
                return -7;
            }
            written += 512;
        }
        kprintf("[OTA/x86] Written %u / %u bytes\r",
                written > size ? size : written, size);

        uint32_t fi = cluster / 128;
        if (x86_ata_read_sector(fat_start + fi, sector) != 0) {
            kprintf("\n[OTA/x86] FAT read error\n"); return -8;
        }
        uint32_t next = ((uint32_t *)sector)[cluster % 128] & 0x0FFFFFFFu;
        if (next >= 0x0FFFFFF8u) break;
        cluster = next;
    }
    kprintf("\n");

    /* Update directory entry file size */
    if (x86_ata_read_sector(dir_sector, sector) != 0) return -9;
    ((x86_fat32_dirent_t *)sector)[dir_entry_idx].file_size = size;
    if (x86_ata_write_sector(dir_sector, sector) != 0) return -10;

    kprintf("[OTA/x86] geodss.efi updated: %u -> %u bytes\n", found_size, size);
    return 0;
}

/* =============================================================================
 * Public API
 * =============================================================================*/
int ota_receive_and_chainload(void)
{
    x86_ota_serial_init();
    uint64_t tpm = x86_tsc_ticks_per_ms();
    uint32_t size = 0;
    int r = x86_ota_receive(&size, tpm);
    if (r != 0) return r;

    kprintf("[OTA/x86] Chain-loading %u bytes to 0x%lx...\n",
            size, (unsigned long)X86_OTA_LOAD_ADDR);
    x86_ota_puts("BOOT\n");
    /* Drain TX FIFO */
    for (int i = 0; i < 1000000; i++) __asm__ volatile ("pause");

    x86_ota_chainload(X86_OTA_RECV_ADDR, size);
    /* Does not return */
}

int ota_receive_and_flash(void)
{
    x86_ota_serial_init();
    uint64_t tpm = x86_tsc_ticks_per_ms();
    uint32_t size = 0;
    int r = x86_ota_receive(&size, tpm);
    if (r != 0) return r;

    kprintf("[OTA/x86] Writing geodss.efi to EFI system partition...\n");
    r = x86_ota_flash_to_disk(X86_OTA_RECV_ADDR, size);
    if (r != 0) {
        kprintf("[OTA/x86] Disk flash failed (%d) — falling back to chain-load\n", r);
        x86_ota_puts("WARN:flash_fail,chainloading\n");
        x86_ota_chainload(X86_OTA_RECV_ADDR, size);
    }

    x86_ota_puts("BOOT\n");
    kprintf("[OTA/x86] Flash complete. Rebooting via ACPI reset...\n");
    /* ACPI reset (port 0xCF9, value 6 = hard reset) */
    outb(0xCF9, 0x06);
    /* Fallback: triple-fault reset */
    __asm__ volatile ("cli; lidt 0; int3" ::: "memory");
    __builtin_unreachable();
}

#endif
