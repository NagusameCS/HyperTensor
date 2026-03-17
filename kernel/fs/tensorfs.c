/* =============================================================================
 * TensorOS - TensorFS Implementation
 * =============================================================================*/

#include "kernel/fs/tensorfs.h"

/* Forward declaration for virtio_blk driver */
extern int virtio_blk_read(uint64_t sector, uint32_t count, void *buf);
extern int virtio_blk_write(uint64_t sector, uint32_t count, const void *buf);

static tfs_inode_t inodes[TFS_MAX_INODES];
static uint32_t inode_count = 0;
static uint64_t next_inode = 1;

void tensorfs_init(void)
{
    kmemset(inodes, 0, sizeof(inodes));
    inode_count = 0;

    /* Create root directory */
    tfs_inode_t *root = &inodes[inode_count++];
    root->inode_num = next_inode++;
    root->type = TFS_FILE_DIR;
    root->name[0] = '/';
    root->name[1] = '\0';
    root->permissions = 0755;

    /* Create standard directories */
    tfs_mkdir("/models");
    tfs_mkdir("/datasets");
    tfs_mkdir("/checkpoints");
    tfs_mkdir("/logs");
    tfs_mkdir("/config");
    tfs_mkdir("/tmp");
}

int tfs_create(const char *path, tfs_file_type_t type)
{
    if (inode_count >= TFS_MAX_INODES) return -1;

    tfs_inode_t *inode = &inodes[inode_count++];
    kmemset(inode, 0, sizeof(*inode));
    inode->inode_num = next_inode++;
    inode->type = type;
    inode->permissions = 0644;

    /* Extract filename from path */
    const char *name = path;
    for (const char *p = path; *p; p++) {
        if (*p == '/') name = p + 1;
    }
    for (int i = 0; i < 255 && name[i]; i++)
        inode->name[i] = name[i];

    return 0;
}

int tfs_mkdir(const char *path)
{
    return tfs_create(path, TFS_FILE_DIR);
}

int tfs_stat(const char *path, tfs_inode_t *inode)
{
    /* Extract filename */
    const char *name = path;
    for (const char *p = path; *p; p++) {
        if (*p == '/') name = p + 1;
    }

    for (uint32_t i = 0; i < inode_count; i++) {
        if (kstrcmp(inodes[i].name, name) == 0 ||
            (path[0] == '/' && path[1] == '\0' && inodes[i].name[0] == '/')) {
            if (inode) *inode = inodes[i];
            return 0;
        }
    }
    return -1; /* Not found */
}

int tfs_readdir(const char *path, tfs_inode_t *entries,
                 uint32_t max, uint32_t *count)
{
    uint32_t found = 0;
    for (uint32_t i = 0; i < inode_count && found < max; i++) {
        /* Simplified: return all entries */
        entries[found++] = inodes[i];
    }
    if (count) *count = found;
    return 0;
}

int tfs_model_list(tfs_inode_t *models, uint32_t max, uint32_t *count)
{
    uint32_t found = 0;
    for (uint32_t i = 0; i < inode_count && found < max; i++) {
        if (inodes[i].type == TFS_FILE_MODEL ||
            inodes[i].type == TFS_FILE_WEIGHTS) {
            models[found++] = inodes[i];
        }
    }
    if (count) *count = found;
    return 0;
}

/* Simple in-memory file data storage */
#define TFS_DATA_POOL_SIZE  (4 * 1024 * 1024)  /* 4 MB total */
#define TFS_MAX_FILE_DATA   (64 * 1024)         /* 64 KB per file max */
static uint8_t tfs_data_pool[TFS_DATA_POOL_SIZE];
static uint64_t tfs_data_pool_used = 0;

/* Per-inode data mapping */
static struct {
    uint8_t *data;
    uint64_t capacity;
} inode_data[TFS_MAX_INODES];

/* File descriptor table */
#define MAX_OPEN_FILES 64
static struct { bool open; uint32_t inode_idx; uint64_t pos; } fd_table[MAX_OPEN_FILES];

int tfs_open(const char *path, uint32_t flags)
{
    /* Find inode */
    const char *name = path;
    for (const char *p = path; *p; p++) {
        if (*p == '/') name = p + 1;
    }

    for (uint32_t i = 0; i < inode_count; i++) {
        if (kstrcmp(inodes[i].name, name) == 0) {
            /* Find free fd */
            for (int fd = 0; fd < MAX_OPEN_FILES; fd++) {
                if (!fd_table[fd].open) {
                    fd_table[fd].open = true;
                    fd_table[fd].inode_idx = i;
                    fd_table[fd].pos = 0;
                    return fd;
                }
            }
            return -1; /* No free fds */
        }
    }
    return -1; /* Not found */
}

int tfs_close(int fd)
{
    if (fd < 0 || fd >= MAX_OPEN_FILES) return -1;
    fd_table[fd].open = false;
    return 0;
}

int tfs_read(int fd, void *buf, uint64_t size, uint64_t offset)
{
    if (fd < 0 || fd >= MAX_OPEN_FILES || !fd_table[fd].open) return -1;
    uint32_t idx = fd_table[fd].inode_idx;
    tfs_inode_t *inode = &inodes[idx];

    if (offset >= inode->size) return 0;
    uint64_t avail = inode->size - offset;
    if (size > avail) size = avail;
    if (size == 0) return 0;

    if (inode_data[idx].data) {
        kmemcpy(buf, inode_data[idx].data + offset, size);
    } else {
        kmemset(buf, 0, size);
    }
    return (int)size;
}

int tfs_write(int fd, const void *buf, uint64_t size, uint64_t offset)
{
    if (fd < 0 || fd >= MAX_OPEN_FILES || !fd_table[fd].open) return -1;
    uint32_t idx = fd_table[fd].inode_idx;
    tfs_inode_t *inode = &inodes[idx];

    uint64_t end = offset + size;
    if (end > TFS_MAX_FILE_DATA) return -1;

    /* Allocate data buffer from pool if needed */
    if (!inode_data[idx].data) {
        uint64_t alloc = (end > 4096) ? end : 4096;
        if (alloc > TFS_MAX_FILE_DATA) alloc = TFS_MAX_FILE_DATA;
        if (tfs_data_pool_used + alloc > TFS_DATA_POOL_SIZE) return -1;
        inode_data[idx].data = &tfs_data_pool[tfs_data_pool_used];
        inode_data[idx].capacity = alloc;
        kmemset(inode_data[idx].data, 0, alloc);
        tfs_data_pool_used += alloc;
    }

    /* Grow if needed */
    if (end > inode_data[idx].capacity) {
        /* Can't grow in-place with static pool, fail */
        if (end > TFS_MAX_FILE_DATA) return -1;
        /* Try to allocate a new larger block */
        uint64_t new_cap = end * 2;
        if (new_cap > TFS_MAX_FILE_DATA) new_cap = TFS_MAX_FILE_DATA;
        if (tfs_data_pool_used + new_cap > TFS_DATA_POOL_SIZE) return -1;
        uint8_t *new_data = &tfs_data_pool[tfs_data_pool_used];
        kmemset(new_data, 0, new_cap);
        kmemcpy(new_data, inode_data[idx].data, inode->size);
        inode_data[idx].data = new_data;
        inode_data[idx].capacity = new_cap;
        tfs_data_pool_used += new_cap;
        /* Old block is leaked — acceptable for embedded OS */
    }

    kmemcpy(inode_data[idx].data + offset, buf, size);
    if (end > inode->size) inode->size = end;
    return (int)size;
}

int tfs_unlink(const char *path)
{
    const char *name = path;
    for (const char *p = path; *p; p++) {
        if (*p == '/') name = p + 1;
    }

    for (uint32_t i = 0; i < inode_count; i++) {
        if (kstrcmp(inodes[i].name, name) == 0) {
            /* Close any open fds pointing to this inode */
            for (int fd = 0; fd < MAX_OPEN_FILES; fd++) {
                if (fd_table[fd].open && fd_table[fd].inode_idx == i)
                    fd_table[fd].open = false;
            }
            /* Clear data reference (pool memory is leaked) */
            inode_data[i].data = NULL;
            inode_data[i].capacity = 0;
            /* Remove inode by shifting */
            for (uint32_t j = i; j + 1 < inode_count; j++) {
                inodes[j] = inodes[j + 1];
                inode_data[j] = inode_data[j + 1];
            }
            inode_count--;
            return 0;
        }
    }
    return -1;
}

/* =============================================================================
 * Disk Persistence Layer
 *
 * On-disk format (via virtio_blk):
 *   Sector 0:      Superblock (512 bytes)
 *   Sectors 1-N:   Inode table (packed tfs_inode_t entries)
 *   Sectors N+1..: File data blocks (referenced by inode_data offsets)
 *
 * The superblock contains a magic number, inode count, data pool usage,
 * and a CRC32 checksum for integrity validation.
 * =============================================================================*/

#define TFS_DISK_MAGIC      0x54465330  /* "TFS0" */
#define TFS_SUPER_SECTOR    0           /* Superblock at sector 0 */
#define TFS_INODE_SECTOR    1           /* Inode table starts at sector 1 */
#define TFS_DATA_SECTOR     256         /* Data pool starts at sector 256 */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t inode_count;
    uint64_t next_inode;
    uint64_t data_pool_used;
    uint32_t checksum;          /* Simple additive checksum */
    uint8_t  reserved[480];     /* Pad to 512 bytes */
} __attribute__((packed)) tfs_superblock_t;

static uint32_t tfs_checksum(const void *data, uint64_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    uint32_t sum = 0;
    for (uint64_t i = 0; i < len; i++)
        sum += p[i];
    return sum;
}

int tfs_sync(void)
{
    /* Write superblock */
    tfs_superblock_t sb;
    kmemset(&sb, 0, sizeof(sb));
    sb.magic = TFS_DISK_MAGIC;
    sb.version = 1;
    sb.inode_count = inode_count;
    sb.next_inode = next_inode;
    sb.data_pool_used = tfs_data_pool_used;
    sb.checksum = tfs_checksum(inodes, inode_count * sizeof(tfs_inode_t));

    if (virtio_blk_write(TFS_SUPER_SECTOR, 1, &sb) != 0) {
        kprintf("[TFS] SYNC ERROR: Failed to write superblock\n");
        return -1;
    }

    /* Write inode table */
    uint32_t inode_bytes = inode_count * sizeof(tfs_inode_t);
    uint32_t inode_sectors = (inode_bytes + 511) / 512;
    if (inode_sectors > 0) {
        /* Need a sector-aligned buffer — use data pool as temp if needed */
        uint32_t aligned_bytes = inode_sectors * 512;
        static uint8_t inode_buf[255 * 512];  /* Max 255 sectors for inode table */
        if (aligned_bytes > sizeof(inode_buf)) {
            kprintf("[TFS] SYNC ERROR: Inode table too large\n");
            return -1;
        }
        kmemset(inode_buf, 0, aligned_bytes);
        kmemcpy(inode_buf, inodes, inode_bytes);
        if (virtio_blk_write(TFS_INODE_SECTOR, inode_sectors, inode_buf) != 0) {
            kprintf("[TFS] SYNC ERROR: Failed to write inode table\n");
            return -1;
        }
    }

    /* Write used portion of data pool */
    if (tfs_data_pool_used > 0) {
        uint32_t data_sectors = (tfs_data_pool_used + 511) / 512;
        if (virtio_blk_write(TFS_DATA_SECTOR, data_sectors, tfs_data_pool) != 0) {
            kprintf("[TFS] SYNC ERROR: Failed to write data pool\n");
            return -1;
        }
    }

    kprintf_debug("[TFS] Synced to disk: %u inodes, %lu bytes data\n",
                  inode_count, tfs_data_pool_used);
    return 0;
}

int tfs_mount(void)
{
    /* Read superblock */
    tfs_superblock_t sb;
    kmemset(&sb, 0, sizeof(sb));
    if (virtio_blk_read(TFS_SUPER_SECTOR, 1, &sb) != 0) {
        kprintf("[TFS] No disk found, using RAM-only mode\n");
        return -1;
    }

    if (sb.magic != TFS_DISK_MAGIC) {
        kprintf("[TFS] No TensorFS filesystem on disk (formatting...)\n");
        /* Format: write empty superblock */
        return tfs_sync();
    }

    if (sb.version != 1) {
        kprintf("[TFS] Unsupported TFS version %u\n", sb.version);
        return -1;
    }

    /* Restore inode table */
    inode_count = sb.inode_count;
    next_inode = sb.next_inode;
    tfs_data_pool_used = sb.data_pool_used;

    if (inode_count > 0) {
        uint32_t inode_bytes = inode_count * sizeof(tfs_inode_t);
        uint32_t inode_sectors = (inode_bytes + 511) / 512;
        static uint8_t inode_buf[255 * 512];
        uint32_t aligned_bytes = inode_sectors * 512;
        if (aligned_bytes > sizeof(inode_buf)) {
            kprintf("[TFS] WARNING: Inode table exceeds buffer\n");
            return -1;
        }
        if (virtio_blk_read(TFS_INODE_SECTOR, inode_sectors, inode_buf) != 0) {
            kprintf("[TFS] ERROR: Failed to read inode table\n");
            return -1;
        }
        kmemcpy(inodes, inode_buf, inode_bytes);

        /* Verify checksum */
        uint32_t computed = tfs_checksum(inodes, inode_bytes);
        if (computed != sb.checksum) {
            kprintf("[TFS] WARNING: Inode checksum mismatch (disk=%u, computed=%u)\n",
                    sb.checksum, computed);
        }
    }

    /* Restore data pool */
    if (tfs_data_pool_used > 0) {
        uint32_t data_sectors = (tfs_data_pool_used + 511) / 512;
        if (virtio_blk_read(TFS_DATA_SECTOR, data_sectors, tfs_data_pool) != 0) {
            kprintf("[TFS] ERROR: Failed to read data pool\n");
            return -1;
        }

        /* Reconstruct inode_data pointers from the restored data pool */
        kmemset(inode_data, 0, sizeof(inode_data));
        uint64_t offset = 0;
        for (uint32_t i = 0; i < inode_count && offset < tfs_data_pool_used; i++) {
            if (inodes[i].size > 0 && inodes[i].type != TFS_FILE_DIR) {
                inode_data[i].data = &tfs_data_pool[offset];
                inode_data[i].capacity = (inodes[i].size > 4096) ?
                    inodes[i].size : 4096;
                if (inode_data[i].capacity > TFS_MAX_FILE_DATA)
                    inode_data[i].capacity = TFS_MAX_FILE_DATA;
                offset += inode_data[i].capacity;
            }
        }
    }

    kprintf("[TFS] Mounted from disk: %u inodes, %lu bytes data\n",
            inode_count, tfs_data_pool_used);
    return 0;
}
