/* =============================================================================
 * TensorOS - TensorFS Implementation
 * =============================================================================*/

#include "kernel/fs/tensorfs.h"

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
