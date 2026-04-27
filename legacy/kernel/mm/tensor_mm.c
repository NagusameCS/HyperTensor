/*
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::.................:::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::.............................::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::......................................:::::::::::::::::::::::::::
 * ::::::::::::::::::::::::......................*%:....................::::::::::::::::::::::::
 * ::::::::::::::::::::::.......................+@@@-......................::::::::::::::::::::::
 * ::::::::::::::::::::........................+@@@@@:.......................:::::::::::::::::::
 * ::::::::::::::::::.........................=@@@@@@@:........................:::::::::::::::::
 * ::::::::::::::::..........................:@@@@@@@@@-........................:::::::::::::::
 * :::::::::::::::..........................-@@@@@@@@@@@=.........................:::::::::::::
 * :::::::::::::...........................=@@@@@@@@@@@@@-.........................::::::::::::::
 * ::::::::::::...........................-@@@@@@@@@@@@@@@..........................:::::::::::
 * :::::::::::............................:%@@@@@@@@@@@@@+...........................:::::::::
 * ::::::::::..............................=@@@@@@@@@@@@%:............................:::::::::
 * ::::::::::...............................*@@@@@@@@@@@=..............................::::::::
 * :::::::::................................:@@@@@@@@@@%:...............................::::::
 * ::::::::..................................*@@@@@@@@@-................................::::::::
 * ::::::::..................:@@+:...........:@@@@@@@@@.............:+-..................:::::::
 * :::::::...................*@@@@@@*-:.......%@@@@@@@+........:-*@@@@@..................:::::::
 * :::::::..................:@@@@@@@@@@@%:....*@@@@@@@:....:=%@@@@@@@@@=.................:::::::
 * :::::::..................*@@@@@@@@@@@@#....=@@@@@@@....:*@@@@@@@@@@@#..................::::::
 * :::::::.................:@@@@@@@@@@@@@@-...=@@@@@@@....*@@@@@@@@@@@@@:.................::::::
 * :::::::.................*@@@@@@@@@@@@@@@:..=@@@@@@#...+@@@@@@@@@@@@@@=.................::::::
 * :::::::................:@@@@@@@@@@@@@@@@*..=@@@@@@#..+@@@@@@@@@@@@@@@+.................::::::
 * :::::::................=@@@@@@@@@@@@@@@@@-.#@@@@@@@.-@@@@@@@@@@@@@@@@*................:::::::
 * :::::::...............:#@@@@@@@@@@@@@@@@@*.@@@@@@@@:@@@@@@@@@@@@@@@@@%:...............:::::::
 * ::::::::..............:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%:...............:::::::
 * ::::::::................:*@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@-...............::::::::
 * :::::::::.................:=#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%-.................::::::::
 * ::::::::::....................:#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@=...................::::::::::
 * ::::::::::.......................:*@@@@@@@@@@@@@@@@@@@@@@@@@#-.....................:::::::::
 * :::::::::::.........................:=@@@@@@@@@@@@@@@@@@*:........................:::::::::::
 * ::::::::::::......................:=%@@@@@@@@@@@@@@@@@@@@#:......................::::::::::::
 * :::::::::::::.............+#%@@@@@@@@@@@@@@%-::*-.:%@@@@@@@@%=:.................::::::::::::::
 * :::::::::::::::...........:#@@@@@@@@@@@#--+%@@@@@@@#=:=%@@@@@@@@@@-............::::::::::::::::
 * ::::::::::::::::............-@@@@@@+-=#@@@@@@@@@@@@@@@@#=-=#@@@@*:............::::::::::::::::
 * ::::::::::::::::::...........:==:...-@@@@@@@@@@@@@@@@@@@@:...:=-............:::::::::::::::::
 * :::::::::::::::::::...................@@@@@@@@@@@@@@@@@-..................::::::::::::::::::::
 * ::::::::::::::::::::::................:#@@@@@@@@@@@@@*:.................::::::::::::::::::::::
 * ::::::::::::::::::::::::...............:*@@%+-.:=#@%-................::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::.............:........................:::::::::::::::::::::::::::
 * :::::::::::::::::::::::::::::::...............................:::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::.....................:::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 * ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
 */

/* =============================================================================
 * TensorOS - Tensor-Aware Memory Manager Implementation
 *
 * Architecture:
 * ┌─────────────────────────────────────────────────────────────┐
 * │                    Unified Virtual Address Space             │
 * ├────────────┬──────────────┬──────────────┬─────────────────┤
 * │  Kernel    │  Tensor Heap │ Model Cache  │   GPU Mapped    │
 * │  (4K pgs)  │  (2MB pgs)   │  (2MB pgs)   │   (MMIO)        │
 * └────────────┴──────────────┴──────────────┴─────────────────┘
 *
 * Key design decisions:
 * 1. Tensor heap uses 2MB pages by default to reduce TLB misses during
 *    large matrix operations.
 * 2. Model weights are cached with an LRU policy, avoiding re-loading
 *    from disk for frequently used models.
 * 3. DMA allocations are aligned and pinned for zero-copy GPU transfers.
 * 4. A slab allocator handles small kernel allocations efficiently.
 * =============================================================================*/

#include "kernel/mm/tensor_mm.h"

/* =============================================================================
 * Multiboot1 Memory Map Parsing
 * =============================================================================*/

/* Global set by entry64.asm before BSS zeroing.  Lives in .data (not BSS). */
volatile uint64_t g_multiboot_info_addr __attribute__((section(".data"))) = 0;

/* Multiboot1 information structure (subset of fields we use) */
struct multiboot_info {
    uint32_t flags;
    uint32_t mem_lower;         /* KB of lower memory (below 1MB) */
    uint32_t mem_upper;         /* KB of upper memory (above 1MB) */
    uint32_t boot_device;
    uint32_t cmdline;
    uint32_t mods_count;
    uint32_t mods_addr;
    uint8_t  syms[16];
    uint32_t mmap_length;       /* total size of memory map buffer */
    uint32_t mmap_addr;         /* physical address of memory map */
} __attribute__((packed));

/* Multiboot1 memory map entry */
struct multiboot_mmap_entry {
    uint32_t size;              /* size of the rest of this entry */
    uint64_t base_addr;
    uint64_t length;
    uint32_t type;              /* 1=available, 2=reserved, 3=ACPI, 4=NVS, 5=bad */
} __attribute__((packed));

#define MBOOT_FLAG_MEM    (1 << 0)   /* mem_lower/mem_upper valid */
#define MBOOT_FLAG_MMAP   (1 << 6)   /* mmap_addr/mmap_length valid */

/* Memory region information parsed from multiboot memory map.
 * Tracks usable RAM below and above the PCI MMIO hole (~3GB-4GB). */
struct mem_regions {
    uint64_t max_addr;      /* highest physical address (for bitmap sizing) */
    uint64_t total_usable;  /* sum of all usable bytes */
    uint64_t low_top;       /* end of usable RAM below 4GB */
    uint64_t high_base;     /* start of usable RAM above 4GB (0 = none) */
    uint64_t high_end;      /* end of usable RAM above 4GB */
};

/* Detect physical memory from multiboot info structure.
 * Returns total usable bytes, or 0 if detection fails.
 * Also populates *regions if non-NULL. */
static uint64_t multiboot_detect_memory(struct mem_regions *regions)
{
    if (regions)
        kmemset(regions, 0, sizeof(*regions));

    if (g_multiboot_info_addr == 0)
        return 0;

    struct multiboot_info *mbi = (struct multiboot_info *)(uintptr_t)g_multiboot_info_addr;
    uint64_t max_addr = 0;
    uint64_t total_usable = 0;

    /* Try full memory map first (most accurate) */
    if (mbi->flags & MBOOT_FLAG_MMAP) {
        uint64_t addr = (uint64_t)mbi->mmap_addr;
        uint64_t end  = addr + mbi->mmap_length;

        while (addr < end) {
            struct multiboot_mmap_entry *entry = (struct multiboot_mmap_entry *)(uintptr_t)addr;
            if (entry->type == 1) {  /* Available RAM */
                uint64_t region_base = entry->base_addr;
                uint64_t region_end  = region_base + entry->length;
                total_usable += entry->length;
                if (region_end > max_addr)
                    max_addr = region_end;
                if (regions) {
                    /* Track low memory top (below 4GB) */
                    if (region_base < 0x100000000ULL) {
                        uint64_t low_end = region_end < 0x100000000ULL ? region_end : 0x100000000ULL;
                        if (low_end > regions->low_top)
                            regions->low_top = low_end;
                    }
                    /* Track high memory range (above 4GB) */
                    if (region_end > 0x100000000ULL) {
                        uint64_t high_start = region_base > 0x100000000ULL ? region_base : 0x100000000ULL;
                        if (regions->high_base == 0 || high_start < regions->high_base)
                            regions->high_base = high_start;
                        if (region_end > regions->high_end)
                            regions->high_end = region_end;
                    }
                }
            }
            /* Advance by entry->size + 4 (the size field itself is not included in size) */
            addr += entry->size + 4;
        }
        if (max_addr > 0) {
            if (regions) {
                regions->max_addr = max_addr;
                regions->total_usable = total_usable;
            }
            kprintf("[MM] Multiboot mmap: %lu MB usable, %lu MB address space\n",
                    total_usable / (1024 * 1024), max_addr / (1024 * 1024));
            return max_addr;
        }
    }

    /* Fallback: basic memory info */
    if (mbi->flags & MBOOT_FLAG_MEM) {
        uint64_t total = ((uint64_t)mbi->mem_upper + 1024) * 1024;  /* mem_upper is KB above 1MB */
        /* Sanity: Q35 machine type often gives garbage mem_upper.
         * If it claims > 16 GB, fall through to CMOS detection. */
        if (total <= 16ULL * 1024 * 1024 * 1024) {
            kprintf("[MM] Multiboot basic: %lu MB RAM\n", total / (1024 * 1024));
            return total;
        }
        kprintf("[MM] Multiboot basic: %lu MB (suspicious, trying CMOS)\n",
                total / (1024 * 1024));
    }

    /* CMOS-based memory detection (QEMU always populates these registers).
     * Reads extended memory (16 MB – 4 GB) from registers 0x34/0x35 in
     * 64 KB units, and high memory (above 4 GB) from 0x5b/0x5c/0x5d. */
    {
        uint8_t v;
        /* Extended memory above 16 MB (64KB units) → covers up to ~4 GB */
        __asm__ volatile("outb %0,%1" : : "a"((uint8_t)0x34), "Nd"((uint16_t)0x70));
        __asm__ volatile("inb %1,%0" : "=a"(v) : "Nd"((uint16_t)0x71));
        uint64_t ext16_lo = v;
        __asm__ volatile("outb %0,%1" : : "a"((uint8_t)0x35), "Nd"((uint16_t)0x70));
        __asm__ volatile("inb %1,%0" : "=a"(v) : "Nd"((uint16_t)0x71));
        uint64_t ext16_hi = v;
        uint64_t ext_above16m = (ext16_lo | (ext16_hi << 8)) * 65536ULL; /* bytes */

        /* High memory above 4 GB (64KB units) */
        __asm__ volatile("outb %0,%1" : : "a"((uint8_t)0x5b), "Nd"((uint16_t)0x70));
        __asm__ volatile("inb %1,%0" : "=a"(v) : "Nd"((uint16_t)0x71));
        uint64_t hi_b0 = v;
        __asm__ volatile("outb %0,%1" : : "a"((uint8_t)0x5c), "Nd"((uint16_t)0x70));
        __asm__ volatile("inb %1,%0" : "=a"(v) : "Nd"((uint16_t)0x71));
        uint64_t hi_b1 = v;
        __asm__ volatile("outb %0,%1" : : "a"((uint8_t)0x5d), "Nd"((uint16_t)0x70));
        __asm__ volatile("inb %1,%0" : "=a"(v) : "Nd"((uint16_t)0x71));
        uint64_t hi_b2 = v;
        uint64_t high_mem = (hi_b0 | (hi_b1 << 8) | (hi_b2 << 16)) * 65536ULL;

        /* Total = first 16 MB + extended + high */
        uint64_t cmos_total = 16ULL * 1024 * 1024 + ext_above16m + high_mem;
        uint64_t cmos_max   = cmos_total; /* max address ~ total for layout */

        if (cmos_total > 32ULL * 1024 * 1024) {
            if (regions) {
                regions->total_usable = cmos_total;
                /* Low RAM top: 16 MB + extended, capped at PCI hole (~3.25 GB) */
                uint64_t low = 16ULL * 1024 * 1024 + ext_above16m;
                if (low > 0xC0000000ULL) low = 0xC0000000ULL; /* cap at ~3 GB */
                regions->low_top = low;
                if (high_mem > 0) {
                    regions->high_base = 0x100000000ULL;        /* 4 GB */
                    regions->high_end  = 0x100000000ULL + high_mem;
                    cmos_max = regions->high_end;
                }
                regions->max_addr = cmos_max;
            }
            kprintf("[MM] CMOS: %lu MB below 4G, %lu MB above 4G, %lu MB total\n",
                    (16ULL * 1024 * 1024 + ext_above16m) / (1024 * 1024),
                    high_mem / (1024 * 1024),
                    cmos_total / (1024 * 1024));
            return cmos_max;
        }
    }

    return 0;
}

/* =============================================================================
 * Physical Memory Bitmap
 * Simple bitmap allocator for physical pages
 * =============================================================================*/

#define PHYS_MEM_MAX_GB     64
#define PHYS_PAGES_4K       (PHYS_MEM_MAX_GB * 1024 * 256) /* 4K pages */
#define BITMAP_SIZE         (PHYS_PAGES_4K / 8)

/* Bitmap for physical pages.  Covers up to PHYS_MEM_MAX_GB (64GB). */
static uint8_t phys_bitmap[BITMAP_SIZE];
static uint64_t phys_mem_total = 0;
static uint64_t phys_mem_free = 0;

/* Track page allocations for kfree: (ptr → page_count) */
#define KFREE_TRACK_MAX 2048
static struct { uint64_t addr; uint32_t pages; } kfree_track[KFREE_TRACK_MAX];
static int kfree_track_count = 0;

static void phys_page_mark_used(uint64_t page_index)
{
    phys_bitmap[page_index / 8] |= (1 << (page_index % 8));
    phys_mem_free -= MM_PAGE_SIZE_4K;
}

static void phys_page_mark_free(uint64_t page_index)
{
    phys_bitmap[page_index / 8] &= ~(1 << (page_index % 8));
    phys_mem_free += MM_PAGE_SIZE_4K;
}

static bool phys_page_is_free(uint64_t page_index)
{
    return !(phys_bitmap[page_index / 8] & (1 << (page_index % 8)));
}

/* Hint for phys_alloc_pages: skip past the bulk kernel region during scan */
static uint64_t phys_alloc_search_start = 0;

static uint64_t phys_alloc_pages(uint64_t count, uint64_t alignment_pages)
{
    uint64_t consecutive = 0;
    uint64_t start = 0;

    /* Start scanning from hint (skips kernel+tensor+model_cache pages) */
    for (uint64_t i = phys_alloc_search_start; i < PHYS_PAGES_4K; i++) {
        if (phys_page_is_free(i)) {
            if (consecutive == 0) {
                /* Check alignment */
                if ((i % alignment_pages) != 0) continue;
                start = i;
            }
            consecutive++;
            if (consecutive == count) {
                /* Mark all pages as used */
                for (uint64_t j = start; j < start + count; j++)
                    phys_page_mark_used(j);
                return start * MM_PAGE_SIZE_4K;
            }
        } else {
            consecutive = 0;
        }
    }
    return 0; /* Out of memory */
}

static void phys_free_pages(uint64_t phys_addr, uint64_t count)
{
    uint64_t start = phys_addr / MM_PAGE_SIZE_4K;
    for (uint64_t i = start; i < start + count; i++)
        phys_page_mark_free(i);
}

/* =============================================================================
 * Tensor Heap - Bump allocator with free list
 * Optimized for large, aligned tensor allocations
 * =============================================================================*/

typedef struct heap_block {
    uint64_t            size;
    bool                free;
    struct heap_block  *next;
    struct heap_block  *prev;
} heap_block_t;

static uint8_t *tensor_heap_base;
static uint64_t tensor_heap_size;
static uint64_t tensor_heap_used;
static heap_block_t *tensor_heap_first;

static void tensor_heap_init(uint8_t *base, uint64_t size)
{
    tensor_heap_base = base;
    tensor_heap_size = size;
    tensor_heap_used = 0;

    /* Initialize with single free block */
    tensor_heap_first = (heap_block_t *)tensor_heap_base;
    tensor_heap_first->size = tensor_heap_size - sizeof(heap_block_t);
    tensor_heap_first->free = true;
    tensor_heap_first->next = NULL;
    tensor_heap_first->prev = NULL;
}

static void *tensor_heap_alloc(uint64_t size, uint32_t alignment)
{
    if (alignment == 0) alignment = 1;
    /* Round up size to alignment (cast to 64-bit to preserve upper address bits) */
    uint64_t amask = ~((uint64_t)alignment - 1);
    size = (size + alignment - 1) & amask;

    /* First-fit search */
    heap_block_t *block = tensor_heap_first;
    while (block) {
        if (block->free && block->size >= size) {
            /* Calculate aligned address */
            uint64_t addr = (uint64_t)(block + 1);
            uint64_t aligned = (addr + alignment - 1) & amask;
            uint64_t padding = aligned - addr;
            uint64_t total_needed = size + padding;

            if (block->size >= total_needed) {
                /* Split block if enough space remaining */
                if (block->size > total_needed + sizeof(heap_block_t) + 64) {
                    heap_block_t *new_block = (heap_block_t *)
                        ((uint8_t *)block + sizeof(heap_block_t) + total_needed);
                    new_block->size = block->size - total_needed - sizeof(heap_block_t);
                    new_block->free = true;
                    new_block->next = block->next;
                    new_block->prev = block;
                    if (block->next)
                        block->next->prev = new_block;
                    block->next = new_block;
                    block->size = total_needed;
                }

                block->free = false;
                tensor_heap_used += block->size;
                return (void *)aligned;
            }
        }
        block = block->next;
    }

    return NULL; /* Out of tensor heap memory */
}

static void tensor_heap_free_block(void *ptr)
{
    if (!ptr) return;

    /* Find the block header */
    heap_block_t *block = tensor_heap_first;
    while (block) {
        uint64_t block_start = (uint64_t)(block + 1);
        uint64_t block_end = block_start + block->size;
        if ((uint64_t)ptr >= block_start && (uint64_t)ptr < block_end) {
            block->free = true;
            tensor_heap_used -= block->size;

            /* Coalesce with next block */
            if (block->next && block->next->free) {
                block->size += sizeof(heap_block_t) + block->next->size;
                block->next = block->next->next;
                if (block->next)
                    block->next->prev = block;
            }

            /* Coalesce with previous block */
            if (block->prev && block->prev->free) {
                block->prev->size += sizeof(heap_block_t) + block->size;
                block->prev->next = block->next;
                if (block->next)
                    block->next->prev = block->prev;
            }
            return;
        }
        block = block->next;
    }
}

/* =============================================================================
 * Model Weight Cache
 * LRU cache for model weights to avoid re-loading from storage
 * =============================================================================*/

static model_cache_t model_cache;
static uint8_t *model_cache_base_ptr;

static void model_cache_init(uint8_t *base, uint64_t size)
{
    kmemset(&model_cache, 0, sizeof(model_cache));
    model_cache_base_ptr = base;
    model_cache.max_size = size;
}

void *model_cache_get(uint64_t model_hash, uint64_t *size)
{
    for (uint32_t i = 0; i < model_cache.count; i++) {
        if (model_cache.entries[i].model_hash == model_hash) {
            model_cache.entries[i].last_access = kstate.uptime_ticks;
            model_cache.entries[i].ref_count++;
            model_cache.hits++;
            if (size) *size = model_cache.entries[i].size;
            return model_cache.entries[i].data;
        }
    }
    model_cache.misses++;
    return NULL;
}

int model_cache_put(uint64_t model_hash, void *data, uint64_t size)
{
    /* Check if already cached */
    for (uint32_t i = 0; i < model_cache.count; i++) {
        if (model_cache.entries[i].model_hash == model_hash)
            return 0; /* Already cached */
    }

    /* Evict if necessary */
    while (model_cache.total_size + size > model_cache.max_size) {
        model_cache_evict_lru();
    }

    if (model_cache.count >= MODEL_CACHE_MAX_ENTRIES)
        model_cache_evict_lru();

    /* Add entry */
    model_cache_entry_t *entry = &model_cache.entries[model_cache.count++];
    entry->model_hash = model_hash;
    entry->data = data;
    entry->size = size;
    entry->last_access = kstate.uptime_ticks;
    entry->ref_count = 1;
    entry->pin_count = 0;
    entry->on_gpu = false;

    model_cache.total_size += size;
    return 0;
}

void model_cache_pin(uint64_t model_hash)
{
    for (uint32_t i = 0; i < model_cache.count; i++) {
        if (model_cache.entries[i].model_hash == model_hash) {
            model_cache.entries[i].pin_count++;
            return;
        }
    }
}

void model_cache_unpin(uint64_t model_hash)
{
    for (uint32_t i = 0; i < model_cache.count; i++) {
        if (model_cache.entries[i].model_hash == model_hash) {
            if (model_cache.entries[i].pin_count > 0)
                model_cache.entries[i].pin_count--;
            return;
        }
    }
}

void model_cache_release(uint64_t model_hash)
{
    for (uint32_t i = 0; i < model_cache.count; i++) {
        if (model_cache.entries[i].model_hash == model_hash) {
            if (model_cache.entries[i].ref_count > 0)
                model_cache.entries[i].ref_count--;
            return;
        }
    }
}

void model_cache_evict_lru(void)
{
    if (model_cache.count == 0) return;

    /* Find least recently used unpinned entry */
    int lru_idx = -1;
    uint64_t oldest = UINT64_MAX;

    for (uint32_t i = 0; i < model_cache.count; i++) {
        if (model_cache.entries[i].pin_count == 0 &&
            model_cache.entries[i].ref_count == 0 &&
            model_cache.entries[i].last_access < oldest) {
            oldest = model_cache.entries[i].last_access;
            lru_idx = i;
        }
    }

    if (lru_idx < 0) return; /* Everything is pinned */

    model_cache.total_size -= model_cache.entries[lru_idx].size;

    /* Shift entries down */
    for (uint32_t i = lru_idx; i < model_cache.count - 1; i++)
        model_cache.entries[i] = model_cache.entries[i + 1];
    model_cache.count--;
}

/* =============================================================================
 * Slab Allocator for small kernel allocations
 * =============================================================================*/

#define SLAB_SIZES      8
#define SLAB_PER_PAGE   4096

typedef struct slab {
    void          *free_list;
    uint32_t       obj_size;
    uint32_t       total;
    uint32_t       used;
    struct slab   *next;
} slab_t;

static slab_t *slab_caches[SLAB_SIZES]; /* 16, 32, 64, 128, 256, 512, 1024, 2048 */
static const uint32_t slab_sizes[SLAB_SIZES] = {16, 32, 64, 128, 256, 512, 1024, 2048};

static int slab_size_index(uint64_t size)
{
    for (int i = 0; i < SLAB_SIZES; i++) {
        if (size <= slab_sizes[i])
            return i;
    }
    return -1; /* Too large for slab */
}

/* Create a new slab page for a given size class.
 * Each slab uses one 4K page: the slab_t header lives at the start,
 * followed by as many objects as fit. Each free object stores a pointer
 * to the next free object (freelist threading). */
static slab_t *slab_create(int idx)
{
    uint64_t phys = phys_alloc_pages(1, 1);
    if (phys == 0) return NULL;

    slab_t *slab = (slab_t *)phys;
    uint32_t obj_size = slab_sizes[idx];

    /* Ensure obj_size can hold a pointer for freelist threading */
    if (obj_size < sizeof(void *)) obj_size = sizeof(void *);

    slab->obj_size = obj_size;
    slab->next = slab_caches[idx];
    slab_caches[idx] = slab;

    /* Carve objects from the page, starting after the slab_t header */
    uint8_t *base = (uint8_t *)slab + ((sizeof(slab_t) + obj_size - 1) & ~(obj_size - 1));
    uint8_t *end = (uint8_t *)slab + MM_PAGE_SIZE_4K;
    slab->free_list = NULL;
    slab->total = 0;
    slab->used = 0;

    while (base + obj_size <= end) {
        *(void **)base = slab->free_list;
        slab->free_list = base;
        slab->total++;
        base += obj_size;
    }

    return slab;
}

/* Check if a pointer falls within a slab page */
static bool ptr_in_slab(slab_t *slab, void *ptr)
{
    uint64_t slab_base = (uint64_t)slab;
    uint64_t addr = (uint64_t)ptr;
    return addr >= slab_base && addr < slab_base + MM_PAGE_SIZE_4K;
}

void *kmalloc(uint64_t size)
{
    int idx = slab_size_index(size);
    if (idx >= 0) {
        /* Walk slab chain looking for a slab with free objects */
        slab_t *slab = slab_caches[idx];
        while (slab) {
            if (slab->free_list) {
                void *obj = slab->free_list;
                slab->free_list = *(void **)obj;
                slab->used++;
                return obj;
            }
            slab = slab->next;
        }
        /* No free objects — create a new slab page */
        slab = slab_create(idx);
        if (slab && slab->free_list) {
            void *obj = slab->free_list;
            slab->free_list = *(void **)obj;
            slab->used++;
            return obj;
        }
    }

    /* Fall through to page allocator for large allocations */
    uint64_t pages = (size + MM_PAGE_SIZE_4K - 1) / MM_PAGE_SIZE_4K;
    uint64_t phys = phys_alloc_pages(pages, 1);
    if (phys == 0) return NULL;

    /* Track allocation for kfree */
    if (kfree_track_count < KFREE_TRACK_MAX) {
        kfree_track[kfree_track_count].addr = phys;
        kfree_track[kfree_track_count].pages = (uint32_t)pages;
        kfree_track_count++;
    }

    return (void *)phys; /* Identity mapped in early boot */
}

void kfree(void *ptr)
{
    if (!ptr) return;
    uint64_t addr = (uint64_t)ptr;

    /* Check if it's a slab allocation — walk each size class chain */
    for (int si = 0; si < SLAB_SIZES; si++) {
        slab_t *slab = slab_caches[si];
        while (slab) {
            if (ptr_in_slab(slab, ptr)) {
                /* Return object to this slab's free list */
                *(void **)ptr = slab->free_list;
                slab->free_list = ptr;
                if (slab->used > 0) slab->used--;
                return;
            }
            slab = slab->next;
        }
    }

    /* Check page allocation tracker */
    for (int i = 0; i < kfree_track_count; i++) {
        if (kfree_track[i].addr == addr) {
            phys_free_pages(addr, kfree_track[i].pages);
            kfree_track[i] = kfree_track[--kfree_track_count];
            return;
        }
    }
}

/* =============================================================================
 * Public API Implementation
 * =============================================================================*/

/* Extend identity mapping beyond the initial 4 GB set up by the boot stub.
 * Uses low-memory pages at 0x7000+ (right after boot page tables 0x1000-0x6FFF)
 * so PD pages are always within the already-mapped first 2 MB.
 * Must be called AFTER multiboot is parsed (so we can safely reuse low memory). */
static void vm_extend_identity_map(uint64_t phys_mem_bytes)
{
    uint64_t max_gb = (phys_mem_bytes + (1ULL << 30) - 1) >> 30;
    if (max_gb <= 4) return;            /* first 4 GB already mapped */
    if (max_gb > 64) max_gb = 64;       /* cap to PDPT capacity */

    volatile uint64_t *pdpt = (volatile uint64_t *)0x2000;

    /* PD page pool at 0x7000+ (first 1 MB, identity-mapped from boot).
     * Supports up to (0x100000 - 0x7000) / 0x1000 = 249 extra GBs. */
    uint64_t pd_pool = 0x7000;

    for (uint64_t g = 4; g < max_gb; g++) {
        uint64_t pd_phys = pd_pool;
        pd_pool += 0x1000;
        if (pd_phys >= 0x100000) break;  /* safety: stay in first 1 MB */

        /* Zero and fill PD: 512 × 2MB huge pages covering this GB */
        volatile uint64_t *pd = (volatile uint64_t *)(uintptr_t)pd_phys;
        uint64_t base = g << 30;
        for (int i = 0; i < 512; i++)
            pd[i] = (base + (uint64_t)i * 0x200000) | 0x83;  /* P+W+Huge */

        /* Point PDPT entry at the new PD */
        pdpt[g] = pd_phys | 0x03;      /* Present + Write */
    }

    /* Flush TLB */
    __asm__ volatile (
        "mov %%cr3, %%rax\n\t"
        "mov %%rax, %%cr3"
        : : : "rax", "memory"
    );
}

/* Helper: bulk-mark pages [start_page, end_page) as used in bitmap. */
static void mark_pages_used_bulk(uint64_t start_page, uint64_t end_page)
{
    uint64_t first_full_byte = (start_page + 7) / 8;
    uint64_t last_full_byte  = end_page / 8;
    for (uint64_t i = start_page; i < first_full_byte * 8 && i < end_page; i++)
        phys_page_mark_used(i);
    if (last_full_byte > first_full_byte) {
        kmemset(&phys_bitmap[first_full_byte], 0xFF, last_full_byte - first_full_byte);
        uint64_t bulk_pages = (last_full_byte - first_full_byte) * 8;
        phys_mem_free -= bulk_pages * MM_PAGE_SIZE_4K;
    }
    for (uint64_t i = last_full_byte * 8; i < end_page; i++)
        phys_page_mark_used(i);
}

void tensor_mm_init(void)
{
    /* Detect physical memory and region layout from multiboot info */
    struct mem_regions regions;
    uint64_t detected = multiboot_detect_memory(&regions);
    if (detected > 0) {
        phys_mem_total = detected;
    } else {
        /* Fallback: assume 4GB (safe default for QEMU) */
        phys_mem_total = 4ULL * 1024 * 1024 * 1024;
        kprintf("[MM] No multiboot memory info, assuming 4096 MB\n");
    }

    /* Cap to our bitmap capacity */
    uint64_t bitmap_max = (uint64_t)BITMAP_SIZE * 8 * MM_PAGE_SIZE_4K;
    if (phys_mem_total > bitmap_max)
        phys_mem_total = bitmap_max;

    phys_mem_free = phys_mem_total;

    /* Mark first 2MB as reserved: BIOS data, VGA framebuffer, ISA ROM,
     * page tables (0x1000-0x4FFF), multiboot stub, stack, etc. */
    for (uint64_t i = 0; i < 0x200; i++)
        phys_page_mark_used(i);

    /* Mark static kernel region as used (text + data + BSS, up to
     * __kernel_end which is right after BSS). */
    uint64_t kernel_start_page = (uint64_t)(uintptr_t)__text_start / MM_PAGE_SIZE_4K;
    uint64_t kernel_end_page = ((uint64_t)(uintptr_t)__kernel_end + MM_PAGE_SIZE_4K - 1) / MM_PAGE_SIZE_4K;
    mark_pages_used_bulk(kernel_start_page, kernel_end_page);

    /* Mark PCI MMIO hole as used (typically ~3GB-4GB, not backed by RAM).
     * This prevents the page allocator from handing out addresses in it. */
    if (regions.low_top > 0 && regions.low_top < 0x100000000ULL) {
        uint64_t hole_start_page = regions.low_top / MM_PAGE_SIZE_4K;
        uint64_t hole_end_page   = 0x100000000ULL / MM_PAGE_SIZE_4K;
        if (hole_end_page > hole_start_page)
            mark_pages_used_bulk(hole_start_page, hole_end_page);
    }

    /* Extend identity mapping to cover all detected RAM (boot only maps 4 GB).
     * Uses dedicated low-memory pool, no dependency on phys allocator. */
    vm_extend_identity_map(phys_mem_total);

    /* ── Dynamic memory partitioning ──────────────────────────────────
     * Model cache goes in high memory (above 4GB) where there is more
     * contiguous space for large GGUF files (2+ GB).
     * Tensor heap goes in low memory (scratch buffers are small).
     *
     * Layout:  [kernel..] [tensor heap] [MMIO hole] [model cache] [free]
     * ─────────────────────────────────────────────────────────────── */
    uint64_t region_start = ((uint64_t)(uintptr_t)__dynamic_region_start + 0x1FFFFF) & ~0x1FFFFFULL;
    uint64_t reserve = 128ULL * 1024 * 1024; /* keep 128 MB for kmalloc/pages */

    uint64_t th_base, th_size, mc_base, mc_size;

    if (regions.high_base > 0 && regions.high_end > regions.high_base + reserve) {
        /* ── Split layout: tensor heap LOW, model cache HIGH ── */

        /* Tensor heap: low memory from region_start to MMIO hole */
        uint64_t low_top_2m = regions.low_top & ~0x1FFFFFULL;
        th_base = region_start;
        th_size = (low_top_2m > region_start) ? (low_top_2m - region_start) : 0;

        /* Model cache: above 4GB (plenty of room for large GGUF files) */
        mc_base = (regions.high_base + 0x1FFFFF) & ~0x1FFFFFULL;
        uint64_t high_avail = (regions.high_end > mc_base + reserve)
                            ? (regions.high_end - mc_base - reserve) : 0;
        mc_size = high_avail & ~0x1FFFFFULL;
    } else {
        /* ── Contiguous layout: everything in low memory ── */
        uint64_t available = 0;
        if (phys_mem_total > region_start + reserve)
            available = phys_mem_total - region_start - reserve;
        mc_size = (available * 60 / 100) & ~0x1FFFFFULL;
        th_size = (available - mc_size)   & ~0x1FFFFFULL;
        th_base = region_start;
        mc_base = region_start + th_size;
    }

    /* Ensure minimums */
    if (th_size < (64ULL << 20)) th_size = 64ULL << 20;
    if (mc_size < (64ULL << 20)) mc_size = 64ULL << 20;

    /* Mark tensor heap region as used in the physical bitmap */
    uint64_t th_start_page = th_base / MM_PAGE_SIZE_4K;
    uint64_t th_end_page   = (th_base + th_size + MM_PAGE_SIZE_4K - 1) / MM_PAGE_SIZE_4K;
    if (th_end_page > phys_mem_total / MM_PAGE_SIZE_4K)
        th_end_page = phys_mem_total / MM_PAGE_SIZE_4K;
    mark_pages_used_bulk(th_start_page, th_end_page);

    /* Mark model cache region as used in the physical bitmap */
    uint64_t mc_start_page = mc_base / MM_PAGE_SIZE_4K;
    uint64_t mc_end_page   = (mc_base + mc_size + MM_PAGE_SIZE_4K - 1) / MM_PAGE_SIZE_4K;
    if (mc_end_page > phys_mem_total / MM_PAGE_SIZE_4K)
        mc_end_page = phys_mem_total / MM_PAGE_SIZE_4K;
    mark_pages_used_bulk(mc_start_page, mc_end_page);

    /* Set allocator search hint past all reserved regions */
    uint64_t max_end_page = th_end_page > mc_end_page ? th_end_page : mc_end_page;
    phys_alloc_search_start = max_end_page;

    /* Initialize tensor heap */
    tensor_heap_init((uint8_t *)th_base, th_size);

    /* Initialize model cache */
    model_cache_init((uint8_t *)mc_base, mc_size);

    /* Initialize slab caches */
    kmemset(slab_caches, 0, sizeof(slab_caches));

    kstate.memory_total_bytes = phys_mem_total;
    kstate.memory_used_bytes = phys_mem_total - phys_mem_free;

    kprintf_debug("[MM] Initialized: %lu MB total, %lu MB free\n",
                  phys_mem_total / (1024 * 1024), phys_mem_free / (1024 * 1024));
    kprintf_debug("[MM] Tensor heap: %lu MB @ %p, Model cache: %lu MB @ %p\n",
                  tensor_heap_size / (1024 * 1024), (void *)tensor_heap_base,
                  model_cache.max_size / (1024 * 1024), (void *)model_cache_base_ptr);
}

void *tensor_mm_alloc(mm_alloc_request_t *req)
{
    if (!req) return NULL;

    /* Determine alignment */
    uint32_t align = req->alignment;
    if (align < MM_ALIGN_TENSOR)
        align = MM_ALIGN_TENSOR;

    switch (req->zone) {
    case MM_ZONE_TENSOR:
        return tensor_heap_alloc(req->size, align);

    case MM_ZONE_KERNEL:
        return kmalloc(req->size);

    case MM_ZONE_DMA:
        /* DMA allocations must be physically contiguous and page-aligned */
        {
            uint64_t pages = (req->size + MM_PAGE_SIZE_4K - 1) / MM_PAGE_SIZE_4K;
            return (void *)phys_alloc_pages(pages, 1);
        }

    case MM_ZONE_MODEL:
        return tensor_heap_alloc(req->size, align);

    default:
        return kmalloc(req->size);
    }
}

void tensor_mm_free(void *ptr)
{
    if (!ptr) return;

    /* Determine which zone this came from */
    uint64_t addr = (uint64_t)ptr;
    uint64_t heap_start = (uint64_t)tensor_heap_base;
    uint64_t heap_end = heap_start + tensor_heap_size;

    if (addr >= heap_start && addr < heap_end) {
        tensor_heap_free_block(ptr);
    } else {
        kfree(ptr);
    }
}

void *tensor_alloc(uint64_t size)
{
    return tensor_heap_alloc(size, MM_ALIGN_TENSOR);
}

void *tensor_alloc_pinned(uint64_t size)
{
    mm_alloc_request_t req = {
        .size = size,
        .alignment = MM_ALIGN_TENSOR,
        .zone = MM_ZONE_TENSOR,
        .flags = MM_ALLOC_PINNED | MM_ALLOC_PREFAULT,
    };
    return tensor_mm_alloc(&req);
}

void *tensor_alloc_dma(uint64_t size)
{
    mm_alloc_request_t req = {
        .size = size,
        .alignment = MM_ALIGN_PAGE,
        .zone = MM_ZONE_DMA,
        .flags = MM_ALLOC_DMA | MM_ALLOC_PINNED,
    };
    return tensor_mm_alloc(&req);
}

void *tensor_alloc_shared(uint64_t size)
{
    mm_alloc_request_t req = {
        .size = size,
        .alignment = MM_ALIGN_TENSOR,
        .zone = MM_ZONE_TENSOR,
        .flags = MM_ALLOC_SHARED,
    };
    return tensor_mm_alloc(&req);
}

void tensor_free(void *ptr)
{
    tensor_mm_free(ptr);
}

/* =============================================================================
 * Memory Maintenance
 * =============================================================================*/

void tensor_mm_defrag(void)
{
    /* Walk the free list and coalesce adjacent free blocks */
    heap_block_t *block = tensor_heap_first;
    while (block && block->next) {
        if (block->free && block->next->free) {
            block->size += sizeof(heap_block_t) + block->next->size;
            block->next = block->next->next;
            if (block->next)
                block->next->prev = block;
        } else {
            block = block->next;
        }
    }
}

void tensor_mm_cache_warmup(void)
{
    /* Touch first pages of cached blocks to bring them into CPU cache */
    heap_block_t *block = tensor_heap_first;
    uint32_t warmed = 0;
    while (block && warmed < 16) {
        if (!block->free) {
            volatile uint8_t dummy = *(volatile uint8_t *)block;
            (void)dummy;
            warmed++;
        }
        block = block->next;
    }
}

/* =============================================================================
 * Statistics & Accessors
 * =============================================================================*/

void *tensor_mm_model_cache_base(void)
{
    return (void *)model_cache_base_ptr;
}

uint64_t tensor_mm_model_cache_max(void)
{
    return model_cache.max_size;
}

uint64_t tensor_mm_heap_size(void)
{
    return tensor_heap_size;
}

uint64_t tensor_mm_cache_size(void)
{
    return model_cache.max_size;
}

uint64_t tensor_mm_free_bytes(void)
{
    return tensor_heap_size - tensor_heap_used;
}

void tensor_mm_get_stats(mm_stats_t *stats)
{
    if (!stats) return;

    stats->total_phys = phys_mem_total;
    stats->free_phys = phys_mem_free;
    stats->tensor_heap_size = tensor_heap_size;
    stats->tensor_heap_used = tensor_heap_used;
    stats->model_cache_size = model_cache.max_size;
    stats->model_cache_used = model_cache.total_size;
}

#ifndef __aarch64__
/* =============================================================================
 * Virtual Memory: 4K Page Mapping
 *
 * The boot loader identity-maps 4GB using 2MB huge pages.
 * This module can "split" a 2MB huge page into 512 × 4K pages and then
 * individually control each 4K mapping.  Primary use: demand paging for
 * model weights &mdash; fault on first access, allocate physical page, map it,
 * and resume.
 *
 * Page table layout (set up by multiboot_stub.asm):
 *   PML4   @ 0x1000
 *   PDPT   @ 0x2000
 *   PD0-7  @ 0x3000-0xA000  (each PD covers 1GB with 512 × 2MB entries)
 * =============================================================================*/

#define PT_PRESENT   0x001ULL
#define PT_WRITE     0x002ULL
#define PT_USER      0x004ULL
#define PT_HUGE      0x080ULL  /* 2MB page in PD entry */
#define PT_NX        (1ULL << 63)
#define PT_ADDR_MASK 0x000FFFFFFFFFF000ULL

/* Pool of pre-allocated 4K page table pages for splitting huge pages */
#define VM_PT_POOL_MAX 16
static uint64_t vm_pt_pool[VM_PT_POOL_MAX]; /* physical addresses of PT pages */
static int vm_pt_pool_count = 0;

/* Split a 2MB huge page into 512 × 4K pages.
 * Returns the physical address of the new page table, or 0 on failure. */
static uint64_t vm_split_huge_page(uint64_t huge_phys_base)
{
    /* Allocate a page for the new page table */
    uint64_t pt_phys;
    if (vm_pt_pool_count > 0) {
        pt_phys = vm_pt_pool[--vm_pt_pool_count];
    } else {
        pt_phys = phys_alloc_pages(1, 1);
        if (pt_phys == 0) return 0;
    }

    /* Fill 512 4K entries mapping the same 2MB region */
    volatile uint64_t *pt = (volatile uint64_t *)(uintptr_t)pt_phys;
    for (int i = 0; i < 512; i++) {
        pt[i] = (huge_phys_base + (uint64_t)i * 4096) | PT_PRESENT | PT_WRITE;
    }

    return pt_phys;
}

/* Map a single 4K page: vaddr → paddr with given flags.
/* Map a 2MB page directly in the PD (no splitting to 4K).
 * vaddr and paddr must be 2MB-aligned.  flags should include PT_HUGE.
 * Returns 0 on success, -1 on failure. */
static int vm_map_2m(uint64_t vaddr, uint64_t paddr, uint64_t flags)
{
    if (vaddr >= 0x1000000000ULL) return -1;  /* 64 GB limit */
    if ((vaddr & 0x1FFFFFULL) || (paddr & 0x1FFFFFULL)) return -1; /* not aligned */

    uint64_t pd_index = (vaddr >> 21) & 0x1FF;
    uint64_t gb_index = (vaddr >> 30) & 0x3F;  /* 0..63 */

    volatile uint64_t *pd = (volatile uint64_t *)(uintptr_t)(0x3000 + gb_index * 0x1000);
    pd[pd_index] = (paddr & 0x000FFFFFFFE00000ULL) | flags;

    /* Invalidate TLB for this 2MB region */
    __asm__ volatile ("invlpg (%0)" : : "r"(vaddr) : "memory");
    return 0;
}

/* Map a single 4K page: vaddr → paddr with given flags.
 * If the PD entry is still a 2MB huge page, splits it first.
 * Returns 0 on success, -1 on failure. */
int vm_map_4k(uint64_t vaddr, uint64_t paddr, uint64_t flags)
{
    /* Works within 64 GB identity-mapped region */
    if (vaddr >= 0x1000000000ULL) return -1;

    /* Navigate page tables */
    uint64_t pd_index  = (vaddr >> 21) & 0x1FF;    /* Which 2MB slot */
    uint64_t gb_index  = (vaddr >> 30) & 0x3F;      /* Which GB (0-63) */
    uint64_t pt_index  = (vaddr >> 12) & 0x1FF;     /* Which 4K slot within 2MB */

    /* PD base addresses: 0x3000 + gb_index * 0x1000 */
    volatile uint64_t *pd = (volatile uint64_t *)(uintptr_t)(0x3000 + gb_index * 0x1000);
    uint64_t pd_entry = pd[pd_index];

    volatile uint64_t *pt;

    if (pd_entry & PT_HUGE) {
        /* Split the 2MB huge page */
        uint64_t huge_phys = pd_entry & 0x000FFFFFFFE00000ULL; /* 2MB-aligned phys addr */
        uint64_t pt_phys = vm_split_huge_page(huge_phys);
        if (pt_phys == 0) return -1;

        /* Replace PD entry: point to new PT, remove HUGE flag */
        pd[pd_index] = pt_phys | PT_PRESENT | PT_WRITE;

        /* Invalidate TLB for the entire 2MB region */
        for (int i = 0; i < 512; i++) {
            uint64_t inv_addr = (gb_index << 30) | (pd_index << 21) | ((uint64_t)i << 12);
            __asm__ volatile ("invlpg (%0)" : : "r"(inv_addr) : "memory");
        }

        pt = (volatile uint64_t *)(uintptr_t)pt_phys;
    } else if (pd_entry & PT_PRESENT) {
        /* Already split — get existing PT */
        pt = (volatile uint64_t *)(uintptr_t)(pd_entry & PT_ADDR_MASK);
    } else {
        return -1; /* PD entry not present */
    }

    /* Set the 4K page table entry */
    pt[pt_index] = (paddr & PT_ADDR_MASK) | flags;

    /* Invalidate TLB for this page */
    __asm__ volatile ("invlpg (%0)" : : "r"(vaddr) : "memory");

    return 0;
}

/* Unmap a 4K page (set entry to not-present). */
void vm_unmap_4k(uint64_t vaddr)
{
    if (vaddr >= 0x1000000000ULL) return;

    uint64_t pd_index = (vaddr >> 21) & 0x1FF;
    uint64_t gb_index = (vaddr >> 30) & 0x3F;
    uint64_t pt_index = (vaddr >> 12) & 0x1FF;

    volatile uint64_t *pd = (volatile uint64_t *)(uintptr_t)(0x3000 + gb_index * 0x1000);
    uint64_t pd_entry = pd[pd_index];

    if ((pd_entry & PT_PRESENT) && !(pd_entry & PT_HUGE)) {
        volatile uint64_t *pt = (volatile uint64_t *)(uintptr_t)(pd_entry & PT_ADDR_MASK);
        pt[pt_index] = 0; /* Not present */
        __asm__ volatile ("invlpg (%0)" : : "r"(vaddr) : "memory");
    }
}

/* Handle a demand-page fault: allocate a physical page, map it, return success.
 * Returns 0 if the fault was handled (caller should IRETQ), -1 if not ours. */
int vm_demand_fault(uint64_t fault_addr)
{
    /* Only handle faults in the upper 2-3 GB range (where model data lives) */
    if (fault_addr < 0x40000000ULL || fault_addr >= 0x100000000ULL) return -1;

    /* Allocate a physical page */
    uint64_t phys = phys_alloc_pages(1, 1);
    if (phys == 0) return -1;

    /* Zero it */
    kmemset((void *)(uintptr_t)phys, 0, 4096);

    /* Map it at the faulting address (4K-aligned) */
    uint64_t page_addr = fault_addr & ~0xFFFULL;
    if (vm_map_4k(page_addr, phys, PT_PRESENT | PT_WRITE) != 0) {
        phys_free_pages(phys, 1);
        return -1;
    }

    return 0;
}

/* =============================================================================
 * W^X Enforcement
 *
 * After boot, split the huge pages covering kernel sections and apply:
 *   .text        → RX  (executable, not writable)
 *   .rodata      → R   (read-only, not writable, not executable)
 *   .data / .bss → RW  (writable, not executable)
 *
 * This prevents code injection (W^X: a page is either writable or executable,
 * never both). Requires NXE bit enabled in EFER (done in multiboot_stub).
 * =============================================================================*/

void vmm_enforce_wx(void)
{
    kprintf_debug("[VMM] W^X: starting rodata\n");
    /* Apply NX to .rodata pages (read-only, no execute, no write) */
    uint64_t ro_start = (uint64_t)(uintptr_t)__rodata_start & ~0xFFFULL;
    uint64_t ro_end   = ((uint64_t)(uintptr_t)__rodata_end + 0xFFF) & ~0xFFFULL;
    uint64_t count_ro = 0;
    for (uint64_t addr = ro_start; addr < ro_end; addr += 0x1000) {
        if (vm_map_4k(addr, addr, PT_PRESENT | PT_NX) == 0)
            count_ro++;
    }

    kprintf_debug("[VMM] W^X: rodata done (%lu pages), starting data\n", count_ro);
    /* Apply NX to .data pages (writable, no execute) */
    uint64_t data_start = (uint64_t)(uintptr_t)__data_start & ~0xFFFULL;
    uint64_t data_end   = ((uint64_t)(uintptr_t)__data_end + 0xFFF) & ~0xFFFULL;
    uint64_t count_data = 0;
    for (uint64_t addr = data_start; addr < data_end; addr += 0x1000) {
        if (vm_map_4k(addr, addr, PT_PRESENT | PT_WRITE | PT_NX) == 0)
            count_data++;
    }

    kprintf_debug("[VMM] W^X: data done (%lu pages), starting bss (0x%lx - 0x%lx)\n", count_data,
                  (uint64_t)(uintptr_t)__bss_start, (uint64_t)(uintptr_t)__bss_end);

    /* Apply NX to .bss pages (writable, no execute).
     * BSS can be very large (hundreds of MB for KV caches etc.),
     * so use 2MB huge pages for the aligned interior that doesn't
     * overlap with already-split .data or .rodata 2MB pages. */
    uint64_t bss_start = (uint64_t)(uintptr_t)__bss_start & ~0xFFFULL;
    uint64_t bss_end   = ((uint64_t)(uintptr_t)__bss_end + 0xFFF) & ~0xFFFULL;
    uint64_t count_bss = 0;
    {
        /* First 2MB-aligned boundary AFTER data_end (to avoid PD entries
         * that were already split for .data or .rodata 4K mappings) */
        uint64_t safe_huge_start = (data_end + 0x1FFFFFULL) & ~0x1FFFFFULL;
        if (safe_huge_start < ((bss_start + 0x1FFFFFULL) & ~0x1FFFFFULL))
            safe_huge_start = (bss_start + 0x1FFFFFULL) & ~0x1FFFFFULL;
        uint64_t huge_end = bss_end & ~0x1FFFFFULL;
        /* Map leading 4K pages (up to first safe 2MB boundary) */
        uint64_t lead_end = (safe_huge_start < bss_end) ? safe_huge_start : bss_end;
        kprintf_debug("[VMM] BSS lead: 0x%lx-0x%lx (%lu 4K pages)\n", bss_start, lead_end, (lead_end - bss_start) / 0x1000);
        for (uint64_t addr = bss_start; addr < lead_end; addr += 0x1000) {
            if (vm_map_4k(addr, addr, PT_PRESENT | PT_WRITE | PT_NX) == 0)
                count_bss++;
        }
        kprintf_debug("[VMM] BSS huge: 0x%lx-0x%lx (%lu 2M pages)\n", safe_huge_start, huge_end, (huge_end - safe_huge_start) / 0x200000);
        /* Map interior with 2MB huge pages — only fully-within-BSS, non-split pages */
        for (uint64_t addr = safe_huge_start; addr < huge_end; addr += 0x200000) {
            if (vm_map_2m(addr, addr, PT_PRESENT | PT_WRITE | PT_NX | PT_HUGE) == 0)
                count_bss += 512;
        }
        /* Map trailing 4K pages */
        uint64_t trail_start = (huge_end > safe_huge_start) ? huge_end : lead_end;
        kprintf_debug("[VMM] BSS trail: 0x%lx-0x%lx (%lu 4K pages)\n", trail_start, bss_end, (bss_end - trail_start) / 0x1000);
        for (uint64_t addr = trail_start; addr < bss_end; addr += 0x1000) {
            if (vm_map_4k(addr, addr, PT_PRESENT | PT_WRITE | PT_NX) == 0)
                count_bss++;
        }
    }

    kprintf_debug("[VMM] W^X: bss done (%lu pages), starting text\n", count_bss);
    /* .text stays as-is from the split: Present + RX (no NX, no Write) —
     * It inherits Present from the huge page split.
     * Remove the Write bit from .text pages */
    uint64_t text_start = (uint64_t)(uintptr_t)__text_start & ~0xFFFULL;
    uint64_t text_end   = ((uint64_t)(uintptr_t)__text_end + 0xFFF) & ~0xFFFULL;
    uint64_t count_text = 0;
    for (uint64_t addr = text_start; addr < text_end; addr += 0x1000) {
        if (vm_map_4k(addr, addr, PT_PRESENT) == 0) /* RX: present, no write, no NX */
            count_text++;
    }

    kprintf("[VMM] W^X enforced: .text=%lu RX, .rodata=%lu R, .data=%lu RW, .bss=%lu RW\n",
            count_text, count_ro, count_data, count_bss);
}

/* When running under a hypervisor (WHPX/Hyper-V), page table modifications
 * crash the VM.  W^X enforcement is skipped so all pages remain RWX via
 * 2MB huge pages — vmm_mark_rx/rw become no-ops. */
int vmm_hypervisor_active = 0;

/* =============================================================================
 * vmm_mark_rx() — Transition pages from RW+NX to RX (W^X compliant).
 * Used by JIT compiler: write code first, then flip to executable.
 * Pages become read+execute only (not writable).
 * =============================================================================*/
int vmm_mark_rx(void *addr, uint32_t size)
{
    if (vmm_hypervisor_active) return 0;  /* pages already RWX under hypervisor */
    uint64_t start = (uint64_t)(uintptr_t)addr & ~0xFFFULL;
    uint64_t end   = ((uint64_t)(uintptr_t)addr + size + 0xFFF) & ~0xFFFULL;
    int count = 0;
    for (uint64_t a = start; a < end; a += 0x1000) {
        if (vm_map_4k(a, a, PT_PRESENT) == 0) /* RX: no write, no NX */
            count++;
    }
    /* Flush TLB for remapped pages */
    for (uint64_t a = start; a < end; a += 0x1000)
        __asm__ volatile("invlpg (%0)" :: "r"(a) : "memory");
    return count;
}

/* vmm_mark_rw() — Transition pages back to RW+NX (for re-compilation) */
int vmm_mark_rw(void *addr, uint32_t size)
{
    if (vmm_hypervisor_active) return 0;  /* pages already RWX under hypervisor */
    uint64_t start = (uint64_t)(uintptr_t)addr & ~0xFFFULL;
    uint64_t end   = ((uint64_t)(uintptr_t)addr + size + 0xFFF) & ~0xFFFULL;
    int count = 0;
    for (uint64_t a = start; a < end; a += 0x1000) {
        if (vm_map_4k(a, a, PT_PRESENT | PT_WRITE | PT_NX) == 0)
            count++;
    }
    for (uint64_t a = start; a < end; a += 0x1000)
        __asm__ volatile("invlpg (%0)" :: "r"(a) : "memory");
    return count;
}
#endif /* !__aarch64__ */
