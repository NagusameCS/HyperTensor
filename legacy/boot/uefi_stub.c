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
 * TensorOS - UEFI Boot Stub
 *
 * Provides a UEFI-compatible entry point that:
 *   1. Gets memory map from UEFI Boot Services
 *   2. Sets up framebuffer from GOP (Graphics Output Protocol)
 *   3. Exits Boot Services
 *   4. Sets up page tables and jumps to kernel_main()
 *
 * This file is compiled as a PE/COFF executable (EFI application) via a
 * special linker script. It coexists with the Multiboot1 boot path — the
 * same kernel binary supports both boot methods.
 *
 * Build: zig cc -target x86_64-uefi -ffreestanding -nostdlib uefi_stub.c
 *        -o BOOTX64.EFI
 * =============================================================================*/

#include "kernel/core/kernel.h"

/* =============================================================================
 * UEFI Type Definitions (subset of UEFI spec)
 * We define these ourselves to avoid depending on external headers.
 * =============================================================================*/

typedef unsigned long long UINTN;
typedef unsigned long long UINT64;
typedef unsigned int       UINT32;
typedef unsigned short     UINT16;
typedef unsigned char      UINT8;
typedef long long          INTN;
typedef unsigned short     CHAR16;
typedef void               VOID;
typedef UINTN              EFI_STATUS;
typedef void              *EFI_HANDLE;
typedef void              *EFI_EVENT;

#define EFI_SUCCESS              0ULL
#define EFI_INVALID_PARAMETER    (0x8000000000000000ULL | 2)
#define EFI_NOT_FOUND            (0x8000000000000000ULL | 14)

typedef struct {
    UINT32 Type;
    UINT64 PhysicalStart;
    UINT64 VirtualStart;
    UINT64 NumberOfPages;
    UINT64 Attribute;
} EFI_MEMORY_DESCRIPTOR;

/* Memory types */
#define EFI_CONVENTIONAL_MEMORY     7
#define EFI_BOOT_SERVICES_CODE      3
#define EFI_BOOT_SERVICES_DATA      4
#define EFI_LOADER_CODE             1
#define EFI_LOADER_DATA             2

/* GUID structure */
typedef struct {
    UINT32 Data1;
    UINT16 Data2;
    UINT16 Data3;
    UINT8  Data4[8];
} EFI_GUID;

/* Graphics Output Protocol */
typedef enum {
    PixelRedGreenBlueReserved8BitPerColor,
    PixelBlueGreenRedReserved8BitPerColor,
    PixelBitMask,
    PixelBltOnly,
} EFI_GRAPHICS_PIXEL_FORMAT;

typedef struct {
    UINT32                     Version;
    UINT32                     HorizontalResolution;
    UINT32                     VerticalResolution;
    EFI_GRAPHICS_PIXEL_FORMAT  PixelFormat;
    UINT32                     PixelInformation[4];
    UINT32                     PixelsPerScanLine;
} EFI_GRAPHICS_OUTPUT_MODE_INFORMATION;

typedef struct {
    UINT32                               MaxMode;
    UINT32                               Mode;
    EFI_GRAPHICS_OUTPUT_MODE_INFORMATION *Info;
    UINTN                                SizeOfInfo;
    UINT64                               FrameBufferBase;
    UINTN                                FrameBufferSize;
} EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE;

typedef struct EFI_GRAPHICS_OUTPUT_PROTOCOL {
    EFI_STATUS (*QueryMode)(struct EFI_GRAPHICS_OUTPUT_PROTOCOL *, UINT32, UINTN *, EFI_GRAPHICS_OUTPUT_MODE_INFORMATION **);
    EFI_STATUS (*SetMode)(struct EFI_GRAPHICS_OUTPUT_PROTOCOL *, UINT32);
    void *Blt;
    EFI_GRAPHICS_OUTPUT_PROTOCOL_MODE *Mode;
} EFI_GRAPHICS_OUTPUT_PROTOCOL;

/* Simple Text Output Protocol */
typedef struct EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL {
    void *Reset;
    EFI_STATUS (*OutputString)(struct EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *, CHAR16 *);
    void *TestString;
    void *QueryMode;
    void *SetMode;
    void *SetAttribute;
    EFI_STATUS (*ClearScreen)(struct EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *);
} EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL;

/* Boot Services (subset) */
typedef struct {
    char _hdr[24]; /* EFI_TABLE_HEADER */
    void *RaiseTPL;
    void *RestoreTPL;
    EFI_STATUS (*AllocatePages)(UINTN, UINTN, UINTN, UINT64 *);
    EFI_STATUS (*FreePages)(UINT64, UINTN);
    EFI_STATUS (*GetMemoryMap)(UINTN *, EFI_MEMORY_DESCRIPTOR *, UINTN *, UINTN *, UINT32 *);
    EFI_STATUS (*AllocatePool)(UINTN, UINTN, void **);
    EFI_STATUS (*FreePool)(void *);
    void *CreateEvent;
    void *SetTimer;
    void *WaitForEvent;
    void *SignalEvent;
    void *CloseEvent;
    void *CheckEvent;
    void *InstallProtocolInterface;
    void *ReinstallProtocolInterface;
    void *UninstallProtocolInterface;
    EFI_STATUS (*HandleProtocol)(EFI_HANDLE, EFI_GUID *, void **);
    void *Reserved;
    void *RegisterProtocolNotify;
    void *LocateHandle;
    void *LocateDevicePath;
    void *InstallConfigurationTable;
    void *LoadImage;
    void *StartImage;
    void *Exit;
    void *UnloadImage;
    EFI_STATUS (*ExitBootServices)(EFI_HANDLE, UINTN);
} EFI_BOOT_SERVICES;

/* System Table */
typedef struct {
    char _hdr[24];
    CHAR16 *FirmwareVendor;
    UINT32 FirmwareRevision;
    EFI_HANDLE ConsoleInHandle;
    void *ConIn;
    EFI_HANDLE ConsoleOutHandle;
    EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL *ConOut;
    EFI_HANDLE StdErrHandle;
    void *StdErr;
    void *RuntimeServices;
    EFI_BOOT_SERVICES *BootServices;
    UINTN NumberOfTableEntries;
    void *ConfigurationTable;
} EFI_SYSTEM_TABLE;

/* GOP GUID */
static EFI_GUID gop_guid = {
    0x9042a9de, 0x23dc, 0x4a38,
    { 0x96, 0xfb, 0x7a, 0xde, 0xd0, 0x80, 0x51, 0x6a }
};

/* =============================================================================
 * Globals passed to kernel
 * =============================================================================*/

/* Exported boot info for the kernel */
volatile uint64_t g_uefi_fb_base;
volatile uint32_t g_uefi_fb_width;
volatile uint32_t g_uefi_fb_height;
volatile uint32_t g_uefi_fb_pitch;
volatile uint64_t g_uefi_mem_total;
volatile int      g_uefi_boot;  /* 1 if booted via UEFI */

/* =============================================================================
 * Helper: Print to UEFI console
 * =============================================================================*/

static EFI_SYSTEM_TABLE *gST;

static void uefi_puts(const char *s)
{
    if (!gST || !gST->ConOut) return;
    CHAR16 buf[256];
    int i = 0;
    while (*s && i < 254) {
        if (*s == '\n') buf[i++] = '\r';
        buf[i++] = (CHAR16)*s++;
    }
    buf[i] = 0;
    gST->ConOut->OutputString(gST->ConOut, buf);
}

/* =============================================================================
 * UEFI Entry Point
 * =============================================================================*/

extern void kernel_main(void);

EFI_STATUS efi_main(EFI_HANDLE ImageHandle, EFI_SYSTEM_TABLE *SystemTable)
{
    gST = SystemTable;
    EFI_BOOT_SERVICES *BS = SystemTable->BootServices;

    uefi_puts("TensorOS UEFI Boot Stub v1.0\n");

    /* =========================================================================
     * Step 1: Get Graphics Output Protocol (framebuffer)
     * =========================================================================*/
    EFI_GRAPHICS_OUTPUT_PROTOCOL *gop = NULL;
    EFI_STATUS status = BS->HandleProtocol(
        SystemTable->ConsoleOutHandle, &gop_guid, (void **)&gop);

    if (status == EFI_SUCCESS && gop && gop->Mode) {
        g_uefi_fb_base   = gop->Mode->FrameBufferBase;
        g_uefi_fb_width  = gop->Mode->Info->HorizontalResolution;
        g_uefi_fb_height = gop->Mode->Info->VerticalResolution;
        g_uefi_fb_pitch  = gop->Mode->Info->PixelsPerScanLine * 4;

        uefi_puts("  GOP framebuffer acquired\n");
    } else {
        uefi_puts("  WARNING: No GOP framebuffer\n");
    }

    /* =========================================================================
     * Step 2: Get Memory Map
     * =========================================================================*/
    UINT8 mmap_buf[8192];
    UINTN mmap_size = sizeof(mmap_buf);
    UINTN map_key = 0;
    UINTN desc_size = 0;
    UINT32 desc_ver = 0;

    status = BS->GetMemoryMap(&mmap_size, (EFI_MEMORY_DESCRIPTOR *)mmap_buf,
                              &map_key, &desc_size, &desc_ver);
    if (status != EFI_SUCCESS) {
        uefi_puts("  ERROR: GetMemoryMap failed\n");
        return status;
    }

    /* Calculate total usable memory */
    uint64_t total_mem = 0;
    UINTN entries = mmap_size / desc_size;
    for (UINTN i = 0; i < entries; i++) {
        EFI_MEMORY_DESCRIPTOR *desc = (EFI_MEMORY_DESCRIPTOR *)(mmap_buf + i * desc_size);
        if (desc->Type == EFI_CONVENTIONAL_MEMORY ||
            desc->Type == EFI_BOOT_SERVICES_CODE ||
            desc->Type == EFI_BOOT_SERVICES_DATA) {
            total_mem += desc->NumberOfPages * 4096;
        }
    }
    g_uefi_mem_total = total_mem;

    uefi_puts("  Memory map acquired\n");

    /* =========================================================================
     * Step 3: Exit Boot Services
     * =========================================================================*/

    /* Must re-fetch map key just before ExitBootServices */
    mmap_size = sizeof(mmap_buf);
    BS->GetMemoryMap(&mmap_size, (EFI_MEMORY_DESCRIPTOR *)mmap_buf,
                     &map_key, &desc_size, &desc_ver);

    status = BS->ExitBootServices(ImageHandle, map_key);
    if (status != EFI_SUCCESS) {
        /* Retry once — map key may have changed */
        mmap_size = sizeof(mmap_buf);
        BS->GetMemoryMap(&mmap_size, (EFI_MEMORY_DESCRIPTOR *)mmap_buf,
                         &map_key, &desc_size, &desc_ver);
        status = BS->ExitBootServices(ImageHandle, map_key);
        if (status != EFI_SUCCESS) {
            uefi_puts("  ERROR: ExitBootServices failed\n");
            return status;
        }
    }

    /* =========================================================================
     * Step 4: We are now in UEFI runtime — no more Boot Services available.
     * Set up kernel environment and jump to kernel_main.
     * =========================================================================*/

    g_uefi_boot = 1;

    /* Enable SSE2 (UEFI may not have configured it) */
    __asm__ volatile(
        "mov %%cr0, %%rax\n"
        "and $~(1 << 2), %%rax\n"   /* Clear CR0.EM */
        "or  $(1 << 1), %%rax\n"    /* Set CR0.MP */
        "mov %%rax, %%cr0\n"
        "mov %%cr4, %%rax\n"
        "or  $(3 << 9), %%rax\n"    /* Set CR4.OSFXSR + OSXMMEXCPT */
        "mov %%rax, %%cr4\n"
        ::: "rax"
    );

    /* Jump to kernel_main — page tables are already identity-mapped by UEFI */
    kernel_main();

    /* Should never reach here */
    for (;;) __asm__ volatile("cli; hlt");
    return EFI_SUCCESS;
}
