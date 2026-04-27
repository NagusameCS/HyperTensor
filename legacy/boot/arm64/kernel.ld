/* =============================================================================
 * TensorOS ARM64 Linker Script for Raspberry Pi
 *
 * RPi GPU firmware loads kernel8.img to 0x80000 (AArch64).
 * Layout:
 *   0x00001000  Page tables (set up by boot.S)
 *   0x00080000  _start (boot.S text)
 *   .text       Kernel code
 *   .rodata     Read-only data
 *   .data       Initialized data
 *   .bss        Zero-initialized data
 *   <heap>      256 MB tensor heap
 *   <cache>     512 MB model cache
 * =============================================================================*/

ENTRY(_start)

SECTIONS
{
    . = 0x80000;

    __text_start = .;

    .text : {
        KEEP(*(.text.boot))       /* Boot stub must be first */
        *(.text .text.*)
    }

    .rodata : ALIGN(4096) {
        *(.rodata .rodata.*)
    }

    .data : ALIGN(4096) {
        *(.data .data.*)
    }

    .bss : ALIGN(4096) {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
    }
    __bss_end = ALIGN(8);

    . = ALIGN(4096);
    __kernel_end = .;

    /* Tensor heap: 256 MB */
    . = ALIGN(1M);
    __tensor_heap_start = .;
    . = . + 256M;
    __tensor_heap_end = .;

    /* Model cache: 512 MB */
    . = ALIGN(1M);
    __model_cache_start = .;
    . = . + 512M;
    __model_cache_end = .;

    /* Git object store: 64 MB */
    . = ALIGN(1M);
    __git_objects_start = .;
    . = . + 64M;
    __git_objects_end = .;

    /DISCARD/ : {
        *(.comment)
        *(.note*)
        *(.eh_frame*)
    }
}
