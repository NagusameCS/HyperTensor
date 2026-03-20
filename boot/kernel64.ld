/* =============================================================================
 * TensorOS - 64-bit Kernel Linker Script
 * Produces a flat binary loaded at 0x200000 (2MB)
 * =============================================================================*/

OUTPUT_FORMAT(elf64-x86-64)
ENTRY(long_mode_entry)

SECTIONS
{
    . = 0x200000;

    .text ALIGN(4K) :
    {
        __text_start = .;
        *(.text.entry)
        *(.text)
        *(.text.*)
        __text_end = .;
    }

    .rodata ALIGN(4K) :
    {
        __rodata_start = .;
        *(.rodata)
        *(.rodata.*)
        __rodata_end = .;
    }

    .data ALIGN(4K) :
    {
        __data_start = .;
        *(.data)
        *(.data.*)
        __data_end = .;
    }

    .bss ALIGN(4K) :
    {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        __bss_end = .;
    }

    . = ALIGN(2M);

    /* Dynamic memory region: tensor_mm_init() partitions all RAM beyond
     * this point into tensor-heap, model-cache, and git-object-store
     * at runtime based on detected physical memory. */
    __dynamic_region_start = .;
    __kernel_end = .;

    /DISCARD/ :
    {
        *(.comment)
        *(.note.*)
        *(.eh_frame)
    }
}
