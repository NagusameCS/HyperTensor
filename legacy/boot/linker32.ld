/* =============================================================================
 * TensorOS - Linker Script (Multiboot1 / i386 ELF container)
 *
 * Uses OUTPUT_FORMAT elf32-i386 so QEMU's multiboot recognizes it.
 * The 64-bit (long mode) code lives in .text64 and runs after the
 * boot stub has switched the CPU to long mode.
 * =============================================================================*/

OUTPUT_FORMAT(elf32-i386)
ENTRY(_start)

SECTIONS
{
    . = 1M;

    /* Multiboot header must be early */
    .multiboot_header :
    {
        *(.multiboot_header)
    }

    /* 32-bit boot stub */
    .text ALIGN(4K) :
    {
        __text_start = .;
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

    /* NOLOAD regions for large reservations */
    . = ALIGN(2M);
    __tensor_region_start = .;

    .tensor_heap ALIGN(2M) (NOLOAD) :
    {
        __tensor_heap_start = .;
        . += 256M;
        __tensor_heap_end = .;
    }

    .model_cache ALIGN(2M) (NOLOAD) :
    {
        __model_cache_start = .;
        . += 512M;
        __model_cache_end = .;
    }

    __tensor_region_end = .;

    .git_objects ALIGN(4K) (NOLOAD) :
    {
        __git_objects_start = .;
        . += 64M;
        __git_objects_end = .;
    }

    __kernel_end = .;

    /DISCARD/ :
    {
        *(.comment)
        *(.note.*)
        *(.eh_frame)
    }
}
