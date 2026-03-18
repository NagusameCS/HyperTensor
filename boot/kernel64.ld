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
    __tensor_region_start = .;

    .tensor_heap ALIGN(2M) (NOLOAD) :
    {
        __tensor_heap_start = .;
        . += 512M;
        __tensor_heap_end = .;
    }

    .model_cache ALIGN(2M) (NOLOAD) :
    {
        __model_cache_start = .;
        . += 2048M;
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
