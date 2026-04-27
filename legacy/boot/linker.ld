/* =============================================================================
 * TensorOS - Linker Script
 * Memory layout for the TensorOS kernel image
 * =============================================================================*/

ENTRY(_start)

SECTIONS
{
    /* Kernel loads at 1MB (standard for multiboot) */
    . = 1M;

    /* Multiboot header must be in first 8KB */
    .multiboot_header :
    {
        *(.multiboot_header)
    }

    /* Code section */
    .text ALIGN(4K) :
    {
        __text_start = .;
        *(.text)
        *(.text.*)
        __text_end = .;
    }

    /* Read-only data */
    .rodata ALIGN(4K) :
    {
        __rodata_start = .;
        *(.rodata)
        *(.rodata.*)
        __rodata_end = .;
    }

    /* Initialized data */
    .data ALIGN(4K) :
    {
        __data_start = .;
        *(.data)
        *(.data.*)
        __data_end = .;
    }

    /* Uninitialized data */
    .bss ALIGN(4K) :
    {
        __bss_start = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        __bss_end = .;
    }

    /* Tensor memory region - reserved high-performance area */
    . = ALIGN(2M);
    __tensor_region_start = .;

    .tensor_heap ALIGN(2M) (NOLOAD) :
    {
        __tensor_heap_start = .;
        . += 256M;  /* 256MB default tensor heap */
        __tensor_heap_end = .;
    }

    .model_cache ALIGN(2M) (NOLOAD) :
    {
        __model_cache_start = .;
        . += 512M;  /* 512MB model weight cache */
        __model_cache_end = .;
    }

    __tensor_region_end = .;

    /* Git object store region */
    . = ALIGN(4K);
    .git_objects ALIGN(4K) (NOLOAD) :
    {
        __git_objects_start = .;
        . += 64M;   /* 64MB git object store */
        __git_objects_end = .;
    }

    __kernel_end = .;

    /* Discard unnecessary sections */
    /DISCARD/ :
    {
        *(.comment)
        *(.note.*)
        *(.eh_frame)
    }
}
