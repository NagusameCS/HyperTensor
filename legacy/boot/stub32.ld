/* =============================================================================
 * TensorOS - 32-bit Multiboot Stub Linker Script
 * Wraps the 64-bit kernel payload in a multiboot1-compliant elf32.
 * =============================================================================*/

OUTPUT_FORMAT(elf32-i386)
ENTRY(_start)

SECTIONS
{
    . = 1M;

    .multiboot_header :
    {
        *(.multiboot_header)
    }

    .text ALIGN(4K) :
    {
        *(.text)
        *(.text.*)
    }

    .rodata ALIGN(4K) :
    {
        *(.rodata)
        *(.rodata.*)
    }

    /* 64-bit kernel payload */
    .kernel64_payload ALIGN(4K) :
    {
        *(.kernel64_payload)
    }

    .data ALIGN(4K) :
    {
        *(.data)
        *(.data.*)
    }

    .bss ALIGN(4K) :
    {
        *(.bss)
        *(.bss.*)
        *(COMMON)
    }

    /DISCARD/ :
    {
        *(.comment)
        *(.note.*)
        *(.eh_frame)
    }
}
