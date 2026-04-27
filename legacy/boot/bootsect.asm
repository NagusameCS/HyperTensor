; =============================================================================
; TensorOS - Minimal boot sector for QEMU -fda or -hda
; Loads the flat-binary kernel from disk sector 1+ into memory at 0x100000
; and jumps to it after setting up long mode. Works with QEMU -drive.
;
; For simplicity, we use QEMU's -kernel with multiboot (elf_i386 format)
; instead. This file is an alternative boot path.
; =============================================================================
; (kept as reference, using multiboot approach instead)
