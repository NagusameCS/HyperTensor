; =============================================================================
; TensorOS Bootloader - Stage 1
; Multiboot1-compliant bootloader for x86_64
; Sets up long mode and transfers control to C kernel
; =============================================================================

section .multiboot_header
align 4
header_start:
    dd 0x1BADB002                   ; Multiboot1 magic number
    dd 0x00000003                   ; Flags: align modules, provide memory map
    dd -(0x1BADB002 + 0x00000003)   ; Checksum
header_end:

; =============================================================================
; Entry point from bootloader (32-bit protected mode)
; =============================================================================
section .text
bits 32

global _start
extern kernel_main
extern klib_early_init

_start:
    ; Save multiboot info pointer
    mov edi, ebx                    ; Multiboot info struct
    mov esi, eax                    ; Multiboot magic

    ; Disable interrupts during setup
    cli

    ; Setup initial stack
    mov esp, stack_top

    ; Verify multiboot2 magic
    cmp esi, 0x36d76289
    jne .no_multiboot

    ; Check for long mode support
    call check_cpuid
    call check_long_mode

    ; Setup paging for long mode
    call setup_page_tables
    call enable_paging

    ; Load 64-bit GDT
    lgdt [gdt64.pointer]

    ; Jump to 64-bit code
    jmp gdt64.code:long_mode_start

.no_multiboot:
    mov dword [0xb8000], 0x4f524f45  ; "ER" in red
    hlt

; =============================================================================
; CPU Feature Detection
; =============================================================================
check_cpuid:
    pushfd
    pop eax
    mov ecx, eax
    xor eax, 1 << 21
    push eax
    popfd
    pushfd
    pop eax
    push ecx
    popfd
    cmp eax, ecx
    je .no_cpuid
    ret
.no_cpuid:
    mov al, "C"
    jmp error

check_long_mode:
    mov eax, 0x80000000
    cpuid
    cmp eax, 0x80000001
    jb .no_long_mode
    mov eax, 0x80000001
    cpuid
    test edx, 1 << 29
    jz .no_long_mode
    ret
.no_long_mode:
    mov al, "L"
    jmp error

; =============================================================================
; Page Table Setup - Identity map first 2GB + tensor memory region
; =============================================================================
setup_page_tables:
    ; Map P4 -> P3
    mov eax, p3_table
    or eax, 0b11                    ; Present + Writable
    mov [p4_table], eax

    ; Map P3 -> P2
    mov eax, p2_table
    or eax, 0b11
    mov [p3_table], eax

    ; Identity map first 1GB using 2MB pages
    mov ecx, 0
.map_p2:
    mov eax, 0x200000              ; 2MB
    mul ecx
    or eax, 0b10000011            ; Present + Writable + Huge
    mov [p2_table + ecx * 8], eax
    inc ecx
    cmp ecx, 512                   ; 512 entries = 1GB
    jne .map_p2

    ; Setup tensor memory region mapping (high memory)
    ; Reserve 0x100000000 - 0x200000000 for tensor workspace
    mov eax, tensor_p3_table
    or eax, 0b11
    mov [p4_table + 8], eax        ; Second P4 entry

    mov eax, tensor_p2_table
    or eax, 0b11
    mov [tensor_p3_table], eax

    ; Map 1GB for tensor operations
    mov ecx, 0
.map_tensor:
    mov eax, 0x200000
    mul ecx
    add eax, 0x40000000           ; Start after first GB
    or eax, 0b10000011
    mov [tensor_p2_table + ecx * 8], eax
    inc ecx
    cmp ecx, 512
    jne .map_tensor

    ret

enable_paging:
    ; Load P4 into CR3
    mov eax, p4_table
    mov cr3, eax

    ; Enable PAE
    mov eax, cr4
    or eax, 1 << 5
    mov cr4, eax

    ; Set long mode bit in EFER MSR
    mov ecx, 0xC0000080
    rdmsr
    or eax, 1 << 8
    wrmsr

    ; Enable paging
    mov eax, cr0
    or eax, 1 << 31
    mov cr0, eax

    ret

error:
    mov dword [0xb8000], 0x4f524f45
    mov byte [0xb8004], al
    hlt

; =============================================================================
; 64-bit Long Mode Entry
; =============================================================================
bits 64

long_mode_start:
    ; Reload segment registers
    mov ax, gdt64.data
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Setup 64-bit stack
    mov rsp, stack_top

    ; Clear screen
    mov rdi, 0xb8000
    mov rcx, 500
    mov rax, 0x0f200f200f200f20    ; Spaces with white-on-black
    rep stosq

    ; Print boot banner
    mov rdi, 0xb8000
    mov rsi, boot_banner
    call print_string

    ; Initialize core subsystems in order
    ; 1. Early serial init
    call klib_early_init

    ; 2. Transfer to kernel main (handles all init)
    call kernel_main

    ; Should never return
    cli
.halt:
    hlt
    jmp .halt

; =============================================================================
; Utility: Print null-terminated string at [rsi] to VGA at [rdi]
; =============================================================================
print_string:
    lodsb
    test al, al
    jz .done
    mov ah, 0x0a                   ; Green on black
    stosw
    jmp print_string
.done:
    ret

; =============================================================================
; Data
; =============================================================================
section .rodata
boot_banner:
    db "TensorOS v0.1.0 - AI-First Operating System", 0

; GDT for 64-bit mode
gdt64:
    dq 0                           ; Null descriptor
.code: equ $ - gdt64
    dq (1<<43)|(1<<44)|(1<<47)|(1<<53) ; Code segment
.data: equ $ - gdt64
    dq (1<<44)|(1<<47)                  ; Data segment
.pointer:
    dw $ - gdt64 - 1
    dq gdt64

; =============================================================================
; BSS - Page tables and stack
; =============================================================================
section .bss
align 4096
p4_table:       resb 4096
p3_table:       resb 4096
p2_table:       resb 4096
tensor_p3_table: resb 4096
tensor_p2_table: resb 4096

; Kernel stack (64KB)
stack_bottom:
    resb 65536
stack_top:
