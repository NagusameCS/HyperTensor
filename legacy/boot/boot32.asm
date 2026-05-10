; =============================================================================
; TensorOS Bootloader - Multiboot1 32-bit stub
; This is compiled as elf32 for QEMU multiboot compatibility.
; It sets up long mode and jumps to the 64-bit kernel.
; =============================================================================

section .multiboot_header
align 4
    dd 0x1BADB002                       ; Multiboot magic
    dd 0x00000003                       ; Flags: page-align, meminfo
    dd -(0x1BADB002 + 0x00000003)       ; Checksum

section .text
bits 32
global _start
extern long_mode_entry

_start:
    ; Save multiboot info
    mov edi, ebx
    mov esi, eax
    cli
    mov esp, stack_top

    ; Verify multiboot magic
    cmp eax, 0x2BADB002
    jne .error

    ; Check CPUID support
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
    je .error

    ; Check long mode
    mov eax, 0x80000000
    cpuid
    cmp eax, 0x80000001
    jb .error
    mov eax, 0x80000001
    cpuid
    test edx, 1 << 29
    jz .error

    ; Setup page tables — identity map first 2GB
    ; P4[0] -> P3
    mov eax, p3_table
    or eax, 0b11
    mov [p4_table], eax

    ; P3[0] -> P2
    mov eax, p2_table
    or eax, 0b11
    mov [p3_table], eax

    ; P2: 512 x 2MB huge pages = 1GB identity map
    mov ecx, 0
.map_p2:
    mov eax, 0x200000
    mul ecx
    or eax, 0b10000011             ; Present + Write + Huge
    mov [p2_table + ecx * 8], eax
    inc ecx
    cmp ecx, 512
    jne .map_p2

    ; Second P3 entry -> P2b for 1-2GB range
    mov eax, p2b_table
    or eax, 0b11
    mov [p3_table + 8], eax

    mov ecx, 0
.map_p2b:
    mov eax, 0x200000
    mul ecx
    add eax, 0x40000000
    or eax, 0b10000011
    mov [p2b_table + ecx * 8], eax
    inc ecx
    cmp ecx, 512
    jne .map_p2b

    ; Load P4 into CR3
    mov eax, p4_table
    mov cr3, eax

    ; Enable PAE (CR4.PAE)
    mov eax, cr4
    or eax, 1 << 5
    mov cr4, eax

    ; Enable long mode (EFER.LME)
    mov ecx, 0xC0000080
    rdmsr
    or eax, 1 << 8
    wrmsr

    ; Enable paging (CR0.PG)
    mov eax, cr0
    or eax, 1 << 31
    mov cr0, eax

    ; Load 64-bit GDT and far jump to 64-bit code
    lgdt [gdt64.pointer]
    jmp gdt64.code:long_mode_start

.error:
    mov dword [0xb8000], 0x4f524f45
    mov dword [0xb8004], 0x4f214f52
    hlt
    jmp .error

; =============================================================================
; 64-bit entry point
; =============================================================================
bits 64
long_mode_start:
    ; Reload segment registers with 64-bit data segment
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
    mov rax, 0x0f200f200f200f20
    rep stosq

    ; Print banner
    mov rdi, 0xb8000
    mov rsi, boot_banner
.print:
    lodsb
    test al, al
    jz .done_print
    mov ah, 0x0a
    stosw
    jmp .print
.done_print:

    ; Jump to C kernel
    call long_mode_entry

    ; Halt
    cli
.halt:
    hlt
    jmp .halt

; =============================================================================
; Data
; =============================================================================
section .rodata
boot_banner:
    db "TensorOS v0.1 - Booting...", 0

align 8
gdt64:
    dq 0                                    ; Null
.code: equ $ - gdt64
    dq (1<<43)|(1<<44)|(1<<47)|(1<<53)     ; 64-bit code
.data: equ $ - gdt64
    dq (1<<44)|(1<<47)                      ; Data
.pointer:
    dw $ - gdt64 - 1
    dq gdt64

; =============================================================================
; BSS
; =============================================================================
section .bss
align 4096
p4_table:   resb 4096
p3_table:   resb 4096
p2_table:   resb 4096
p2b_table:  resb 4096

align 16
stack_bottom:
    resb 65536
stack_top:
