; =============================================================================
; TensorOS Multiboot1 Boot Stub (elf32-i386)
;
; QEMU's -kernel flag requires a 32-bit ELF with multiboot header.
; This stub:
;   1. Sets up page tables for identity-mapping first 2GB
;   2. Enables long mode (64-bit)
;   3. Jumps to the 64-bit kernel payload at KERNEL64_LOAD_ADDR
;
; The 64-bit kernel binary is included via incbin and placed at a
; known address by the linker.
; =============================================================================

bits 32

; Multiboot1 header
section .multiboot_header
align 4
mboot_header:
    dd 0x1BADB002                       ; Magic
    dd 0x00000003                       ; Flags: align + meminfo
    dd -(0x1BADB002 + 0x00000003)       ; Checksum

section .text
global _start

KERNEL64_LOAD_ADDR equ 0x200000         ; 2MB — where the 64-bit kernel lives

_start:
    cli
    mov esp, stack_top

    ; Save multiboot registers (eax=magic, ebx=info) before using them
    push eax
    push ebx

    ; === IMMEDIATE serial test (32-bit mode) ===
    mov dx, 0x3F8
    mov al, 'B'
    out dx, al
    mov al, 'O'
    out dx, al
    mov al, 'O'
    out dx, al
    mov al, 'T'
    out dx, al
    mov al, 13    ; \r
    out dx, al
    mov al, 10    ; \n
    out dx, al

    ; Restore multiboot registers
    pop ebx
    pop eax

    ; Check multiboot magic
    cmp eax, 0x2BADB002
    jne .error

    ; Save multiboot info pointer to safe location before CPUID clobbers EBX
    ; Address 0x500 is in the conventional BIOS free area (0x500-0x7BFF)
    mov [0x500], ebx

    ; --- Check CPUID ---
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

    ; --- Check long mode ---
    mov eax, 0x80000000
    cpuid
    cmp eax, 0x80000001
    jb .error
    mov eax, 0x80000001
    cpuid
    test edx, 1 << 29
    jz .error

    ; --- Copy 64-bit kernel to KERNEL64_LOAD_ADDR ---
    ; Serial checkpoint: '1' = starting copy
    mov dx, 0x3F8
    mov al, '1'
    out dx, al

    mov esi, kernel64_payload
    mov edi, KERNEL64_LOAD_ADDR
    mov ecx, kernel64_size
    rep movsb

    ; Serial checkpoint: '2' = copy done, setting up page tables
    mov dx, 0x3F8
    mov al, '2'
    out dx, al

    ; --- Setup page tables: identity map first 4GB ---
    ; Zero page tables (6 pages: PML4 + PDPT + 4 PDs)
    mov edi, 0x1000
    mov cr3, edi
    xor eax, eax
    mov ecx, 6144
    rep stosd                     ; Clear 24KB (6 pages)

    ; PML4[0] -> PDPT at 0x2000
    mov dword [0x1000], 0x2003
    ; PDPT[0] -> PD at 0x3000  (1st GB: 0x00000000-0x3FFFFFFF)
    mov dword [0x2000], 0x3003
    ; PDPT[1] -> PD at 0x4000  (2nd GB: 0x40000000-0x7FFFFFFF)
    mov dword [0x2008], 0x4003
    ; PDPT[2] -> PD at 0x5000  (3rd GB: 0x80000000-0xBFFFFFFF)
    mov dword [0x2010], 0x5003
    ; PDPT[3] -> PD at 0x6000  (4th GB: 0xC0000000-0xFFFFFFFF -- LAPIC/IOAPIC)
    mov dword [0x2018], 0x6003

    ; PD: 512 x 2MB huge pages = 1GB (0x00000000-0x3FFFFFFF)
    mov edi, 0x3000
    mov eax, 0x00000083           ; Present + Write + Huge
    mov ecx, 512
.fill_pd0:
    mov [edi], eax
    add eax, 0x200000
    add edi, 8
    dec ecx
    jnz .fill_pd0

    ; PD: next 512 x 2MB huge pages = 1GB (0x40000000-0x7FFFFFFF)
    mov edi, 0x4000
    mov eax, 0x40000083
    mov ecx, 512
.fill_pd1:
    mov [edi], eax
    add eax, 0x200000
    add edi, 8
    dec ecx
    jnz .fill_pd1

    ; PD: next 512 x 2MB huge pages = 1GB (0x80000000-0xBFFFFFFF)
    mov edi, 0x5000
    mov eax, 0x80000083
    mov ecx, 512
.fill_pd2:
    mov [edi], eax
    add eax, 0x200000
    add edi, 8
    dec ecx
    jnz .fill_pd2

    ; PD: next 512 x 2MB huge pages = 1GB (0xC0000000-0xFFFFFFFF)
    ; This GB contains LAPIC (0xFEE00000) and IOAPIC (0xFEC00000)
    mov edi, 0x6000
    mov eax, 0xC0000083
    mov ecx, 512
.fill_pd3:
    mov [edi], eax
    add eax, 0x200000
    add edi, 8
    dec ecx
    jnz .fill_pd3

    ; --- Enable PAE ---
    ; Serial checkpoint: '3' = enabling long mode
    mov dx, 0x3F8
    mov al, '3'
    out dx, al

    mov eax, cr4
    or eax, 1 << 5
    mov cr4, eax

    ; --- Set LME in EFER ---
    mov ecx, 0xC0000080
    rdmsr
    or eax, 1 << 8
    wrmsr

    ; --- Enable paging ---
    ; Serial checkpoint: '4' = enabling paging
    mov dx, 0x3F8
    mov al, '4'
    out dx, al

    mov eax, cr0
    or eax, 1 << 31
    mov cr0, eax

    ; --- Load 64-bit GDT ---
    lgdt [gdt64_ptr]

    ; --- Far jump to 64-bit code ---
    jmp 0x08:realm64

.error:
    ; Print "ERR" in red on VGA
    mov dword [0xb8000], 0x4f524f45
    mov dword [0xb8004], 0x4f214f52
.halt:
    hlt
    jmp .halt

; =============================================================================
; 64-bit trampoline
; =============================================================================
bits 64
realm64:
    ; Load data segments
    mov ax, 0x10
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    ; Setup stack in high memory
    mov rsp, stack_top

    ; Clear screen
    mov rdi, 0xb8000
    mov rcx, 500
    mov rax, 0x0f200f200f200f20
    rep stosq

    ; Print banner
    mov rdi, 0xb8000
    mov rsi, banner
.print:
    lodsb
    test al, al
    jz .jmp_kernel
    mov ah, 0x0a
    stosw
    jmp .print

.jmp_kernel:
    ; === Serial test from 64-bit mode ===
    mov dx, 0x3F8
    mov al, 'L'
    out dx, al
    mov al, 'O'
    out dx, al
    mov al, 'N'
    out dx, al
    mov al, 'G'
    out dx, al
    mov al, 13
    out dx, al
    mov al, 10
    out dx, al

    ; Jump to 64-bit kernel at KERNEL64_LOAD_ADDR
    ; Pass multiboot info pointer in RDI (System V ABI first argument)
    xor rdi, rdi
    mov edi, [0x500]              ; Load saved multiboot info pointer
    mov rax, KERNEL64_LOAD_ADDR
    jmp rax

    cli
.halt64:
    hlt
    jmp .halt64

; =============================================================================
; Data
; =============================================================================
section .rodata
banner:
    db "TensorOS v0.1 - Long mode OK, jumping to kernel...", 0

align 8
gdt64:
    dq 0x0000000000000000          ; Null
    dq 0x00AF9A000000FFFF          ; 64-bit Code: base=0, limit=0xFFFFF, G=1, L=1, P=1, DPL=0, Type=code/read
    dq 0x00CF92000000FFFF          ; 64-bit Data: base=0, limit=0xFFFFF, G=1, P=1, DPL=0, Type=data/write
gdt64_ptr:
    dw $ - gdt64 - 1
    dq gdt64                              ; 8-byte base for 64-bit LGDT

; =============================================================================
; 64-bit kernel payload (flat binary)
; =============================================================================
section .kernel64_payload
align 4096
kernel64_payload:
    incbin "build/kernel64.bin"
kernel64_end:
kernel64_size equ kernel64_end - kernel64_payload

; =============================================================================
; BSS
; =============================================================================
section .bss
align 16
stack_bottom:
    resb 65536
stack_top:
