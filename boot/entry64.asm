;; =============================================================================
;; TensorOS - 64-bit kernel entry point
;; 
;; This MUST be the first object linked so it ends up at 0x200000.
;; The multiboot stub jumps here after setting up long mode.
;; =============================================================================

[BITS 64]
section .text

global long_mode_entry
extern klib_early_init
extern kernel_main

long_mode_entry:
    ;; Set up a minimal stack frame
    ;; (stack was already set in multiboot stub)

    ;; Enable SSE2 (required before any C code using float/double)
    mov rax, cr0
    and ax, 0xFFFB          ; Clear CR0.EM (bit 2) - no x87 emulation
    or  ax, 0x0002          ; Set CR0.MP (bit 1) - monitor coprocessor
    mov cr0, rax
    mov rax, cr4
    or  ax, 0x0600          ; Set CR4.OSFXSR (bit 9) + OSXMMEXCPT (bit 10)
    mov cr4, rax

    ;; Initialize serial port
    call klib_early_init

    ;; Direct serial test: print "ENTRY64\r\n" to COM1
    mov dx, 0x3F8
.serial_msg:
    ;; Wait for TX ready
    push rdx
    mov dx, 0x3FD
.wait_tx:
    in al, dx
    test al, 0x20
    jz .wait_tx
    pop rdx

    ;; Print 'E'
    mov al, 'E'
    out dx, al

    ;; Quick wait for next char
    push rdx
    mov dx, 0x3FD
.w2: in al, dx
    test al, 0x20
    jz .w2
    pop rdx
    mov al, 'N'
    out dx, al

    push rdx
    mov dx, 0x3FD
.w3: in al, dx
    test al, 0x20
    jz .w3
    pop rdx
    mov al, 'T'
    out dx, al

    push rdx
    mov dx, 0x3FD
.w4: in al, dx
    test al, 0x20
    jz .w4
    pop rdx
    mov al, 'R'
    out dx, al

    push rdx
    mov dx, 0x3FD
.w5: in al, dx
    test al, 0x20
    jz .w5
    pop rdx
    mov al, 'Y'
    out dx, al

    push rdx
    mov dx, 0x3FD
.w6: in al, dx
    test al, 0x20
    jz .w6
    pop rdx
    mov al, '6'
    out dx, al

    push rdx
    mov dx, 0x3FD
.w7: in al, dx
    test al, 0x20
    jz .w7
    pop rdx
    mov al, '4'
    out dx, al

    push rdx
    mov dx, 0x3FD
.w8: in al, dx
    test al, 0x20
    jz .w8
    pop rdx
    mov al, 0x0D   ; CR
    out dx, al

    push rdx
    mov dx, 0x3FD
.w9: in al, dx
    test al, 0x20
    jz .w9
    pop rdx
    mov al, 0x0A   ; LF
    out dx, al

    ;; Call kernel_main (System V AMD64 ABI: no args needed)
    call kernel_main

    ;; Should never return
    cli
.halt:
    hlt
    jmp .halt
