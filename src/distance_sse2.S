.intel_syntax noprefix

.section .rodata
.p2align 4
ones16:
    .byte 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1

.text
.globl approximate_distance_sse2_asm

# Windows x64 ABI (MinGW-w64):
# RCX = vp
# RDX = vn
# R8  = wp
# R9  = wn
# 5° arg (D) sullo stack a [RSP + 40] all'ingresso

approximate_distance_sse2_asm:
    # Leggi D PRIMA di cambiare RSP (altrimenti l'offset cambia)
    mov r10, qword ptr [rsp+40]     # r10 = D

    # --- salva GPR non-volatili che useremo ---
    push r12
    push r13
    push r14
    push r15

    # --- salva XMM non-volatili che useremo (XMM6-11) ---
    sub rsp, 96
    movdqu xmmword ptr [rsp+0],  xmm6
    movdqu xmmword ptr [rsp+16], xmm7
    movdqu xmmword ptr [rsp+32], xmm8
    movdqu xmmword ptr [rsp+48], xmm9
    movdqu xmmword ptr [rsp+64], xmm10
    movdqu xmmword ptr [rsp+80], xmm11

    # Costanti:
    pxor xmm5, xmm5                                # xmm5 = zero (volatile)
    movdqu xmm6, xmmword ptr [rip + ones16]         # xmm6 = 0x01 bytes (NON-volatile -> salvato)
    pcmpeqb xmm7, xmm7                              # xmm7 = 0xFF bytes (all ones) (NON-volatile -> salvato)

    # fast-path D == 256
    cmp r10, 256
    je  .d256

    # ---- generic path ----
    pxor xmm8,  xmm8      # acc_pp
    pxor xmm9,  xmm9      # acc_nn
    pxor xmm10, xmm10     # acc_pn
    pxor xmm11, xmm11     # acc_np

    mov r11, r10
    shr r11, 4            # blocks = D/16
    test r11, r11
    jz .reduce_generic

.loop_generic:
    movdqu xmm0, xmmword ptr [rcx]   # vp
    movdqu xmm1, xmmword ptr [rdx]   # vn
    movdqu xmm2, xmmword ptr [r8]    # wp
    movdqu xmm3, xmmword ptr [r9]    # wn

    # ---- pp: count((vp & wp) != 0) ----
    movdqa xmm4, xmm0
    pand   xmm4, xmm2
    pcmpeqb xmm4, xmm5        # 0xFF where ==0
    pxor    xmm4, xmm7        # 0xFF where !=0
    pand    xmm4, xmm6        # 0x01 where !=0
    psadbw  xmm4, xmm5
    paddq   xmm8, xmm4

    # ---- nn: count((vn & wn) != 0) ----
    movdqa xmm4, xmm1
    pand   xmm4, xmm3
    pcmpeqb xmm4, xmm5
    pxor    xmm4, xmm7
    pand    xmm4, xmm6
    psadbw  xmm4, xmm5
    paddq   xmm9, xmm4

    # ---- pn: count((vp & wn) != 0) ----
    movdqa xmm4, xmm0
    pand   xmm4, xmm3
    pcmpeqb xmm4, xmm5
    pxor    xmm4, xmm7
    pand    xmm4, xmm6
    psadbw  xmm4, xmm5
    paddq   xmm10, xmm4

    # ---- np: count((vn & wp) != 0) ----
    movdqa xmm4, xmm1
    pand   xmm4, xmm2
    pcmpeqb xmm4, xmm5
    pxor    xmm4, xmm7
    pand    xmm4, xmm6
    psadbw  xmm4, xmm5
    paddq   xmm11, xmm4

    add rcx, 16
    add rdx, 16
    add r8,  16
    add r9,  16
    dec r11
    jnz .loop_generic

.reduce_generic:
    # riduzione acc_pp -> rax
    movq rax, xmm8
    psrldq xmm8, 8
    movq r11, xmm8
    add rax, r11         # pp

    # riduzione acc_nn -> r12
    movq r12, xmm9
    psrldq xmm9, 8
    movq r11, xmm9
    add r12, r11         # nn

    # riduzione acc_pn -> r13
    movq r13, xmm10
    psrldq xmm10, 8
    movq r11, xmm10
    add r13, r11         # pn

    # riduzione acc_np -> r14
    movq r14, xmm11
    psrldq xmm11, 8
    movq r11, xmm11
    add r14, r11         # np

    # remainder (D % 16) scalare
    mov r11, r10
    and r11, 15
    jz .final

.rem_loop:
    # pp += ((vp & wp) != 0)
    mov al, byte ptr [rcx]
    and al, byte ptr [r8]
    setne dl
    movzx r15, dl
    add rax, r15

    # nn += ((vn & wn) != 0)
    mov al, byte ptr [rdx]
    and al, byte ptr [r9]
    setne dl
    movzx r15, dl
    add r12, r15

    # pn += ((vp & wn) != 0)
    mov al, byte ptr [rcx]
    and al, byte ptr [r9]
    setne dl
    movzx r15, dl
    add r13, r15

    # np += ((vn & wp) != 0)
    mov al, byte ptr [rdx]
    and al, byte ptr [r8]
    setne dl
    movzx r15, dl
    add r14, r15

    inc rcx
    inc rdx
    inc r8
    inc r9
    dec r11
    jnz .rem_loop

.final:
    # return = pp + nn - pn - np
    add rax, r12
    sub rax, r13
    sub rax, r14
    jmp .epilogue

# --------------------------
# fast-path D == 256 (16x16B)
# --------------------------
.d256:
    pxor xmm8,  xmm8
    pxor xmm9,  xmm9
    pxor xmm10, xmm10
    pxor xmm11, xmm11

    .set off, 0
    .rept 16
        movdqu xmm0, xmmword ptr [rcx+off]
        movdqu xmm1, xmmword ptr [rdx+off]
        movdqu xmm2, xmmword ptr [r8 +off]
        movdqu xmm3, xmmword ptr [r9 +off]

        movdqa xmm4, xmm0
        pand   xmm4, xmm2
        pcmpeqb xmm4, xmm5
        pxor    xmm4, xmm7
        pand    xmm4, xmm6
        psadbw  xmm4, xmm5
        paddq   xmm8, xmm4

        movdqa xmm4, xmm1
        pand   xmm4, xmm3
        pcmpeqb xmm4, xmm5
        pxor    xmm4, xmm7
        pand    xmm4, xmm6
        psadbw  xmm4, xmm5
        paddq   xmm9, xmm4

        movdqa xmm4, xmm0
        pand   xmm4, xmm3
        pcmpeqb xmm4, xmm5
        pxor    xmm4, xmm7
        pand    xmm4, xmm6
        psadbw  xmm4, xmm5
        paddq   xmm10, xmm4

        movdqa xmm4, xmm1
        pand   xmm4, xmm2
        pcmpeqb xmm4, xmm5
        pxor    xmm4, xmm7
        pand    xmm4, xmm6
        psadbw  xmm4, xmm5
        paddq   xmm11, xmm4

        .set off, off+16
    .endr

    movq rax, xmm8
    psrldq xmm8, 8
    movq r11, xmm8
    add rax, r11

    movq r12, xmm9
    psrldq xmm9, 8
    movq r11, xmm9
    add r12, r11

    movq r13, xmm10
    psrldq xmm10, 8
    movq r11, xmm10
    add r13, r11

    movq r14, xmm11
    psrldq xmm11, 8
    movq r11, xmm11
    add r14, r11

    add rax, r12
    sub rax, r13
    sub rax, r14
    jmp .epilogue

.epilogue:
    # ripristina XMM non-volatili
    movdqu xmm6,  xmmword ptr [rsp+0]
    movdqu xmm7,  xmmword ptr [rsp+16]
    movdqu xmm8,  xmmword ptr [rsp+32]
    movdqu xmm9,  xmmword ptr [rsp+48]
    movdqu xmm10, xmmword ptr [rsp+64]
    movdqu xmm11, xmmword ptr [rsp+80]
    add rsp, 96

    # ripristina GPR non-volatili
    pop r15
    pop r14
    pop r13
    pop r12
    ret
