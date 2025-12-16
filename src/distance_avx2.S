.intel_syntax noprefix
.text
.globl approximate_distance_avx2_asm

# Windows x64 ABI (MinGW):
# RCX = vp
# RDX = vn
# R8  = wp
# R9  = wn
# 5° arg (D) è sullo stack a [RSP + 40] all'ingresso
#
# ritorno: int in EAX

approximate_distance_avx2_asm:
    # D in r10
    mov r10, qword ptr [rsp+40]

    # Se D == 256: percorso super-ottimizzato (8 blocchi da 32)
    cmp r10, 256
    je  .d256

    # -------------------------
    # Percorso generico: D variabile
    # -------------------------
    vpxor ymm8,  ymm8,  ymm8    # acc_pp (4xqword)
    vpxor ymm9,  ymm9,  ymm9    # acc_nn
    vpxor ymm10, ymm10, ymm10   # acc_pn
    vpxor ymm11, ymm11, ymm11   # acc_np
    vpxor ymm15, ymm15, ymm15   # zero

    # blocks = D/32 in r11
    mov r11, r10
    shr r11, 5
    test r11, r11
    jz .reduce_generic

.loop_generic:
    # carica una sola volta vp,vn,wp,wn
    vmovdqu ymm0, ymmword ptr [rcx]   # vp
    vmovdqu ymm1, ymmword ptr [rdx]   # vn
    vmovdqu ymm2, ymmword ptr [r8]    # wp
    vmovdqu ymm3, ymmword ptr [r9]    # wn

    # pp = sum(vp & wp)
    vpand   ymm4, ymm0, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq  ymm8, ymm8, ymm4

    # nn = sum(vn & wn)
    vpand   ymm4, ymm1, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq  ymm9, ymm9, ymm4

    # pn = sum(vp & wn)
    vpand   ymm4, ymm0, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq  ymm10, ymm10, ymm4

    # np = sum(vn & wp)
    vpand   ymm4, ymm1, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq  ymm11, ymm11, ymm4

    add rcx, 32
    add rdx, 32
    add r8,  32
    add r9,  32

    dec r11
    jnz .loop_generic

.reduce_generic:
    # Riduzione: converto acc YMM -> 4 scalari (pp,nn,pn,np) in shadow space
    # Useremo:
    # [rsp+8]  = pp
    # [rsp+16] = nn
    # [rsp+24] = pn
    # [rsp+32] = np

    # ---- pp (ymm8) ----
    vextracti128 xmm0, ymm8, 0
    vextracti128 xmm1, ymm8, 1
    paddq xmm0, xmm1
    movq r11, xmm0
    psrldq xmm0, 8
    movq rax, xmm0
    add r11, rax
    mov qword ptr [rsp+8], r11

    # ---- nn (ymm9) ----
    vextracti128 xmm0, ymm9, 0
    vextracti128 xmm1, ymm9, 1
    paddq xmm0, xmm1
    movq r11, xmm0
    psrldq xmm0, 8
    movq rax, xmm0
    add r11, rax
    mov qword ptr [rsp+16], r11

    # ---- pn (ymm10) ----
    vextracti128 xmm0, ymm10, 0
    vextracti128 xmm1, ymm10, 1
    paddq xmm0, xmm1
    movq r11, xmm0
    psrldq xmm0, 8
    movq rax, xmm0
    add r11, rax
    mov qword ptr [rsp+24], r11

    # ---- np (ymm11) ----
    vextracti128 xmm0, ymm11, 0
    vextracti128 xmm1, ymm11, 1
    paddq xmm0, xmm1
    movq r11, xmm0
    psrldq xmm0, 8
    movq rax, xmm0
    add r11, rax
    mov qword ptr [rsp+32], r11

    # Resto (D % 32) in r11
    mov r11, r10
    and r11, 31
    jz  .final_compute

.rem_loop:
    # pp += (vp & wp)
    mov  al, byte ptr [rcx]
    and  al, byte ptr [r8]
    movzx rax, al
    add  qword ptr [rsp+8], rax

    # nn += (vn & wn)
    mov  al, byte ptr [rdx]
    and  al, byte ptr [r9]
    movzx rax, al
    add  qword ptr [rsp+16], rax

    # pn += (vp & wn)
    mov  al, byte ptr [rcx]
    and  al, byte ptr [r9]
    movzx rax, al
    add  qword ptr [rsp+24], rax

    # np += (vn & wp)
    mov  al, byte ptr [rdx]
    and  al, byte ptr [r8]
    movzx rax, al
    add  qword ptr [rsp+32], rax

    inc rcx
    inc rdx
    inc r8
    inc r9
    dec r11
    jnz .rem_loop

.final_compute:
    # return = pp + nn - pn - np
    mov rax, qword ptr [rsp+8]
    add rax, qword ptr [rsp+16]
    sub rax, qword ptr [rsp+24]
    sub rax, qword ptr [rsp+32]

    vzeroupper
    ret


    # -------------------------
    # Percorso D == 256 (8 blocchi, no loop, no resto)
    # -------------------------
.d256:
    vpxor ymm8,  ymm8,  ymm8
    vpxor ymm9,  ymm9,  ymm9
    vpxor ymm10, ymm10, ymm10
    vpxor ymm11, ymm11, ymm11
    vpxor ymm15, ymm15, ymm15

    # macro concettuale: per offset in {0,32,64,...,224}
    # carica vp,vn,wp,wn una volta e aggiorna 4 accumulatori

    # ---- offset 0 ----
    vmovdqu ymm0, ymmword ptr [rcx+0]
    vmovdqu ymm1, ymmword ptr [rdx+0]
    vmovdqu ymm2, ymmword ptr [r8+0]
    vmovdqu ymm3, ymmword ptr [r9+0]
    vpand ymm4, ymm0, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm8, ymm8, ymm4
    vpand ymm4, ymm1, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm9, ymm9, ymm4
    vpand ymm4, ymm0, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm10, ymm10, ymm4
    vpand ymm4, ymm1, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm11, ymm11, ymm4

    # ---- offset 32 ----
    vmovdqu ymm0, ymmword ptr [rcx+32]
    vmovdqu ymm1, ymmword ptr [rdx+32]
    vmovdqu ymm2, ymmword ptr [r8+32]
    vmovdqu ymm3, ymmword ptr [r9+32]
    vpand ymm4, ymm0, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm8, ymm8, ymm4
    vpand ymm4, ymm1, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm9, ymm9, ymm4
    vpand ymm4, ymm0, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm10, ymm10, ymm4
    vpand ymm4, ymm1, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm11, ymm11, ymm4

    # ---- offset 64 ----
    vmovdqu ymm0, ymmword ptr [rcx+64]
    vmovdqu ymm1, ymmword ptr [rdx+64]
    vmovdqu ymm2, ymmword ptr [r8+64]
    vmovdqu ymm3, ymmword ptr [r9+64]
    vpand ymm4, ymm0, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm8, ymm8, ymm4
    vpand ymm4, ymm1, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm9, ymm9, ymm4
    vpand ymm4, ymm0, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm10, ymm10, ymm4
    vpand ymm4, ymm1, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm11, ymm11, ymm4

    # ---- offset 96 ----
    vmovdqu ymm0, ymmword ptr [rcx+96]
    vmovdqu ymm1, ymmword ptr [rdx+96]
    vmovdqu ymm2, ymmword ptr [r8+96]
    vmovdqu ymm3, ymmword ptr [r9+96]
    vpand ymm4, ymm0, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm8, ymm8, ymm4
    vpand ymm4, ymm1, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm9, ymm9, ymm4
    vpand ymm4, ymm0, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm10, ymm10, ymm4
    vpand ymm4, ymm1, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm11, ymm11, ymm4

    # ---- offset 128 ----
    vmovdqu ymm0, ymmword ptr [rcx+128]
    vmovdqu ymm1, ymmword ptr [rdx+128]
    vmovdqu ymm2, ymmword ptr [r8+128]
    vmovdqu ymm3, ymmword ptr [r9+128]
    vpand ymm4, ymm0, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm8, ymm8, ymm4
    vpand ymm4, ymm1, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm9, ymm9, ymm4
    vpand ymm4, ymm0, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm10, ymm10, ymm4
    vpand ymm4, ymm1, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm11, ymm11, ymm4

    # ---- offset 160 ----
    vmovdqu ymm0, ymmword ptr [rcx+160]
    vmovdqu ymm1, ymmword ptr [rdx+160]
    vmovdqu ymm2, ymmword ptr [r8+160]
    vmovdqu ymm3, ymmword ptr [r9+160]
    vpand ymm4, ymm0, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm8, ymm8, ymm4
    vpand ymm4, ymm1, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm9, ymm9, ymm4
    vpand ymm4, ymm0, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm10, ymm10, ymm4
    vpand ymm4, ymm1, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm11, ymm11, ymm4

    # ---- offset 192 ----
    vmovdqu ymm0, ymmword ptr [rcx+192]
    vmovdqu ymm1, ymmword ptr [rdx+192]
    vmovdqu ymm2, ymmword ptr [r8+192]
    vmovdqu ymm3, ymmword ptr [r9+192]
    vpand ymm4, ymm0, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm8, ymm8, ymm4
    vpand ymm4, ymm1, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm9, ymm9, ymm4
    vpand ymm4, ymm0, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm10, ymm10, ymm4
    vpand ymm4, ymm1, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm11, ymm11, ymm4

    # ---- offset 224 ----
    vmovdqu ymm0, ymmword ptr [rcx+224]
    vmovdqu ymm1, ymmword ptr [rdx+224]
    vmovdqu ymm2, ymmword ptr [r8+224]
    vmovdqu ymm3, ymmword ptr [r9+224]
    vpand ymm4, ymm0, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm8, ymm8, ymm4
    vpand ymm4, ymm1, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm9, ymm9, ymm4
    vpand ymm4, ymm0, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm10, ymm10, ymm4
    vpand ymm4, ymm1, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq ymm11, ymm11, ymm4

    # Riduzione e calcolo finale (pp+nn-pn-np) senza remainder
    # pp
    vextracti128 xmm0, ymm8, 0
    vextracti128 xmm1, ymm8, 1
    paddq xmm0, xmm1
    movq rax, xmm0
    psrldq xmm0, 8
    movq r11, xmm0
    add rax, r11
    mov qword ptr [rsp+8], rax

    # nn
    vextracti128 xmm0, ymm9, 0
    vextracti128 xmm1, ymm9, 1
    paddq xmm0, xmm1
    movq rax, xmm0
    psrldq xmm0, 8
    movq r11, xmm0
    add rax, r11
    mov qword ptr [rsp+16], rax

    # pn
    vextracti128 xmm0, ymm10, 0
    vextracti128 xmm1, ymm10, 1
    paddq xmm0, xmm1
    movq rax, xmm0
    psrldq xmm0, 8
    movq r11, xmm0
    add rax, r11
    mov qword ptr [rsp+24], rax

    # np
    vextracti128 xmm0, ymm11, 0
    vextracti128 xmm1, ymm11, 1
    paddq xmm0, xmm1
    movq rax, xmm0
    psrldq xmm0, 8
    movq r11, xmm0
    add rax, r11
    mov qword ptr [rsp+32], rax

    mov rax, qword ptr [rsp+8]
    add rax, qword ptr [rsp+16]
    sub rax, qword ptr [rsp+24]
    sub rax, qword ptr [rsp+32]

    vzeroupper
    ret
