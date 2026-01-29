.intel_syntax noprefix

.section .rodata
.p2align 4
ones16:
 # Definizione di una costante di 16 byte tutti impostati a 1, usata per il conteggio dei bit
    .byte 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1

.text
.globl approximate_distance_sse2_asm

# Windows x64 ABI (MinGW-w64):
# RCX = vp
# RDX = vn
# R8  = wp
# R9  = wn
# 5° arg (D) è sullo stack. Originalmente a [RSP + 40].
# Dopo il nostro 'sub rsp, 104', sarà a [RSP + 40 + 104] = [RSP + 144].

approximate_distance_sse2_asm:
    # --- PROLOGO: SALVATAGGIO REGISTRI NON-VOLATILI ---
    # Dobbiamo salvare XMM6, XMM7, XMM8, XMM9, XMM10, XMM11
     # In Windows, i registri XMM6-XMM15 e R12-R15 devono essere preservati
    # 6 registri * 16 byte = 96 byte. Arrotondiamo a 104 per allineamento stack 16-byte.
    sub rsp, 104

    movdqu xmmword ptr [rsp + 0],  xmm6
    movdqu xmmword ptr [rsp + 16], xmm7
    movdqu xmmword ptr [rsp + 32], xmm8
    movdqu xmmword ptr [rsp + 48], xmm9
    movdqu xmmword ptr [rsp + 64], xmm10
    movdqu xmmword ptr [rsp + 80], xmm11

    # Salvataggio dei registri General Purpose (GPR) tramite push.
    push r12
    push r13
    push r14
    push r15

    # --- INIZIO LOGICA ---
    # Leggi D (offset 144 + 4*8 bytes di push = 176)
    # Calcolo: 40 (shadow) + 104 (xmm save) + 32 (gpr save) = 176. L'offset tiene conto di shadow space, XMM e push GPR.

    mov r10, qword ptr [rsp + 176]

    # Costanti:
    pxor xmm5, xmm5                                 # xmm5 = zero (volatile)
    movdqu xmm6, xmmword ptr [rip + ones16]         # xmm6 = 1s carica la maschera di 1
    pcmpeqb xmm7, xmm7                              # xmm7 = crea un amschera di tutti i bit a 1

    # fast-path D == 256
    cmp r10, 256 #se la dimensione D è 256, usa la versione srotolata (unrolled).
    je  .d256

    # ---- generic path ----
    # Inizializzazione accumulatori a zero per i 4 termini della formula binaria
    pxor xmm8,  xmm8      # acc_pp (v+ * w+)
    pxor xmm9,  xmm9      # acc_nn (v- * w-)
    pxor xmm10, xmm10     # acc_pn (v+ * w-)
    pxor xmm11, xmm11     # acc_np (v- * w+)

    mov r11, r10
    shr r11, 4            # r11 = D / 16 (numero di blocchi da 16 byte/128 bit).
    test r11, r11
    jz .reduce_generic

.loop_generic:
# Caricamento dei dati dai puntatori (16 byte alla volta).
    movdqu xmm0, xmmword ptr [rcx]   # vp
    movdqu xmm1, xmmword ptr [rdx]   # vn
    movdqu xmm2, xmmword ptr [r8]    # wp
    movdqu xmm3, xmmword ptr [r9]    # wn

    # ---- pp ----
    movdqa xmm4, xmm0
    pand   xmm4, xmm2 # xmm4 = vp & wp (AND bit-a-bit)
    # Trasformazione dei bit in valori sommabili (0 o 1 per byte):
    pcmpeqb xmm4, xmm5  # Confronta con zero: FF se uguale, 00 se diverso.
    pxor    xmm4, xmm7 # Inverte i bit (NOT): FF se bit era presente.
    pand    xmm4, xmm6 # Maschera con 1: il byte diventa 01 se bit presente, 00 altrimenti
    psadbw  xmm4, xmm5 # Somma orizzontale dei byte (Population Count SSE2).
    paddq   xmm8, xmm4 # Accumulo nel totale pp.

    # ---- nn ---- stesse operazioni di prima
    movdqa xmm4, xmm1
    pand   xmm4, xmm3
    pcmpeqb xmm4, xmm5
    pxor    xmm4, xmm7
    pand    xmm4, xmm6
    psadbw  xmm4, xmm5
    paddq   xmm9, xmm4

    # ---- pn ----
    movdqa xmm4, xmm0
    pand   xmm4, xmm3
    pcmpeqb xmm4, xmm5
    pxor    xmm4, xmm7
    pand    xmm4, xmm6
    psadbw  xmm4, xmm5
    paddq   xmm10, xmm4

    # ---- np ----
    movdqa xmm4, xmm1
    pand   xmm4, xmm2
    pcmpeqb xmm4, xmm5
    pxor    xmm4, xmm7
    pand    xmm4, xmm6
    psadbw  xmm4, xmm5
    paddq   xmm11, xmm4

    # Aggiornamento puntatori (16 byte) e contatore.
    add rcx, 16
    add rdx, 16
    add r8,  16
    add r9,  16
    dec r11
    jnz .loop_generic

.reduce_generic:
# Trasferimento dei valori dagli accumulatori XMM ai registri GPR (64-bit).
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

    # gestione del resto: remainder (D % 16) scalare
    mov r11, r10
    and r11, 15
    jz .final

.rem_loop:
# Calcolo scalare per gli ultimi byte rimanenti.
    # pp += ((vp & wp) != 0)
    mov al, byte ptr [rcx]
    and al, byte ptr [r8]
    setne dl # dl = 1 se il risultato AND non è zero.

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
# fast-path D == 256 (LOOP UNROLLING)
# --------------------------
 # Versione srotolata 16 volte (16 * 16 byte = 256 byte) per eliminare i salti.
    # La logica di calcolo è identica al loop generico ma ripetuta linearmente.
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

        # pp
        movdqa xmm4, xmm0
        pand   xmm4, xmm2
        pcmpeqb xmm4, xmm5
        pxor    xmm4, xmm7
        pand    xmm4, xmm6
        psadbw  xmm4, xmm5
        paddq   xmm8, xmm4

        # nn
        movdqa xmm4, xmm1
        pand   xmm4, xmm3
        pcmpeqb xmm4, xmm5
        pxor    xmm4, xmm7
        pand    xmm4, xmm6
        psadbw  xmm4, xmm5
        paddq   xmm9, xmm4

        # pn
        movdqa xmm4, xmm0
        pand   xmm4, xmm3
        pcmpeqb xmm4, xmm5
        pxor    xmm4, xmm7
        pand    xmm4, xmm6
        psadbw  xmm4, xmm5
        paddq   xmm10, xmm4

        # np
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
    # Ripristina GPR
    pop r15
    pop r14
    pop r13
    pop r12

    # Ripristina XMM
    movdqu xmm6,  xmmword ptr [rsp + 0]
    movdqu xmm7,  xmmword ptr [rsp + 16]
    movdqu xmm8,  xmmword ptr [rsp + 32]
    movdqu xmm9,  xmmword ptr [rsp + 48]
    movdqu xmm10, xmmword ptr [rsp + 64]
    movdqu xmm11, xmmword ptr [rsp + 80]

    add rsp, 104
    ret
