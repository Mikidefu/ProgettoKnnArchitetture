.intel_syntax noprefix
.text
.globl approximate_distance_avx2_asm

# Windows x64 ABI (MinGW):
# RCX = vp valori positivi del primo vettore (query)
# RDX = vn valori negativi del primo vettore (query)
# R8  = wp valori positivi del secondo vettore (dataset)
# R9  = wn valori negativi del secondo vettore (dataset)
# (D) = sullo stack a [RSP + 40] (dopo la shadow area del chiamante) è la dimensionalità dei vettori

approximate_distance_avx2_asm:
    # --- PROLOGUE: SALVATAGGIO REGISTRI NON-VOLATILI (ABI WINDOWS) ---
    # Dobbiamo preservare XMM6-XMM15 (la parte bassa 128 bit).
    # Noi usiamo XMM8, XMM9, XMM10, XMM11, XMM15.

    # Alloco spazio sullo stack (deve essere allineato a 16 byte)
    # 5 registri * 16 byte = 80 byte -> arrotondo a 88 o 96 per allineamento stack
    sub rsp, 88

    vmovdqu xmmword ptr [rsp + 0],  xmm8
    vmovdqu xmmword ptr [rsp + 16], xmm9
    vmovdqu xmmword ptr [rsp + 32], xmm10
    vmovdqu xmmword ptr [rsp + 48], xmm11
    vmovdqu xmmword ptr [rsp + 64], xmm15

    # -------------------------------------------------------------
    # CORPO DELLA FUNZIONE
    # -------------------------------------------------------------

    # D in r10. Nota: ora RSP è cambiato di 88 byte, quindi l'argomento
    # che era a [rsp+40] ora è a [rsp + 40 + 88] = [rsp + 128]

    mov r10, qword ptr [rsp+128]

    # Se D == 256: percorso super-ottimizzato
    cmp r10, 256  #faccio una compare
    je  .d256

    # --- Percorso Generico -------------------------------------------------------
    #faccio l'operazione xor su uno stesso registro per mettere il registro a 0. Lo faccio su tutti i registri
    vpxor ymm8,  ymm8,  ymm8    # ymm8 = acc_pp (v+ * w+)
    vpxor ymm9,  ymm9,  ymm9    # ymm9 = acc_nn (v- * w-)
    vpxor ymm10, ymm10, ymm10   # ymm10 = acc_pn (v+ * w-)
    vpxor ymm11, ymm11, ymm11   # ymm11 = acc_np (v- * w+)
    vpxor ymm15, ymm15, ymm15   # ymm15 = costante zero

    mov r11, r10 # Copio D in r11
    shr r11, 5  # blocks = D / 32
    test r11, r11 #se D<32 vado direttamente alla gestione dei resti
    jz .reduce_generic

.loop_generic:
# Carico i blocchi di bit (32 byte alla volta)
    vmovdqu ymm0, ymmword ptr [rcx] #ymm0 = v+
    vmovdqu ymm1, ymmword ptr [rdx] #ymm1 = v-
    vmovdqu ymm2, ymmword ptr [r8] #ymm2 = w+
    vmovdqu ymm3, ymmword ptr [r9] #ymm3 = w-

AND logico per trovare i bit comuni e PSADBW per contarli

    # v+ AND w+ (pp)
    vpand   ymm4, ymm0, ymm2  # Trova i bit a 1 comuni
    vpsadbw ymm4, ymm4, ymm15   # Conta i bit impostati (somma differenze assolute con zero)
    vpaddq  ymm8, ymm8, ymm4    # Accumulo in ymm8

    # v- AND w- (nn)
    vpand   ymm4, ymm1, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq  ymm9, ymm9, ymm4

    # v+ AND w- (pn)
    vpand   ymm4, ymm0, ymm3
    vpsadbw ymm4, ymm4, ymm15
    vpaddq  ymm10, ymm10, ymm4

    # v- AND w+ (np)
    vpand   ymm4, ymm1, ymm2
    vpsadbw ymm4, ymm4, ymm15
    vpaddq  ymm11, ymm11, ymm4

    #Avanzo i puntatori degli indirizzi di 32 byte
    add rcx, 32
    add rdx, 32
    add r8,  32
    add r9,  32
    dec r11
    jnz .loop_generic

.reduce_generic:
    # Dove salvare i dati temporanei per la riduzione?
    # Non può usare la "Shadow Area" originale [RSP+8] perché ora lì ci sono i salvataggi XMM (I registri non volatili che stiamo usando e che non vanno sovrascritti).
    # Si decide di usare i registri GPR (General Purpose Registers) volatili come RAX e R11 per non toccare lo stack.

    # ---- pp (ymm8) ----
    vextracti128 xmm0, ymm8, 0 # Estraggo i 128 bit bassi di YMM8 e li metto in xmm0
    vextracti128 xmm1, ymm8, 1 # Estraggo i 128 bit alti di YMM8 e li metto in xmm1
    paddq xmm0, xmm1 #sommo le due metà a 64 bit
    movq r11, xmm0  # Sposto i primi 64 bit del risultato in R11
    psrldq xmm0, 8  # Sposto i restanti 64 bit nella parte bassa di XMM0
    movq rax, xmm0 # Li sposto in RAX
    add r11, rax    # r11 = pp totale

    # ---- nn (ymm9) ---- faccio le stesse operazioni di prima
    vextracti128 xmm0, ymm9, 0
    vextracti128 xmm1, ymm9, 1
    paddq xmm0, xmm1
    movq r12, xmm0
    # Risparmiamo r12. Usiamo il risultato direttamente.
    # Calcoliamo (pp + nn - pn - np) in un unico registro RAX man mano.

    # RAX = pp
    mov rax, r11

    # ---- nn (ymm9) ----
    psrldq xmm0, 8 # ricalcolo nn
    # ... riscriviamo la logica finale per non usare stack o troppi registri ...

    # STRATEGIA MIGLIORE PER NON USARE TROPPI GPR:
    # Usiamo lo spazio stack profondo per i parziali, oppure calcoliamo
    # (pp + nn) - (pn + np) usando XMM0 come accumulatore temporaneo.

    # pp + nn
    vpaddq ymm8, ymm8, ymm9
    # pn + np
    vpaddq ymm10, ymm10, ymm11

    # res = (pp+nn) - (pn+np)
    vpsubq ymm8, ymm8, ymm10  #Ho eseguito le varie operazioni

    # Ora riduciamo ymm8 (che contiene il delta finale)
    vextracti128 xmm0, ymm8, 0
    vextracti128 xmm1, ymm8, 1
    paddq xmm0, xmm1
    movq rax, xmm0
    psrldq xmm0, 8
    movq r11, xmm0
    add rax, r11  # RAX contiene il risultato parziale dai blocchi

    # Resto (D % 32) Calcolo quanti byte rimangono (resto della divisione per 32)
    mov r11, r10
    and r11, 31
    jz  .epilogue_restore

.rem_loop:
# Calcolo scalare dei 4 termini per i byte rimanenti
    # pp: (vp & wp)
    mov  r10b, byte ptr [rcx]
    and  r10b, byte ptr [r8]
    movzx r10d, r10b
    add  rax, r10   # +pp

    # nn: (vn & wn)
    mov  r10b, byte ptr [rdx]
    and  r10b, byte ptr [r9]
    movzx r10d, r10b
    add  rax, r10   # +nn

    # pn: (vp & wn)
    mov  r10b, byte ptr [rcx]
    and  r10b, byte ptr [r9]
    movzx r10d, r10b
    sub  rax, r10   # -pn

    # np: (vn & wp)
    mov  r10b, byte ptr [rdx]
    and  r10b, byte ptr [r8]
    movzx r10d, r10b
    sub  rax, r10   # -np

    inc rcx
    inc rdx
    inc r8
    inc r9 #incremento i registri
    dec r11
    jnz .rem_loop

    jmp .epilogue_restore


.d256:
    # Implementazione tramite Loop Unrolling: calcolo manuale di 8 blocchi da 32 byte senza salti intermedi

    vpxor ymm8,  ymm8,  ymm8
    vpxor ymm9,  ymm9,  ymm9
    vpxor ymm10, ymm10, ymm10
    vpxor ymm11, ymm11, ymm11
    vpxor ymm15, ymm15, ymm15

    # Srotolamento manuale (invariato) per offset 0, 32, 64, 96, 128, 160, 192, 224]

    # Offset 0
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

    # Offset 32
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

    # Offset 64
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

    # Offset 96
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

    # Offset 128
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

    # Offset 160
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

    # Offset 192
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

    # Offset 224
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

    # Calcolo finale ottimizzato (pp+nn) - (pn+np) uguale a quello di prima
    vpaddq ymm8, ymm8, ymm9
    vpaddq ymm10, ymm10, ymm11
    vpsubq ymm8, ymm8, ymm10

    vextracti128 xmm0, ymm8, 0
    vextracti128 xmm1, ymm8, 1
    paddq xmm0, xmm1
    movq rax, xmm0
    psrldq xmm0, 8
    movq r11, xmm0
    add rax, r11

.epilogue_restore:
    # --- EPILOGO: RIPRISTINO REGISTRI ---
    # Ripristino i registri originali dallo stack e dealloco lo spazio
    vmovdqu xmm8,  xmmword ptr [rsp + 0]
    vmovdqu xmm9,  xmmword ptr [rsp + 16]
    vmovdqu xmm10, xmmword ptr [rsp + 32]
    vmovdqu xmm11, xmmword ptr [rsp + 48]
    vmovdqu xmm15, xmmword ptr [rsp + 64]

    add rsp, 88

    vzeroupper # Pulisce lo stato dei registri YMM
    ret # Ritorna RAX (distanza approssimata finale) al chiamante C.
