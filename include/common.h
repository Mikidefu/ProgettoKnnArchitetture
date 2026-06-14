#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>
#include <stddef.h>

/*
 * Tipo scalare e allineamento usati dai wrapper Python.
 * Le estensioni a 64 bit (double) compilano con -DQP_DOUBLE.
 *   - 32 bit: float,  registri XMM 128 bit  -> allineamento 16 byte
 *   - 64 bit: double, registri YMM 256 bit  -> allineamento 32 byte
 */
#if defined(QP_DOUBLE)
    typedef double type;
    #define align 32
#else
    typedef float type;
    #define align 16
#endif

/*
 * Struttura "ponte" tra Python (NumPy) e il codice C/Assembly.
 * Contiene i parametri del problema, i puntatori ai dati gestiti da NumPy
 * e i buffer dei risultati.
 */
typedef struct {
    type   *DS;        // dataset (N x D), gestito da NumPy
    size_t *P;         // (opzionale) indici dei pivot; gestito dall'indice
    int     h;         // numero di pivot
    int     k;         // numero di vicini
    int     x;         // parametro di quantizzazione
    int     N;         // righe del dataset
    int     D;         // colonne/feature
    void   *index;     // indice costruito (Index*)
    type   *Q;         // query (nq x D), gestito da NumPy
    int     nq;        // numero di query
    int    *id_nn;     // identificativi dei vicini (nq x k)
    type   *dist_nn;   // distanze reali dai vicini (nq x k)
    int     silent;    // modalità silenziosa
} params;

#endif
