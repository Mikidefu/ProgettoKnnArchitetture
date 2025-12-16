#ifndef INDEX_H
#define INDEX_H

#include <stddef.h>
#include <stdint.h>
#include "matrix.h"

// Struttura che contiene l'indice delle distanze approssimate
typedef struct {
    size_t n;         // numero di punti nel dataset
    size_t h;         // numero di pivot
    size_t D;         // dimensione dei vettori

    size_t *pivot_ids;   // gli indici scelti come pivot

    uint8_t *vp_all;  // v+ per tutto il dataset  (n * D)
    uint8_t *vn_all;  // v- per tutto il dataset

    uint8_t *vp_piv;  // v+ per i pivot (h * D)
    uint8_t *vn_piv;  // v- per i pivot

    int *dist;        // matrice distanze approssimate: size = n * h
} Index;

// Funzioni principali
Index *build_index(const MatrixF32 *ds, int h, int x);     // 32 bit
Index *build_index_f64(const MatrixF64 *ds, int h, int x); // 64 bit
void free_index(Index *idx);





#endif
