#ifndef INDEX_H
#define INDEX_H

#include <stddef.h>
#include <stdint.h>
#include "matrix.h"

// Indice delle distanze approssimate
typedef struct {
    size_t n;         // n punti nel dataset
    size_t h;         // n pivot
    size_t D;         // dimensione vettori

    size_t *pivot_ids;   // pivot scelti

    uint8_t *vp_all;  // v+ dataset  (n * D)
    uint8_t *vn_all;  // v- dataset

    uint8_t *vp_piv;  // v+ per i pivot (h * D)
    uint8_t *vn_piv;  // v- per i pivot

    int *dist;        // matrice distanze approssimate: dimensione = n * h
} Index;

// Funzioni
Index *build_index(const MatrixF32 *ds, int h, int x);     // 32 bit
Index *build_index_f64(const MatrixF64 *ds, int h, int x); // 64 bit
void free_index(Index *idx);





#endif
