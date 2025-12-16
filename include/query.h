#ifndef QUERY_H
#define QUERY_H

#include <stdint.h>
#include <float.h>
#include "matrix.h"
#include "index.h"

// Struttura vicini con distanza approssimata + reale
typedef struct {
    int   id;
    float dist_approx;
    float dist_real;
} Neighbor;

// KNN per UNA query
void knn_query_single(const MatrixF32 *ds,
                      const Index *idx,
                      const float *q,
                      int k,
                      int x,
                      Neighbor *neighbors);

// KNN per TUTTE le query
void knn_query_all(const MatrixF32 *ds,
                   const Index *idx,
                   const MatrixF32 *queries,
                   int k,
                   int x,
                   Neighbor *results);

#endif
