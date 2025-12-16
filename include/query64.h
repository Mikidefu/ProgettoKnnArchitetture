#ifndef QUERY64_H
#define QUERY64_H

#include "matrix.h"
#include "index.h"

typedef struct {
    int    id;
    double dist_approx;
    double dist_real;
} Neighbor64;

void knn_query_single_f64(const MatrixF64 *ds,
                          const Index *idx,
                          const double *q,
                          int k,
                          int x,
                          Neighbor64 *neighbors);

void knn_query_all_f64(const MatrixF64 *ds,
                       const Index *idx,
                       const MatrixF64 *queries,
                       int k,
                       int x,
                       Neighbor64 *results);

#endif
