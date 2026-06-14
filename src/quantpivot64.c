#include <stdlib.h>
#include "common.h"
#include "matrix.h"
#include "index.h"
#include "query64.h"

/*
 * Back-end 64 bit (Double Precision, AVX2), versione seriale.
 * Compilato con -DQP_DOUBLE e -DUSE_AVX_ASM: approximate_distance() invoca
 * la routine assembly approximate_distance_avx2_asm.
 */

void fit(params *input) {
    MatrixF64 ds;
    ds.n    = (uint32_t)input->N;
    ds.d    = (uint32_t)input->D;
    ds.data = input->DS;

    input->index = (void *)build_index_f64(&ds, input->h, input->x);
}

void predict(params *input) {
    Index *idx = (Index *)input->index;
    if (!idx) return;

    MatrixF64 ds; ds.n = (uint32_t)input->N;  ds.d = (uint32_t)input->D; ds.data = input->DS;
    MatrixF64 qs; qs.n = (uint32_t)input->nq; qs.d = (uint32_t)input->D; qs.data = input->Q;

    int k = input->k;
    Neighbor64 *res = (Neighbor64 *)malloc((size_t)input->nq * (size_t)k * sizeof(Neighbor64));
    if (!res) return;

    knn_query_all_f64(&ds, idx, &qs, k, input->x, res);

    for (int i = 0; i < input->nq; i++) {
        for (int j = 0; j < k; j++) {
            input->id_nn[i * k + j]   = res[i * k + j].id;
            input->dist_nn[i * k + j] = res[i * k + j].dist_real;
        }
    }

    free(res);
}
