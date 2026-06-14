#include <stdlib.h>
#include "common.h"
#include "matrix.h"
#include "index.h"
#include "query.h"

/*
 * Back-end 32 bit (Single Precision, SSE2).
 * fit()  -> costruisce l'indice a pivot con distanze approssimate d~(v,p).
 * predict() -> esegue il K-NN per tutte le query (pruning + d~ + euclidea reale).
 * Il calcolo della distanza approssimata passa per approximate_distance(), che
 * sotto -DUSE_SSE2_ASM invoca la routine assembly approximate_distance_sse2_asm.
 */

void fit(params *input) {
    MatrixF32 ds;
    ds.n    = (uint32_t)input->N;
    ds.d    = (uint32_t)input->D;
    ds.data = input->DS;

    input->index = (void *)build_index(&ds, input->h, input->x);
}

void predict(params *input) {
    Index *idx = (Index *)input->index;
    if (!idx) return;

    MatrixF32 ds; ds.n = (uint32_t)input->N;  ds.d = (uint32_t)input->D; ds.data = input->DS;
    MatrixF32 qs; qs.n = (uint32_t)input->nq; qs.d = (uint32_t)input->D; qs.data = input->Q;

    int k = input->k;
    Neighbor *res = (Neighbor *)malloc((size_t)input->nq * (size_t)k * sizeof(Neighbor));
    if (!res) return;

    knn_query_all(&ds, idx, &qs, k, input->x, res);

    for (int i = 0; i < input->nq; i++) {
        for (int j = 0; j < k; j++) {
            input->id_nn[i * k + j]   = res[i * k + j].id;
            input->dist_nn[i * k + j] = res[i * k + j].dist_real;
        }
    }

    free(res);
}
