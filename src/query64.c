#include "query64.h"
#include "quantization.h"
#include "distance.h"

#include <float.h>
#include <stdlib.h>

// worst neighbor (max dist_approx)
static int find_worst_neighbor64(const Neighbor64 *neighbors, int k)
{
    int worst = 0;
    double maxd = neighbors[0].dist_approx;
    for (int i = 1; i < k; i++) {
        if (neighbors[i].dist_approx > maxd) {
            maxd = neighbors[i].dist_approx;
            worst = i;
        }
    }
    return worst;
}

void knn_query_single_f64(const MatrixF64 *ds,
                          const Index *idx,
                          const double *q,
                          int k,
                          int x,
                          Neighbor64 *neighbors)
{
    if (!ds || !idx || !q || !neighbors) return;

    size_t n = ds->n;
    size_t D = ds->d;
    int h = (int)idx->h;

    if (k > (int)n) k = (int)n;

    for (int i = 0; i < k; i++) {
        neighbors[i].id          = -1;
        neighbors[i].dist_approx = DBL_MAX;
        neighbors[i].dist_real   = DBL_MAX;
    }

    uint8_t *vp_q = (uint8_t*)malloc(D);
    uint8_t *vn_q = (uint8_t*)malloc(D);
    if (!vp_q || !vn_q) {
        free(vp_q); free(vn_q);
        return;
    }

    quantize_vector_f64(q, vp_q, vn_q, D, x);

    int *dq_pivot = (int*)malloc(h * sizeof(int));
    if (!dq_pivot) {
        free(vp_q); free(vn_q);
        return;
    }

    for (int j = 0; j < h; j++) {
        const uint8_t *vpj = &idx->vp_piv[j * D];
        const uint8_t *vnj = &idx->vn_piv[j * D];
        dq_pivot[j] = approximate_distance(vp_q, vn_q, vpj, vnj, D);
    }

    for (size_t i = 0; i < n; i++) {

        int d_star = 0;
        for (int j = 0; j < h; j++) {
            int dvp = idx->dist[i * h + j];
            int diff = dvp - dq_pivot[j];
            if (diff < 0) diff = -diff;
            if (diff > d_star) d_star = diff;
        }

        int worst = find_worst_neighbor64(neighbors, k);
        double worst_approx = neighbors[worst].dist_approx;

        if ((double)d_star >= worst_approx)
            continue;

        const uint8_t *vpi = &idx->vp_all[i * D];
        const uint8_t *vni = &idx->vn_all[i * D];

        int d_approx = approximate_distance(vp_q, vn_q, vpi, vni, D);

        if ((double)d_approx < worst_approx) {
            neighbors[worst].id          = (int)i;
            neighbors[worst].dist_approx = (double)d_approx;
        }
    }

    for (int i = 0; i < k; i++) {
        if (neighbors[i].id < 0) {
            neighbors[i].dist_real = DBL_MAX;
            continue;
        }
        const double *v = &ds->data[(size_t)neighbors[i].id * D];
        neighbors[i].dist_real = euclidean_distance_f64(q, v, D);
    }

    // come per la versione 32 bit, NON ordiniamo qui:
    // manteniamo l'ordine impostato dal processo di selezione

    free(vp_q);
    free(vn_q);
    free(dq_pivot);
}

void knn_query_all_f64(const MatrixF64 *ds,
                       const Index *idx,
                       const MatrixF64 *queries,
                       int k,
                       int x,
                       Neighbor64 *results)
{
    if (!ds || !idx || !queries || !results) return;

    for (size_t qi = 0; qi < queries->n; qi++) {
        const double *q = &queries->data[qi * queries->d];
        Neighbor64 *nb = &results[qi * k];
        knn_query_single_f64(ds, idx, q, k, x, nb);
    }
}
