#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "query.h"
#include "quantization.h"
#include "distance.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// Trova il vicino peggiore (max distanza approssimata)
static int find_worst_neighbor(const Neighbor *neighbors, int k)
{
    int worst = 0;
    float maxd = neighbors[0].dist_approx;

    for (int i = 1; i < k; i++) {
        if (neighbors[i].dist_approx > maxd) {
            maxd = neighbors[i].dist_approx;
            worst = i;
        }
    }
    return worst;
}

// KNN per UNA query
void knn_query_single(const MatrixF32 *ds,
                      const Index *idx,
                      const float *q,
                      int k,
                      int x,
                      Neighbor *neighbors)
{
    if (!ds || !idx || !q || !neighbors) return;

    size_t n = ds->n;
    size_t D = ds->d;
    int h = (int)idx->h;

    if (k > (int)n) k = (int)n;

    // Inizializzazione dei vicini
    for (int i = 0; i < k; i++) {
        neighbors[i].id          = -1;
        neighbors[i].dist_approx = FLT_MAX;
        neighbors[i].dist_real   = FLT_MAX;
    }

    // Quantizzazione query
    uint8_t *vp_q = (uint8_t *)malloc(D * sizeof(uint8_t));
    uint8_t *vn_q = (uint8_t *)malloc(D * sizeof(uint8_t));
    if (!vp_q || !vn_q) {
        free(vp_q);
        free(vn_q);
        return;
    }

    quantize_vector(q, vp_q, vn_q, D, x);

    // Distanze approssimata query-pivot
    int *dq_pivot = (int *)malloc(h * sizeof(int));
    if (!dq_pivot) {
        free(vp_q);
        free(vn_q);
        return;
    }

    for (int j = 0; j < h; j++) {
        const uint8_t *vpj = &idx->vp_piv[j * D];
        const uint8_t *vnj = &idx->vn_piv[j * D];
        dq_pivot[j] = approximate_distance(vp_q, vn_q, vpj, vnj, D);
    }

    // Scansione punti del dataset
    for (size_t i = 0; i < n; i++) {

        // Limite inferiore calcolato attraverso pivot:
        int d_star = 0;
        for (int j = 0; j < h; j++) {
            int dvp = idx->dist[i * h + j];  // ˜d(v_i, p_j)
            int diff = dvp - dq_pivot[j];
            if (diff < 0) diff = -diff;
            if (diff > d_star) d_star = diff;
        }

        // Distanza approssimata peggiore nella lista
        int worst = find_worst_neighbor(neighbors, k);
        float worst_approx = neighbors[worst].dist_approx;

        // PRUNING
        if ((float)d_star >= worst_approx)
            continue;

        // Calcolo distanza approssimata tra v_i e la query
        const uint8_t *vpi = &idx->vp_all[i * D];
        const uint8_t *vni = &idx->vn_all[i * D];

        int d_approx = approximate_distance(vp_q, vn_q, vpi, vni, D);

        if ((float)d_approx < worst_approx) {
            neighbors[worst].id          = (int)i;
            neighbors[worst].dist_approx = (float)d_approx;
        }
    }

    // Calcolo distanza reale per i candidati trovati
    for (int i = 0; i < k; i++) {
        if (neighbors[i].id < 0) {
            neighbors[i].dist_real = FLT_MAX;
            continue;
        }

        const float *v = &ds->data[(size_t)neighbors[i].id * D];
        neighbors[i].dist_real = euclidean_distance(q, v, D);
    }

    free(vp_q);
    free(vn_q);
    free(dq_pivot);
}

// KNN per tutte le query
void knn_query_all(const MatrixF32 *ds,
                   const Index *idx,
                   const MatrixF32 *queries,
                   int k,
                   int x,
                   Neighbor *results)
{
    if (!ds || !idx || !queries || !results) return;

    #pragma omp parallel for schedule(dynamic)
    for (size_t qi = 0; qi < queries->n; qi++) {
        const float *q = &queries->data[qi * queries->d];
        Neighbor *nb = &results[qi * k];

        // Ogni thread elabora una query da solo
        knn_query_single(ds, idx, q, k, x, nb);
    }
}
