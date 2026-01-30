#include "index.h"
#include "quantization.h"
#include "distance.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// ------------------ SELEZIONE DEI PIVOT ----------------------
//
// j = Da 0 ad h-1
// pivot[j] = floor(n/h) * j
//
// --------------------------------------------------------------

static void select_pivots(size_t *pivot_ids, size_t n, int h) {
    size_t step = n / h;
    for (int j = 0; j < h; j++) {
        pivot_ids[j] = step * j;
    }
}

Index *build_index(const MatrixF32 *ds, int h, int x) {

    if (!ds || h <= 0 || x <= 0) return NULL;

    size_t n = ds->n;
    size_t D = ds->d;

    // Allocazione struttura Index
    Index *idx = calloc(1, sizeof(Index));
    if (!idx) return NULL;

    idx->n = n;
    idx->h = h;
    idx->D = D;

    // Allocazione pivot_ids (identificativi)
    idx->pivot_ids = malloc(h * sizeof(size_t));
    if (!idx->pivot_ids) {
        free(idx);
        return NULL;
    }

    select_pivots(idx->pivot_ids, n, h);

    // Allocazione vettori quantizzati per tutto il dataset
    idx->vp_all = malloc(n * D * sizeof(uint8_t));
    idx->vn_all = malloc(n * D * sizeof(uint8_t));

    if (!idx->vp_all || !idx->vn_all) {
        free(idx->pivot_ids);
        free(idx->vp_all);
        free(idx->vn_all);
        free(idx);
        return NULL;
    }

    // Quantizzazione dataset
    for (size_t i = 0; i < n; i++) {
        const float *row = &ds->data[i * D];
        uint8_t *vp = &idx->vp_all[i * D];
        uint8_t *vn = &idx->vn_all[i * D];

        quantize_vector(row, vp, vn, D, x);
    }

    // Allocazione quantizzazioni dei SOLI pivot
    idx->vp_piv = malloc(h * D * sizeof(uint8_t));
    idx->vn_piv = malloc(h * D * sizeof(uint8_t));

    if (!idx->vp_piv || !idx->vn_piv) {
        free(idx->pivot_ids);
        free(idx->vp_all);
        free(idx->vn_all);
        free(idx->vp_piv);
        free(idx->vn_piv);
        free(idx);
        return NULL;
    }

    // Copia dei pivot già quantizzati
    for (int j = 0; j < h; j++) {
        size_t p = idx->pivot_ids[j];

        memcpy(&idx->vp_piv[j * D], &idx->vp_all[p * D], D * sizeof(uint8_t));
        memcpy(&idx->vn_piv[j * D], &idx->vn_all[p * D], D * sizeof(uint8_t));
    }

    // Allocazione matrice distanze
    idx->dist = malloc(n * h * sizeof(int));
    if (!idx->dist) {
        free(idx->pivot_ids);
        free(idx->vp_all);  free(idx->vn_all);
        free(idx->vp_piv);  free(idx->vn_piv);
        free(idx);
        return NULL;
    }

    // Calcolo distanze approssimate d(v,p)
    for (size_t i = 0; i < n; i++) {

        const uint8_t *vpi = &idx->vp_all[i * D];
        const uint8_t *vni = &idx->vn_all[i * D];

        for (int j = 0; j < h; j++) {

            const uint8_t *vpj = &idx->vp_piv[j * D];
            const uint8_t *vnj = &idx->vn_piv[j * D];

            int d = approximate_distance(vpi, vni, vpj, vnj, D);

            idx->dist[i * h + j] = d;
        }
    }

    return idx;
}

Index *build_index_f64(const MatrixF64 *ds, int h, int x)
{
    if (!ds || h <= 0 || x <= 0) return NULL;

    size_t n = ds->n;
    size_t D = ds->d;

    Index *idx = calloc(1, sizeof(Index));
    if (!idx) return NULL;

    idx->n = n;
    idx->h = (size_t)h;
    idx->D = D;

    idx->pivot_ids = malloc(h * sizeof(size_t));
    if (!idx->pivot_ids) {
        free(idx);
        return NULL;
    }

    select_pivots(idx->pivot_ids, n, h);

    idx->vp_all = malloc(n * D * sizeof(uint8_t));
    idx->vn_all = malloc(n * D * sizeof(uint8_t));
    if (!idx->vp_all || !idx->vn_all) {
        free(idx->pivot_ids);
        free(idx->vp_all);
        free(idx->vn_all);
        free(idx);
        return NULL;
    }

    // Quantizzazione dataset (double)
    for (size_t i = 0; i < n; i++) {
        const double *row = &ds->data[i * D];
        uint8_t *vp = &idx->vp_all[i * D];
        uint8_t *vn = &idx->vn_all[i * D];
        quantize_vector_f64(row, vp, vn, D, x);
    }

    // Pivot
    idx->vp_piv = malloc(h * D * sizeof(uint8_t));
    idx->vn_piv = malloc(h * D * sizeof(uint8_t));
    if (!idx->vp_piv || !idx->vn_piv) {
        free(idx->pivot_ids);
        free(idx->vp_all); free(idx->vn_all);
        free(idx->vp_piv); free(idx->vn_piv);
        free(idx);
        return NULL;
    }

    for (int j = 0; j < h; j++) {
        size_t p = idx->pivot_ids[j];
        memcpy(&idx->vp_piv[j * D], &idx->vp_all[p * D], D * sizeof(uint8_t));
        memcpy(&idx->vn_piv[j * D], &idx->vn_all[p * D], D * sizeof(uint8_t));
    }

    idx->dist = malloc(n * h * sizeof(int));
    if (!idx->dist) {
        free(idx->pivot_ids);
        free(idx->vp_all); free(idx->vn_all);
        free(idx->vp_piv); free(idx->vn_piv);
        free(idx);
        return NULL;
    }

    for (size_t i = 0; i < n; i++) {
        const uint8_t *vpi = &idx->vp_all[i * D];
        const uint8_t *vni = &idx->vn_all[i * D];

        for (int j = 0; j < h; j++) {
            const uint8_t *vpj = &idx->vp_piv[j * D];
            const uint8_t *vnj = &idx->vn_piv[j * D];

            int d = approximate_distance(vpi, vni, vpj, vnj, D);
            idx->dist[i * h + j] = d;
        }
    }

    return idx;
}


// --------------------------------------------------------------
// PULIZIA MEMORIA
// --------------------------------------------------------------

void free_index(Index *idx) {
    if (!idx) return;
    free(idx->pivot_ids);
    free(idx->vp_all);
    free(idx->vn_all);
    free(idx->vp_piv);
    free(idx->vn_piv);
    free(idx->dist);
    free(idx);
}
