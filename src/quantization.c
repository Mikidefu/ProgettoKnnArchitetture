#include "quantization.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    float  value;
    size_t index;
} AbsIndexF32;

typedef struct {
    double value;
    size_t index;
} AbsIndexF64;

// Confronto per qsort (decrescente)
static int cmp_abs_desc_f32(const void *a, const void *b)
{
    const AbsIndexF32 *A = (const AbsIndexF32*)a;
    const AbsIndexF32 *B = (const AbsIndexF32*)b;
    if (A->value < B->value) return 1;
    if (A->value > B->value) return -1;
    return 0;
}

static int cmp_abs_desc_f64(const void *a, const void *b)
{
    const AbsIndexF64 *A = (const AbsIndexF64*)a;
    const AbsIndexF64 *B = (const AbsIndexF64*)b;
    if (A->value < B->value) return 1;
    if (A->value > B->value) return -1;
    return 0;
}

// ====================== FLOAT32 ======================

void quantize_vector(const float *v, uint8_t *vp, uint8_t *vn, size_t D, int x)
{
    memset(vp, 0, D * sizeof(uint8_t));
    memset(vn, 0, D * sizeof(uint8_t));

    AbsIndexF32 *absvals = (AbsIndexF32*)malloc(D * sizeof(AbsIndexF32));
    if (!absvals) return;

    for (size_t i = 0; i < D; i++) {
        float val = v[i];
        absvals[i].value = (val >= 0.0f) ? val : -val;
        absvals[i].index = i;
    }

    qsort(absvals, D, sizeof(AbsIndexF32), cmp_abs_desc_f32);

    if (x > (int)D) x = (int)D;

    for (int j = 0; j < x; j++) {
        size_t idx = absvals[j].index;
        if (v[idx] >= 0.0f)
            vp[idx] = 1;
        else
            vn[idx] = 1;
    }

    free(absvals);
}

// ====================== FLOAT64 (Double) ======================

void quantize_vector_f64(const double *v, uint8_t *vp, uint8_t *vn, size_t D, int x)
{
    memset(vp, 0, D * sizeof(uint8_t));
    memset(vn, 0, D * sizeof(uint8_t));

    AbsIndexF64 *absvals = (AbsIndexF64*)malloc(D * sizeof(AbsIndexF64));
    if (!absvals) return;

    for (size_t i = 0; i < D; i++) {
        double val = v[i];
        absvals[i].value = (val >= 0.0) ? val : -val;
        absvals[i].index = i;
    }

    qsort(absvals, D, sizeof(AbsIndexF64), cmp_abs_desc_f64);

    if (x > (int)D) x = (int)D;

    for (int j = 0; j < x; j++) {
        size_t idx = absvals[j].index;
        if (v[idx] >= 0.0)
            vp[idx] = 1;
        else
            vn[idx] = 1;
    }

    free(absvals);
}
