#include "distance.h"
#include <math.h>

int approximate_distance_scalar(
    const uint8_t *vp, const uint8_t *vn,
    const uint8_t *wp, const uint8_t *wn,
    size_t D
) {
    int pp = 0, nn = 0, pn = 0, np = 0;
    for (size_t i = 0; i < D; i++) {
        if (vp[i] & wp[i]) pp++;
        if (vn[i] & wn[i]) nn++;
        if (vp[i] & wn[i]) pn++;
        if (vn[i] & wp[i]) np++;
    }
    return pp + nn - pn - np;
}

#ifdef USE_AVX_ASM
extern int approximate_distance_avx2_asm(
    const uint8_t *vp, const uint8_t *vn,
    const uint8_t *wp, const uint8_t *wn,
    size_t D
);
#endif

int approximate_distance(
    const uint8_t *vp, const uint8_t *vn,
    const uint8_t *wp, const uint8_t *wn,
    size_t D
) {
#ifdef USE_AVX_ASM
    return approximate_distance_avx2_asm(vp, vn, wp, wn, D);
#else
    return approximate_distance_scalar(vp, vn, wp, wn, D);
#endif
}

float euclidean_distance(const float *a, const float *b, size_t D)
{
    float sum = 0.0f;
    for (size_t i = 0; i < D; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

double euclidean_distance_f64(const double *a, const double *b, size_t D)
{
    double sum = 0.0;
    for (size_t i = 0; i < D; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}
