#include "distance.h"
#include <math.h>

int approximate_distance_scalar(
    const uint8_t *vp, const uint8_t *vn,
    const uint8_t *wp, const uint8_t *wn,
    size_t D
) {
    int pp = 0, nn = 0, pn = 0, np = 0;
    for (size_t i = 0; i < D; i++) {
        pp += (vp[i] & wp[i]) ? 1 : 0;
        nn += (vn[i] & wn[i]) ? 1 : 0;
        pn += (vp[i] & wn[i]) ? 1 : 0;
        np += (vn[i] & wp[i]) ? 1 : 0;
    }
    return pp + nn - pn - np;
}

#ifdef USE_SSE2_ASM
extern int approximate_distance_sse2_asm(
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
#ifdef USE_SSE2_ASM
    return approximate_distance_sse2_asm(vp, vn, wp, wn, D);
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
