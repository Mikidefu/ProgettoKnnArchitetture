#include "distance.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#if defined(USE_AVX)
    #include <immintrin.h>   // AVX/AVX2 + SSE inclusi
#elif defined(USE_SSE2)
    #include <emmintrin.h>   // solo SSE2
#endif

// =====================================================================
// Distanza approssimata ˜d(v,w)
//  ˜d = (v+·w+) + (v−·w−) − (v+·w−) − (v−·w+)
// vp, vn, wp, wn sono vettori di uint8_t con valori 0/1
// =====================================================================

int approximate_distance(
    const uint8_t *vp, const uint8_t *vn,
    const uint8_t *wp, const uint8_t *wn,
    size_t D
) {

#if defined(USE_AVX)
    // ================== VERSIONE AVX2 (256 bit) ==================
    // Richiede: -DUSE_AVX -mavx2
    size_t blocks = D / 32;   // 32 byte per vettore AVX
    size_t offset = blocks * 32;

    __m256i zero    = _mm256_setzero_si256();
    __m256i acc_pp  = zero;
    __m256i acc_nn  = zero;
    __m256i acc_pn  = zero;
    __m256i acc_np  = zero;

    for (size_t b = 0; b < blocks; b++) {

        const uint8_t *p_vp = vp + b * 32;
        const uint8_t *p_vn = vn + b * 32;
        const uint8_t *p_wp = wp + b * 32;
        const uint8_t *p_wn = wn + b * 32;

        __m256i vvp = _mm256_loadu_si256((const __m256i*)p_vp);
        __m256i vvn = _mm256_loadu_si256((const __m256i*)p_vn);
        __m256i vwp = _mm256_loadu_si256((const __m256i*)p_wp);
        __m256i vwn = _mm256_loadu_si256((const __m256i*)p_wn);

        __m256i m_pp = _mm256_and_si256(vvp, vwp);
        __m256i m_nn = _mm256_and_si256(vvn, vwn);
        __m256i m_pn = _mm256_and_si256(vvp, vwn);
        __m256i m_np = _mm256_and_si256(vvn, vwp);

        __m256i lo, hi;

        // v+·w+
        lo = _mm256_unpacklo_epi8(m_pp, zero);
        hi = _mm256_unpackhi_epi8(m_pp, zero);
        acc_pp = _mm256_add_epi16(acc_pp, lo);
        acc_pp = _mm256_add_epi16(acc_pp, hi);

        // v-·w-
        lo = _mm256_unpacklo_epi8(m_nn, zero);
        hi = _mm256_unpackhi_epi8(m_nn, zero);
        acc_nn = _mm256_add_epi16(acc_nn, lo);
        acc_nn = _mm256_add_epi16(acc_nn, hi);

        // v+·w-
        lo = _mm256_unpacklo_epi8(m_pn, zero);
        hi = _mm256_unpackhi_epi8(m_pn, zero);
        acc_pn = _mm256_add_epi16(acc_pn, lo);
        acc_pn = _mm256_add_epi16(acc_pn, hi);

        // v-·w+
        lo = _mm256_unpacklo_epi8(m_np, zero);
        hi = _mm256_unpackhi_epi8(m_np, zero);
        acc_np = _mm256_add_epi16(acc_np, lo);
        acc_np = _mm256_add_epi16(acc_np, hi);
    }

    // Somma orizzontale dei 16 valori 16-bit in ciascun accumulatore
    int pp = 0, nn = 0, pn = 0, np = 0;
    uint16_t tmp16[16];

    _mm256_storeu_si256((__m256i*)tmp16, acc_pp);
    for (int i = 0; i < 16; ++i) pp += tmp16[i];

    _mm256_storeu_si256((__m256i*)tmp16, acc_nn);
    for (int i = 0; i < 16; ++i) nn += tmp16[i];

    _mm256_storeu_si256((__m256i*)tmp16, acc_pn);
    for (int i = 0; i < 16; ++i) pn += tmp16[i];

    _mm256_storeu_si256((__m256i*)tmp16, acc_np);
    for (int i = 0; i < 16; ++i) np += tmp16[i];

    // Resto scalare (se D non multiplo di 32)
    for (size_t i = offset; i < D; i++) {
        if (vp[i] & wp[i]) pp++;
        if (vn[i] & wn[i]) nn++;
        if (vp[i] & wn[i]) pn++;
        if (vn[i] & wp[i]) np++;
    }

    return pp + nn - pn - np;

#elif defined(USE_SSE2)
    // ================== VERSIONE SSE2 (128 bit) ==================
    size_t blocks = D / 16;
    size_t offset = blocks * 16;

    __m128i zero    = _mm_setzero_si128();
    __m128i acc_pp  = zero;
    __m128i acc_nn  = zero;
    __m128i acc_pn  = zero;
    __m128i acc_np  = zero;

    for (size_t b = 0; b < blocks; b++) {

        const uint8_t *p_vp = vp + b * 16;
        const uint8_t *p_vn = vn + b * 16;
        const uint8_t *p_wp = wp + b * 16;
        const uint8_t *p_wn = wn + b * 16;

        __m128i vvp = _mm_loadu_si128((const __m128i*)p_vp);
        __m128i vvn = _mm_loadu_si128((const __m128i*)p_vn);
        __m128i vwp = _mm_loadu_si128((const __m128i*)p_wp);
        __m128i vwn = _mm_loadu_si128((const __m128i*)p_wn);

        __m128i m_pp = _mm_and_si128(vvp, vwp);
        __m128i m_nn = _mm_and_si128(vvn, vwn);
        __m128i m_pn = _mm_and_si128(vvp, vwn);
        __m128i m_np = _mm_and_si128(vvn, vwp);

        __m128i lo, hi;

        // v+·w+
        lo = _mm_unpacklo_epi8(m_pp, zero);
        hi = _mm_unpackhi_epi8(m_pp, zero);
        acc_pp = _mm_add_epi16(acc_pp, lo);
        acc_pp = _mm_add_epi16(acc_pp, hi);

        // v-·w-
        lo = _mm_unpacklo_epi8(m_nn, zero);
        hi = _mm_unpackhi_epi8(m_nn, zero);
        acc_nn = _mm_add_epi16(acc_nn, lo);
        acc_nn = _mm_add_epi16(acc_nn, hi);

        // v+·w-
        lo = _mm_unpacklo_epi8(m_pn, zero);
        hi = _mm_unpackhi_epi8(m_pn, zero);
        acc_pn = _mm_add_epi16(acc_pn, lo);
        acc_pn = _mm_add_epi16(acc_pn, hi);

        // v-·w+
        lo = _mm_unpacklo_epi8(m_np, zero);
        hi = _mm_unpackhi_epi8(m_np, zero);
        acc_np = _mm_add_epi16(acc_np, lo);
        acc_np = _mm_add_epi16(acc_np, hi);
    }

    int pp = 0, nn = 0, pn = 0, np = 0;
    uint16_t tmp16[8];

    _mm_storeu_si128((__m128i*)tmp16, acc_pp);
    for (int i = 0; i < 8; ++i) pp += tmp16[i];

    _mm_storeu_si128((__m128i*)tmp16, acc_nn);
    for (int i = 0; i < 8; ++i) nn += tmp16[i];

    _mm_storeu_si128((__m128i*)tmp16, acc_pn);
    for (int i = 0; i < 8; ++i) pn += tmp16[i];

    _mm_storeu_si128((__m128i*)tmp16, acc_np);
    for (int i = 0; i < 8; ++i) np += tmp16[i];

    for (size_t i = offset; i < D; i++) {
        if (vp[i] & wp[i]) pp++;
        if (vn[i] & wn[i]) nn++;
        if (vp[i] & wn[i]) pn++;
        if (vn[i] & wp[i]) np++;
    }

    return pp + nn - pn - np;

#else
    // ================== VERSIONE SCALARE ==================
    int pp = 0, nn = 0, pn = 0, np = 0;

    for (size_t i = 0; i < D; i++) {
        if (vp[i] & wp[i]) pp++;
        if (vn[i] & wn[i]) nn++;
        if (vp[i] & wn[i]) pn++;
        if (vn[i] & wp[i]) np++;
    }

    return pp + nn - pn - np;
#endif
}

// =====================================================================
// Distanza euclidea reale float32 (scalare, basta così)
// =====================================================================

float euclidean_distance(const float *a, const float *b, size_t D)
{
    float sum = 0.0f;
    for (size_t i = 0; i < D; i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

// =====================================================================
// Distanza euclidea reale float64 (AVX quando disponibile)
// =====================================================================

double euclidean_distance_f64(const double *a, const double *b, size_t D)
{
#if defined(USE_AVX)
    size_t blocks = D / 4;   // 4 double per __m256d
    size_t offset = blocks * 4;

    __m256d acc = _mm256_setzero_pd();

    for (size_t i = 0; i < blocks; ++i) {
        __m256d va = _mm256_loadu_pd(a + 4 * i);
        __m256d vb = _mm256_loadu_pd(b + 4 * i);
        __m256d diff = _mm256_sub_pd(va, vb);
        __m256d sq   = _mm256_mul_pd(diff, diff);
        acc = _mm256_add_pd(acc, sq);
    }

    double tmp[4];
    _mm256_storeu_pd(tmp, acc);
    double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (size_t i = offset; i < D; ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sqrt(sum);
#else
    double sum = 0.0;
    for (size_t i = 0; i < D; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
#endif
}
