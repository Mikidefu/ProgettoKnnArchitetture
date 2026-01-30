#ifndef DISTANCE_H
#define DISTANCE_H

#include <stdint.h>
#include <stddef.h>

// v+: vettore binario del primo punto
// v-: vettore binario negativo del primo punto
// w+: vettore binario del secondo punto
// w-: vettore binario negativo del secondo punto
// D : dimensione del vettore
//
// Restituisce la distanza approssimata d(v,w)

int approximate_distance(
    const uint8_t *vp, const uint8_t *vn,
    const uint8_t *wp, const uint8_t *wn,
    size_t D
);

// Distanza euclidea reale float32
float euclidean_distance(const float *a, const float *b, size_t D);

// Distanza euclidea reale float64
double euclidean_distance_f64(const double *a, const double *b, size_t D);

#endif



