#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <stddef.h>
#include <stdint.h>

// v  = vettore originale
// vp = vettore v+ quantizzato
// vn = vettore v- quantizzato
// D  = dimensione del vettore
// x  = n elementi da quantizzare

// Versione float32
void quantize_vector(const float *v, uint8_t *vp, uint8_t *vn, size_t D, int x);


// Versione float64
void quantize_vector_f64(const double *v, uint8_t *vp, uint8_t *vn, size_t D, int x);

#endif
