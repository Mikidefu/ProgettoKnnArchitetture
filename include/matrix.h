#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>

// Matrice float32
typedef struct {
    uint32_t n;
    uint32_t d;
    float   *data;
} MatrixF32;

// Matrice float64 (double)
typedef struct {
    uint32_t n;
    uint32_t d;
    double  *data;
} MatrixF64;

// Matrice risultati indice int32
typedef struct {
    uint32_t n;
    uint32_t d;
    int32_t *data;
} MatrixI32;

// Caricamento/svuotamento float32
int  load_matrix_f32(const char *path, MatrixF32 *m);
void free_matrix_f32(MatrixF32 *m);

// Caricamento/svuotamento float64
int  load_matrix_f64(const char *path, MatrixF64 *m);
void free_matrix_f64(MatrixF64 *m);

// Caricamento/svuotamento int32
int  load_matrix_i32(const char *path, MatrixI32 *m);
void free_matrix_i32(MatrixI32 *m);

#endif
