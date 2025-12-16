#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>

// Funzione interna per leggere header n,d
static int read_header(FILE *f, uint32_t *n, uint32_t *d)
{
    if (fread(n, sizeof(uint32_t), 1, f) != 1) return -1;
    if (fread(d, sizeof(uint32_t), 1, f) != 1) return -1;
    return 0;
}

// ===================== FLOAT32 =====================

int load_matrix_f32(const char *path, MatrixF32 *m)
{
    if (!path || !m) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("fopen");
        return -1;
    }

    uint32_t n = 0, d = 0;
    if (read_header(f, &n, &d) != 0) {
        fclose(f);
        return -1;
    }

    size_t count = (size_t)n * (size_t)d;
    float *data = (float*)malloc(count * sizeof(float));
    if (!data) {
        fclose(f);
        return -1;
    }

    size_t read = fread(data, sizeof(float), count, f);
    fclose(f);

    if (read != count) {
        free(data);
        return -1;
    }

    m->n = n;
    m->d = d;
    m->data = data;

    return 0;
}

void free_matrix_f32(MatrixF32 *m)
{
    if (!m) return;
    free(m->data);
    m->data = NULL;
    m->n = m->d = 0;
}

// ===================== FLOAT64 (double) =====================

int load_matrix_f64(const char *path, MatrixF64 *m)
{
    if (!path || !m) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("fopen");
        return -1;
    }

    uint32_t n = 0, d = 0;
    if (read_header(f, &n, &d) != 0) {
        fclose(f);
        return -1;
    }

    size_t count = (size_t)n * (size_t)d;
    double *data = (double*)malloc(count * sizeof(double));
    if (!data) {
        fclose(f);
        return -1;
    }

    size_t read = fread(data, sizeof(double), count, f);
    fclose(f);

    if (read != count) {
        free(data);
        return -1;
    }

    m->n = n;
    m->d = d;
    m->data = data;

    return 0;
}

void free_matrix_f64(MatrixF64 *m)
{
    if (!m) return;
    free(m->data);
    m->data = NULL;
    m->n = m->d = 0;
}

// ===================== INT32 (risultati IDs) =====================

int load_matrix_i32(const char *path, MatrixI32 *m)
{
    if (!path || !m) return -1;

    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("fopen");
        return -1;
    }

    uint32_t n = 0, d = 0;
    if (read_header(f, &n, &d) != 0) {
        fclose(f);
        return -1;
    }

    size_t count = (size_t)n * (size_t)d;
    int32_t *data = (int32_t*)malloc(count * sizeof(int32_t));
    if (!data) {
        fclose(f);
        return -1;
    }

    size_t read = fread(data, sizeof(int32_t), count, f);
    fclose(f);

    if (read != count) {
        free(data);
        return -1;
    }

    m->n = n;
    m->d = d;
    m->data = data;

    return 0;
}

void free_matrix_i32(MatrixI32 *m)
{
    if (!m) return;
    free(m->data);
    m->data = NULL;
    m->n = m->d = 0;
}
