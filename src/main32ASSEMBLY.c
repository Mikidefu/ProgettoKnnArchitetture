#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "config.h"
#include "matrix.h"
#include "index.h"
#include "query.h"
#include "compare.h"
#include "distance.h"
#include "quantization.h"

static double ms(clock_t start, clock_t end) {
    return 1000.0 * (double)(end - start) / CLOCKS_PER_SEC;
}

int main(int argc, char **argv)
{
    printf("argc = %d\n", argc);

#ifdef USE_SSE2_ASM
    printf("[INFO] BUILD MODE : 32-bit FLOAT (F32)\n");
    printf("[INFO] DISTANCE   : SSE2 ASM (external procedure)\n");
#elif defined(USE_SSE2)
    printf("[INFO] DISTANCE   : SSE2 intrinsics\n");
#else
    printf("[INFO] DISTANCE   : SCALAR C\n");
#endif

    Config cfg = {0};
    if (parse_args(argc, argv, &cfg) != 0) {
        printf("Uso: %s -d dataset.ds2 -q query.ds2 -h <pivot> -k <vicini> -x <quant>\n", argv[0]);
        return 1;
    }

    printf("\n=============================\n");
    printf("  PARAMETRI DI INPUT (SSE2)\n");
    printf("=============================\n");
    printf("Dataset: %s\n", cfg.ds_path);
    printf("Query  : %s\n", cfg.q_path);
    printf("h (pivot): %d\n", cfg.h);
    printf("k (vicini): %d\n", cfg.k);
    printf("x (quantizzazione): %d\n\n", cfg.x);

    MatrixF32 ds = {0};
    MatrixF32 qs = {0};

    if (load_matrix_f32(cfg.ds_path, &ds) != 0) {
        printf("ERRORE: impossibile leggere dataset '%s'\n", cfg.ds_path);
        return 1;
    }
    if (load_matrix_f32(cfg.q_path, &qs) != 0) {
        printf("ERRORE: impossibile leggere query '%s'\n", cfg.q_path);
        free_matrix_f32(&ds);
        return 1;
    }
    if (ds.d != qs.d) {
        printf("ERRORE: dataset D=%u, query D=%u — incompatibili.\n", ds.d, qs.d);
        free_matrix_f32(&ds);
        free_matrix_f32(&qs);
        return 1;
    }

    printf("Dataset caricato: %u x %u\n", ds.n, ds.d);
    printf("Query caricate : %u x %u\n\n", qs.n, qs.d);

    printf("Costruzione indice...\n");
    clock_t t0 = clock();
    Index *idx = build_index(&ds, cfg.h, cfg.x);
    clock_t t1 = clock();

    if (!idx) {
        printf("ERRORE: impossibile costruire indice.\n");
        free_matrix_f32(&ds);
        free_matrix_f32(&qs);
        return 1;
    }

    printf("Indice costruito.\n");
    printf("Tempo build_index(): %.2f ms\n\n", ms(t0, t1));

    int k = cfg.k;
    Neighbor *results = malloc((size_t)qs.n * (size_t)k * sizeof(Neighbor));
    if (!results) {
        printf("ERRORE: allocazione risultati fallita.\n");
        free_index(idx);
        free_matrix_f32(&ds);
        free_matrix_f32(&qs);
        return 1;
    }

    printf("Esecuzione K-NN su %u query...\n", qs.n);
    clock_t t2 = clock();
    knn_query_all(&ds, idx, &qs, k, cfg.x, results);
    clock_t t3 = clock();

    printf("K-NN completato.\n");
    printf("Tempo knn_query_all(): %.2f ms\n\n", ms(t2, t3));

    MatrixI32 ref_ids = {0};
    MatrixF32 ref_dst = {0};

    if (load_matrix_i32("data/results_ids_2000x8_k8_x64_32.ds2", &ref_ids) != 0) {
        printf("ERRORE lettura results_ids (32).\n");
        free(results);
        free_index(idx);
        free_matrix_f32(&ds);
        free_matrix_f32(&qs);
        return 1;
    }
    if (load_matrix_f32("data/results_dst_2000x8_k8_x64_32.ds2", &ref_dst) != 0) {
        printf("ERRORE lettura results_dst (32).\n");
        free_matrix_i32(&ref_ids);
        free(results);
        free_index(idx);
        free_matrix_f32(&ds);
        free_matrix_f32(&qs);
        return 1;
    }

    printf("Confronto con i risultati ufficiali...\n\n");
    compare_results(&qs, results, &ref_ids, &ref_dst, k);

    printf("=====================================\n");
    printf("         BENCHMARK 32-bit SSE2        \n");
    printf("=====================================\n");
    printf("build_index()     : %.2f ms\n", ms(t0, t1));
    printf("knn_query_all()   : %.2f ms\n", ms(t2, t3));
    printf("Totale runtime    : %.2f ms\n", ms(t0, t3));
    printf("=====================================\n\n");

    free_matrix_i32(&ref_ids);
    free_matrix_f32(&ref_dst);
    free(results);
    free_index(idx);
    free_matrix_f32(&ds);
    free_matrix_f32(&qs);

    printf("Esecuzione completata.\n");
    return 0;
}
