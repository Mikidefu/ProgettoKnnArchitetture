#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "config.h"
#include "matrix.h"
#include "index.h"
#include "query64.h"
#include "compare64.h"
#include "distance.h"
#include "quantization.h"

// ---------------------------------------------
// Utility: ms()
// ---------------------------------------------
double ms(clock_t start, clock_t end) {
    return 1000.0 * (double)(end - start) / CLOCKS_PER_SEC;
}

int main(int argc, char **argv)
{
    printf("argc = %d\n", argc);

#ifdef USE_AVX
    printf("[INFO] Compilato con AVX\n");
#elif defined(USE_SSE2)
    printf("[INFO] Compilato con SSE2\n");
#else
    printf("[INFO] Compilato SENZA SIMD (SCALAR)\n");
#endif

    // -----------------------------------------------------
    // PARSING ARGOMENTI
    // -----------------------------------------------------
    Config cfg = {0};

    if (parse_args(argc, argv, &cfg) != 0) {
        printf("Uso: %s -d dataset.ds2 -q query.ds2 -h <pivot> -k <vicini> -x <quant>\n",
               argv[0]);
        return 1;
    }

    printf("\n=============================\n");
    printf("     PARAMETRI DI INPUT (64-bit)\n");
    printf("=============================\n");
    printf("Dataset: %s\n", cfg.ds_path);
    printf("Query  : %s\n", cfg.q_path);
    printf("h pivot: %d\n", cfg.h);
    printf("k      : %d\n", cfg.k);
    printf("x      : %d\n\n", cfg.x);

    // -----------------------------------------------------
    // CARICAMENTO DATASET / QUERY (DOUBLE)
    // -----------------------------------------------------
    MatrixF64 ds = {0};
    MatrixF64 qs = {0};

    if (load_matrix_f64(cfg.ds_path, &ds) != 0) {
        printf("ERRORE: impossibile leggere dataset '%s'\n", cfg.ds_path);
        return 1;
    }

    if (load_matrix_f64(cfg.q_path, &qs) != 0) {
        printf("ERRORE: impossibile leggere query '%s'\n", cfg.q_path);
        free_matrix_f64(&ds);
        return 1;
    }

    if (ds.d != qs.d) {
        printf("ERRORE: dataset D=%u, query D=%u — dimensioni incompatibili.\n",
               ds.d, qs.d);
        free_matrix_f64(&ds);
        free_matrix_f64(&qs);
        return 1;
    }

    printf("Dataset caricato: %u x %u\n", ds.n, ds.d);
    printf("Query caricate : %u x %u\n\n", qs.n, qs.d);

    // -----------------------------------------------------
    // COSTRUZIONE INDICE + BENCHMARK
    // -----------------------------------------------------
    printf("Costruzione indice (64-bit)...\n");

    clock_t t0 = clock();
    Index *idx = build_index_f64(&ds, cfg.h, cfg.x);
    clock_t t1 = clock();

    if (!idx) {
        printf("ERRORE: impossibile costruire indice.\n");
        free_matrix_f64(&ds);
        free_matrix_f64(&qs);
        return 1;
    }

    printf("Indice costruito.\n");
    printf("Tempo build_index(): %.2f ms\n\n", ms(t0, t1));

    // -----------------------------------------------------
    // ESECUZIONE KNN SU TUTTE LE QUERY + BENCHMARK
    // -----------------------------------------------------
    int k = cfg.k;

    Neighbor64 *results = malloc(qs.n * k * sizeof(Neighbor64));
    if (!results) {
        printf("ERRORE: impossibile allocare memoria risultati.\n");
        free_index(idx);
        free_matrix_f64(&ds);
        free_matrix_f64(&qs);
        return 1;
    }

    printf("Esecuzione K-NN (double) su %u query...\n", qs.n);

    clock_t t2 = clock();
    knn_query_all_f64(&ds, idx, &qs, k, cfg.x, results);
    clock_t t3 = clock();

    printf("K-NN completato.\n");
    printf("Tempo knn_query_all(): %.2f ms\n\n", ms(t2, t3));

    // -----------------------------------------------------
    // CARICAMENTO RISULTATI UFFICIALI 64-BIT
    // -----------------------------------------------------
    MatrixI32 ref_ids = {0};
    MatrixF64 ref_dst = {0};

    if (load_matrix_i32("data/results_ids_2000x8_k8_x64_64.ds2", &ref_ids) != 0) {
        printf("ERRORE lettura results_ids.\n");
        free(results);
        free_index(idx);
        free_matrix_f64(&ds);
        free_matrix_f64(&qs);
        return 1;
    }

    if (load_matrix_f64("data/results_dst_2000x8_k8_x64_64.ds2", &ref_dst) != 0) {
        printf("ERRORE lettura results_dst.\n");
        free_matrix_i32(&ref_ids);
        free(results);
        free_index(idx);
        free_matrix_f64(&ds);
        free_matrix_f64(&qs);
        return 1;
    }

    if (ref_ids.d != (uint32_t)k || ref_dst.d != (uint32_t)k) {
        printf("ERRORE: k nei file ufficiali non corrisponde.\n");
        return 1;
    }

    if (ref_ids.n != qs.n) {
        printf("ERRORE: numero di query nei risultati ufficiali (%u) diverso da quello nelle query (%u)\n",
               ref_ids.n, qs.n);
        return 1;
    }

    // -----------------------------------------------------
    // CONFRONTO RISULTATI
    // -----------------------------------------------------
    printf("Confronto con i risultati ufficiali (double)...\n\n");

    compare_results_f64(&qs, results, &ref_ids, &ref_dst, k);

    // -----------------------------------------------------
    // STAMPA RIEPILOGO BENCHMARK
    // -----------------------------------------------------
    printf("=====================================\n");
    printf("        BENCHMARK (DOUBLE)          \n");
    printf("=====================================\n");
    printf("build_index()     : %.2f ms\n", ms(t0, t1));
    printf("knn_query_all()   : %.2f ms\n", ms(t2, t3));
    printf("Totale runtime    : %.2f ms\n", ms(t0, t3));
    printf("=====================================\n\n");

    // -----------------------------------------------------
    // CLEANUP MEMORIA
    // -----------------------------------------------------
    free_matrix_i32(&ref_ids);
    free_matrix_f64(&ref_dst);

    free(results);
    free_index(idx);

    free_matrix_f64(&ds);
    free_matrix_f64(&qs);

    printf("Esecuzione 64-bit completata senza errori.\n");

    return 0;
}
