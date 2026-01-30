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

#ifdef _OPENMP
#include <omp.h>
#endif

// ---------------------------------------------
// Funzione tempo
// ---------------------------------------------
double calc_time_ms(clock_t c_start, clock_t c_end, double w_start, double w_end) {
#ifdef _OPENMP
    return (w_end - w_start) * 1000.0;
#else
    return 1000.0 * (double)(c_end - c_start) / CLOCKS_PER_SEC;
#endif
}

int main(int argc, char **argv)
{
    printf("argc = %d\n", argc);

    // -----------------------------------------------------
    // INFO COMPILAZIONE
    // -----------------------------------------------------
#ifdef USE_AVX_ASM
    printf("[INFO] BUILD MODE : 64-bit DOUBLE (ASM AVX2)\n");
#elif defined(USE_AVX)
    printf("[INFO] BUILD MODE : 64-bit DOUBLE (AVX Intrinsics)\n");
#else
    printf("[INFO] BUILD MODE : 64-bit DOUBLE (SCALAR)\n");
#endif

#ifdef _OPENMP
    printf("[INFO] OpenMP     : ATTIVO (Max threads: %d)\n", omp_get_max_threads());
#else
    printf("[INFO] OpenMP     : DISATTIVATO (Single Thread)\n");
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
    printf("     PARAMETRI DI INPUT\n");
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

    clock_t c0 = clock();
    double w0 = 0;
    #ifdef _OPENMP
    w0 = omp_get_wtime();
    #endif

    Index *idx = build_index_f64(&ds, cfg.h, cfg.x);

    clock_t c1 = clock();
    double w1 = 0;
    #ifdef _OPENMP
    w1 = omp_get_wtime();
    #endif

    if (!idx) {
        printf("ERRORE: impossibile costruire indice.\n");
        free_matrix_f64(&ds);
        free_matrix_f64(&qs);
        return 1;
    }

    double time_build = calc_time_ms(c0, c1, w0, w1);
    printf("Indice costruito.\n");
    printf("Tempo build_index(): %.2f ms\n\n", time_build);

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

    clock_t c2 = clock();
    double w2 = 0;
    #ifdef _OPENMP
    w2 = omp_get_wtime();
    #endif

    knn_query_all_f64(&ds, idx, &qs, k, cfg.x, results);

    clock_t c3 = clock();
    double w3 = 0;
    #ifdef _OPENMP
    w3 = omp_get_wtime();
    #endif

    double time_query = calc_time_ms(c2, c3, w2, w3);
    printf("K-NN completato.\n");
    printf("Tempo knn_query_all(): %.2f ms\n\n", time_query);

    // -----------------------------------------------------
    // CARICAMENTO RISULTATI 64-BIT
    // -----------------------------------------------------
    MatrixI32 ref_ids = {0};
    MatrixF64 ref_dst = {0};
    int skip_check = 0;

    // Carichiamo i risultati, in caso di fallimento saltiamo solo il check.
    if (load_matrix_i32("data/results_ids_2000x8_k8_x64_64.ds2", &ref_ids) != 0) {
        printf("ATTENZIONE: File results_ids non trovato. Il confronto verrà saltato.\n");
        skip_check = 1;
    } else if (load_matrix_f64("data/results_dst_2000x8_k8_x64_64.ds2", &ref_dst) != 0) {
        printf("ATTENZIONE: File results_dst non trovato. Il confronto verrà saltato.\n");
        free_matrix_i32(&ref_ids);
        skip_check = 1;
    }

    if (!skip_check) {
        if (ref_ids.d != (uint32_t)k || ref_dst.d != (uint32_t)k) {
            printf("ERRORE: k nei file ufficiali non corrisponde.\n");
            // Non usciamo, stampiamo solo l'errore e saltiamo il check
        } else if (ref_ids.n != qs.n) {
            printf("ERRORE: numero di query nei risultati ufficiali (%u) diverso da quello nelle query (%u)\n",
                   ref_ids.n, qs.n);
        } else {
            // -----------------------------------------------------
            // CONFRONTO RISULTATI
            // -----------------------------------------------------
            printf("Confronto con i risultati ufficiali (double)...\n\n");
            compare_results_f64(&qs, results, &ref_ids, &ref_dst, k);
        }

        // Cleanup check data
        free_matrix_i32(&ref_ids);
        free_matrix_f64(&ref_dst);
    }

    // -----------------------------------------------------
    // STAMPA RIEPILOGO BENCHMARK FINALE
    // -----------------------------------------------------
    printf("=====================================\n");
    printf("        BENCHMARK (DOUBLE)          \n");
    printf("=====================================\n");
    printf("build_index()     : %.2f ms\n", time_build);
    printf("knn_query_all()   : %.2f ms\n", time_query);
    printf("Totale runtime    : %.2f ms\n", time_build + time_query);
    printf("=====================================\n\n");

    // -----------------------------------------------------
    // PULIZIA MEMORIA
    // -----------------------------------------------------
    free(results);
    free_index(idx);

    free_matrix_f64(&ds);
    free_matrix_f64(&qs);

    printf("Esecuzione 64-bit completata.\n");

    return 0;
}
