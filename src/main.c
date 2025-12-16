#include <stdio.h>
#include <stdlib.h>

#include "config.h"
#include "matrix.h"
#include "index.h"
#include "query.h"
#include "compare.h"
#include "distance.h"
#include "quantization.h"

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

// ---------------------------------------------
// Funzione di utilità: millisecondi
// ---------------------------------------------
double ms(clock_t start, clock_t end) {
    return 1000.0 * (double)(end - start) / CLOCKS_PER_SEC;
}

int main(int argc, char **argv)
{
    printf("argc = %d\n", argc); //Debug
    #ifdef USE_SSE2
    printf("[INFO] Compilato con USE_SSE2\n");
    #else
    printf("[INFO] Compilato SENZA USE_SSE2 (SCALAR)\n");
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
    printf("h (pivot): %d\n", cfg.h);
    printf("k (vicini): %d\n", cfg.k);
    printf("x (quantizzazione): %d\n\n", cfg.x);

    // -----------------------------------------------------
    // CARICAMENTO DATASET / QUERY
    // -----------------------------------------------------
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
        printf("ERRORE: dataset D=%u, query D=%u — dimensioni incompatibili.\n",
               ds.d, qs.d);
        free_matrix_f32(&ds);
        free_matrix_f32(&qs);
        return 1;
    }

    printf("Dataset caricato: %u x %u\n", ds.n, ds.d);
    printf("Query caricate : %u x %u\n\n", qs.n, qs.d);

    // -----------------------------------------------------
    // COSTRUZIONE INDICE + BENCHMARK
    // -----------------------------------------------------
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

    // -----------------------------------------------------
    // ESECUZIONE KNN SU TUTTE LE QUERY + BENCHMARK
    // -----------------------------------------------------
    int k = cfg.k;

    Neighbor *results = malloc(qs.n * k * sizeof(Neighbor));
    if (!results) {
        printf("ERRORE: impossibile allocare memoria risultati.\n");
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

    // -----------------------------------------------------
    // CARICAMENTO RISULTATI UFFICIALI
    // -----------------------------------------------------
    MatrixI32 ref_ids = {0};
    MatrixF32 ref_dst = {0};

    if (load_matrix_i32("data/results_ids_2000x8_k8_x64_32.ds2", &ref_ids) != 0) {
        printf("ERRORE lettura results_ids.\n");
        free(results);
        free_index(idx);
        free_matrix_f32(&ds);
        free_matrix_f32(&qs);
        return 1;
    }

    if (load_matrix_f32("data/results_dst_2000x8_k8_x64_32.ds2", &ref_dst) != 0) {
        printf("ERRORE lettura results_dst.\n");
        free_matrix_i32(&ref_ids);
        free(results);
        free_index(idx);
        free_matrix_f32(&ds);
        free_matrix_f32(&qs);
        return 1;
    }

    if (ref_ids.d != (uint32_t)k || ref_dst.d != (uint32_t)k) {
        printf("ERRORE: k nei files ufficiali non corrisponde a quello scelto.\n");
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
    printf("Confronto con i risultati ufficiali...\n\n");

    compare_results(&qs, results, &ref_ids, &ref_dst, k);

    // -----------------------------------------------------
    // STAMPA RIEPILOGO BENCHMARK
    // -----------------------------------------------------
    printf("=====================================\n");
    printf("               BENCHMARK             \n");
    printf("=====================================\n");
    printf("build_index()     : %.2f ms\n", ms(t0, t1));
    printf("knn_query_all()   : %.2f ms\n", ms(t2, t3));
    printf("Totale runtime    : %.2f ms\n", ms(t0, t3));
    printf("=====================================\n\n");

    // -----------------------------------------------------
    // CLEANUP MEMORIA
    // -----------------------------------------------------
    free_matrix_i32(&ref_ids);
    free_matrix_f32(&ref_dst);

    free(results);
    free_index(idx);

    free_matrix_f32(&ds);
    free_matrix_f32(&qs);

    printf("Esecuzione completata senza errori.\n");

    return 0;
}




void test_quant() {
    float v[5] = {2.3, -6.7, 1.1, 4.5, -0.4};
    uint8_t vp[5], vn[5];

    quantize_vector(v, vp, vn, 5, 2);

    printf("v+: ");
    for (int i = 0; i < 5; i++) printf("%d ", vp[i]);
    printf("\nv-: ");
    for (int i = 0; i < 5; i++) printf("%d ", vn[i]);
    printf("\n");
}

void test_distance() {
    float v[4] = {2.3, -6.7, 1.5, 4.5};
    float w[4] = {-3.2, 0.5, 7.1, -2.0};

    uint8_t vp[4], vn[4];
    uint8_t wp[4], wn[4];

    quantize_vector(v, vp, vn, 4, 2);
    quantize_vector(w, wp, wn, 4, 2);

    int d = approximate_distance(vp, vn, wp, wn, 4);
    printf("Distanza ˜d = %d\n", d);
}

