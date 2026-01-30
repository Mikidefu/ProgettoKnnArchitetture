#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// --------------------------------------------------------
// CONFIGURAZIONE PERCORSI E ARGOMENTI
// --------------------------------------------------------

typedef struct {
    char *label;        // Etichetta
    char *exe_path;     // Percorso eseguibile
    int   is_64bit;     // 1 --> 64-bit, 0 --> 32-bit
} TestConfig;

// Lista di tutte le configurazioni richieste
TestConfig TESTS[] = {
    // 32-BIT SCALARE / SSE
    { "32-bit Scalar (C)",          "bin\\Release_Scalar\\progetto_knn.exe",        0 },
    { "32-bit SSE2 (Intrinsic)",    "bin\\Release_SSE2\\progetto_knn.exe",          0 },
    { "32-bit SSE2 (Assembly)",     "bin\\Release_SSE2ASSEMBLY\\progetto_knn.exe",  0 },
    { "32-bit SSE2 + OpenMP",       "bin\\Release_SSE2_OpenMP\\progetto_knn.exe",   0 },

    // 64-BIT SCALARE / AVX
    { "64-bit Scalar (C)",          "bin\\Release_Scalar64\\progetto_knn.exe",      1 },
    { "64-bit AVX2 (Intrinsic)",    "bin\\Release_AVX64\\progetto_knn.exe",         1 },
    { "64-bit AVX2 (Assembly)",     "bin\\Release_AVX64ASSEMBLY\\progetto_knn.exe", 1 },
    { "64-bit AVX2 + OpenMP",       "bin\\Release_AVX64_OpenMP\\progetto_knn.exe",  1 },

    { NULL, NULL, 0 }
};

// Argomenti
const char *ARGS_32 = "-d data/dataset_2000x256_32.ds2 -q data/query_2000x256_32.ds2 -h 16 -k 8 -x 64";
const char *ARGS_64 = "-d data/dataset_2000x256_64.ds2 -q data/query_2000x256_64.ds2 -h 16 -k 8 -x 64";

// Struttura per salvare i risultati
typedef struct {
    double build_time;
    double query_time;
    double total_time;
    int    valid;
} TestResult;

TestResult results[20];

// --------------------------------------------------------
// FUNZIONI DI SUPPORTO
// --------------------------------------------------------

// Esegue il comando e parsa l'output per trovare i tempi
TestResult run_test(int test_idx) {
    TestResult res = {0, 0, 0, 0};
    char command[1024];
    char line[1024];

    TestConfig cfg = TESTS[test_idx];
    const char *args = cfg.is_64bit ? ARGS_64 : ARGS_32;

    // Verifica esistenza file
    FILE *fcheck = fopen(cfg.exe_path, "r");
    if (!fcheck) {
        printf("[SKIP] File non trovato: %s\n", cfg.exe_path);
        return res;
    }
    fclose(fcheck);

    printf("Running: %-30s ... ", cfg.label);
    fflush(stdout);

    // Costruisce il comando
    sprintf(command, ".\\%s %s", cfg.exe_path, args);

    // Apre la pipe per leggere l'output del programma
    FILE *pipe = _popen(command, "r");
    if (!pipe) {
        printf("ERRORE: impossibile eseguire comando.\n");
        return res;
    }

    // Legge riga per riga l'output
    while (fgets(line, sizeof(line), pipe)) {
        if (strstr(line, "build_index()")) {
            char *p = strchr(line, ':');
            if (p) res.build_time = atof(p + 1);
        }
        else if (strstr(line, "knn_query_all()")) {
            char *p = strchr(line, ':');
            if (p) res.query_time = atof(p + 1);
        }
    }

    _pclose(pipe);

    res.total_time = res.build_time + res.query_time;
    if (res.total_time > 0) {
        res.valid = 1;
        printf("DONE (%.2f ms)\n", res.total_time);
    } else {
        printf("FAILED (No output parsing)\n");
    }

    return res;
}

void write_html_report(const char *filename) {
    FILE *f = fopen(filename, "w");
    if (!f) return;

    // Minimi per 32-bit e 64-bit
    double min_build_32 = DBL_MAX, min_query_32 = DBL_MAX, min_tot_32 = DBL_MAX;
    double min_build_64 = DBL_MAX, min_query_64 = DBL_MAX, min_tot_64 = DBL_MAX;

    // Calcolo minimi
    for (int i = 0; TESTS[i].label != NULL; i++) {
        if (!results[i].valid) continue;

        if (TESTS[i].is_64bit) {
            if (results[i].build_time < min_build_64) min_build_64 = results[i].build_time;
            if (results[i].query_time < min_query_64) min_query_64 = results[i].query_time;
            if (results[i].total_time < min_tot_64)   min_tot_64   = results[i].total_time;
        } else {
            if (results[i].build_time < min_build_32) min_build_32 = results[i].build_time;
            if (results[i].query_time < min_query_32) min_query_32 = results[i].query_time;
            if (results[i].total_time < min_tot_32)   min_tot_32   = results[i].total_time;
        }
    }

    // HTML + CSS
    fprintf(f, "<!DOCTYPE html><html><head><title>Benchmark Report</title>");
    fprintf(f, "<style>");
    fprintf(f, "body { font-family: sans-serif; padding: 20px; background: #f4f4f4; }");
    fprintf(f, "h1 { color: #333; }");
    fprintf(f, "table { border-collapse: collapse; width: 100%%; background: white; }");
    fprintf(f, "th, td { padding: 10px; text-align: right; border-bottom: 1px solid #ddd; }");
    fprintf(f, "th { background-color: #333; color: white; text-align: left; }");
    fprintf(f, ".label { text-align: left; font-weight: bold; }");
    fprintf(f, ".best { background-color: #d4edda; font-weight: bold; }");
    fprintf(f, "</style></head><body>");

    fprintf(f, "<h1>Benchmark Report: K-NN Optimization</h1>");
    fprintf(f, "<p>Generato automaticamente tramite Launcher.</p>");

    fprintf(f, "<table>");
    fprintf(f, "<thead><tr>");
    fprintf(f, "<th>Configurazione</th><th>Build (ms)</th><th>Query (ms)</th><th>Totale (ms)</th>");
    fprintf(f, "</tr></thead><tbody>");

    for (int i = 0; TESTS[i].label != NULL; i++) {
        if (!results[i].valid) continue;

        int is64 = TESTS[i].is_64bit;

        fprintf(f, "<tr>");
        fprintf(f, "<td class='label'>%s</td>", TESTS[i].label);

        // BUILD
        if ((!is64 && results[i].build_time == min_build_32) ||
            ( is64 && results[i].build_time == min_build_64))
            fprintf(f, "<td class='best'>%.2f</td>", results[i].build_time);
        else
            fprintf(f, "<td>%.2f</td>", results[i].build_time);

        // QUERY
        if ((!is64 && results[i].query_time == min_query_32) ||
            ( is64 && results[i].query_time == min_query_64))
            fprintf(f, "<td class='best'>%.2f</td>", results[i].query_time);
        else
            fprintf(f, "<td>%.2f</td>", results[i].query_time);

        // TOTAL
        if ((!is64 && results[i].total_time == min_tot_32) ||
            ( is64 && results[i].total_time == min_tot_64))
            fprintf(f, "<td class='best'>%.2f</td>", results[i].total_time);
        else
            fprintf(f, "<td>%.2f</td>", results[i].total_time);

        fprintf(f, "</tr>");
    }

    fprintf(f, "</tbody></table>");

    fprintf(f,
        "<p><i>* Valori più bassi indicano prestazioni migliori.<br>"
        "Le celle evidenziate rappresentano la miglior prestazione "
        "all’interno del gruppo 32-bit o 64-bit.</i></p>");

    fprintf(f, "</body></html>");
    fclose(f);

    printf("\nReport HTML corretto salvato in: %s\n", filename);
}


int main() {
    printf("==================================================\n");
    printf("      AUTOMATED BENCHMARK SUITE (FULL)            \n");
    printf("==================================================\n\n");

    for (int i = 0; TESTS[i].label != NULL; i++) {
        results[i] = run_test(i);
    }

    write_html_report("report_completo.html");

    FILE *ftxt = fopen("report_completo.txt", "w");
    if (ftxt) {
        fprintf(ftxt, "%-30s | %10s | %10s | %10s\n", "CONFIG", "BUILD", "QUERY", "TOTAL");
        fprintf(ftxt, "------------------------------------------------------------------\n");
        for (int i = 0; TESTS[i].label != NULL; i++) {
            if(results[i].valid) {
                fprintf(ftxt, "%-30s | %10.2f | %10.2f | %10.2f\n",
                        TESTS[i].label, results[i].build_time, results[i].query_time, results[i].total_time);
            }
        }
        fclose(ftxt);
    }

    printf("\nPremere INVIO per uscire...");
    getchar();
    return 0;
}
