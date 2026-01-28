# ProgettoKnnArchitetture

Questo progetto implementa un sistema ad alte prestazioni per il calcolo dei **K-Nearest Neighbors (K-NN)** utilizzando distanze approssimate (basate su quantizzazione e pivot) e distanze esatte (Euclidea).

Il focus principale è l'ottimizzazione architetturale su processori x86-64, confrontando diverse tecniche:
* **Scalare (C puro)**
* **SIMD Intrinsics** (SSE2 per 32-bit, AVX2 per 64-bit)
* **Assembly puro** (procedure ottimizzate manualmente)
* **Multi-threading** (OpenMP)

## Requisiti di Sistema

Per eseguire correttamente tutte le configurazioni, la macchina deve disporre di:
* **Sistema Operativo:** Windows (tramite MinGW-w64) o Linux.
* **CPU:** Supporto hardware per istruzioni **AVX2** (per le versioni 64-bit) e **SSE2**.
* **Compilatore:** GCC con supporto OpenMP (flag `-fopenmp` attivo).
* **IDE:** Code::Blocks (progetto configurato con Virtual Targets).

## Struttura del Progetto

* `src/`: Contiene i codici sorgente C (`.c`) e Assembly (`.S`).
* `include/`: Contiene gli header files (`.h`).
* `data/`: Deve contenere i dataset e le query (formato `.ds2`).
    * *Nota: Assicurarsi che i file di input siano presenti in questa cartella prima dell'esecuzione.*
* `bin/`: Cartella di output dove verranno generati i vari eseguibili, organizzati in sottocartelle.

---

## Istruzioni per la Compilazione (Build)

Il progetto è configurato su Code::Blocks con un **Virtual Target** che permette di compilare tutte le 8 versioni simultaneamente.

1.  Aprire il file `progetto_knn.cbp` con **Code::Blocks**.
2.  Nel menu a tendina dei target (nella barra degli strumenti in alto), selezionare **`Build_TUTTO`**.
3.  Cliccare sul pulsante **Rebuild** (icona con le due frecce blu, oppure premere `Ctrl+F11`).

> **Cosa succede:** Code::Blocks compilerà separatamente tutte le versioni (Scalar, SSE, AVX, ASM, OpenMP) e posizionerà i rispettivi `.exe` nelle sottocartelle di `bin/` (es. `bin/Release_AVX64_OpenMP/`).
> Attendere fino alla comparsa del messaggio "0 errors".

---

## Esecuzione del Benchmark Automatizzato

Per avviare la suite di test completa e generare il report comparativo:

1.  Assicurarsi di aver completato la fase di **Compilazione** (vedi punto precedente).
2.  Nel menu a tendina dei target, selezionare **`Benchmark_Report`**.
3.  Cliccare su **Run** (icona triangolo verde, oppure premere `F9`).

### Output del Benchmark
Il launcher eseguirà in sequenza tutte le configurazioni e genererà due file nella root del progetto:
* **`report_completo.txt`**: Una tabella testuale riassuntiva.
* **`report_completo.html`**: Un report grafico dettagliato dove le celle verdi indicano i tempi migliori per ogni categoria.

---

## Esecuzione Manuale (Singola Configurazione)

Se si desidera eseguire o debuggare una singola versione specifica:

1.  Selezionare il target desiderato dal menu a tendina (es. `Release_AVX64_OpenMP`).
2.  Cliccare su **Run** (`F9`).

### Parametri da Riga di Comando
Il programma accetta i seguenti argomenti (già configurati nell'IDE, ma modificabili):

```bash
./progetto_knn.exe -d data/dataset.ds2 -q data/query.ds2 -h <pivot> -k <vicini> -x <quant>

* `-d`: Percorso del file Dataset
* `-q`: Percorso del file Query
* `-h`: Numero di Pivot (es. 16)
* `-k`: Numero di K vicini da cercare (es. 8)
* `-x`: Fattore di quantizzazione (es. 64)

---

##  Configurazioni Implementate

Il benchmark confronta le seguenti modalità operative:

| Architettura | Modalità | Descrizione |
|:---:|:---|:---|
| **32-bit** | `Scalar` | Codice C standard (Single Precision) |
| **32-bit** | `SSE2 Intrinsic` | Vettorizzazione C con intrinseci SSE |
| **32-bit** | `SSE2 Assembly` | Procedura critica scritta in Assembly x86 esterno |
| **32-bit** | `SSE2 + OpenMP` | **Massima Performance** (SSE2 + Multi-core) |
| **64-bit** | `Scalar` | Codice C standard (Double Precision) |
| **64-bit** | `AVX2 Intrinsic` | Vettorizzazione C con intrinseci AVX2 |
| **64-bit** | `AVX2 Assembly` | Procedura critica scritta in Assembly AVX2 esterno |
| **64-bit** | `AVX2 + OpenMP` | **Massima Performance** (AVX2 + Multi-core) |
