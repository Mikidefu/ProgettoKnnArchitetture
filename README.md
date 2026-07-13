# ProgettoKnnArchitetture

Questo progetto implementa un sistema ad alte prestazioni per il calcolo dei **K-Nearest Neighbors (K-NN)** utilizzando distanze approssimate (basate su quantizzazione e pivot) e distanze esatte (Euclidea).

Il focus principale è l'ottimizzazione architetturale su processori x86-64, confrontando diverse tecniche:
* **Scalare (C puro)**
* **SIMD Intrinsics** (SSE2 per 32-bit, AVX2 per 64-bit)
* **Assembly puro** (procedure ottimizzate manualmente)
* **Multi-threading** (OpenMP)

## Documentazione

* [`docs/ARCHITETTURA.md`](docs/ARCHITETTURA.md) — documentazione tecnica completa: algoritmo, struttura del codice C/Assembly, le 8 configurazioni di build, la libreria Python e la verifica.
* [`docs/GUIDA_UTILIZZO.md`](docs/GUIDA_UTILIZZO.md) — guida pratica: installazione, uso della libreria e degli eseguibili, come vedere i risultati.
* [`examples/esempio_knn.py`](examples/esempio_knn.py) — esempio eseguibile che lancia le tre varianti e confronta con i risultati di riferimento.

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
* `python/` Cartella con tutti i file necesssari per integrazione in python. In particolare contiene la cartella 
                Gruppo_Ferrari_deFusco_Cuconato al cui interso c'è l'organizzazione inmoduli per le diverse architetture richieste 


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

## INSTALLAZIONE DEL PACCHETTO IN PYTHON

Il pacchetto `Gruppo_Ferrari_DeFusco_Cuconato` espone tre moduli (`quantpivot32`,
`quantpivot64`, `quantpivot64omp`), ognuno con una classe `QuantPivot` con i metodi
`fit(dataset, n_pivots, quant_level)` e `predict(query, k)`. Il calcolo della distanza
approssimata è vettorizzato con **intrinseci SIMD** (SSE2 a 32 bit, AVX2 a 64 bit),
portabili su ogni piattaforma; la variante `quantpivot64omp` parallelizza le query con **OpenMP**.

### Requisiti
Serve soltanto un **compilatore C** e **NumPy** — nessun toolchain particolare:
* **Linux / macOS:** `gcc` o `clang` (di norma già presenti); su Linux servono gli header
  di Python (pacchetto `python3-dev`).
* **Windows:** MSVC (Build Tools per Visual Studio) **oppure** MinGW-w64.

`setup.py` sceglie automaticamente i flag corretti per il compilatore in uso.

### Installazione
Dalla **cartella principale del progetto** (quella che contiene `setup.py`):

```bash
pip install .
```

Il comando compila le estensioni con il compilatore di sistema e installa il pacchetto.
Verifica (da una cartella **diversa** da `python/`, per non mascherare i moduli compilati
con i sorgenti):

```bash
python -c "from Gruppo_Ferrari_DeFusco_Cuconato.quantpivot64omp import QuantPivot; print('OK')"
```

> **Nota (Windows/PowerShell):** se `python` apre lo *stub* del Microsoft Store, usa il
> percorso completo dell'interprete; se compili con MinGW aggiungi la sua cartella `bin` al
> PATH. Dettagli in [`docs/GUIDA_UTILIZZO.md`](docs/GUIDA_UTILIZZO.md).

### Esempio d'uso
```python
import numpy as np
from Gruppo_Ferrari_DeFusco_Cuconato.quantpivot64omp import QuantPivot

DS = np.ascontiguousarray(dataset, dtype=np.float64)  # (N, D)
Q  = np.ascontiguousarray(query,   dtype=np.float64)  # (nq, D)

model = QuantPivot().fit(DS, n_pivots=16, quant_level=64)
ids, dists = model.predict(Q, k=8)   # ids: (nq, k) int32, dists: (nq, k) distanze euclidee reali
```
> Nota: usare `float32` per `quantpivot32`, `float64` per `quantpivot64`/`quantpivot64omp`.

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
