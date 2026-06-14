# Documentazione tecnica — Progetto K-NN approssimato (Architetture 2025)

> Questo documento spiega **come è costruito** l'intero progetto: l'algoritmo, la
> struttura del codice C/Assembly, le 8 configurazioni di build, la libreria Python
> e la metodologia di verifica. Le descrizioni riflettono il **codice realmente
> presente nel repository** (verificato riga per riga), non una descrizione idealizzata.

---

## 1. Obiettivo

Dato un dataset `DS` di `n` punti in `R^D` e un insieme di `nq` query, per ogni query
`q` si cercano i `k` punti del dataset più vicini. Il calcolo esatto (distanza
euclidea verso tutti i punti) costa `O(n·D)` per query: troppo per dataset grandi.

Il progetto applica due tecniche di accelerazione previste dalla traccia:

1. **Distanza approssimata** tramite *quantizzazione binaria* → riduce il costo della
   singola distanza (lavora su vettori di bit invece che su `D` float/double).
2. **Indicizzazione a pivot** → riduce il numero di distanze esatte da calcolare,
   scartando in anticipo i punti palesemente lontani (pruning).

E confronta diverse implementazioni hardware-aware: **scalare C**, **intrinseci SIMD**
(SSE2 a 32 bit / AVX2 a 64 bit), **assembly scritto a mano** e **multi-threading OpenMP**.

---

## 2. L'algoritmo

### 2.1 Quantizzazione binaria — `quantize_vector` (`src/quantization.c`)
Per un vettore `v` di `D` componenti e un parametro `x`, si producono due vettori
binari `v⁺` e `v⁻` di `D` byte (valori 0/1):

- si individuano le **`x` componenti con `|v[i]|` massimo**;
- per ciascuna di esse: se `v[i] ≥ 0` allora `v⁺[i]=1`, altrimenti `v⁻[i]=1`;
- tutte le altre posizioni restano 0.

Implementazione: si costruisce un array di coppie `(|v[i]|, i)`, lo si ordina in modo
decrescente con `qsort`, e si marcano le prime `x` posizioni. (Versioni `float` e
`double` distinte: `quantize_vector` / `quantize_vector_f64`.)

### 2.2 Distanza approssimata `d̃` — `approximate_distance` (`src/distance*.c`)
Date due quantizzazioni `(v⁺,v⁻)` e `(w⁺,w⁻)`:

```
d̃(v,w) = (v⁺·w⁺) + (v⁻·w⁻) − (v⁺·w⁻) − (v⁻·w⁺)
```

I quattro termini sono prodotti scalari fra vettori di bit, cioè conteggi di posizioni
in cui entrambi i bit valgono 1 (`pp`, `nn`, `pn`, `np`). Il risultato è un **intero**.

> ⚠️ **Nota concettuale importante.** `d̃` è di fatto una *misura di similarità*: vale
> il **massimo** (`= x`) quando `v=w` e può essere **negativa**. Non è una metrica e non
> soddisfa la disuguaglianza triangolare. L'algoritmo della traccia la usa comunque come
> "distanza" (più piccola = più vicino). Questa è una caratteristica **della traccia**, non
> un difetto dell'implementazione: i file di riferimento (golden) sono generati con la
> stessa logica, quindi i risultati coincidono al 100%.

### 2.3 Selezione pivot e indice — `select_pivots` / `build_index` (`src/index.c`)
- Si scelgono `h` pivot ai punti di indice `⌊n/h⌋·j` per `j = 0 … h−1`.
- Si quantizza l'intero dataset (`v⁺/v⁻` per ogni punto).
- Si pre-calcola la matrice `dist[i][j] = d̃(v_i, p_j)` (dimensione `n×h`), che costituisce
  l'**indice** riutilizzato per tutte le query.

### 2.4 Querying con pruning — `knn_query_single(_f64)` (`src/query.c`, `src/query64.c`)
Per ogni query `q`:
1. la si quantizza e si calcola `d̃(q, p_j)` per ogni pivot;
2. la lista dei `k` vicini è inizializzata a distanza `+∞`;
3. per ogni punto `v_i` del dataset:
   - **lower bound** `d* = max_j |d̃(v_i,p_j) − d̃(q,p_j)|`;
   - se `d* ≥` (distanza approssimata del peggiore attualmente in lista) → **scarta** `v_i`;
   - altrimenti calcola `d̃(q, v_i)`; se è migliore del peggiore, lo **sostituisce**;
4. alla fine, per i `k` candidati rimasti si calcola la **distanza euclidea reale**
   (`euclidean_distance` / `_f64`) che diventa il valore `δ` restituito.

L'output per query sono `k` coppie `⟨id, δ⟩`, nell'ordine degli "slot" interni (non
riordinati per distanza: è la convenzione con cui sono stati generati anche i golden).

---

## 3. Struttura del repository

```
ProgettoKnnArchitetture/
├── include/                 # header
│   ├── matrix.h             #   strutture MatrixF32/F64/I32 + I/O .ds2
│   ├── quantization.h       #   quantize_vector(_f64)
│   ├── index.h              #   Index + build_index(_f64) / free_index
│   ├── query.h / query64.h  #   Neighbor(64) + knn_query_*
│   ├── distance.h           #   approximate_distance + euclidean_distance(_f64)
│   ├── config.h / compare*.h
│   └── common.h             #   [Python] struct `params`, `type`, `align`
├── src/                     # sorgenti C + Assembly
│   ├── matrix.c             #   lettura file .ds2 (binari)
│   ├── quantization.c       #   quantizzazione (qsort top-x)
│   ├── index.c              #   pivot + costruzione indice d̃(v,p)
│   ├── query.c / query64.c  #   K-NN con pruning (32 / 64 bit)
│   ├── distance.c           #   distanze: scalare + INTRINSECI SSE2/AVX2
│   ├── distance32ASSEMBLY.c #   wrapper che chiama l'asm SSE2 (USE_SSE2_ASM)
│   ├── distance64ASSEMBLY.c #   wrapper che chiama l'asm AVX2 (USE_AVX_ASM)
│   ├── distance_sse2.S      #   ASSEMBLY: approximate_distance_sse2_asm
│   ├── distance_avx2.S      #   ASSEMBLY: approximate_distance_avx2_asm
│   ├── config.c             #   parsing argomenti CLI (-d -q -h -k -x)
│   ├── compare.c/compare64.c#   confronto con i file golden
│   ├── main.c / main64.c    #   eseguibili scalare/intrinseci
│   ├── main32ASSEMBLY.c …   #   eseguibili versione assembly
│   ├── mainReport.c         #   launcher benchmark → report_completo.txt/html
│   ├── quantpivot32.c …     #   [Python] adattatori fit()/predict()
│   └── quantpivot32_py.c …  #   [Python] wrapper Python C-API
├── data/                    # dataset, query e risultati golden (.ds2)
├── python/                  # packaging della libreria (setup.py, pyproject.toml)
├── doc/                     # relazione LaTeX + PDF
├── docs/                    # questa documentazione + guida d'uso
├── examples/                # script di esempio eseguibile
└── progetto_knn.cbp         # progetto Code::Blocks (8+ target)
```

### Formato dei file `.ds2`
Binario: header di **2 interi `uint32`** (`n`, `d`), seguiti da `n·d` valori in
ordine *row-major*. Il tipo dei valori dipende dal file:
- dataset/query: `float32` (suffisso `_32`) o `float64` (`_64`);
- `results_ids_*`: `int32` (gli identificativi dei vicini);
- `results_dst_*`: `float32` (`_32`) o `float64` (`_64`) (le distanze euclidee).

---

## 4. Il nucleo C e le macro di compilazione

Il **back-end di calcolo è intercambiabile** tramite macro di preprocessore, mantenendo
un'unica logica di alto livello (`index.c`, `query*.c`). Il punto di commutazione è la
funzione `approximate_distance` e, per la versione intrinseca, anche `euclidean_distance_f64`.

| File compilato | Macro | `approximate_distance` esegue | Euclidea |
|---|---|---|---|
| `distance.c` | *(nessuna)* | ciclo **scalare** C | scalare |
| `distance.c` | `USE_SSE2` | **intrinseci SSE2** (128 bit) | scalare (f32) |
| `distance.c` | `USE_AVX` | **intrinseci AVX2** (256 bit) | **intrinseci AVX** (f64) |
| `distance32ASSEMBLY.c` | `USE_SSE2_ASM` | **asm** `approximate_distance_sse2_asm` | scalare |
| `distance64ASSEMBLY.c` | `USE_AVX_ASM` | **asm** `approximate_distance_avx2_asm` | scalare |

> Nota onesta sul codice: la **distanza euclidea NON è implementata in assembly** in
> nessuna variante (nelle build "Assembly" è un ciclo scalare C; solo la versione
> *intrinseca AVX* vettorizza l'euclidea a 64 bit). L'assembly accelera esclusivamente la
> **distanza approssimata** `d̃`, che è l'operazione dominante grazie al pruning.

Altri moduli:
- `matrix.c`: I/O dei `.ds2` con `malloc` semplice (nessun allineamento forzato — non
  necessario perché l'asm usa load *non allineate*, vedi §5).
- `config.c`: parsing di `-d -q -h -k -x`. Obbligatori `-d`/`-q`.
- `compare.c`/`compare64.c`: confronto risultati calcolati vs golden (tolleranza `1e-3`).
- `main.c`/`main64.c` (intrinseci/scalare) e `main32ASSEMBLY.c`/`main64ASSEMBLY.c` (asm).

---

## 5. Le routine Assembly (`distance_sse2.S`, `distance_avx2.S`)

Sono file **GAS** (GNU assembler) in sintassi Intel (`.intel_syntax noprefix`),
assemblati da `gcc`. Calcolano `d̃ = pp + nn − pn − np`.

- **ABI Windows x64 (MinGW):** i 4 puntatori `v⁺,v⁻,w⁺,w⁻` arrivano in `RCX, RDX, R8, R9`;
  la dimensione `D` è sullo stack. Nel prologo si salvano i registri non-volatili usati
  (XMM6–XMM11 e GPR `r12–r15` nella versione SSE2) e si ripristinano nell'epilogo.
- **Conteggio dei bit:** per ogni blocco si esegue `PAND` fra le maschere e poi `PSADBW`
  (Sum of Absolute Differences) per sommare orizzontalmente i byte → un *population count*
  vettoriale (si è scelto `PSADBW` invece di `POPCNT` per portabilità).
- **Load non allineate:** `movdqu` (SSE2) / `vmovdqu` (AVX2): nessun vincolo di
  allineamento sui dati di input.
- **Fast path `D=256`:** poiché il dataset ha `D=256`, esiste un percorso completamente
  *srotolato* (16 blocchi da 16 byte in SSE2; 8 blocchi da 32 byte in AVX2) che elimina i
  salti di ciclo. È il percorso effettivamente usato dal workload.
- **Riduzione finale:** i quattro accumulatori vengono ridotti a interi a 64 bit e
  combinati come `pp + nn − pn − np`, restituito in `RAX`.

> Limitazioni note (non attivate dal workload `D=256`, ma presenti): nella versione SSE2 il
> ciclo scalare dei "resti" (per `D` non multiplo di 16) corrompe l'accumulatore; nella
> versione AVX2 il percorso generico (per `D≠256`) scrive il registro non-volatile `r12`
> senza salvarlo. Entrambi i percorsi *fast `D=256`* sono invece corretti.

---

## 6. Le 8 configurazioni di build (Code::Blocks)

`progetto_knn.cbp` definisce i target seguenti (più `Debug`, `Release`,
`Benchmark_Report` e il virtual target `Build_TUTTO` che li compila tutti).

| # | Target | Bit | Tecnica | Sorgenti chiave | Flag/Macro |
|---|---|---|---|---|---|
| 1 | `Release_Scalar` | 32 | Scalare C | main.c, distance.c, query.c | — |
| 2 | `Release_SSE2` | 32 | Intrinseci SSE2 | main.c, distance.c | `-msse2 -DUSE_SSE2` |
| 3 | `Release_SSE2ASSEMBLY` | 32 | **Assembly** SSE2 | main32ASSEMBLY.c, distance32ASSEMBLY.c, distance_sse2.S | `-msse2 -DUSE_SSE2_ASM` |
| 4 | `Release_SSE2_OpenMP` | 32 | Assembly SSE2 + OpenMP | come #3 | `-fopenmp -msse2 -DUSE_SSE2_ASM` |
| 5 | `Release_Scalar64` | 64 | Scalare C | main64.c, distance.c, query64.c | — |
| 6 | `Release_AVX64` | 64 | Intrinseci AVX2 | main64.c, distance.c | `-mavx2 -DUSE_AVX` |
| 7 | `Release_AVX64ASSEMBLY` | 64 | **Assembly** AVX2 | main64ASSEMBLY.c, distance64ASSEMBLY.c, distance_avx2.S | `-mavx2 -DUSE_AVX_ASM` |
| 8 | `Release_AVX64_OpenMP` | 64 | Assembly AVX2 + OpenMP | come #7 | `-fopenmp -mavx2 -DUSE_AVX_ASM` |

Il parallelismo OpenMP agisce sul **loop delle query** (`#pragma omp parallel for` in
`knn_query_all`/`knn_query_all_f64`): ogni thread elabora query indipendenti.

---

## 7. La libreria Python

Il pacchetto si chiama **`Gruppo_Ferrari_DeFusco_Cuconato`** ed espone tre sotto-moduli,
ciascuno con una classe `QuantPivot` (`fit` / `predict`):

| Modulo | Precisione | Back-end |
|---|---|---|
| `quantpivot32` | `float32` | SSE2 assembly |
| `quantpivot64` | `float64` | AVX2 assembly |
| `quantpivot64omp` | `float64` | AVX2 assembly + OpenMP |

### 7.1 Architettura a tre strati
```
  Python (NumPy)
      │  fit(dataset, n_pivots, quant_level) / predict(query, k)
      ▼
  quantpivot32_py.c   ← wrapper Python C-API: valida gli array NumPy,
      │                  riempie la struct `params`, gestisce i riferimenti,
      │                  costruisce gli array di output (id, dist) con capsule
      ▼
  quantpivot32.c      ← adattatore: traduce `params` in MatrixF32/Index/Neighbor
      │                  e chiama l'algoritmo verificato
      ▼
  index.c / query.c   ← algoritmo K-NN  →  distance32ASSEMBLY.c → distance_sse2.S
```

- **`include/common.h`** definisce la struct `params` (puntatori a dataset/query, parametri
  `h,k,x`, dimensioni, puntatore all'indice, buffer dei risultati) e, in base alla macro
  `QP_DOUBLE`, il tipo scalare `type` (`float`/`double`) e l'allineamento `align` (16/32).
- **`quantpivot{32,64,64omp}.c`**: `fit()` costruisce l'indice (`build_index`/`_f64`),
  `predict()` esegue `knn_query_all`/`_f64` e copia `id` e `dist_real` nei buffer di output.
  La distanza approssimata passa per l'assembly grazie alle macro `USE_SSE2_ASM`/`USE_AVX_ASM`.
- **`quantpivot{32,64,64omp}_py.c`**: lo strato C-API (tipo Python, metodi `fit`/`predict`,
  conversione NumPy↔C, gestione memoria con `PyCapsule`).

### 7.2 Build con `python/setup.py`
`setup.py` dichiara **tre estensioni** (una per modulo), ognuna compilata dai rispettivi
sorgenti C + il file `.S`, con:
- macro `USE_SSE2_ASM` (32 bit) oppure `USE_AVX_ASM`+`QP_DOUBLE` (64 bit);
- flag `-O3 -msse2` / `-O3 -mavx2` (+ `-fopenmp` per la variante OMP);
- una classe **`build_ext` personalizzata** che (1) abilita i sorgenti `.S`/`.s` (distutils
  non li riconosce di default) e (2) forza il compilatore **MinGW (`mingw32`)** su Windows,
  indispensabile perché i file `.S` GAS e i flag GCC non sono supportati da MSVC.

### 7.3 Toolchain e processo di build
- Richiede **MinGW-w64** (GCC). Su Windows: `winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT`.
- Build del wheel: `python setup.py bdist_wheel` (dalla cartella `python/`, con `mingw64/bin` nel PATH).
- Wheel **autosufficiente**: `delvewheel repair` incorpora le DLL runtime di GCC
  (`libgcc_s_seh`, `libgomp`, `libwinpthread`) così il pacchetto funziona senza MinGW installato.

Vedi `docs/GUIDA_UTILIZZO.md` per i comandi esatti.

---

## 8. Metodologia di verifica

La correttezza è stata verificata in modo **indipendente** (`_verify/reference_knn.py`):
una reimplementazione dell'algoritmo in puro NumPy. Risultato:

- riproduce **esattamente** i golden `data/results_*.ds2` — **2000/2000 query**, stessi
  identificativi (nello stesso ordine) e stesse distanze entro tolleranza — sia a 32 sia a
  64 bit, con `h=16, k=8, x=64`;
- il determinismo è garantito (nessun pareggio di modulo, nessun valore nullo nei dati).

Inoltre il pacchetto Python compilato è stato verificato end-to-end
(`_verify/test_package.py`, `_verify/test_installed.py`): tutti e tre i moduli, passando per
l'assembly, riproducono i golden **2000/2000**.

---

## 9. Prestazioni (da `report_completo.txt`)

Tempi misurati sulle 2000 query, `D=256`, `h=16, k=8, x=64` (BUILD = costruzione indice,
QUERY = ricerca):

| Configurazione | BUILD (ms) | QUERY (ms) | Speedup query vs scalare |
|---|---:|---:|---:|
| 32-bit Scalar (C) | 171 | 7743 | 1× |
| 32-bit SSE2 (intrinseci) | 133 | 4013 | ~1.9× |
| 32-bit SSE2 (assembly) | 83 | 458 | ~16.9× |
| 32-bit SSE2 + OpenMP | 145 | 122 | ~63× |
| 64-bit Scalar (C) | 173 | 7485 | 1× |
| 64-bit AVX2 (intrinseci) | 117 | 3505 | ~2.1× |
| 64-bit AVX2 (assembly) | 101 | 870 | ~8.6× |
| 64-bit AVX2 + OpenMP | 103 | 110 | ~68× |

> I valori dipendono dalla macchina; quelli sopra sono quelli salvati in
> `report_completo.txt`. (La relazione LaTeX cita numeri di una run diversa: per coerenza,
> fare riferimento a `report_completo.txt` come dato ufficiale.)

---

## 10. Riepilogo dei punti di forza e delle limitazioni

**Punti di forza**
- Algoritmo conforme alla traccia e **numericamente corretto** (verificato 2000/2000).
- Back-end intercambiabili tramite macro: confronto pulito scalare / intrinseci / asm / OpenMP.
- Assembly corretto sul percorso `D=256` (quello realmente usato), con riduzione via `PSADBW`.
- Libreria Python funzionante, con wheel autosufficiente.

**Limitazioni note (documentate, non bloccanti per il workload D=256)**
- Distanza euclidea non vettorizzata in assembly (solo scalare o intrinseca).
- Bug latenti nei percorsi assembly "generici" (`D≠256`): resto SSE2 e `r12` in AVX2.
- Validazione dei parametri `h/k/x` incompleta (la traccia chiede "messaggio + terminazione").
- La relazione LaTeX contiene affermazioni tecniche non allineate al codice (POPCNT, HADDPS,
  MOVAPS allineate, quantizzazione in asm, CLI argparse): da correggere prima della consegna.
