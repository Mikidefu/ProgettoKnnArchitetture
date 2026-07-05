# Guida all'uso — Progetto K-NN approssimato

Guida pratica per **installare**, **usare** e **vedere i risultati** del progetto, sia
tramite la **libreria Python** sia tramite gli **eseguibili C**.

Indice:
1. [Prerequisiti](#1-prerequisiti)
2. [Installare la libreria Python](#2-installare-la-libreria-python)
3. [Usare la libreria Python](#3-usare-la-libreria-python)
4. [Vedere i risultati: esempio e verifica](#4-vedere-i-risultati-esempio-e-verifica)
5. [Eseguibili C e benchmark](#5-eseguibili-c-e-benchmark)
6. [Formato dei dati e dei risultati](#6-formato-dei-dati-e-dei-risultati)
7. [Risoluzione problemi](#7-risoluzione-problemi)

---

## 1. Prerequisiti

| Per... | Serve |
|---|---|
| **Usare** un wheel già compilato | Python 3.x + NumPy |
| **Compilare** la libreria da sorgente | Python 3.x, NumPy, **MinGW-w64 (GCC)** |
| **Compilare** gli eseguibili C | Code::Blocks + MinGW-w64 (oppure GCC) |

Installare MinGW-w64 su Windows (una tantum):
```powershell
winget install -e --id BrechtSanders.WinLibs.POSIX.UCRT
```
Annotare il percorso della cartella `mingw64\bin` (es.
`C:\Users\<utente>\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_*\mingw64\bin`).

---

## 2. Installare la libreria Python

> **⚠️ Quale `python` usare (PowerShell su Windows).** Spesso il comando `python` punta allo
> *stub "Alias di esecuzione app"* del Microsoft Store (`...\WindowsApps\python.exe`), che
> non è un interprete reale → errore *"Impossibile eseguire il programma 'python.exe'"*. Usa
> l'interprete vero tramite percorso completo, memorizzandolo in una variabile di sessione:
> ```powershell
> $py = "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\python.exe"   # adatta se diverso
> & $py --version
> ```
> Per trovare il tuo: `Get-Command python -All` / `where.exe python`. **Attenzione:** `py`
> può avviare un *altro* Python privo di NumPy/setuptools — assicurati di usare l'interprete
> in cui hai installato le dipendenze. Nei comandi seguenti usa **`& $py`** al posto di
> `python`. Inoltre **PowerShell non espande i caratteri jolly `*`** passati a programmi
> esterni (`pip`, `delvewheel`): indica il nome completo del file, oppure espandi il glob con
> `Get-ChildItem` (come mostrato sotto).

### Opzione A — installare il wheel già pronto (consigliata)
Nel repository è presente un wheel **autosufficiente** (le DLL di GCC sono incluse:
non serve avere MinGW installato per usarlo):

```powershell
# (definisci prima $py come nella nota sopra)
& $py -m pip install python\dist\repaired\gruppo_ferrari_defusco_cuconato-1.0-cp314-cp314-win_amd64.whl
```

### Opzione B — ricompilare da sorgente
Con la cartella `mingw64\bin` nel PATH della sessione:

```powershell
# interprete Python corretto (vedi nota sopra) e cartella bin di MinGW
$py    = "$env:LOCALAPPDATA\Python\pythoncore-3.14-64\python.exe"   # adatta se diverso
$mingw = (Resolve-Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_*\mingw64\bin").Path
$env:Path = "$mingw;" + $env:Path

# 1) entra nella cartella di packaging
cd python

# 2) costruisci il wheel (usa automaticamente MinGW)
& $py setup.py bdist_wheel

# 3) (consigliato) rendi il wheel autosufficiente includendo le DLL runtime
& $py -m pip install delvewheel
$src = (Get-ChildItem dist\*.whl | Select-Object -First 1).FullName    # espande il glob
& $py -m delvewheel repair $src --add-path "$mingw" -w dist\repaired

# 4) installa (espandi il glob: PowerShell non lo fa per gli exe esterni)
$whl = (Get-ChildItem dist\repaired\*.whl | Select-Object -First 1).FullName
& $py -m pip install --force-reinstall $whl
```

In alternativa, installazione diretta (richiede però MinGW nel PATH **anche a runtime**):
```powershell
& $py -m pip install . --no-build-isolation
```

Verifica rapida dell'installazione — **da una cartella diversa da `python\`** (es. la root
del progetto), altrimenti la cartella sorgente `python\Gruppo_Ferrari_DeFusco_Cuconato\`
maschera il pacchetto installato e l'import fallisce:
```powershell
cd ..    # esci da python\ verso la root del progetto
& $py -c "from Gruppo_Ferrari_DeFusco_Cuconato.quantpivot64omp import QuantPivot; print('OK')"
```

---

## 3. Usare la libreria Python

Ogni modulo espone la classe `QuantPivot` con due metodi:

| Metodo | Firma | Cosa fa |
|---|---|---|
| `fit` | `fit(dataset, n_pivots, quant_level, silent=1)` | costruisce l'indice a pivot. Ritorna `self` (concatenabile). |
| `predict` | `predict(query, k, silent=0)` | esegue il K-NN. Ritorna la tupla `(ids, dists)`. |

- `dataset` / `query`: array NumPy **2D** `(N, D)` / `(nq, D)`, **C-contigui**.
  - `quantpivot32` → `dtype=float32`
  - `quantpivot64` / `quantpivot64omp` → `dtype=float64`
- `n_pivots` = `h`, `quant_level` = `x`, `k` = numero di vicini.
- Ritorno: `ids` `(nq, k)` `int32` (indici nel dataset) e `dists` `(nq, k)` (distanze
  **euclidee reali** verso quei vicini).

Esempio minimo:
```python
import numpy as np
from Gruppo_Ferrari_DeFusco_Cuconato.quantpivot64omp import QuantPivot

# dataset e query (float64 per la variante a 64 bit)
DS = np.ascontiguousarray(dataset, dtype=np.float64)   # (N, 256)
Q  = np.ascontiguousarray(query,   dtype=np.float64)   # (nq, 256)

model = QuantPivot().fit(DS, n_pivots=16, quant_level=64)
ids, dists = model.predict(Q, k=8)

print(ids[0])     # indici degli 8 vicini della prima query
print(dists[0])   # distanze euclidee corrispondenti
```

Per la versione a 32 bit basta cambiare import e dtype:
```python
from Gruppo_Ferrari_DeFusco_Cuconato.quantpivot32 import QuantPivot
DS = np.ascontiguousarray(dataset, dtype=np.float32)
Q  = np.ascontiguousarray(query,   dtype=np.float32)
```

> **Allineamento.** I wrapper richiedono che gli array siano allineati (16 byte per 32 bit,
> 32 byte per 64 bit). Gli array NumPy "normali" possono non esserlo: vedi la funzione
> `aligned()` in `examples/esempio_knn.py` per ottenere copie allineate in modo sicuro.

---

## 4. Vedere i risultati: esempio e verifica

### 4.1 Script di esempio (consigliato)
`examples/esempio_knn.py` carica i dati di `data/`, esegue **tutte e tre** le varianti,
stampa i vicini di una query campione, i **tempi** e il **confronto con i golden**:

```powershell
python examples\esempio_knn.py
```

Output atteso (estratto):
```
=== quantpivot64omp (AVX2 + OpenMP, float64) ===
Query #0 — primi 8 vicini (id : distanza euclidea):
   id=1623  dist=21.4537
   ...
Tempo fit:  ... ms | Tempo predict: ... ms
Confronto con i golden: 2000/2000 query corrette  ✔
```

### 4.2 Verifica formale (riproduzione dei golden)
Gli script in `_verify/` confrontano l'output con i file di riferimento ufficiali:

```powershell
# reimplementazione indipendente in NumPy vs golden (non richiede la libreria compilata)
python _verify\reference_knn.py

# libreria compilata (percorso assembly) vs golden
python _verify\test_package.py
```
Entrambi devono riportare **2000/2000** per 32 e 64 bit.

---

## 5. Eseguibili C e benchmark

Gli eseguibili si compilano con **Code::Blocks** aprendo `progetto_knn.cbp`:

1. Selezionare il target `Build_TUTTO` e premere **Rebuild** (compila tutte le 8 versioni).
2. Per una singola versione, selezionare es. `Release_AVX64_OpenMP` e premere **Run**.

Esecuzione manuale da riga di comando:
```bash
progetto_knn.exe -d data/dataset_2000x256_32.ds2 -q data/query_2000x256_32.ds2 -h 16 -k 8 -x 64
```
| Flag | Significato | Esempio |
|---|---|---|
| `-d` | file dataset `.ds2` | `data/dataset_2000x256_32.ds2` |
| `-q` | file query `.ds2` | `data/query_2000x256_32.ds2` |
| `-h` | numero di pivot | `16` |
| `-k` | numero di vicini | `8` |
| `-x` | parametro di quantizzazione | `64` |

> A 32 bit l'eseguibile confronta automaticamente con `data/results_*_x64_32.ds2` e si
> aspetta `k=8`: per altri valori di `k` o senza quei file segnala un errore.

### Benchmark comparativo
Il target **`Benchmark_Report`** (sorgente `mainReport.c`) esegue in sequenza tutte le
configurazioni e genera nella root del progetto:
- **`report_completo.txt`** — tabella testuale dei tempi (BUILD / QUERY / TOTAL);
- **`report_completo.html`** — report grafico (celle verdi = tempi migliori).

---

## 6. Formato dei dati e dei risultati

File `.ds2` (binari): header `uint32 n`, `uint32 d`, poi `n·d` valori row-major.

| File | Forma | Tipo valori |
|---|---|---|
| `dataset_2000x256_32.ds2` | 2000 × 256 | `float32` |
| `dataset_2000x256_64.ds2` | 2000 × 256 | `float64` |
| `query_2000x256_{32,64}.ds2` | 2000 × 256 | `float32` / `float64` |
| `results_ids_2000x8_k8_x64_{32,64}.ds2` | 2000 × 8 | `int32` (id dei vicini) |
| `results_dst_2000x8_k8_x64_32.ds2` | 2000 × 8 | `float32` (distanze) |
| `results_dst_2000x8_k8_x64_64.ds2` | 2000 × 8 | `float64` (distanze) |

Lettura in Python:
```python
import numpy as np
def load_ds2(path, dtype):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
        return np.fromfile(f, dtype=dtype, count=int(n)*int(d)).reshape(int(n), int(d))
```

**Interpretare l'output:** per ogni query, `ids[q]` sono gli indici (riga nel dataset) degli
`k` vicini e `dists[q]` le loro distanze euclidee reali. Confrontando `ids`/`dists` con
`results_ids_*`/`results_dst_*` si ottiene la percentuale di query corrette (deve essere
100% con `h=16, k=8, x=64`).

---

## 7. Risoluzione problemi

| Sintomo | Causa | Soluzione |
|---|---|---|
| `Impossibile eseguire il programma 'python.exe'` (PowerShell) | `python` punta allo stub alias del Microsoft Store | usare il percorso completo dell'interprete vero (`$py`, vedi §2), oppure disattivare gli alias in *Impostazioni → App → Alias di esecuzione app* |
| `Invalid wheel filename (wrong number of parts): '*'` o `looks like a filename, but the file does not exist` | PowerShell non espande `*` per gli exe esterni | passare il nome completo del file, o espandere con `$whl = (Get-ChildItem dist\repaired\*.whl).FullName` e poi `& $py -m pip install $whl` |
| `ModuleNotFoundError: ..._quantpivot32` all'import | stai lanciando Python da dentro `python\`: la cartella sorgente maschera il pacchetto installato | spostarsi in un'altra cartella (es. la root del progetto) prima di importare/eseguire |
| `ImportError: DLL load failed` | mancano le DLL di GCC | usare il wheel **repaired** (autosufficiente), oppure `import os; os.add_dll_directory(r"C:\...\mingw64\bin")` prima dell'import |
| `ValueError: Input array not aligned` | array NumPy non allineato | usare `aligned()` (vedi `examples/esempio_knn.py`) o `np.ascontiguousarray` su buffer allineato |
| `TypeError: Data must be float32/float64` | dtype sbagliato | `float32` per `quantpivot32`, `float64` per `quantpivot64`/`omp` |
| build: `unknown file type '.S'` | compilatore non MinGW | assicurarsi che `setup.py` usi `mingw32` e che `gcc` sia nel PATH |
| build: `cl` / errori MSVC | è stato scelto MSVC | MSVC non assembla i `.S`: usare MinGW (vedi §2) |
| `RuntimeError: Model not fitted` | `predict` prima di `fit` | chiamare `fit(...)` prima di `predict(...)` |

Per i dettagli su **come è costruito** il progetto, vedere
[`docs/ARCHITETTURA.md`](ARCHITETTURA.md).
