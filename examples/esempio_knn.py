"""
Esempio d'uso della libreria Gruppo_Ferrari_DeFusco_Cuconato.

Carica i dataset/query di `data/`, esegue le tre varianti (SSE2, AVX2, AVX2+OpenMP),
stampa i vicini di una query campione e i tempi, e confronta con i file golden.

Uso:
    python examples/esempio_knn.py
"""
import os
import sys
import time
import glob
import numpy as np

# --- (best effort) rende reperibili le DLL di MinGW se è installato un wheel NON repaired ---
for base in glob.glob(os.path.join(os.environ.get("LOCALAPPDATA", ""),
                                   r"Microsoft\WinGet\Packages\BrechtSanders.WinLibs*\mingw64\bin")):
    if os.path.isdir(base):
        try:
            os.add_dll_directory(base)
        except OSError:
            pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

from Gruppo_Ferrari_DeFusco_Cuconato.quantpivot32 import QuantPivot as QP32
from Gruppo_Ferrari_DeFusco_Cuconato.quantpivot64 import QuantPivot as QP64
from Gruppo_Ferrari_DeFusco_Cuconato.quantpivot64omp import QuantPivot as QP64OMP

H, K, X = 16, 8, 64  # pivot, vicini, quantizzazione


def load_ds2(path, dtype):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
        return np.fromfile(f, dtype=dtype, count=int(n) * int(d)).reshape(int(n), int(d))


def aligned(a, alignment=64):
    """Copia di `a` con puntatore base allineato a `alignment` byte (richiesto dai wrapper)."""
    a = np.ascontiguousarray(a)
    buf = np.empty(a.nbytes + alignment, dtype=np.uint8)
    off = (-buf.ctypes.data) % alignment
    out = buf[off:off + a.nbytes].view(a.dtype).reshape(a.shape)
    out[...] = a
    return out


def run(tag, QP, dtype, prec):
    print(f"\n=== {tag} ===")
    DS = aligned(load_ds2(os.path.join(DATA, f"dataset_2000x256_{prec}.ds2"), dtype))
    Q = aligned(load_ds2(os.path.join(DATA, f"query_2000x256_{prec}.ds2"), dtype))
    gids = load_ds2(os.path.join(DATA, f"results_ids_2000x8_k8_x64_{prec}.ds2"), np.int32)
    gdst = load_ds2(os.path.join(DATA, f"results_dst_2000x8_k8_x64_{prec}.ds2"), dtype)

    t0 = time.perf_counter()
    model = QP().fit(DS, n_pivots=H, quant_level=X, silent=1)
    t1 = time.perf_counter()
    ids, dists = model.predict(Q, k=K, silent=1)
    t2 = time.perf_counter()
    ids = np.asarray(ids)
    dists = np.asarray(dists)

    # vicini della prima query
    print(f"Query #0 - primi {K} vicini (id : distanza euclidea):")
    for j in range(K):
        print(f"   id={int(ids[0, j]):5d}   dist={float(dists[0, j]):.4f}")

    print(f"Tempo fit: {1000*(t1-t0):7.1f} ms | Tempo predict: {1000*(t2-t1):7.1f} ms")

    # confronto con i golden
    atol = 1e-3 if prec == "32" else 1e-9
    ok_rows = int((np.all(ids == gids, axis=1) &
                   np.all(np.isclose(dists, gdst, atol=atol, rtol=0), axis=1)).sum())
    flag = "OK" if ok_rows == len(gids) else "MISMATCH"
    print(f"Confronto con i golden: {ok_rows}/{len(gids)} query corrette  [{flag}]")
    return ok_rows == len(gids)


def main():
    print("Parametri: h =", H, " k =", K, " x =", X, " | dataset 2000 x 256")
    ok = True
    ok &= run("quantpivot32   (SSE2 assembly, float32)", QP32, np.float32, "32")
    ok &= run("quantpivot64   (AVX2 assembly, float64)", QP64, np.float64, "64")
    ok &= run("quantpivot64omp(AVX2 + OpenMP, float64)", QP64OMP, np.float64, "64")
    print("\n" + ("TUTTE LE VARIANTI CORRETTE [OK]" if ok else "ATTENZIONE: rilevati mismatch"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
