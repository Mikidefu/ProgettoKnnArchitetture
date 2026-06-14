"""
Verifica end-to-end del pacchetto Python compilato (percorso Assembly):
fit()/predict() devono riprodurre i file golden data/results_*.ds2.
"""
import os, sys
import numpy as np

MINGW = r"C:\Users\mikid\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin"
if os.path.isdir(MINGW):
    os.add_dll_directory(MINGW)  # per libgomp/libgcc/libwinpthread del modulo OpenMP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from Gruppo_Ferrari_DeFusco_Cuconato.quantpivot32 import QuantPivot as QP32
from Gruppo_Ferrari_DeFusco_Cuconato.quantpivot64 import QuantPivot as QP64
from Gruppo_Ferrari_DeFusco_Cuconato.quantpivot64omp import QuantPivot as QP64OMP


def load(p, dt):
    with open(p, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
        return np.fromfile(f, dtype=dt, count=int(n) * int(d)).reshape(int(n), int(d))


def aligned(a, alignment=64):
    a = np.ascontiguousarray(a)
    buf = np.empty(a.nbytes + alignment, dtype=np.uint8)
    off = (-buf.ctypes.data) % alignment
    out = buf[off:off + a.nbytes].view(a.dtype).reshape(a.shape)
    out[...] = a
    return out


def check(tag, QP, dt, dst_dt):
    prec = "32" if dt == np.float32 else "64"
    DS = aligned(load(os.path.join(ROOT, f"data/dataset_2000x256_{prec}.ds2"), dt))
    Q = aligned(load(os.path.join(ROOT, f"data/query_2000x256_{prec}.ds2"), dt))
    gids = load(os.path.join(ROOT, f"data/results_ids_2000x8_k8_x64_{prec}.ds2"), np.int32).astype(np.int64)
    gdst = load(os.path.join(ROOT, f"data/results_dst_2000x8_k8_x64_{prec}.ds2"), dst_dt).astype(np.float64)

    model = QP()
    model.fit(DS, n_pivots=16, quant_level=64, silent=1)
    ids, dst = model.predict(Q, k=8, silent=1)
    ids = np.asarray(ids).astype(np.int64)
    dst = np.asarray(dst).astype(np.float64)

    id_rows = int(np.all(ids == gids, axis=1).sum())
    atol = 1e-3 if prec == "32" else 1e-9
    full = int((np.all(ids == gids, axis=1) & np.all(np.isclose(dst, gdst, atol=atol, rtol=0), axis=1)).sum())
    print(f"[{tag}] ids identici (ordine): {id_rows}/{len(gids)}   ids+dist: {full}/{len(gids)}   "
          f"{'OK' if full == len(gids) else 'MISMATCH'}")
    return full == len(gids)


ok = True
ok &= check("quantpivot32  (SSE2 asm)", QP32, np.float32, np.float32)
ok &= check("quantpivot64  (AVX2 asm)", QP64, np.float64, np.float64)
ok &= check("quantpivot64omp (AVX2+OMP)", QP64OMP, np.float64, np.float64)
print("\nRISULTATO:", "TUTTO CORRETTO" if ok else "ALMENO UN MISMATCH")
sys.exit(0 if ok else 1)
