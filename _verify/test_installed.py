"""Verifica del wheel INSTALLATO (self-contained): nessun add_dll_directory,
nessun python/ in sys.path -> importa dal site-packages."""
import os, sys
import numpy as np

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"

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


def check(tag, QP, dt, prec):
    DS = aligned(load(os.path.join(DATA, f"dataset_2000x256_{prec}.ds2"), dt))
    Q = aligned(load(os.path.join(DATA, f"query_2000x256_{prec}.ds2"), dt))
    gids = load(os.path.join(DATA, f"results_ids_2000x8_k8_x64_{prec}.ds2"), np.int32).astype(np.int64)
    gdst = load(os.path.join(DATA, f"results_dst_2000x8_k8_x64_{prec}.ds2"), dt).astype(np.float64)
    ids, dst = QP().fit(DS, n_pivots=16, quant_level=64, silent=1).predict(Q, k=8, silent=1)
    ids = np.asarray(ids).astype(np.int64); dst = np.asarray(dst).astype(np.float64)
    atol = 1e-3 if prec == "32" else 1e-9
    full = int((np.all(ids == gids, axis=1) & np.all(np.isclose(dst, gdst, atol=atol, rtol=0), axis=1)).sum())
    print(f"[{tag}] ids+dist corretti: {full}/{len(gids)}  {'OK' if full == len(gids) else 'MISMATCH'}")
    return full == len(gids)


print("import OK da:", QP32.__module__)
ok = check("quantpivot32", QP32, np.float32, "32")
ok &= check("quantpivot64", QP64, np.float64, "64")
ok &= check("quantpivot64omp", QP64OMP, np.float64, "64")
print("\nWHEEL INSTALLATO:", "TUTTO CORRETTO" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
