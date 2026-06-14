"""
Independent reference re-implementation of the project spec (Progetto Architetture 2025)
to verify the *veracity of the calculations* against the committed golden files
data/results_ids_*.ds2 and data/results_dst_*.ds2.

It reproduces, exactly, the C algorithm in src/index.c + src/query.c:
  - quantize: top-x abs components -> vp/vn (0/1)
  - approx distance d~ = pp + nn - pn - np   (integer)
  - pivot selection: ids = floor(n/h)*j, j=0..h-1
  - querying with the |d~(v,p)-d~(q,p)| lower-bound pruning
  - final neighbour distances recomputed with the *real* Euclidean distance
  - output kept in the C "slot order" (NOT sorted) so it is byte-comparable to golden
"""
import numpy as np
import sys

def load_ds2(path, dtype):
    with open(path, "rb") as f:
        n, d = np.fromfile(f, dtype=np.uint32, count=2)
        data = np.fromfile(f, dtype=dtype, count=int(n) * int(d))
    return data.reshape(int(n), int(d))

def quantize(M, x):
    """Return vp, vn as (N,D) uint8 arrays, replicating quantize_vector."""
    N, D = M.shape
    vp = np.zeros((N, D), np.uint8)
    vn = np.zeros((N, D), np.uint8)
    # indices of the x largest |value| per row. C uses qsort desc on |v|;
    # argpartition picks the same set (ties impossible on continuous floats).
    absM = np.abs(M)
    xx = min(x, D)
    idx = np.argpartition(absM, D - xx, axis=1)[:, D - xx:]
    rows = np.arange(N)[:, None]
    sel = M[rows, idx]
    pos = sel >= 0
    vp[rows, idx] = pos.astype(np.uint8)
    vn[rows, idx] = (~pos).astype(np.uint8)
    return vp, vn

def approx_matrix(VP, VN, WP, WN):
    """d~ between every row of (VP,VN) and every row of (WP,WN). Returns (A,B) int."""
    pp = VP @ WP.T
    nn = VN @ WN.T
    pn = VP @ WN.T
    npp = VN @ WP.T
    return (pp + nn - pn - npp).astype(np.int64)

def select_pivots(n, h):
    step = n // h
    return np.array([step * j for j in range(h)], dtype=np.int64)

def knn_run(DS, Q, h, k, x, prune=True):
    n, D = DS.shape
    nq = Q.shape[0]
    if k > n: k = n

    VP, VN = quantize(DS, x)          # dataset quant
    QP, QN = quantize(Q, x)           # query quant
    piv = select_pivots(n, h)
    PVP, PVN = VP[piv], VN[piv]

    # index: d~(v_i, p_j)  shape (n,h)
    idx_dist = approx_matrix(VP, VN, PVP, PVN)        # (n,h)
    # d~(q, p_j) shape (nq,h)
    dq_piv = approx_matrix(QP, QN, PVP, PVN)          # (nq,h)
    # d~(q, v) full (only used for correctness/oracle and for the per-query loop)
    dq_v = approx_matrix(QP, QN, VP, VN)              # (nq,n)

    INF = np.float64(np.finfo(np.float32).max)        # FLT_MAX like the C
    ids_out = np.full((nq, k), -1, np.int64)
    dapprox_out = np.full((nq, k), INF)

    pruned_count = 0
    total = nq * n
    for qi in range(nq):
        nb_id = [-1] * k
        nb_ap = [INF] * k
        d_star_row = np.max(np.abs(idx_dist - dq_piv[qi]), axis=1)  # (n,)
        dqv_row = dq_v[qi]
        for i in range(n):
            # worst = first slot with max approx dist
            worst = 0
            maxd = nb_ap[0]
            for s in range(1, k):
                if nb_ap[s] > maxd:
                    maxd = nb_ap[s]; worst = s
            worst_ap = nb_ap[worst]
            if prune and float(d_star_row[i]) >= worst_ap:
                pruned_count += 1
                continue
            da = float(dqv_row[i])
            if da < worst_ap:
                nb_id[worst] = i
                nb_ap[worst] = da
        ids_out[qi] = nb_id
        dapprox_out[qi] = nb_ap

    # final real Euclidean distance for the chosen ids, in slot order
    dst_out = np.full((nq, k), INF)
    for qi in range(nq):
        for s in range(k):
            vid = ids_out[qi, s]
            if vid >= 0:
                diff = Q[qi] - DS[vid]
                dst_out[qi, s] = np.sqrt(np.dot(diff, diff))
    return ids_out, dst_out, dapprox_out, pruned_count, total

def compare(tag, ids, dst, gold_ids, gold_dst, atol):
    id_match_rows = np.all(ids == gold_ids, axis=1)
    # distance comparison with tolerance, only where id matched position-wise
    dst_close = np.isclose(dst, gold_dst, atol=atol, rtol=0)
    full_row = id_match_rows & np.all(dst_close, axis=1)
    # set-level (order-independent) id agreement
    set_match = np.array([set(a.tolist()) == set(b.tolist())
                          for a, b in zip(ids, gold_ids)])
    print(f"[{tag}]")
    print(f"  rows with IDENTICAL ordered ids : {id_match_rows.sum()}/{len(ids)}")
    print(f"  rows with SAME id SET           : {set_match.sum()}/{len(ids)}")
    print(f"  rows fully matching (ids+dst)   : {full_row.sum()}/{len(ids)}")
    return id_match_rows.all(), full_row.all()

def main():
    for prec, dtype, dst_dtype, atol in [("32", np.float32, np.float32, 1e-3),
                                         ("64", np.float64, np.float64, 1e-9)]:
        print("=" * 70)
        print(f"PRECISION {prec}-bit")
        print("=" * 70)
        DS = load_ds2(f"data/dataset_2000x256_{prec}.ds2", dtype)
        Q  = load_ds2(f"data/query_2000x256_{prec}.ds2", dtype)
        gold_ids = load_ds2(f"data/results_ids_2000x8_k8_x64_{prec}.ds2", np.int32).astype(np.int64)
        gold_dst = load_ds2(f"data/results_dst_2000x8_k8_x64_{prec}.ds2", dst_dtype).astype(np.float64)
        k = gold_ids.shape[1]
        x = 64
        # find h that reproduces golden (filename omits h). Try common values.
        candidates = [16, 8, 32, 4, 64, 2, 10, 20, 100, 50, 256, 128]
        best = None
        for h in candidates:
            ids, dst, ap, pruned, total = knn_run(DS, Q, h, k, x, prune=True)
            rows_ok = np.all(ids == gold_ids, axis=1).sum()
            print(f"  h={h:4d}: identical-ordered rows = {rows_ok}/{len(ids)}  (pruned {pruned}/{total} = {100*pruned/total:.1f}%)")
            if best is None or rows_ok > best[1]:
                best = (h, rows_ok)
            if rows_ok == len(ids):
                break
        h = best[0]
        print(f"  -> best h = {h} ({best[1]}/{len(DS)} rows)")
        print("-" * 70)
        ids_p, dst_p, ap_p, pruned, total = knn_run(DS, Q, h, k, x, prune=True)
        ids_n, dst_n, ap_n, _, _        = knn_run(DS, Q, h, k, x, prune=False)
        compare("PRUNED vs GOLDEN", ids_p, dst_p, gold_ids, gold_dst, atol)
        compare("NO-PRUNE vs GOLDEN", ids_n, dst_n, gold_ids, gold_dst, atol)
        # soundness: pruned == no-prune?
        same = np.all(ids_p == ids_n)
        print(f"[SOUNDNESS] pruned ids == no-prune ids : {same} "
              f"({np.all(ids_p==ids_n,axis=1).sum()}/{len(ids_p)} rows)")
        print()

if __name__ == "__main__":
    main()
