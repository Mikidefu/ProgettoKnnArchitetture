import subprocess
import os
import time

def run_knn_c(ds_path, q_path, h, k, x, variant="sse"):
    # Costruisce il percorso relativo corretto partendo dalla root del progetto
    # INSERIRE IL PATH DELLE VARIE VERSIONI DOPO AVER BUILDATO, IO NON POSSO
    exe_map = {
        "sse": "bin/Release_SSE2/progetto_knn.exe",
        "avx": "bin/Release_AVX/progetto_knn.exe",
        "scalar": "bin/Release_Scalar/progetto_knn.exe"
    }

    exe_path = exe_map.get(variant, exe_map["sse"])

    if not os.path.exists(exe_path):
        print(f"Errore: Eseguibile non trovato in {exe_path}")
        return

    cmd = [exe_path, "-d", ds_path, "-q", q_path, "-h", str(h), "-k", str(k), "-x", str(x)]

    print(f"--- Test Variante: {variant.upper()} ---")
    start_py = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)

    end_py = time.time()

    if result.returncode == 0:
        print(result.stdout)
        print(f"Tempo misurato da Python: {int((end_py - start_py) * 1000)} ms")
    else:
        print("Errore C:", result.stderr)

# Esempio: esegue prima SSE poi AVX per confrontarli
run_knn_c("data/dataset_2000x256_32.ds2", "data/query_2000x256_32.ds2", 16, 8, 64, "sse")
run_knn_c("data/dataset_2000x256_64.ds2", "data/query_2000x256_64.ds2", 16, 8, 64, "avx")