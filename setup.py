import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import numpy as np

# setup.py vive nella ROOT del progetto: da qui sono visibili sia i sorgenti C
# (src/, include/) sia il pacchetto Python (python/). Questo permette
#   pip install .
# dalla cartella del progetto, senza percorsi "../" e senza che la cartella
# sorgente del pacchetto possa "mascherare" i moduli compilati (shadowing).

PKG = "Gruppo_Ferrari_DeFusco_Cuconato"
INCLUDE_DIRS = [np.get_include(), "include"]

# Sorgenti C condivisi (il calcolo passa per gli INTRINSECI SIMD in distance.c,
# portabili su Linux/gcc, Windows/MSVC e macOS/clang).
CORE = ("index.c", "quantization.c", "matrix.c", "distance.c")


def s(*names):
    return [os.path.join("src", n) for n in names]


def make_ext(modname, wrapper, query_src, macros, simd, omp):
    ext = Extension(
        f"{PKG}.{modname}._{modname}",
        sources=s(wrapper, query_src, *CORE),
        include_dirs=INCLUDE_DIRS,
        define_macros=macros,
    )
    # attributi letti da build_ext_portable per scegliere i flag giusti
    ext._simd = simd
    ext._omp = omp
    return ext


# 32-bit SSE2 (float) | 64-bit AVX2 (double) | 64-bit AVX2 + OpenMP
module32 = make_ext("quantpivot32", "quantpivot32_py.c", "query.c",
                    [("USE_SSE2", None)], "sse2", False)
module64 = make_ext("quantpivot64", "quantpivot64_py.c", "query64.c",
                    [("USE_AVX", None), ("QP_DOUBLE", None)], "avx2", False)
module64omp = make_ext("quantpivot64omp", "quantpivot64omp_py.c", "query64.c",
                       [("USE_AVX", None), ("QP_DOUBLE", None)], "avx2", True)


class build_ext_portable(build_ext):
    """Sceglie i flag SIMD/OpenMP corretti in base al compilatore rilevato,
    così il pacchetto si compila con il toolchain di sistema su ogni piattaforma
    (gcc/clang su Linux/macOS, MSVC o MinGW su Windows) senza dipendenze esterne."""

    def build_extensions(self):
        msvc = self.compiler.compiler_type == "msvc"
        for ext in self.extensions:
            simd = getattr(ext, "_simd", "sse2")
            omp = getattr(ext, "_omp", False)
            if msvc:
                ca = ["/O2"]
                if simd == "avx2":
                    ca.append("/arch:AVX2")
                if omp:
                    ca.append("/openmp")
                ext.extra_compile_args, ext.extra_link_args = ca, []
            else:  # gcc / clang / mingw32
                ca = ["-O3", "-mavx2" if simd == "avx2" else "-msse2"]
                la = []
                if omp:
                    ca.append("-fopenmp")
                    la.append("-fopenmp")
                ext.extra_compile_args, ext.extra_link_args = ca, la
        super().build_extensions()


setup(
    packages=find_packages(where="python", exclude=["build", "build.*"]),
    package_dir={"": "python"},
    ext_modules=[module32, module64, module64omp],
    cmdclass={"build_ext": build_ext_portable},
)
