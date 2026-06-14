import os
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import numpy as np


class build_ext_asm(build_ext):
    """build_ext che (1) accetta i sorgenti Assembly .S/.s e (2) usa MinGW
    (gcc) su Windows, indispensabile per i file GAS e i flag GCC/OpenMP."""

    def finalize_options(self):
        super().finalize_options()
        if os.name == "nt" and not self.compiler:
            self.compiler = "mingw32"

    def build_extensions(self):
        for ext in (".S", ".s"):
            if ext not in self.compiler.src_extensions:
                self.compiler.src_extensions.append(ext)
        super().build_extensions()

# I sorgenti C/Assembly stanno nella root del repo. Si usano percorsi RELATIVI
# (rispetto a python/, dove gira setup.py): i path assoluti sono rifiutati da bdist_wheel.
SRC = os.path.join(os.pardir, "src")
INC = os.path.join(os.pardir, "include")


def srcs(*names):
    return [os.path.join(SRC, n) for n in names]


PKG = "Gruppo_Ferrari_DeFusco_Cuconato"
INCLUDE_DIRS = [np.get_include(), INC]

# Estensione 32-bit SSE2 (Single Precision) -> Assembly approximate_distance_sse2_asm
module32 = Extension(
    f"{PKG}.quantpivot32._quantpivot32",
    sources=srcs(
        "quantpivot32_py.c",
        "index.c",
        "query.c",
        "distance32ASSEMBLY.c",
        "distance_sse2.S",
        "quantization.c",
        "matrix.c",
    ),
    include_dirs=INCLUDE_DIRS,
    define_macros=[("USE_SSE2_ASM", None)],
    extra_compile_args=["-O3", "-msse2"],
)

# Estensione 64-bit AVX2 (Double Precision), seriale -> approximate_distance_avx2_asm
module64 = Extension(
    f"{PKG}.quantpivot64._quantpivot64",
    sources=srcs(
        "quantpivot64_py.c",
        "index.c",
        "query64.c",
        "distance64ASSEMBLY.c",
        "distance_avx2.S",
        "quantization.c",
        "matrix.c",
    ),
    include_dirs=INCLUDE_DIRS,
    define_macros=[("USE_AVX_ASM", None), ("QP_DOUBLE", None)],
    extra_compile_args=["-O3", "-mavx2"],
)

# Estensione 64-bit AVX2 + OpenMP
module64omp = Extension(
    f"{PKG}.quantpivot64omp._quantpivot64omp",
    sources=srcs(
        "quantpivot64omp_py.c",
        "index.c",
        "query64.c",
        "distance64ASSEMBLY.c",
        "distance_avx2.S",
        "quantization.c",
        "matrix.c",
    ),
    include_dirs=INCLUDE_DIRS,
    define_macros=[("USE_AVX_ASM", None), ("QP_DOUBLE", None)],
    extra_compile_args=["-O3", "-mavx2", "-fopenmp"],
    extra_link_args=["-fopenmp"],
)

setup(
    packages=find_packages(),
    ext_modules=[module32, module64, module64omp],
    cmdclass={"build_ext": build_ext_asm},
)
