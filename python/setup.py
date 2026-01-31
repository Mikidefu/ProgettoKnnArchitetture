from setuptools import setup, Extension, find_packages
import numpy as np

gruppo = 'gruppoX'

# Estensione 32-bit SSE
module32 = Extension(
    f"{gruppo}.quantpivot32._quantpivot32",
    sources=[
        'src/quantpivot32_py.c',
        'src/distance32ASSEMBLY.c',
        'src/distance_sse2.S',
        'src/quantization.c',
        'src/matrix.c'
    ],
    include_dirs=[np.get_include(), 'include'],
    extra_compile_args=['-O3', '-msse2']
)

# Estensione 64-bit AVX con OpenMP
module64omp = Extension(
    f"{gruppo}.quantpivot64omp._quantpivot64omp",
    sources=[
        'src/quantpivot64omp_py.c',
        'src/distance64ASSEMBLY.c',
        'src/distance_avx2.S',
        'src/quantization.c',
        'src/matrix.c'
    ],
    include_dirs=[np.get_include(), 'include'],
    extra_compile_args=['-O3', '-mavx2', '-fopenmp'],
    extra_link_args=['-fopenmp']
)

setup(
    name=gruppo,
    version='1.0',
    packages=find_packages(),
    ext_modules=[module32, module64omp],
    install_requires=['numpy'],
)