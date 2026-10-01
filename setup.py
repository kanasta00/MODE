from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        name="soft_dtw_fast",
        sources=["soft_dtw_fast.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="soft_dtw_fast",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
        },
    ),
)