# -*- coding: utf-8 -*-

# Usage: python setup.py build_ext --inplace


from distutils.core import setup
from Cython.Build import cythonize
import numpy as np



setup(
    name = "PriorityQueue",
    include_dirs=[np.get_include()],
    ext_modules = cythonize("PriorityQueue.pyx")
)


setup(
    name = "TreeOfShapes",
    include_dirs=[np.get_include()],
    ext_modules = cythonize("TreeOfShapes.pyx")
)

