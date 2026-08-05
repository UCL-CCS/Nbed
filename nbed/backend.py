def get_array_backend(use_gpu=False):
    if use_gpu:
        try:
            import cupy as cp
            return cp
        except ImportError:
            raise RuntimeError("CuPy is not installed.")
    else:
        import numpy as np
        return np

def set_array_backend(use_gpu):
    global xp
    xp = get_array_backend(use_gpu)


def get_pyscf_backend(use_gpu=False):
    if use_gpu:
        try:
            import gpu4pyscf as pyscf_backend
            return pyscf_backend
        except ImportError:
            raise RuntimeError("CuPy is not installed.")
    else:
        import pyscf as pyscf_backend
        return pyscf


def set_array_backend(use_gpu):
    global xp
    xp = get_array_backend(use_gpu)

def set_pyscf_backend(use_gpu):
    global pyscf
    pyscf = get_pyscf_backend(use_gpu)

xp = get_array_backend(use_gpu=False)    # Default to NumPy
pyscf = get_pyscf_backend(use_gpu=False) # Default to CPU PySCF