import numpy as np
import ctypes

lib = ctypes.cdll.LoadLibrary("/".join(__file__.split('/')[:-1])+"/libZooSVD.so")

def print_matrix(A):
    """
    Print the A matrix by calling a Cpp function.
    Mostly for debug purpose.
    """
    if A.dtype == np.dtype('float32'): 
        c_dtype=ctypes.c_float
        is_complex = False
        func = lib.print_matrixf
    elif A.dtype == np.dtype('float64'): 
        c_dtype=ctypes.c_double
        is_complex = False
        func = lib.print_matrixd
    if A.dtype == np.dtype('complex64'): 
        c_dtype=ctypes.c_float
        is_complex = True
        func = lib.print_matrixf
    elif A.dtype == np.dtype('complex128'): 
        c_dtype=ctypes.c_double
        is_complex = True
        func = lib.print_matrixd
    func(
        A.ctypes.data_as(ctypes.POINTER(c_dtype)),
        A.ctypes.shape,
        is_complex
    )
    return
