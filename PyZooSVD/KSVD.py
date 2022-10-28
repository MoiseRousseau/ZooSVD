import numpy as np
import ctypes

lib = ctypes.cdll.LoadLibrary("/".join(__file__.split('/')[:-1])+"/libZooSVD.so")

def KSVD(A):

    U = np.zeros((A.shape[0],min(A.shape)),A.dtype)
    V = np.zeros((min(A.shape),A.shape[1]),A.dtype) #eigen return V and not V^T
    s = np.zeros(min(A.shape),A.dtype)
    executable = "/".join(__file__.split('/')[:-1]) + "/KSVD_main"
    
    lib.wrap_KSVD(
        A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        A.ctypes.shape,
        U.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_char_p(executable.encode()),
    )
    return
