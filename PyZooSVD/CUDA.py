import numpy as np
import ctypes

lib = ctypes.cdll.LoadLibrary("/".join(__file__.split('/')[:-1])+"/libZooSVD.so")

def CUDA_SVD(A, tol=0.2e-7, max_sweeps=100):
    UT = np.zeros((min(A.shape),A.shape[0]),A.dtype)
    V = np.zeros((min(A.shape),A.shape[1]),A.dtype)
    if A.dtype == np.dtype('float64'): 
        s = np.zeros(min(A.shape),A.dtype)
        lib.wrap_CUDA_dgesvdj(
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            A.ctypes.shape,
            UT.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            V.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_double(tol),
            ctypes.c_int(max_sweeps),
        )
    return UT.transpose(),s,V
