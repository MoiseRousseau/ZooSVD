import numpy as np
import ctypes

lib = ctypes.cdll.LoadLibrary("/".join(__file__.split('/')[:-1])+"/libZooSVD.so")

def Lapack_SVD(A, driver="gesvd"):
    # TODO: A is changed!
    # initialize output
    U = np.zeros((A.shape[0],min(A.shape)),A.dtype)
    VT = np.zeros((min(A.shape),A.shape[1]),A.dtype)
    #set the right function
    if driver == "gesvd":
        if A.dtype == np.dtype('float32'): 
            s = np.zeros(min(A.shape),A.dtype)
            lib.wrap_sgesvd(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                A.ctypes.shape,
                U.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                VT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        elif A.dtype == np.dtype('float64'):
            s = np.zeros(min(A.shape),A.dtype)
            lib.wrap_dgesvd(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                A.ctypes.shape,
                U.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                VT.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        elif A.dtype == np.dtype('complex64'): 
            s = np.zeros(min(A.shape),dtype='float32')
            lib.wrap_cgesvd(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                A.ctypes.shape,
                U.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                VT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        elif A.dtype == np.dtype('complex128'): 
            s = np.zeros(min(A.shape),dtype='float64')
            lib.wrap_zgesvd(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                A.ctypes.shape,
                U.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                VT.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
    elif driver == "gesdd":
        if A.dtype == np.dtype('float32'): 
            s = np.zeros(min(A.shape),A.dtype)
            lib.wrap_sgesdd(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                A.ctypes.shape,
                U.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                VT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        elif A.dtype == np.dtype('float64'):
            s = np.zeros(min(A.shape),A.dtype)
            lib.wrap_dgesdd(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                A.ctypes.shape,
                U.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                VT.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        elif A.dtype == np.dtype('complex64'): 
            s = np.zeros(min(A.shape),dtype='float32')
            lib.wrap_cgesdd(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                A.ctypes.shape,
                U.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                VT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        elif A.dtype == np.dtype('complex128'): 
            s = np.zeros(min(A.shape),dtype='float64')
            lib.wrap_zgesdd(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                A.ctypes.shape,
                U.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                VT.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
    return U,s,VT
