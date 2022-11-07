import numpy as np
import ctypes

lib = ctypes.cdll.LoadLibrary("/".join(__file__.split('/')[:-1])+"/libZooSVD.so")

def Eigen_SVD(A, driver="bidiagdc", jacobi_preconditionner="ColPivHouseholderQR"):
    """
    A wrapper around Eigen implementation of SVD Jacobi (jacobi) and Bidiagonalisation Divide and Conquer (bidiagdd) algorithm.
    See: https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
    
    About Jacobi Preconditionner:
    * ColPivHouseholderQR: default in Eigen, safe
    * FullPivHouseholderQR: very safe, slowest, no Thin unitary, does not work
    * HouseholderQR: fatest, less safe.
    """
    U = np.zeros((A.shape[0],min(A.shape)),A.dtype)
    V = np.zeros((A.shape[1],min(A.shape)),A.dtype) #eigen return V and not V^T
    s = np.zeros(min(A.shape),A.dtype)
    if driver == "jacobi":
        if jacobi_preconditionner == "None": 
            if A.shape[0] != A.shape[1]:
                raise ValueError("No preconditionner only available for square matrix")
                prec = 0
        elif jacobi_preconditionner == "HouseholderQR": prec = 1
        elif jacobi_preconditionner == "ColPivHouseholderQR": prec = 2
        elif jacobi_preconditionner == "FullPivHouseholderQR": 
            if A.shape[0] != A.shape[1]:
                raise ValueError("FullPivHouseholderQR preconditionner only available for square matrix")
            prec = 3
        else:
            raise ValueError(f"Unknown preconditionner \"{jacobi_preconditionner}\". Available preconditionner are \"None\", \"HouseholderQR\", \"ColPivHouseholderQR\" and \"FullPivHouseholderQR\"")
        if A.dtype == np.dtype('float32'):
            lib.wrap_sEigenJacobi(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                A.ctypes.shape,
                U.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(prec),
            )
        elif A.dtype == np.dtype('float64'):
            lib.wrap_dEigenJacobi(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                A.ctypes.shape,
                U.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                V.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_int(prec),
            )
        else:
            raise NotImplementedError(f"Not implemented for {A.dtype}")
    elif driver == "bidiagdc":
        if A.dtype == np.dtype('float32'):
            lib.wrap_sEigenBDC(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                A.ctypes.shape,
                U.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                s.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        elif A.dtype == np.dtype('float64'):
            lib.wrap_dEigenBDC(
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                A.ctypes.shape,
                U.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                s.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                V.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
    else:
        raise ValueError("Unknown driver. Available driver are \"jacobi\" and \"bidiagdc\"")
    return U,s,V.transpose()
