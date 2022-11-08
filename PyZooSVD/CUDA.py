import numpy as np
import ctypes

lib = ctypes.cdll.LoadLibrary("/".join(__file__.split('/')[:-1])+"/libZooSVD.so")

def CUDA_SVD(A, driver="Jacobi", tol=0.2e-7, max_sweeps=100, PD_pertubation=2e-8):
    """
    Perform the SVD using GPU with the CUDA library.
    
    When driver="Jacobi", the tol argument can be specified to return a truncated SVD up to the given tolerance.
    """
    if A.dtype == np.dtype('float32'):
        s = np.zeros(min(A.shape),A.dtype)
        data_type = 0
    elif A.dtype == np.dtype('float64'):
        s = np.zeros(min(A.shape),A.dtype)
        data_type = 1
    elif A.dtype == np.dtype('complex64'):
        s = np.zeros(min(A.shape),dtype="float32")
        data_type = 4
    elif A.dtype == np.dtype('complex128'):
        s = np.zeros(min(A.shape),dtype="float64")
        data_type = 5

    if driver == "Jacobi":
        UT = np.zeros((min(A.shape),A.shape[0]),A.dtype)
        V = np.zeros((min(A.shape),A.shape[1]),A.dtype)
        lib.wrap_CUDA_Xgesvdj(
            A.ctypes.data_as(ctypes.c_void_p),
            A.ctypes.shape,
            UT.ctypes.data_as(ctypes.c_void_p),
            s.ctypes.data_as(ctypes.c_void_p),
            V.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_double(tol),
            ctypes.c_int(max_sweeps),
            ctypes.c_int(data_type),
        )
        return UT.transpose(),s,V

    elif driver == "Polar-Decomposition":
        UT = np.zeros((min(A.shape),A.shape[0]),A.dtype)
        V = np.zeros((min(A.shape),A.shape[1]),A.dtype)
        lib.wrap_CUDA_Xgesvdp(
            A.ctypes.data_as(ctypes.c_void_p),
            A.ctypes.shape,
            UT.ctypes.data_as(ctypes.c_void_p),
            s.ctypes.data_as(ctypes.c_void_p),
            V.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(data_type),
            ctypes.c_double(PD_pertubation),
        )
        return UT.transpose(),s,V

    elif driver == "QR":
        if A.shape[0] > A.shape[1]: A_ = A.transpose()
        else: A_ = A
        U = np.zeros((A_.shape[0],min(A_.shape)),A_.dtype)
        V = np.zeros((min(A_.shape),A_.shape[1]),A_.dtype)
        lib.wrap_CUDA_Xgesvd(
            A.ctypes.data_as(ctypes.c_void_p),
            A_.ctypes.shape,
            U.ctypes.data_as(ctypes.c_void_p),
            s.ctypes.data_as(ctypes.c_void_p),
            V.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(data_type),
        )
        if A.shape[0] > A.shape[1]: 
            return V.transpose(),s,U.transpose()
        else:
            return U,s,V

    else:
        print(f"Unknown driver \"{driver}\". Available drivers are \"Jacobi\", \"QR\" and \"Polar-Decomposition\"")
        return 
