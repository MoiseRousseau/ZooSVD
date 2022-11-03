import numpy as np
import ctypes

lib = ctypes.cdll.LoadLibrary("/".join(__file__.split('/')[:-1])+"/libZooSVD.so")

def CUDA_SVD(A, driver="Jacobi", tol=0.2e-7, max_sweeps=100, PD_pertubation=2e-8):
    """
    Perform the SVD using GPU with the CUDA library.
    
    When driver="Jacobi", the tol argument can be specified to return a truncated SVD up to the given tolerance.
    """
    driver_int = {"Jacobi":0, "QR":1, "Polar-Decomposition":2}

    if driver == "Jacobi":
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

    elif driver == "Polar-Decomposition":
        UT = np.zeros((min(A.shape),A.shape[0]),A.dtype)
        V = np.zeros((min(A.shape),A.shape[1]),A.dtype)
        if A.dtype == np.dtype('float32'): 
            s = np.zeros(min(A.shape),A.dtype)
            datatype = 0
        elif A.dtype == np.dtype('float64'): 
            s = np.zeros(min(A.shape),A.dtype)
            datatype = 1
        elif A.dtype == np.dtype('complex64'): 
            s = np.zeros(min(A.shape),dtype="float32")
            datatype = 4
        elif A.dtype == np.dtype('complex128'): 
            s = np.zeros(min(A.shape),dtype="float64")
            datatype = 5
        lib.wrap_CUDA_dgesvdp(
            A.ctypes.data_as(ctypes.c_void_p),
            A.ctypes.shape,
            UT.ctypes.data_as(ctypes.c_void_p),
            s.ctypes.data_as(ctypes.c_void_p),
            V.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(datatype),
            ctypes.c_double(PD_pertubation),
        )
        return UT.transpose(),s,V

    elif driver == "QR":
        if A.shape[0] > A.shape[1]: A_ = A.transpose()
        else: A_ = A
        U = np.zeros((A_.shape[0],min(A_.shape)),A_.dtype)
        V = np.zeros((min(A_.shape),A_.shape[1]),A_.dtype)
        if A_.dtype == np.dtype('float32'): 
            s = np.zeros(min(A_.shape),A_.dtype)
            datatype = 0
        elif A_.dtype == np.dtype('float64'): 
            s = np.zeros(min(A_.shape),A_.dtype)
            datatype = 1
        elif A_.dtype == np.dtype('complex64'): 
            s = np.zeros(min(A_.shape),dtype="float32")
            datatype = 4
        elif A_.dtype == np.dtype('complex128'): 
            s = np.zeros(min(A_.shape),dtype="float64")
            datatype = 5
        lib.wrap_CUDA_dgesvd(
            A_.ctypes.data_as(ctypes.c_void_p),
            A_.ctypes.shape,
            UT.ctypes.data_as(ctypes.c_void_p),
            s.ctypes.data_as(ctypes.c_void_p),
            V.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(datatype),
            ctypes.c_double(PD_pertubation),
        )
        if A.shape[0] > A.shape[1]: 
            return V.transpose(),s,UT.transpose()
        else:
            return UT,s,V

    else:
        print(f"Unknown driver \"{driver}\". Available drivers are \"Jacobi\", \"QR\" and \"Polar-Decomposition\"")
        return 
