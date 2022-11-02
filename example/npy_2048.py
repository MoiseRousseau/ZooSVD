import numpy as np
from scipy.linalg import svd as scipy_svd
import time
import sys
sys.path.append('../')
import PyZooSVD

if __name__ == "__main__":
    
    #load a numpy array name arr.npy of size 2048x2048
    f_mat = "arr.npy"
    A = np.load(f_mat)
    size = 2048
    #A = A[:size,:size].astype('float64')
    print("Matrix shape:", A.shape, "/", "Data type:", A.dtype)
    
    test_func = ["NumPy", "CUDA_gesvdj"]
    
    if "Scipy" in test_func:
        t_start = time.time()
        U,s,V = scipy_svd(A, full_matrices=False, check_finite=False, lapack_driver="gesdd")
        print(f"SciPy SVD: {time.time() - t_start:.6f} second")
    
    if "NumPy" in test_func:
        t_start = time.time()
        U,s,V = np.linalg.svd(A)
        print(f"NumPy SVD: {time.time() - t_start:.6f} second")
    
    if "LAPACK_gesdd" in test_func:
        t_start = time.time()
        U,s,V = PyZooSVD.Lapack_SVD(A.copy(), driver="gesdd")
        print(f"LAPACK_gesdd: {time.time() - t_start:.6f} second")
    
    if "LAPACK_gesvd" in test_func:
        t_start = time.time()
        U,s,V = PyZooSVD.Lapack_SVD(A.copy(), driver="gesvd")
        print(f"LAPACK_gesvd: {time.time() - t_start:.6f} second")
    
    if "Eigen_BDC" in test_func:
        t_start = time.time()
        U,s,V = PyZooSVD.Eigen_SVD(A.copy(), driver="bidiagdc")
        print(f"Eigen_BDC: {time.time() - t_start:.6f} second")
        
    if "Eigen_Jacobi" in test_func:
        t_start = time.time()
        U,s,V = PyZooSVD.Eigen_SVD(
            A.copy(), 
            driver="jacobi",
            jacobi_preconditionner="HouseholderQR"
        )
        print(f"Eigen_Jacobi: {time.time() - t_start:.6f} second")
    
    if "CUDA_gesvdj" in test_func:
        t_start = time.time()
        U,s,V = PyZooSVD.CUDA_SVD(A.copy(), tol=2e-8, max_sweeps=100)
        print(f"CUDA_gesvdj: {time.time() - t_start:.6f} second")
    
    
