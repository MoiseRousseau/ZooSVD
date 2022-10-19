import numpy as np
import sys
sys.path.append("../")
import PyZooSVD

if __name__ == "__main__":
    
    dtype='float64'
    size = (5,3)
    A = np.random.random(size).astype(dtype) #+ 1j*np.random.random(size).astype(dtype)
    
    #test_func = ["Scipy", "NumPy", "LAPACK", "Eigen"]
    test_func = ["NumPy", "Eigen"]
    
    print("The A matrix from NumPy:")
    print(A)
    print("The A matrix from C++:")
    PyZooSVD.print_matrix(A)
    
    if "NumPy" in test_func:
        print("SVD of A using NumPy:")
        U,s,VT = np.linalg.svd(A)
        print(U,s,VT)
        #print(np.linalg.norm( A - U @ np.diag(s) @ VT ))
    
    if "LAPACK" in test_func:
        print("SVD of A using LAPACK:")
        U,s,VT = PyZooSVD.Lapack_SVD(A.copy(), driver="gesvd") #A is modified!
        print(U,s,VT)
        print(U @ np.diag(s) @ VT)
    
    if "Eigen" in test_func:
        print("SVD of A using Eigen:")
        U,s,VT = PyZooSVD.Eigen_SVD(
            A.copy(),
            driver="jacobi",
            jacobi_preconditionner="FullPivHouseholderQR"
        )
        print(U,s,VT)
        print(U @ np.diag(s) @ VT)
    
