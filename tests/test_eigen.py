import pytest
import sys
sys.path.append('../')
import PyZooSVD
import numpy as np

class Test_EIGEN:
    """
    This test the result of the SVD using the different wrappers
    """
    
    np.random.seed(1)
    #create some real test matrices with different size with their SVD
    sizes = [(3,3), (5,3), (3,8)]
    mats = []
    for size in sizes:
        #double type
        A = np.random.random(size)-0.5
        mats.append(A)
        mats.append(A.astype("float32"))
    tol = {
        np.dtype("float32") : 2e-6,
        np.dtype("float64") : 1e-10,
    }
    
    @pytest.mark.real
    @pytest.mark.parametrize("mat", mats)
    def test_EIGEN_bidiagdc(self, mat):
        print("Eigen-BiDiagDC:", mat.size, mat.dtype)
        self.SVD(mat, "bidiagdc")
    
    @pytest.mark.real
    @pytest.mark.parametrize("mat", [x for x in mats])
    def test_EIGEN_jacobi(self, mat):
        print("Eigen-jacobi:", mat.size, mat.dtype)
        self.SVD(mat, "jacobi")

    def SVD(self, mat, driver):
        U,s,V = PyZooSVD.Eigen_SVD(mat, driver=driver)
        norm = np.linalg.norm( mat - U @ np.diag(s) @ V ) 
        print("Error norm:", norm)
        test = norm < self.tol[mat.dtype]
        if not test:
            print(U,s,V)
            print(np.linalg.svd(mat, full_matrices=False))
        assert test
        
