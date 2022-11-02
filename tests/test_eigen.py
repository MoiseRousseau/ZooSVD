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
        #complex 128
        #A_ = A + 1j * np.random.random(size)-0.5
        #U,s,V = np.linalg.svd(A_)
        #mats.append([A_, U,s,V])
    
    @pytest.mark.parametrize("mat", [x for x in mats])
    def test_EIGEN_bidiagdc(self, mat):
        print("Eigen:", mat.size, mat.dtype)
        U,s,V = PyZooSVD.Eigen_SVD(mat, driver="bidiagdc")
        norm = np.linalg.norm( mat - U @ np.diag(s) @ V ) 
        print("Error norm:", norm)
        assert norm < 1e-8
    
    @pytest.mark.parametrize("mat", [x for x in mats])
    def test_EIGEN_jacobi(self, mat):
        print("Eigen:", mat.size, mat.dtype)
        U,s,V = PyZooSVD.Eigen_SVD(mat, driver="jacobi")
        norm = np.linalg.norm( mat - U @ np.diag(s) @ V ) 
        print("Error norm:", norm)
        assert norm < 1e-8
        
