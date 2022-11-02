import pytest
import sys
sys.path.append('../')
import PyZooSVD
import numpy as np

class Test_LAPACK:
    """
    This test the result of the SVD using the different wrappers
    """
    
    np.random.seed(2)
    #create some real test matrices with different size with their SVD
    sizes = [(3,3), (5,3), (3,8)]
    mats = []
    for size in sizes:
        #double type
        A = np.random.random(size)-0.5
        mats.append(A)
        #complex 128
        A_ = A + 1j * np.random.random(size)-0.5
        mats.append(A_)
    
    @pytest.mark.parametrize("mat", [x for x in mats])
    def test_LAPACK(self, mat):
        print("LAPACK:", mat.shape, mat.dtype)
        #TODO! mat is modified!!!!
        U,s,V = PyZooSVD.Lapack_SVD(mat.copy(), driver="gesdd")
        norm = np.linalg.norm( mat - U @ np.diag(s) @ V ) 
        print("Error norm:", norm)
        assert norm < 1e-8
        
