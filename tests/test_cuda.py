import pytest
import sys
sys.path.append('../')
import PyZooSVD
import numpy as np

class Test_CUDA:
    """
    This test the result of the SVD using CUDA function
    """
    
    np.random.seed(2)
    #create some real test matrices with different size
    sizes = [(3,3), (5,3), (3,8)]
    mats = []
    for size in sizes:
        #double type
        A = np.random.random(size)-0.5
        mats.append(A)
        #complex 128
        #A_ = A + 1j * np.random.random(size)-0.5
        #mats.append(A_)
    
    @pytest.mark.gpu
    @pytest.mark.parametrize("mat", [x for x in mats])
    def test_CUDA_jacobi(self, mat):
        print("CUDA-Jacobi:", mat.shape, mat.dtype)
        U,s,V = PyZooSVD.CUDA_SVD(mat, driver="Jacobi")
        norm = np.linalg.norm( mat - U @ np.diag(s) @ V ) 
        print("Error norm:", norm)
        test = norm < 1e-8
        if not test:
            print(U,s,V)
            print(np.linalg.svd(mat, full_matrices=False))
        assert test
    
    @pytest.mark.gpu
    @pytest.mark.parametrize("mat", [x for x in mats])
    def test_CUDA_PD(self, mat):
        print("CUDA-Polar-Decomposition:", mat.shape, mat.dtype)
        U,s,V = PyZooSVD.CUDA_SVD(mat, driver="Polar-Decomposition")
        print("Error norm:", norm)
        test = norm < 1e-8
        if not test:
            print(U,s,V)
            print(np.linalg.svd(mat, full_matrices=False))
        assert test
    
    @pytest.mark.gpu
    @pytest.mark.parametrize("mat", [x for x in mats])
    def test_CUDA_QR(self, mat):
        print("CUDA-QR:", mat.shape, mat.dtype)
        U,s,V = PyZooSVD.CUDA_SVD(mat, driver="QR")
        print("Error norm:", norm)
        test = norm < 1e-8
        if not test:
            print(U,s,V)
            print(np.linalg.svd(mat, full_matrices=False))
        assert test
        
