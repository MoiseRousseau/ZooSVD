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
    mats_real = []
    mats_complex = []
    for size in sizes:
        #double type
        A = np.random.random(size)-0.5
        mats_real.append(A)
        mats_real.append(A.astype("float32"))
        #complex 128
        A_ = A + 1j * np.random.random(size)-0.5
        mats_complex.append(A_)
        mats_complex.append(A_.astype("complex64"))
    tol = {
        np.dtype("float32") : 2e-6,
        np.dtype("float64") : 1e-10,
        np.dtype("complex64") : 2e-6,
        np.dtype("complex128") : 1e-10,
    }
    
    @pytest.mark.gpu
    @pytest.mark.real
    @pytest.mark.parametrize("mat", mats_real)
    def test_CUDA_jacobi_r(self, mat):
        print("CUDA-Jacobi:", mat.shape, mat.dtype)
        self.SVD(mat, "Jacobi")

    @pytest.mark.gpu
    @pytest.mark.complex
    @pytest.mark.parametrize("mat", mats_complex)
    def test_CUDA_jacobi_c(self, mat):
        print("CUDA-Jacobi:", mat.shape, mat.dtype)
        self.SVD(mat, "Jacobi")

    @pytest.mark.gpu
    @pytest.mark.real
    @pytest.mark.parametrize("mat", mats_real)
    def test_CUDA_PD_r(self, mat):
        print("CUDA-Polar-Decomposition:", mat.shape, mat.dtype)
        self.SVD(mat, "Polar-Decomposition")

    @pytest.mark.gpu
    @pytest.mark.complex
    @pytest.mark.parametrize("mat", mats_complex)
    def test_CUDA_PD_c(self, mat):
        print("CUDA-Polar-Decomposition:", mat.shape, mat.dtype)
        self.SVD(mat, "Polar-Decomposition")

    @pytest.mark.gpu
    @pytest.mark.real
    @pytest.mark.parametrize("mat", mats_real)
    def test_CUDA_QR_r(self, mat):
        print("CUDA-QR:", mat.shape, mat.dtype)
        self.SVD(mat, "QR")

    @pytest.mark.gpu
    @pytest.mark.complex
    @pytest.mark.parametrize("mat", mats_complex)
    def test_CUDA_QR_c(self, mat):
        print("CUDA-QR:", mat.shape, mat.dtype)
        self.SVD(mat, "QR")

    def SVD(self, mat, method):
        U,s,V = PyZooSVD.CUDA_SVD(mat, driver=method)
        norm = np.linalg.norm( mat - U @ np.diag(s) @ V )
        print("Error norm:", norm)
        test = norm < self.tol[mat.dtype]
        if not test and not (mat.dtype == np.dtype("complex64") or mat.dtype == np.dtype("complex128")):
            #different results even if the SVD is correct in complex mode
            print(U,s,V)
            print(np.linalg.svd(mat, full_matrices=False))
        assert test
