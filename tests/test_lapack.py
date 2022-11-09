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

    @pytest.mark.real
    @pytest.mark.parametrize("mat", mats_real)
    def test_LAPACK_gesdd_r(self, mat):
        print("LAPACK-gesdd:", mat.shape, mat.dtype)
        self.SVD(mat, "gesdd")
    
    @pytest.mark.complex
    @pytest.mark.parametrize("mat", mats_complex)
    def test_LAPACK_gesdd_c(self, mat):
        print("LAPACK-gesdd:", mat.shape, mat.dtype)
        self.SVD(mat, "gesdd")

    @pytest.mark.real
    @pytest.mark.parametrize("mat", mats_real)
    def test_LAPACK_gesvd_r(self, mat):
        print("LAPACK-gesvd:", mat.shape, mat.dtype)
        self.SVD(mat, "gesvd")

    @pytest.mark.complex
    @pytest.mark.parametrize("mat", mats_complex)
    def test_LAPACK_gesvd_c(self, mat):
        print("LAPACK-gesvd:", mat.shape, mat.dtype)
        self.SVD(mat, "gesvd")

    def SVD(self, mat, driver):
        #TODO: mat is modified!
        U,s,V = PyZooSVD.Lapack_SVD(mat.copy(), driver=driver)
        norm = np.linalg.norm( mat - U @ np.diag(s) @ V )
        print("Error norm:", norm)
        test = norm < self.tol[mat.dtype]
        if not test:
            print(U,s,V)
            print(np.linalg.svd(mat, full_matrices=False))
        assert test
