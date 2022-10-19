#include <vector>
#include <array>
#include <lapacke.h>
#include <iostream>

/*
This file is mostly for test and developpement purpose.
Actually, NumPy is already linked to LAPACKE.
*/

extern "C" {
    //gesvd
    void wrap_sgesvd(float*, long int[2], float*, float*, float*);
    void wrap_dgesvd(double*, long int[2], double*, double*, double*);
    void wrap_cgesvd(float*, long int[2], float*, float*, float*);
    void wrap_zgesvd(double*, long int[2], double*, double*, double*);
    //gesdd
    void wrap_sgesdd(float*, long int[2], float*, float*, float*);
    void wrap_dgesdd(double*, long int[2], double*, double*, double*);
    void wrap_cgesdd(float*, long int[2], float*, float*, float*);
    void wrap_zgesdd(double*, long int[2], double*, double*, double*);
}

/*
LAPACKE SVD for float type
Matrix must be stored in CSC fashion
Return pointer to U, s and V. Size are already know by Python.
*/
void wrap_sgesvd(
    float* mat_val, long int shape[2], float* U, float* s, float* VT
) {
    //A in NumPy is stored in row, but LAPACK ask for column. 
    //Thus we calculate here the SVD of A^T
    char jobu = 'S';
    char jobvt = 'S';
    lapack_int info,m,n,lda,ldu,ldvt;
    
    m = shape[0];
    n = shape[1];
    lda = shape[1];
    ldu = std::min(shape[0],shape[1]);
    ldvt = shape[1];
    
    std::vector<float> work;
    work.resize(50*std::max(shape[0],shape[1])); //TODO: this influence the calculation time...
    
    info = LAPACKE_sgesvd(
        LAPACK_ROW_MAJOR,
        jobu, jobvt,
        m, n,
        mat_val, lda,
        s,
        U, ldu,
        VT, ldvt,
        work.data()
    );
    if (info != 0) std::cout << info << std::endl;
    
    return;
}


/*
LAPACKE SVD for double type
Matrix must be stored in CSC fashion
Return pointer to U, s and V. Size are already know by Python.
*/
void wrap_dgesvd(
    double* mat_val, long int shape[2], double* U, double* s, double* VT
) {
    //A in NumPy is stored in row, but LAPACK ask for column. 
    //Thus we calculate here the SVD of A^T
    char jobu = 'S';
    char jobvt = 'S';
    lapack_int info,m,n,lda,ldu,ldvt;
    
    m = shape[0];
    n = shape[1];
    lda = shape[1];
    ldu = std::min(shape[0],shape[1]);
    ldvt = shape[1];
    
    std::vector<double> work;
    work.resize(50*std::max(shape[0],shape[1]));
    
    info = LAPACKE_dgesvd(
        LAPACK_ROW_MAJOR,
        jobu, jobvt,
        m, n,
        mat_val, lda,
        s,
        U, ldu,
        VT, ldvt,
        work.data()
    );
    
    return;
}

void wrap_cgesvd(
    float* mat_val, long int shape[2], float* U, float* s, float* VT
) {
    char jobu = 'S';
    char jobvt = 'S';
    lapack_int info,m,n,lda,ldu,ldvt;
    m = shape[0];
    n = shape[1];
    lda = shape[1];
    ldu = std::min(shape[0],shape[1]);
    ldvt = shape[1];
    std::vector<float> work;
    work.resize(10*std::max(shape[0],shape[1])*2);
    info = LAPACKE_cgesvd(
        LAPACK_ROW_MAJOR,
        jobu, jobvt,
        m, n,
        (lapack_complex_float*) mat_val, lda,
        s, 
        (lapack_complex_float*) U, ldu, 
        (lapack_complex_float*) VT, ldvt, 
        work.data()
    );
    return;
}

void wrap_zgesvd(
    double* mat_val, long int shape[2], double* U, double* s, double* VT
) {
    char jobu = 'S';
    char jobvt = 'S';
    lapack_int info,m,n,lda,ldu,ldvt;
    m = shape[0];
    n = shape[1];
    lda = shape[1];
    ldu = std::min(shape[0],shape[1]);
    ldvt = shape[1];
    std::vector<double> work;
    work.resize(10*std::max(shape[0],shape[1])*2);
    info = LAPACKE_zgesvd(
        LAPACK_ROW_MAJOR,
        jobu, jobvt,
        m, n,
        (lapack_complex_double*) mat_val, lda,
        s, 
        (lapack_complex_double*) U, ldu, 
        (lapack_complex_double*) VT, ldvt, 
        work.data()
    );
    return;
}


/*
LAPACKE SDD for float type
Matrix must be stored in CSC fashion
Return pointer to U, s and V. Size are already know by Python.
*/
void wrap_sgesdd(
    float* mat_val, long int shape[2], float* U, float* s, float* VT
) {
    //A in NumPy is stored in row, but LAPACK ask for column. 
    //Thus we calculate here the SVD of A^T
    char jobz = 'S';
    lapack_int info,m,n,lda,ldu,ldvt;
    m = shape[0];
    n = shape[1];
    lda = shape[1];
    ldu = std::min(shape[0],shape[1]);
    ldvt = shape[0];
    
    info = LAPACKE_sgesdd(
        LAPACK_ROW_MAJOR,
        jobz,
        m, n,
        mat_val, lda,
        s,
        U, ldu,
        VT, ldvt
    );
    return;
}

void wrap_dgesdd(
    double* mat_val, long int shape[2], double* U, double* s, double* VT
) {
    //A in NumPy is stored in row, but LAPACK ask for column. 
    //Thus we calculate here the SVD of A^T
    char jobz = 'S';
    lapack_int info,m,n,lda,ldu,ldvt;
    m = shape[0];
    n = shape[1];
    lda = shape[1];
    ldu = std::min(shape[0],shape[1]);
    ldvt = shape[0];
    
    info = LAPACKE_dgesdd(
        LAPACK_ROW_MAJOR,
        jobz,
        m, n,
        mat_val, lda,
        s,
        U, ldu,
        VT, ldvt
    );
    return;
}

void wrap_cgesdd(
    float* mat_val, long int shape[2], float* U, float* s, float* VT
) {
    char jobz = 'S';
    lapack_int info,m,n,lda,ldu,ldvt;
    m = shape[0];
    n = shape[1];
    lda = shape[1];
    ldu = std::min(shape[0],shape[1]);
    ldvt = shape[1];
    info = LAPACKE_cgesdd(
        LAPACK_ROW_MAJOR,
        jobz,
        m, n,
        (lapack_complex_float*) mat_val, lda,
        s, 
        (lapack_complex_float*) U, ldu,
        (lapack_complex_float*) VT, ldvt
    );
    return;
}

void wrap_zgesdd(
    double* mat_val, long int shape[2], double* U, double* s, double* VT
) {
    char jobz = 'S';
    lapack_int info,m,n,lda,ldu,ldvt;
    m = shape[0];
    n = shape[1];
    lda = shape[1];
    ldu = std::min(shape[0],shape[1]);
    ldvt = shape[1];
    info = LAPACKE_zgesdd(
        LAPACK_ROW_MAJOR,
        jobz,
        m, n,
        (lapack_complex_double*) mat_val, lda,
        s, 
        (lapack_complex_double*) U, ldu, 
        (lapack_complex_double*) VT, ldvt
    );
    return;
}

