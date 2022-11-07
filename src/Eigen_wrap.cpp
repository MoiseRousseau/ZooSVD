#include <Eigen/Dense>
#include <Eigen/SVD>
#include <iostream>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matd;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matf;
template<typename T,int> void main_EigenJacobi(T*, const long int[2], T*, T*, T*);

extern "C" {
    void wrap_dEigenJacobi(double*, const long int[2], double*, double*, double*, const int);
    void wrap_sEigenJacobi(float*, const long int[2], float*, float*, float*, const int);
    
    void wrap_dEigenBDC(double*, const long int[2], double*, double*, double*);
    void wrap_sEigenBDC(float*, const long int[2], float*, float*, float*);
}

/*
Eigen Two-sided Jacobi SVD for float type
See: https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
*/
void wrap_sEigenJacobi(
    float* mat_val, const long int shape[2], float* U, float* s, float* VT, const int prec
) {
    if (prec == 0) {
        main_EigenJacobi<float, Eigen::NoQRPreconditioner>(mat_val, shape, U, s, VT);
    }
    else if (prec == 1) {
        main_EigenJacobi<float, Eigen::HouseholderQRPreconditioner>(mat_val, shape, U, s, VT);
    }
    else if (prec == 2) {
        main_EigenJacobi<float, Eigen::ColPivHouseholderQRPreconditioner>(mat_val, shape, U, s, VT);
    }
    else if (prec == 3) {
        main_EigenJacobi<float, Eigen::FullPivHouseholderQRPreconditioner>(mat_val, shape, U, s, VT);
    }
    return;
}

/*
Eigen Two-sided Jacobi SVD for double type
*/
void wrap_dEigenJacobi(
    double* mat_val, const long int shape[2], double* U, double* s, double* VT, const int prec
) {
    if (prec == 0) {
        main_EigenJacobi<double, Eigen::NoQRPreconditioner>(mat_val,shape,U,s,VT);
    }
    else if (prec == 1) {
        main_EigenJacobi<double, Eigen::HouseholderQRPreconditioner>(mat_val,shape,U,s,VT);
    }
    else if (prec == 2) {
        main_EigenJacobi<double, Eigen::ColPivHouseholderQRPreconditioner>(mat_val,shape,U,s,VT);
    }
    else if (prec == 3) {
        main_EigenJacobi<double, Eigen::FullPivHouseholderQRPreconditioner>(mat_val,shape,U,s,VT);
    }
    return;
}


/*
The main routine for Eigen_Jacobi templated with the type and preconditionner
*/
template<typename T, int QRPreconditionner>
void main_EigenJacobi(
    T* mat_val, const long int shape[2], T* U, T* s, T* V
) {
    Eigen::Map< Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> > A(mat_val, shape[0], shape[1]);
    //This work for Eigen 3.4.0 but there is a change for Eigen 3.4.90 as in the docs
    Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,QRPreconditionner> svd;
    svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::ComputationInfo info = svd.info();
    if (info != 0) std::cout << info << std::endl;
    //get output
    long int m = std::min(shape[0],shape[1]);
    for (long int i=0; i<m; i++) {
        s[i] = svd.singularValues()[i];
    }
    for (long int i=0; i<shape[0]; i++) {
        for (long int j=0; j<m; j++) {
            U[m*i+j] = svd.matrixU()(i,j);
        }
    }
    for (long int i=0; i<shape[1]; i++) {
        for (long int j=0; j<m; j++) {
            V[m*i+j] = svd.matrixV()(i,j);
        }
    }
    return;
}



/*
Eigen Bidiag Divide and Conquer for double type
*/
void wrap_dEigenBDC(
    double* mat_val, const long int shape[2], double* U, double* s, double* V
) {
    Eigen::Map<Matd> A(mat_val, shape[0], shape[1]);
    //This work for Eigen 3.4.0 but there is a change for Eigen 3.4.90 as in the docs
    Eigen::BDCSVD<Matd> svd;
    svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV); //Eigen::HouseholderQRPreconditioner
    Eigen::ComputationInfo info = svd.info();
    if (info != 0) std::cout << info << std::endl;
    //get output
    long int m = std::min(shape[0],shape[1]);
    for (long int i=0; i<m; i++) {
        s[i] = svd.singularValues()[i];
    }
    for (long int i=0; i<shape[0]; i++) {
        for (long int j=0; j<m; j++) {
            U[m*i+j] = svd.matrixU()(i,j);
        }
    }
    for (long int i=0; i<shape[1]; i++) {
        for (long int j=0; j<m; j++) {
            V[m*i+j] = svd.matrixV()(i,j);
        }
    }
    return;
}

/*
Eigen Bidiag Divide and Conquer for float type
*/
void wrap_sEigenBDC(
    float* mat_val, const long int shape[2], float* U, float* s, float* V
) {
    Eigen::Map<Matf> A(mat_val, shape[0], shape[1]);
    //This work for Eigen 3.4.0 but there is a change for Eigen 3.4.90 as in the docs
    Eigen::BDCSVD<Matf> svd;
    svd.compute(A, Eigen::ComputeThinU | Eigen::ComputeThinV); //Eigen::HouseholderQRPreconditioner
    Eigen::ComputationInfo info = svd.info();
    if (info != 0) std::cout << info << std::endl;
    //get output
    long int m = std::min(shape[0],shape[1]);
    for (long int i=0; i<m; i++) {
        s[i] = svd.singularValues()[i];
    }
    for (long int i=0; i<shape[0]; i++) {
        for (long int j=0; j<m; j++) {
            U[m*i+j] = svd.matrixU()(i,j);
        }
    }
    for (long int i=0; i<shape[1]; i++) {
        for (long int j=0; j<m; j++) {
            V[m*i+j] = svd.matrixV()(i,j);
        }
    }
    return;
}


