// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/gesvdj/cusolver_gesvdj_example.cu
// https://docs.nvidia.com/cuda/cusolver/index.html

#include <assert.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>


#define CUSOLVER_CHECK(err) \
    do {  \
        cusolverStatus_t err_ = (err); \
        if (err_ != CUSOLVER_STATUS_SUCCESS) { \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);  \
            throw std::runtime_error("cusolver error"); \
        }  \
    } while (0)

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error"); \
        } \
    } while (0)


void check_CUDA_device();

extern "C" {
    void wrap_CUDA_dgesvdj(double*, const long int[2], double*, double*, double*, const double, const int);
}


void check_CUDA_device()
{
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) 
    {
        std::cout << "No CUDA-capable device found, exiting..." << std::endl;
        exit(1);
    }
}


void wrap_CUDA_dgesvdj(
    double* mat_val, const long int shape[2], double* U, double* s, double* V, const double tol, const int max_sweeps
) {
    //We do a little hack here...
    //Numpy arrays are stored row major, but CUDA expect them in column major
    //Thus, when passing Numpy array to CUDA, we are passing the transpose of the matrix
    //and we do the SVD of the matrix transpose and adjust the result

    //LAPACK param
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; //compute singular value and singular vectors
    const int econ = 1 ; // economy size
    const int n = shape[0];
    const int m = shape[1];
    const int lda = m;
    const int ldu = m;
    const int ldv = n;
    const int min_nm = std::min(n,m);
    
    //-1. Check CUDA card
    check_CUDA_device();
    
    //0. initiate variable and context
    cusolverDnHandle_t cusolverH;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    
    //1. Get user parameters (tolerance and max sweeps)
    gesvdjInfo_t gesvdj_params;
    CUSOLVER_CHECK(cusolverDnCreateGesvdjInfo(&gesvdj_params));
    CUSOLVER_CHECK(cusolverDnXgesvdjSetTolerance(
        gesvdj_params,
        tol
    ));
    CUSOLVER_CHECK(cusolverDnXgesvdjSetMaxSweeps(
        gesvdj_params,
        max_sweeps
    ));
    
    //2. Import the matrices on the GPU memory
    double* d_mat_val = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_mat_val, sizeof(double)*m*n));
    CUDA_CHECK(cudaMemcpy(d_mat_val, mat_val, sizeof(double)*m*n, cudaMemcpyHostToDevice));
    double* d_U = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_U, sizeof(double)*m*min_nm));
    double* d_S = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_S, sizeof(double)*min_nm));
    double* d_V = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_V, sizeof(double)*n*min_nm));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_info, sizeof(int)));
    
    //3. Prepare workspace
    int lwork;
    CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(
        cusolverH, 
        jobz, econ,
        m, n, //matrix size
        d_mat_val, lda, //the device matrix and leading dimension
        d_S,  //the singular value
        d_U, ldu, //the U matrix
        d_V, ldv,  //the V matrix
        &lwork,
        gesvdj_params
    ));
    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_work , sizeof(double)*lwork));
    
    //4. Do the SVD
    cusolverDnDgesvdj(
        cusolverH, 
        jobz, econ,
        m, n, //matrix size
        d_mat_val, lda, //the device matrix and leading dimension
        d_S,  //the singular value
        d_U, ldu, //the U matrix
        d_V, ldv,  //the V matrix
        d_work, lwork,
        d_info,
        gesvdj_params
    ); //we do check the error in info latter

    //5. Copy back the result on host memory
    //Here is the hack (d_V in U and d_U in V)
    CUDA_CHECK(cudaMemcpy(U, d_V, sizeof(double)*n*min_nm, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s, d_S, sizeof(double)*min_nm, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(V, d_U, sizeof(double)*min_nm*m, cudaMemcpyDeviceToHost));
    int info;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    //Finalize
    if ( 0 > info )
    {
        std::cout << "There was a problem with the " << -info << "-th parameter in cusolverDnDgesvdj. Please create an issue on GitHub!" << std::endl;
        exit(1);
    }
    else if (0 < info) {
        std::cout << "WARNING: gesvdj does not converge (error code " << info << ")" << std::endl;
    }
    cusolverDnDestroyGesvdjInfo(gesvdj_params); 
    cusolverDnDestroy(cusolverH);
    cudaFree(d_mat_val);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_V);
    cudaFree(d_info);
    cudaFree(d_work);
    cudaDeviceReset();
    return;
}
