// https://stackoverflow.com/questions/57403017/cuda-cusolver-gesvdj-with-large-matrix
// https://docs.nvidia.com/cuda/cusolver/index.html

#include <assert.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <iostream>


extern "C" {
    void wrap_CUDA_dgesvdj(double*, const long int[2], double*, double*, double*, const double&, const int&);
}

void wrap_CUDA_dgesvdj(
    double* mat_val, const long int shape[2], double* U, double* s, double* VT, const double& tol, const int& max_sweeps
) {
    //LAPACK param
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; //compute singular value and singular vectors
    const int econ = 1 ; // economy size
    const int m = shape[0];
    const int n = shape[1];
    const int lda = shape[1];
    const int ldu = std::min(shape[0],shape[1]);
    const int ldvt = shape[1];
    
    //-1/ Check CUDA card
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) 
    {
        std::cout << "No CUDA-capable device found, exiting..." << std::endl;
        exit(1);
    }
    
    //0. initiate variable and context
    cusolverStatus_t status;
    cusolverDnHandle_t cusolverH;
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    
    //1. Get user parameters (tolerance and max sweeps)
    gesvdjInfo_t gesvdj_params;
    status = cusolverDnCreateGesvdjInfo(&gesvdj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    status = cusolverDnXgesvdjSetTolerance(
        gesvdj_params,
        tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    status = cusolverDnXgesvdjSetMaxSweeps(
        gesvdj_params,
        max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    
    //2. Import the matrices on the GPU memory
    cudaError_t success;
    double* d_mat_val = nullptr;
    success = cudaMalloc((void**) &d_mat_val, sizeof(double)*m*n); assert(cudaSuccess == success);
    success = cudaMemcpy(d_mat_val, mat_val, sizeof(double)*m*n, cudaMemcpyHostToDevice); assert(cudaSuccess == success);
    double* d_U = nullptr;
    success = cudaMalloc((void**) &d_U, sizeof(double)*ldu*shape[1]); assert(cudaSuccess == success);
    success = cudaMemcpy(d_U, U, sizeof(double)*m*n, cudaMemcpyHostToDevice); assert(cudaSuccess == success);
    double* d_S = nullptr;
    success = cudaMalloc((void**) &d_S, sizeof(double)*ldu); assert(cudaSuccess == success);
    success = cudaMemcpy(d_S, s, sizeof(double)*ldu, cudaMemcpyHostToDevice); assert(cudaSuccess == success);
    double* d_VT = nullptr;
    success = cudaMalloc((void**) &d_VT, sizeof(double)*ldvt*shape[0]); assert(cudaSuccess == success);
    success = cudaMemcpy(d_VT, VT, sizeof(double)*ldvt*shape[0], cudaMemcpyHostToDevice); assert(cudaSuccess == success);
    
    //3. Prepare workspace
    int lwork;
    status = cusolverDnDgesvdj_bufferSize(
        cusolverH, 
        jobz, econ,
        m, n, //matrix size
        d_mat_val, lda, //the device matrix and leading dimension
        d_S,  //the singular value
        d_U, ldu, //the U matrix
        d_VT, ldvt,  //the VT matrix
        &lwork,
        gesvdj_params
    );
    assert(CUSOLVER_STATUS_SUCCESS == status);
    double* d_work = nullptr;
    success = cudaMalloc((void**)&d_work , sizeof(double)*lwork); assert(cudaSuccess == success);
    
    //4. Do the SVD
    int info = 0;
    status = cusolverDnDgesvdj(
        cusolverH, 
        jobz, econ,
        m, n, //matrix size
        d_mat_val, lda, //the device matrix and leading dimension
        d_S,  //the singular value
        d_U, ldu, //the U matrix
        d_VT, ldvt,  //the VT matrix
        d_work, lwork,
        &info,
        gesvdj_params
    );
    success = cudaDeviceSynchronize();
    assert(cudaSuccess == success);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    
    if ( 0 > info )
    {
        std::cout << "There was a problem with the " << -info << "-th parameter in cusolverDnDgesvdj. Please create an issue on GitHub!" << std::endl;
        exit(1);
    }
    else{
        std::cout << "WARNING: gesvdj does not converge (error code " << info << ")" << std::endl;
    }
    
    //5. Copy back the result on host memory
    success = cudaMemcpy(U, d_U, sizeof(double)*ldu*shape[1], cudaMemcpyDeviceToHost); assert(cudaSuccess == success);
    success = cudaMemcpy(s, d_S, sizeof(double)*ldu, cudaMemcpyDeviceToHost); assert(cudaSuccess == success);
    success = cudaMemcpy(VT, d_VT, sizeof(double)*ldvt*shape[0], cudaMemcpyDeviceToHost); assert(cudaSuccess == success);
    success = cudaDeviceSynchronize(); assert(cudaSuccess == success);
    
    //Finalize
    cusolverDnDestroyGesvdjInfo(gesvdj_params); 
    cusolverDnDestroy(cusolverH);
    cudaFree(d_mat_val);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_work);
    return;
}
