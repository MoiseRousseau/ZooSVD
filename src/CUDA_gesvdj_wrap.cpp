// https://docs.nvidia.com/cuda/cusolver/index.html

//We do a little hack here...
//Numpy arrays are stored row major, but CUDA expect them in column major
//Thus, when passing Numpy array to CUDA, we are passing the transpose of the matrix
//and we do the SVD of the matrix transpose and adjust the result

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


int check_CUDA_device();
void calculate_type_size(const int&, int&, int&);

extern "C" {
    void wrap_CUDA_Xgesvdj(void*, const long int[2], void*, void*, void*, const double, const int, const int);
    void wrap_CUDA_Xgesvd(void*, const long int[2], void*, void*, void*, const int);
    void wrap_CUDA_Xgesvdp(void*, const long int[2], void*, void*, void*, const int, double);
}


int check_CUDA_device()
{
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    if (devCount == 0) 
    {
        std::cout << "No CUDA-capable device found, exiting..." << std::endl;
        return 1;
    }
    return 0;
}


void wrap_CUDA_Xgesvdj(
    void* mat_val, const long int shape[2], void* U, void* s, void* V, const double tol, const int max_sweeps, const int data_type
) {
    // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/gesvdj/cusolver_gesvdj_example.cu
    
    //-1. Check CUDA card
    if (check_CUDA_device()) return;
    
    //LAPACK param
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; //compute singular value and singular vectors
    const int econ = 1 ; // economy size
    const int n = shape[0];
    const int m = shape[1];
    const int lda = m;
    const int ldu = m;
    const int ldv = n;
    const int min_nm = std::min(n,m);
    
    int DataType_size, DataType_size_S;
    calculate_type_size(data_type, DataType_size, DataType_size_S);

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
    void* d_mat_val = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_mat_val, DataType_size*m*n));
    CUDA_CHECK(cudaMemcpy(d_mat_val, mat_val, DataType_size*m*n, cudaMemcpyHostToDevice));
    double* d_U = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_U, DataType_size*m*min_nm));
    double* d_S = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_S, DataType_size_S*min_nm));
    double* d_V = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_V, DataType_size*n*min_nm));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_info, sizeof(int)));
    
    //3. Prepare workspace and 4. Do the SVD
    int lwork;
    void* d_work = nullptr;
    if (data_type == 0) {
        CUSOLVER_CHECK(cusolverDnSgesvdj_bufferSize(
            cusolverH,
            jobz, econ,
            m, n, //matrix size
            (float*) d_mat_val, lda, //the device matrix and leading dimension
            (float*) d_S,  //the singular value
            (float*) d_U, ldu, //the U matrix
            (float*) d_V, ldv,  //the V matrix
            &lwork,
            gesvdj_params
        ));
        //reserve space
        CUDA_CHECK(cudaMalloc((void**)&d_work , DataType_size*lwork));
        cusolverDnSgesvdj(
            cusolverH,
            jobz, econ,
            m, n, //matrix size
            (float*) d_mat_val, lda, //the device matrix and leading dimension
            (float*) d_S,  //the singular value
            (float*) d_U, ldu, //the U matrix
            (float*) d_V, ldv,  //the V matrix
            (float*) d_work, lwork,
            d_info,
            gesvdj_params
         ); //we do check the error in info latter
    }
    else if (data_type == 1) {
        CUSOLVER_CHECK(cusolverDnDgesvdj_bufferSize(
            cusolverH, 
            jobz, econ,
            m, n, //matrix size
            (double*) d_mat_val, lda, //the device matrix and leading dimension
            (double*) d_S,  //the singular value
            (double*) d_U, ldu, //the U matrix
            (double*) d_V, ldv,  //the V matrix
            &lwork,
            gesvdj_params
        ));
	//reserve space
        CUDA_CHECK(cudaMalloc((void**)&d_work , DataType_size*lwork));
        cusolverDnDgesvdj(
            cusolverH, 
            jobz, econ,
            m, n, //matrix size
            (double*) d_mat_val, lda, //the device matrix and leading dimension
            (double*) d_S,  //the singular value
            (double*) d_U, ldu, //the U matrix
            (double*) d_V, ldv,  //the V matrix
            (double*) d_work, lwork,
            d_info,
            gesvdj_params
         ); //we do check the error in info latter
    }
    else if (data_type == 4) {
        CUSOLVER_CHECK(cusolverDnCgesvdj_bufferSize(
            cusolverH,
            jobz, econ,
            m, n, //matrix size
            (cuComplex*) d_mat_val, lda, //the device matrix and leading dimension
            (float*) d_S,  //the singular value
            (cuComplex*) d_U, ldu, //the U matrix
            (cuComplex*) d_V, ldv,  //the V matrix
            &lwork,
            gesvdj_params
        ));
        //reserve space
        CUDA_CHECK(cudaMalloc((void**)&d_work , DataType_size*lwork));
        cusolverDnCgesvdj(
            cusolverH,
            jobz, econ,
            m, n, //matrix size
            (cuComplex*) d_mat_val, lda, //the device matrix and leading dimension
            (float*) d_S,  //the singular value
            (cuComplex*) d_U, ldu, //the U matrix
            (cuComplex*) d_V, ldv,  //the V matrix
            (cuComplex*) d_work, lwork,
            d_info,
            gesvdj_params
         ); //we do check the error in info latter
    }
    else if (data_type == 5) {
        CUSOLVER_CHECK(cusolverDnZgesvdj_bufferSize(
            cusolverH,
            jobz, econ,
            m, n, //matrix size
            (cuDoubleComplex*) d_mat_val, lda, //the device matrix and leading dimension
            (double*) d_S,  //the singular value
            (cuDoubleComplex*) d_U, ldu, //the U matrix
            (cuDoubleComplex*) d_V, ldv,  //the V matrix
            &lwork,
            gesvdj_params
        ));
        //reserve space
        CUDA_CHECK(cudaMalloc((void**)&d_work , DataType_size*lwork));
        cusolverDnZgesvdj(
            cusolverH,
            jobz, econ,
            m, n, //matrix size
            (cuDoubleComplex*) d_mat_val, lda, //the device matrix and leading dimension
            (double*) d_S,  //the singular value
            (cuDoubleComplex*) d_U, ldu, //the U matrix
            (cuDoubleComplex*) d_V, ldv,  //the V matrix
            (cuDoubleComplex*) d_work, lwork,
            d_info,
            gesvdj_params
         ); //we do check the error in info latter
    }

    //5. Copy back the result on host memory
    //Here is the hack (d_V in U and d_U in V)
    CUDA_CHECK(cudaMemcpy(U, d_V, DataType_size*n*min_nm, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s, d_S, DataType_size_S*min_nm, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(V, d_U, DataType_size*min_nm*m, cudaMemcpyDeviceToHost));
    int info;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    //Finalize
    if ( 0 > info )
    {
        std::cout << "There was a problem with the " << -info << "-th parameter in cusolverDnDgesvdj. Please create an issue on GitHub!" << std::endl;
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



void calculate_type_size(
    const int& CUDADataType, 
    int &DataType_size,
    int &DataType_size_S
) {
    if (CUDADataType == CUDA_R_16F) 
        {DataType_size = 2; DataType_size_S = 2;}
    else if  (CUDADataType == CUDA_R_32F) 
        {DataType_size = 4; DataType_size_S = 4;}
    else if (CUDADataType == CUDA_R_64F) 
        {DataType_size = 8; DataType_size_S = 8;}
    else if (CUDADataType == CUDA_C_16F) 
        {DataType_size = 4; DataType_size_S = 2;}
    else if (CUDADataType == CUDA_C_32F) 
        {DataType_size = 8; DataType_size_S = 4;}
    else if (CUDADataType == CUDA_C_64F) 
        {DataType_size = 16; DataType_size_S = 8;}
    else 
        {throw std::runtime_error("Unknown CUDA data type. Can't continue");}
}

void wrap_CUDA_Xgesvd(
    void* mat_val, const long int shape[2], void* U, void* s, void* V, const int _CUDADataType
) {
    //Remark 1: gesvd only supports m>=n.
    //CUDADataType:
    //CUDA_R_16F = 2
    //CUDA_R_32F = 0
    //CUDA_R_64F = 1
    //CUDA_C_16F = 6
    //CUDA_C_32F = 4
    //CUDA_C_64F = 5
    
    //-1. Check CUDA card
    if (check_CUDA_device()) return;
    
    //0. initiate context
    const int n = shape[0];
    const int m = shape[1];
    const int lda = m;
    const int ldu = m;
    const int min_nm = std::min(n,m);
    const int ldvt = min_nm;
    const signed char jobuv = 'S'; //economy SVD
    cusolverDnHandle_t cusolverH;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);
    //cusolverDnSetAdvOptions(params, 0, NULL);
    
    //2. Import the matrices on the GPU memory
    int DataType_size, DataType_size_S;
    calculate_type_size(_CUDADataType, DataType_size, DataType_size_S);
    void* d_mat_val = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_mat_val, DataType_size*m*n));
    CUDA_CHECK(cudaMemcpy(d_mat_val, mat_val, DataType_size*m*n, cudaMemcpyHostToDevice));
    void* d_U = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_U, DataType_size*m*min_nm));
    void* d_S = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_S, DataType_size_S*min_nm));
    void* d_VT = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_VT, DataType_size*n*min_nm));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_info, sizeof(int)));
    
    //3. Prepare workspace
    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    cudaDataType_t CUDADataType = (cudaDataType_t) _CUDADataType;
    //S type (always real)
    cudaDataType_t CUDADataType_S;
    if (_CUDADataType < 3) CUDADataType_S = CUDADataType;
    else CUDADataType_S = (cudaDataType_t) (_CUDADataType - 4);
    CUSOLVER_CHECK(cusolverDnXgesvd_bufferSize(
        cusolverH,
        params,
        jobuv, jobuv,
        m, n,
        CUDADataType, d_mat_val, lda,
        CUDADataType_S, d_S,
        CUDADataType, d_U, ldu,
        CUDADataType, d_VT, ldvt,
        CUDADataType, // computeType
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost
    ));
    void* bufferOnDevice = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &bufferOnDevice, DataType_size*workspaceInBytesOnDevice));
    void* bufferOnHost = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &bufferOnHost, DataType_size*workspaceInBytesOnHost));
    
    //4. Perform SVD
    //Remark 2: the routine returns VT, not V. 
    cusolverDnXgesvd(
        cusolverH,
        params,
        jobuv, jobuv,
        m, n,
        CUDADataType, d_mat_val, lda,
        CUDADataType_S, d_S,
        CUDADataType, d_U, ldu,
        CUDADataType, d_VT, ldvt,
        CUDADataType, // computeType
        bufferOnDevice, workspaceInBytesOnDevice,
        bufferOnHost, workspaceInBytesOnHost,
        d_info
    ); //we do check the error in info latter
    
    //5. Copy back the result on host memory
    //Here is the hack (d_VT in U and d_U in V)
    CUDA_CHECK(cudaMemcpy(U, d_VT, DataType_size*n*min_nm, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s, d_S, DataType_size_S*min_nm, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(V, d_U, DataType_size*min_nm*m, cudaMemcpyDeviceToHost));
    int info;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    //Finalize
    if ( 0 > info )
    {
        std::cout << "There was a problem with the " << -info << "-th parameter in cusolverDnXgesvd. Please create an issue on GitHub!" << std::endl;
    }
    else if (0 < info) {
        std::cout << "WARNING: gesvd did not converge. ";
        std::cout << "Number of unconverged superdiagonals: " << info << std::endl;
    }
    cusolverDnDestroy(cusolverH);
    cusolverDnDestroyParams(params);
    cudaFree(d_mat_val);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_info);
    cudaFree(bufferOnHost);
    cudaFree(bufferOnDevice);
    cudaDeviceReset();
    return;
}


void wrap_CUDA_Xgesvdp(
    void* mat_val, const long int shape[2], void* U, void* s, void* V, const int _CUDADataType, double pertub
) {
    //CUDADataType:
    //CUDA_R_16F = 2
    //CUDA_R_32F = 0
    //CUDA_R_64F = 1
    //CUDA_C_16F = 6
    //CUDA_C_32F = 4
    //CUDA_C_64F = 5
    
    //0. initiate context
    if (check_CUDA_device()) return;
    const int n = shape[0];
    const int m = shape[1];
    const int lda = m;
    const int ldu = m;
    const int ldv = n;
    const int min_nm = std::min(n,m);
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; //compute singular value and singular vectors
    const int econ = 1 ; // economy size
    
    cusolverDnHandle_t cusolverH;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);
    //cusolverDnSetAdvOptions(params, NULL, NULL);
    
    //2. Import the matrices on the GPU memory
    int DataType_size, DataType_size_S;
    calculate_type_size(_CUDADataType, DataType_size, DataType_size_S);
    void* d_mat_val = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_mat_val, DataType_size*m*n));
    CUDA_CHECK(cudaMemcpy(d_mat_val, mat_val, DataType_size*m*n, cudaMemcpyHostToDevice));
    void* d_U = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_U, DataType_size*m*min_nm));
    void* d_S = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_S, DataType_size_S*min_nm));
    void* d_VT = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_VT, DataType_size*n*min_nm));
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &d_info, sizeof(int)));
    
    //3. Prepare workspace
    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    const cudaDataType_t CUDADataType = (cudaDataType_t) _CUDADataType;
    //S type (always real)
    cudaDataType_t CUDADataType_S;
    if (_CUDADataType < 3) CUDADataType_S = CUDADataType;
    else CUDADataType_S = (cudaDataType_t) (_CUDADataType - 4);
    CUSOLVER_CHECK(cusolverDnXgesvdp_bufferSize(
        cusolverH,
        params,
        jobz, econ,
        m, n,
        CUDADataType, d_mat_val, lda,
        CUDADataType_S, d_S,
        CUDADataType, d_U, ldu,
        CUDADataType, d_VT, ldv,
        CUDADataType, // computeType
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost
    ));
    void* bufferOnDevice = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &bufferOnDevice, DataType_size*workspaceInBytesOnDevice));
    void* bufferOnHost = nullptr;
    CUDA_CHECK(cudaMalloc((void**) &bufferOnHost, DataType_size*workspaceInBytesOnHost));
    
    //4. Perform SVD
    //Remark 2: the routine returns VT, not V. 
    cusolverDnXgesvdp(
        cusolverH,
        params,
        jobz, econ,
        m, n,
        CUDADataType, d_mat_val, lda,
        CUDADataType_S, d_S,
        CUDADataType, d_U, ldu,
        CUDADataType, d_VT, ldv,
        CUDADataType, // computeType
        bufferOnDevice, workspaceInBytesOnDevice,
        bufferOnHost, workspaceInBytesOnHost,
        d_info,
        &pertub
    ); //we do check the error in info latter
    
    //5. Copy back the result on host memory
    //Here is the hack (d_VT in U and d_U in V)
    CUDA_CHECK(cudaMemcpy(U, d_VT, DataType_size*n*min_nm, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s, d_S, DataType_size_S*min_nm, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(V, d_U, DataType_size*min_nm*m, cudaMemcpyDeviceToHost));
    int info;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    //Finalize
    if ( 0 > info )
    {
        std::cout << "There was a problem with the " << -info << "-th parameter in cusolverDnXgesvdp. Please create an issue on GitHub!" << std::endl;
    }
    
    cusolverDnDestroy(cusolverH);
    cusolverDnDestroyParams(params);
    cudaFree(d_mat_val);
    cudaFree(d_S);
    cudaFree(d_U);
    cudaFree(d_VT);
    cudaFree(d_info);
    cudaFree(bufferOnHost);
    cudaFree(bufferOnDevice);
    cudaDeviceReset();
    return;
}
