cmake_minimum_required(VERSION 3.15)

# set the project name and version
project(ZooSVD 
    LANGUAGES CXX 
    VERSION 0.1
    DESCRIPTION "A collection of High Performance Computational routines performing Singular Value Decomposition of Dense matrix")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

#------------------
# Default dependencies
#------------------
#default source
set(LIB_SOURCES src/Eigen_wrap.cpp src/LAPACK_wrap.cpp src/utils.cpp)

#LAPACK
find_package(LAPACK REQUIRED)
set(LAPACKE_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(LAPACKE REQUIRED)
message(STATUS "LAPACKE lib: ${LAPACKE_LIBRARIES}")

#EIGEN
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 dir: ${EIGEN3_INCLUDE_DIR}")

#ScaLAPACK
#find_package(ScaLAPACK REQUIRED)
#message(STATUS "ScaLAPACK lib: ${SCALAPACK_LIBRARIES}")

#------------------
# Touchy dependencies to handle separately
#------------------
if(WITH_KSVD)
    set(MPI_TO_FIND 1)
    message(STATUS "Configure with Kaust SVD wrapper")
    add_executable (KSVD_main 
        src/KSVD_main.cpp
    )
    set(LIB_SOURCES ${LIB_SOURCES} src/KSVD_wrap.cpp)
    install(TARGETS KSVD_main DESTINATION ${CMAKE_SOURCE_DIR}/PyZooSVD)
endif()


if(WITH_CUDA)
    message(STATUS "Configure with CUDA wrapper")
    find_package(CUDAToolkit REQUIRED)
    set(LIB_SOURCES ${LIB_SOURCES} src/CUDA_gesvdj_wrap.cpp)
    set(CUDA_LIBRARIES CUDA::cudart CUDA::cusolver)
endif()

if(MPI_TO_FIND)
    include(FindMPI)
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
    message(STATUS "Switch to compiler: ${CMAKE_CXX_COMPILER}")
endif()

#------------------
# Define target 
#------------------
include_directories(src)
add_library (ZooSVD SHARED ${LIB_SOURCES})
target_link_libraries(ZooSVD 
    ${LAPACKE_LIBRARIES}
    ${SCALAPACK_LIBRARIES}
    ${CUDA_LIBRARIES}
)
target_include_directories ( ZooSVD PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
)

#------------------
# Parallelization using OpenMP
#------------------
#find_package(OpenMP)
#if(OpenMP_CXX_FOUND)
  #target_link_libraries(ZooSVD OpenMP::OpenMP_CXX)
#endif()

install(TARGETS ZooSVD DESTINATION ${CMAKE_SOURCE_DIR}/PyZooSVD)
