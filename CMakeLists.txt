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
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#------------------
# Define target 
#------------------
include_directories(src)
add_library (ZooSVD SHARED
    src/Eigen_wrap.cpp
    src/LAPACK_wrap.cpp
    src/utils.cpp
)

#------------------
# Dependencies
#------------------
#LAPACK
find_package(LAPACK REQUIRED)
set(LAPACKE_DIR "${CMAKE_SOURCE_DIR}/external")
find_package(LAPACKE REQUIRED)
#set(LAPACKE_LIBRARIES "/usr/lib/x86_64-linux-gnu/liblapacke.so")
message(STATUS "LAPACKE lib: ${LAPACKE_LIBRARIES}")
target_link_libraries(ZooSVD ${LAPACKE_LIBRARIES})

#EIGEN
find_package(
    Eigen3 REQUIRED
    HINTS ${EIGEN3_PATH}
)
message(STATUS "Eigen3 lib: ${EIGEN3_INCLUDE_DIR}")
target_include_directories ( ZooSVD PRIVATE ${EIGEN3_INCLUDE_DIR} )

#------------------
# Parallelization using OpenMP
#------------------
#find_package(OpenMP)
#if(OpenMP_CXX_FOUND)
  #target_link_libraries(ZooSVD OpenMP::OpenMP_CXX)
#endif()

install(TARGETS ZooSVD DESTINATION ${CMAKE_SOURCE_DIR}/PyZooSVD)
