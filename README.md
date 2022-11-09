# ZooSVD

A collection of High Performance Computational routines wrapped in Python to perform Singular Value Decomposition of dense matrix in NumPy format.

## Getting started

Dependencies: Eigen, CUDA (for GPU wrappers)

Compile the wrappers lib for Python:
```
mkdir build && cd build
cmake .. DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=1
make
make install
```

Then add the path to the folder `PyZooSVD` in your Python path to be able to import PyZooSVD. 
For example, add at the head of your python script:
```
import sys
sys.path.append("path_to_ZooSVD_folder") #replace the previous
import PyZooSVD
```

Documentation to come...


## Available wrappers

The below table references the available wrappers with their algorithm and the data type they can operate (`s` for `float32`, `d` for `float64`, `c` for `complex64` and `z` for `complex128`).

| Wrapper | Algorithm | Data type | Comments |
|---------|-----------|-----------|----------|
| LAPACK `driver="gesvd"` | Householder reflections, Bidiagonisation | s,d,c,z | Equivalent to NumPy (slightly slower) |
| LAPACK `driver="gesdd"` | Divide and Conquer | s,d,c,z | Equivalent to NumPy (slightly slower) |
| Eigen `driver="jacobi"` | Two-sided Jacobi (with QR preconditionners) | s,d | Very slow |
| Eigen `driver="bidiagdc"` | Divide and Conquer | s,d | Slow (to be reviewed) |
| SCALAPACK | To come |  |  |
| CUDA `driver="Jacobi"` | Jacobi | s,d,c,z | GPU algorithm |
| CUDA `driver="Polar-Decomposition"` | Polar decomposition | s,d,c,z | GPU algorithm |
| CUDA `driver="QR"` | Householder reflections, Bidiagonisation | s,d,c,z | GPU equivalent of LAPACK `gesvd` |

More wrappers to come...


## Performance

## CPU performance (Laptop)

Time in second on a i7-1165G7 @ 2.8 Ghz (4 cores, 4 threads) for the SVD of a square matrix of the given size in `float64` format:

| Matrix size | 512 | 1024 | 2048 | 4096 | 8192 | 10240 | 12288 |
|-------------|-----|------|------|------|------|-------|-------|
| NumPy | 0.0651 | 0.358 | 2.49 | 19.7 | 141.2 | 291.9 | 504.6 |
| LAPACK `driver="gesvd"` | |  | 20.52 |
| LAPACK `driver="gesdd"` | 0.068 | 0.400 | 2.59 | 20.2 | 
| Eigen `driver="jacobi"` | | | > 300 |
| Eigen `driver="bidiagdc"` | | | 6.71 |


## GPU performance

Time in second on a NVidia A40 (48 GB GDDR6) and NVidia A100 (40 GB HBM2) GPUs for the SVD of a square matrix of the given size in `float64` format (NumPy on a 4 core CPUs as a reference):

| Matrix size | 512 | 1024 | 2048 | 4096 | 8192 | 10240 | 12288 |
|-------------|-----|------|------|------|------|-------|-------|
| NumPy | **0.0651** | **0.358** | 2.49 | 19.7 | 141.2 | 291.9 | 504.6 |
| CUDA - QR (A40) | 0.922 | 1.25 | 3.10 | 14.2 | 84.8 | 152.6 | 262.6 |
| CUDA - Polar-D (A40) | 0.87 | 1.09 | 2.19 | 8.86 | 78.2 | 106.0 | 180.6 |
| CUDA - Jacobi (A40) | 0.799 | 0.889 | 1.26 | 4.24 | 28.5 | 54.9 | 107.7 |
| CUDA - QR (A100) | 1.12 | 1.29 | 2.33 | 7.12 | 35.57 | 61.48 | 94.87 |
| CUDA - Jacobi (A100) | 1.05 | 1.11 | 1.54 | 4.67 | 25.3 | 48.5 | - |
| CUDA - Polar-D (A100) | 1.01 | 1.07 | **1.17** | **1.82** | **5.48** | **8.98** | **13.6** |

GPU are faster than CPU implementations of SVD for matrix size above 1024.
Fastest implementation are the Polar-Decomposition algorithm on NVidia A100 GPU, which is around 50 times faster than the CPU implementation of NumPy for large matrix (i7-1165G7 on 4 cores).

