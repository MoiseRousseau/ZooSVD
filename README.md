# ZooSVD

A collection of High Performance Computational routines wrapped in Python to perform Singular Value Decomposition of dense matrix in NumPy format.

## Getting started

Compile the wrappers lib for Python:
```
mkdir build && cd build
cmake .. DCMAKE_BUILD_TYPE=Release
make
make install
```

## Available wrappers

The below table references the available wrapper with their algorithm and the data type they can operate (`s` for `float32`, `d` for `float64`, `c` for `complex64` and `z` for `complex128`).

| Wrapper | Algorithm | Data type |
|---------|-----------|-----------|
| LAPACK `driver="gesvd"` | Householder reflections and Bidiagonisation | s,d,c,z |
| LAPACK `driver="gesdd"` | Divide and Conquer | s,d,c,z |
| Eigen `driver="jacobi" | Two-sided Jacobi (with QR preconditionners) | s,d |
| Eigen `driver="bidiagdc" | Divide and Conquer | s,d |

More wrappers to come...

## Performance

Time on a i7-1165G7 @ 2.8 Ghz (4 cores, 8 threads) for the SVD of a 2048x2048 matrix in `float64` format:

| Function | Time (s) |
|----------|-----------|
| NumPy | 2.28 |
| LAPACK `driver="gesvd"` | 20.52 |
| LAPACK `driver="gesdd"` | 2.31 |
| Eigen `driver="jacobi" | > 300 |
| Eigen `driver="bidiagdc" | 6.71 |
