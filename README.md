# Simple Matrix Multiplication using CUDA

We've used 6x6 matrices for matrix multiplication. The matrices are hardcoded with values of 1 for first matrix and 2 for second matrix.

We've introduced Tile Width which divides the matrix into 9 blocks. These blocks have 2x2 matrix and by using blockIdx and threadIdx we are getting the number for calculation.
