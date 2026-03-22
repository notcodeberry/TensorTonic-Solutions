import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    n, m = A.shape
    
    transposed_matrix = np.zeros((m,n))

    for i in range(n):
        for j in range(m):
            transposed_matrix[j][i] = A[i][j]

    return transposed_matrix
            