import numpy as np

def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    # Write code here

    batch_size = len(X)
    input_dim = len(X[0])
    output_dim = len(W[0])
    
    Y = [[0 for _ in range(output_dim)] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(output_dim):
            for k in range(input_dim):
                Y[i][j] += X[i][k] * W[k][j]
            Y[i][j] += b[j]
                
    return Y