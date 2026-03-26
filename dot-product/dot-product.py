import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here
    if len(x) != len(y):
        raise ValueError 

    x, y = np.array(x), np.array(y)

    dot_product = np.dot(x,y)

    return dot_product