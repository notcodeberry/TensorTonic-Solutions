import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here

    x = np.array(x)
    
    relu = np.maximum(x,0)

    return relu