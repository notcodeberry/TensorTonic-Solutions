import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    
    pos_encodings = np.zeros((seq_length, d_model))
    
    # Your code here

    for pos in range(seq_length):
        for i in range(d_model):
            if i % 2 == 0:
                pe = np.sin(pos/(10000)**((2*i)/d_model))
            else:
                pe = np.cos(pos/(10000)**((2*i)/d_model))
            pos_encodings[pos][i] = pe

    return pos_encodings 