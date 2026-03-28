import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """

    # Your code here

    """
    pos_encodings = np.zeros((seq_length, d_model))
    
    for pos in range(seq_length):
        for i in range(d_model):
            if i % 2 == 0:
                pe = np.sin(pos/(10000)**((2*i)/d_model))
            else:
                pe = np.cos(pos/(10000)**((2*i)/d_model))
            pos_encodings[pos][i] = pe

    return pos_encodings 
    """
    
    pos = np.arange(seq_length).reshape(seq_length,1)
    i = np.arange(d_model).reshape(1,d_model)

    angle_rates = 1 / (10000**((2 * (i//2)) / d_model))
    angles = pos * angle_rates #(seq_len, d_model)

    pe = np.zeros((seq_length, d_model))

    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe
