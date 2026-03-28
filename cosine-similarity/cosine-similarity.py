import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    norm = np.linalg.norm(a) * np.linalg.norm(b)
        
    cosine_similarity = np.dot(a,b) / norm

    if norm == 0:
        cosine_similarity = 0.0
    
    return cosine_similarity
    