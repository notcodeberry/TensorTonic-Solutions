import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    bow = np.zeros((len(vocab), ), dtype=int)

    for word in range(len(vocab)):
        for tok in range(len(tokens)): 
            if vocab[word] == tokens[tok]:
                bow[word] += 1

    return bow
            
        