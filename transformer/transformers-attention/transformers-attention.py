import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k = torch.tensor(Q.shape[2])
        
    weights = torch.matmul(Q, K.transpose(1,2)) / torch.sqrt(d_k)
    weights_softmax = torch.softmax(weights, dim=-1)

    attention = torch.matmul(weights_softmax, V)
    
    return attention