import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    
    B, seq_len, d_model = Q.shape[0], Q.shape[1], Q.shape[2]
    
    d_k = d_model // num_heads

    W_q_multi = np.random.randn(d_model, num_heads * d_k) #np.tile(W_q, (1, num_heads)) 
    W_k_multi = np.random.randn(d_model, num_heads * d_k) #np.tile(W_k, (1, num_heads)) # (seq_len, h*d_k)
    W_v_multi = np.random.randn(d_model, num_heads * d_k) #np.tile(W_v, (1, num_heads))
    
    qw_q = (Q @ W_q_multi).reshape(B, seq_len, num_heads, d_k).transpose(0,2,1,3)
    kw_k = (K @ W_k_multi).reshape(B, seq_len, num_heads, d_k).transpose(0,2,1,3) 
    vw_v = (V @ W_v_multi).reshape(B, seq_len, num_heads, d_k).transpose(0,2,1,3)
    
    out = softmax((qw_q @ kw_k.transpose(0,1,3,2)) / np.sqrt(d_k)) @ vw_v #(batch, h, seq_len, d_k)

    out = out.transpose(0,2,1,3).reshape(B, seq_len, d_k * num_heads)
    
    multi_head_attention = out @ W_o

    return multi_head_attention