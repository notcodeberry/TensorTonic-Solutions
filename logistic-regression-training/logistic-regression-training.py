import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)

    n, d = X.shape
    
    w = np.zeros((d,))
    b = 0.0

    for epoch in range(steps):
        for i in range(n):
            z = X[i] @ w + b
            p_i = _sigmoid(z)
            
            loss = (-1/n)*((y[i]*np.log(p_i) + (1-y[i])*np.log(1-p_i)).sum())
            dl_dw = (1/n)*X[i].T*(p_i - y[i])
            dl_db = (1/n)*(p_i - y[i])
            
            w += - lr * dl_dw
            b += - lr * dl_db
        
    # Write code here
    return (w,b)