import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if y_pred.shape != y_true.shape:
        return None 

    sum = 0
    
    for i in range(len(y_pred)):
        sum += (y_pred[i] - y_true[i])**2

    return sum / len(y_pred)
        
