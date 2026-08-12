def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    
    for step in range(steps): 
        f_prim_x = 2*a*x + b
        
        x = x - lr * f_prim_x

    return x 