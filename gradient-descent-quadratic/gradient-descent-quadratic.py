def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    y_prime = lambda x: 2*a*x + b

    x = x0
        
    for i in range(steps):
        x = x - lr * y_prime(x)

    return x 
    