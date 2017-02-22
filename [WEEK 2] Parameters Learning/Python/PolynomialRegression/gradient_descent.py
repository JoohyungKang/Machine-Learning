from calc_linear_cost import *

def gradient_descent(x, y, t, a, iters):
    """
    @param x: training data (x-axis)
    @param y: training data (y-axis)
    @param t: parameters theta (Linear model)
    @param a: learning rate alpha
    @param iters: iteration
    @return : Optimized parameters
    """
    
    # Parameters learning
    for i in range(iters):
        # Calculate parameters for update
        temp = t - a * calc_linear_cost(x, y, t)
        
        # Update parameters
        t = temp
    
    return t

# end function