import numpy as np
from calc_linear_cost import *

def gradient_descent(x, y, t, a):
    """
    @param x: training data (x-axis)
    @param y: training data (y-axis)
    @param t: parameters theta (Linear model)
    @param a: learning rate alpha
    @return : Optimized parameters
    """
    
    # Stop condition
    threshold = 0.3
    
    # Parameters learning
    while True:
        # Calculate parameters for update
        temp = t - a * calc_linear_cost(x, y, t)
        
        # Learning stop condition
        if min(abs(temp - t)) <= threshold:
            break
        
        # Update parameters
        t = temp
    
    return t
# end function