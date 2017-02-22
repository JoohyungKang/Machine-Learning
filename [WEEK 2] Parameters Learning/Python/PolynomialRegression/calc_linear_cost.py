import numpy as np

def calc_linear_cost(x, y, t):
    """
    @param x: training data (x-axis)
    @param y: training data (y-axis)
    @param t: parameters theta (Linear model)
    @return : cost variables
    """
    
    # hypothesis
    h = np.dot(x, t)
        
    # error
    loss = h - y
    
    # cost variables 
    return np.dot(x.T, loss) / len(y)

# end function