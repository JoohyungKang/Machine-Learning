import numpy as np

def normal_equation(x, y, t):
    """
    @param x: training data (x-axis)
    @param y: training data (y-axis)
    @param t: parameters theta (Linear model)
    @return : Optimized parameters
    """
    
    # Parameters learning
    # Normal Equation: Inv(X^T*X)*X^T*Y
    return ((np.linalg.inv(x.T.dot(x))).dot(x.T)).dot(y)

# end function