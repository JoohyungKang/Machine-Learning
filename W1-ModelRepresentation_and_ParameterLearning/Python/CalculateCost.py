def calc_linear_cost(a, b, x, y, d=0):
    """"
    @param a: parameter (Linear model)
    @param b: parameter (Linear model)
    @param x: training data (x-axis)
    @param y: training data (y-axis)
    @param m: size of training sets
    @param d: flag for parameter
    @return : Cost
    """
    total_loss = 0

    for i in range(len(x)):
        loss = ((a * x[i] + b) - y[i])
        if d == 0:
            loss *= x[i]
        total_loss += loss
    # end for

    return total_loss / len(x)
# end function

