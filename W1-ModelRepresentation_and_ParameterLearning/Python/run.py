from numpy import *
from gradient_descent import *

# Training Sets
x = array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8], [1, 9], [1, 10], [1, 11], [1, 12]])
y = array([[-890], [-1411], [-1560], [-2220], [-2091], [-2878], [-3537], [-3268], [-3920], [-4163], [-5471], [-5157]])
if len(x) != len(y):
    print('The size of two arrays is different\n')
    exit()
    
# Parameters setting
theta = array([[0.9], [0.9]])   # a=0.9, b=0.9
alpha = 0.002                   # learning rate

# Optimized parameters (Gradient Descent)
optimized = gradient_descent(x, y, theta, alpha)

# Results
print("Parameters : ", optimized)