import matplotlib.pyplot as plt
import numpy as np
from gradient_descent import *
from numpy import average

# Degree of polynomial
d = 5;

# Training Sets
data = np.loadtxt('TrainingSets.txt', delimiter=',', unpack=True, dtype='int32')
y = data[2:,].T
x = np.ones((d+1, len(y)))
# Create the X values: x + x^2 + ... + x^d
for i in range(1, d+1):
    x[i] = data[1:2,] ** i
x = x.T

if len(x) != len(y):
    print('The size of two arrays is different\n')
    exit()
    
# Normalize features-X
for i in range(1, d+1):
    x[:,i] = (x[:,i] - min(x[:,i])) / (max(x[:,i]) - min(x[:,i]))
    
# Parameters setting
theta = np.ones((d+1, 1))
alpha = 0.7                       # learning rate
iters = 1000000                   # iteration count

# Optimized parameters (Gradient Descent)
optimized = gradient_descent(x, y, theta, alpha, iters)

# Results
print("Parameters : ", optimized)

# Representation : Polynomial function
polynomial_y = np.dot(x, optimized) 

# plot
plt.plot(x[:,1].T, y, 'r o')
plt.plot(x[:,1].T, polynomial_y, 'r')
plt.show()