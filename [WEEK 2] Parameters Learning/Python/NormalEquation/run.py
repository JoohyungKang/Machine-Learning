import matplotlib.pyplot as plt
import numpy as np
from normal_equation import *

# Degree of polynomial
d = 4;

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
    
# Parameters setting
theta = np.ones((d+1, 1))

# Optimized parameters (Normal Equation)
optimized = normal_equation(x, y, theta)

# Results
print("Parameters : ", optimized)

# Representation : Polynomial function
polynomial_y = np.dot(x, optimized) 

# plot
plt.title('Learn parameters by Normal Equation')
plt.xlim([0, max(x[:,1].T)+1])
plt.plot(x[:,1].T, y, 'g o')
plt.plot(x[:,1].T, polynomial_y, 'r')
plt.show()