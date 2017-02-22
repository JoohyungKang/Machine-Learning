import matplotlib.pyplot as plt
import numpy as np
from gradient_descent import *

# Training Sets
data = np.loadtxt('TrainingSets.txt', delimiter=',', unpack=True, dtype='int32')
x = data[0:2,].T
y = data[2: ,].T
if len(x) != len(y):
    print('The size of two arrays is different\n')
    exit()
    
# Parameters setting
theta = np.array([[0.9], [0.9]])    # b=0.9, a=0.9
alpha = 0.002                       # learning rate

# Optimized parameters (Gradient Descent)
optimized = gradient_descent(x, y, theta, alpha)

# Results
print("Parameters : ", optimized)

# plot
plt.plot(x[:,1].T, y, 'r o')
plt.plot(x[:,1].T, optimized[1]*(x[:,1].T) + optimized[0], 'r')
plt.show()