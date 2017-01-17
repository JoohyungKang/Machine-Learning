from CalculateCost import *

# Training Sets
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
y = [-890, -1411, -1560, -2220, -2091, -2878, -3537, -3268, -3920, -4163, -5471, -5157]
if len(x) != len(y):
    print('The size of two arrays is different\n')
    exit()

# Parameter settings
a = 0.9
b = 0.9
learningRate = 0.002
threshold = 0.3

# Parameters learning
while True:
    # Calculate parameters for update
    tempA = a - learningRate * calc_linear_cost(a, b, x, y)
    tempB = b - learningRate * calc_linear_cost(a, b, x, y, d=1)

    # Learning stop condition
    if abs(tempA - a) <= threshold and abs(tempB - b) <= threshold:
        break

    # Update
    a = tempA
    b = tempB

# end while

# Results
print("Parameter 'a': ", a)
print("Parameter 'b': ", b)