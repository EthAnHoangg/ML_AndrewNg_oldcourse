import numpy as np
import os
from matplotlib import pyplot
data = np.loadtxt(os.path.join('ex1data1.txt'), delimiter=',')
X, y = data[:, 0], data[:, 1]
m = y.size  # number of training examples
def plotData(x, y):
    fig = pyplot.figure()  # open a new figure
    pyplot.plot(x, y, 'ro', mec= 'k')
    pyplot.xlabel('Population in the city in 10,000s')
    pyplot.ylabel('Profit in $10,000')
X = np.stack([np.ones(m), X], axis=1)
def computeCost(X, y, theta):
    m = y.size  # number of training examples
    J = 0
    for i in range (m):
        J = J + (np.dot(theta.T, X[i]) - y[i])**2
    return J/(2*m)
def gradientDescent_true(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    J_history = [] # Use a python list to save cost in every iteration
    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        theta = theta - (np.dot(theta, X.T) - y).dot(X) * alpha / m
        # =====================================================================
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))    
    return theta, J_history


def gradientDescent_false(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.shape[0]  # number of training examples
    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    J_history = [] # Use a python list to save cost in every iteration
    def decreasing(X, y, j, theta):
        res = (np.dot(theta.T, X.T) - y).dot(X.T[j]) * alpha/m
        return res
    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        for j in range (len(theta)):
            theta[j] = theta[j] -  decreasing(X, y, j, theta) 
        # ===================================================================== 
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta)) 
    return theta, J_history

theta = np.zeros(2)
# some gradient descent settings
iterations = 1500
alpha = 0.01
theta, J_history = gradientDescent_true(X ,y, theta, alpha, iterations)
plotData(X[:,1],y)
pyplot.plot(X[:,1], theta.dot(X.T))
pyplot.show()
