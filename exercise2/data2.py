# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

#Optimization module in scipy
from scipy import optimize

import utils

# Load Data
# The first two columns contains the X values and the third column
# contains the label (y).
data = np.loadtxt(os.path.join('Data', 'ex2data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]

def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression.
    Computes the predictions for X using a threshold at 0.5 
    (i.e., if sigmoid(theta.T*x) >= 0.5, predict 1)
    
    Parameters
    ----------
    theta : array_like
        Parameters for logistic regression. A vecotor of shape (n+1, ).
    
    X : array_like
        The data to use for computing predictions. The rows is the number 
        of points to compute predictions, and columns is the number of
        features.

    Returns
    -------
    p : array_like
        Predictions and 0 or 1 for each row in X. 
    
    Instructions
    ------------
    Complete the following code to make predictions using your learned 
    logistic regression parameters.You should set p to a vector of 0's and 1's    
    """
    m = X.shape[0] # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    # ====================== YOUR CODE HERE ======================

    p = sigmoid(np.dot(X, theta)) >= 0.5
    
    # ============================================================
    return p

def plotData(X, y):
    """
    Plots the data points X and y into a new figure. Plots the data 
    points with * for the positive examples and o for the negative examples.
    
    Parameters
    ----------
    X : array_like
        An Mx2 matrix representing the dataset. 
    
    y : array_like
        Label values for the dataset. A vector of size (M, ).
    
    Instructions
    ------------
    Plot the positive and negative examples on a 2D plot, using the
    option 'k*' for the positive examples and 'ko' for the negative examples.    
    """
    # Create New Figure
    fig = pyplot.figure()

    # ====================== YOUR CODE HERE ======================
    pos = y == 1
    neg = y == 0

    pyplot.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    pyplot.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
    
    # ============================================================

from math import e
def sigmoid(z):
    """
    Compute sigmoid function given the input z.
    
    Parameters
    ----------
    z : array_like
        The input to the sigmoid function. This can be a 1-D vector 
        or a 2-D matrix. 
    
    Returns
    -------
    g : array_like
        The computed sigmoid function. g has the same shape as z, since
        the sigmoid is computed element-wise on z.
        
    Instructions
    ------------
    Compute the sigmoid of each value of z (z can be a matrix, vector or scalar).
    """
    # convert input to a numpy array
    z = np.array(z)
    
    # You need to return the following variables correctly 
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================
    g = 1. /(1+e**(-z))
    

    # =============================================================
    return g

def costFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression. 
    
    Parameters
    ----------
    theta : array_like
        The parameters for logistic regression. This a vector
        of shape (n+1, ).
    
    X : array_like
        The input dataset of shape (m x n+1) where m is the total number
        of data points and n is the number of features. We assume the 
        intercept has already been added to the input.
    
    y : arra_like
        Labels for the input. This is a vector of shape (m, ).
    
    Returns
    -------
    J : float
        The computed value for the cost function. 
    
    grad : array_like
        A vector of shape (n+1, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.
        
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to 
    the cost. Compute the partial derivatives and set grad to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================

    cost = -np.dot(y,np.log(sigmoid(np.dot(X, np.transpose(theta))))) - np.dot((1-y),np.log(1-sigmoid(np.dot(X, np.transpose(theta)))))

    J = 1/m*np.sum(cost)

    for i in range(grad.size):
        grad[i] = 1/m * np.dot(
            sigmoid(np.dot(X, np.transpose(theta))) - y,
            X[:, i]
        )
        
    
    # =============================================================
    return J, grad

# plotData(X, y)
# # Labels and Legend
# pyplot.xlabel('Microchip Test 1')
# pyplot.ylabel('Microchip Test 2')

# # Specified in plot order
# pyplot.legend(['y = 1', 'y = 0'], loc='upper right')
# pyplot.show()

X = utils.mapFeature(X[:, 0], X[:, 1])

def costFunctionReg(theta, X, y, lambda_):
    """
    Compute cost and gradient for logistic regression with regularization.
    
    Parameters
    ----------
    theta : array_like
        Logistic regression parameters. A vector with shape (n, ). n is 
        the number of features including any intercept. If we have mapped
        our initial features into polynomial features, then n is the total 
        number of polynomial features. 
    
    X : array_like
        The data set with shape (m x n). m is the number of examples, and
        n is the number of features (after feature mapping).
    
    y : array_like
        The data labels. A vector with shape (m, ).
    
    lambda_ : float
        The regularization parameter. 
    
    Returns
    -------
    J : float
        The computed value for the regularized cost function. 
    
    grad : array_like
        A vector of shape (n, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.
    
    Instructions
    ------------
    Compute the cost `J` of a particular choice of theta.
    Compute the partial derivatives and set `grad` to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ===================== YOUR CODE HERE ======================

    reg = lambda_/(2*m)*np.sum(np.array(theta[1:])**2)

    J, _ = costFunction(theta, X, y)

    J = J + reg

    grad[0] = 1/m*np.sum(
            np.dot(
                sigmoid(np.dot(X, theta)) - y,
                X[:,0]
            )
        )

    for i in range(1, grad.size):
        grad[i] = 1/m*np.sum(
            np.dot(
                sigmoid(np.dot(X, theta)) - y,
                X[:,i]
            )
        ) + lambda_/m*theta[i]
    
    
    # =============================================================
    return J, grad

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
# DO NOT use `lambda` as a variable name in python
# because it is a python keyword
lambda_ = 1

# # Compute and display initial cost and gradient for regularized logistic
# # regression
# cost, grad = costFunctionReg(initial_theta, X, y, lambda_)

# print('Cost at initial theta (zeros): {:.3f}'.format(cost))
# print('Expected cost (approx)       : 0.693\n')

# print('Gradient at initial theta (zeros) - first five values only:')
# print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
# print('Expected gradients (approx) - first five values only:')
# print('\t[0.0085, 0.0188, 0.0001, 0.0503, 0.0115]\n')


# # Compute and display cost and gradient
# # with all-ones theta and lambda = 10
# test_theta = np.ones(X.shape[1])
# cost, grad = costFunctionReg(test_theta, X, y, 10)

# print('------------------------------\n')
# print('Cost at test theta    : {:.2f}'.format(cost))
# print('Expected cost (approx): 3.16\n')

# print('Gradient at test theta - first five values only:')
# print('\t[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*grad[:5]))
# print('Expected gradients (approx) - first five values only:')
# print('\t[0.3460, 0.1614, 0.1948, 0.2269, 0.0922]')

options = {"maxiter": 100}

res = optimize.minimize(costFunctionReg, initial_theta, (X,y, lambda_), jac=True, method='TNC', options=options)

cost = res.fun

theta = res.x

utils.plotDecisionBoundary(plotData, theta, X, y)
pyplot.legend(['y = 1', 'y = 0'])
pyplot.xlabel('Microchip test 1')
pyplot.ylabel('Microchip test 2')
pyplot.title('lambda = %0.2f' % lambda_)
pyplot.grid(False)

# Compute accuracy on our training set
p = predict(theta, X)

print('Train Accuracy: %.1f %%' % (np.mean(p == y) * 100))
print('Expected accuracy (with lambda = 1): 83.1 % (approx)\n')

pyplot.show()
