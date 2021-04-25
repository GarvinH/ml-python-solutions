# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils

# Load from ex5data1.mat, where all variables will be store in a dictionary
data = loadmat(os.path.join('Data', 'ex5data1.mat'))

# Extract train, test, validation data from dictionary
# and also convert y's form 2-D matrix (MATLAB format) to a numpy vector
X, y = data['X'], data['y'][:, 0]
Xtest, ytest = data['Xtest'], data['ytest'][:, 0]
Xval, yval = data['Xval'], data['yval'][:, 0]

# m = Number of examples
m = y.size

# Plot training data
# pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1)
# pyplot.xlabel('Change in water level (x)')
# pyplot.ylabel('Water flowing out of the dam (y)');

# pyplot.show()


def linearRegCostFunction(X, y, theta, lambda_=0.0):
    """
    Compute cost and gradient for regularized linear regression 
    with multiple variables. Computes the cost of using theta as
    the parameter for linear regression to fit the data points in X and y. 
    
    Parameters
    ----------
    X : array_like
        The dataset. Matrix with shape (m x n + 1) where m is the 
        total number of examples, and n is the number of features 
        before adding the bias term.
    
    y : array_like
        The functions values at each datapoint. A vector of
        shape (m, ).
    
    theta : array_like
        The parameters for linear regression. A vector of shape (n+1,).
    
    lambda_ : float, optional
        The regularization parameter.
    
    Returns
    -------
    J : float
        The computed cost function. 
    
    grad : array_like
        The value of the cost function gradient w.r.t theta. 
        A vector of shape (n+1, ).
    
    Instructions
    ------------
    Compute the cost and gradient of regularized linear regression for
    a particular choice of theta.
    You should set J to the cost and grad to the gradient.
    """
    # Initialize some useful values
    m = y.size # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================

    hypothesis = np.dot(X, theta.T)

    J = 1 / (2*m) * np.sum(
        np.square(hypothesis - y)
    ) + lambda_ / (2*m) * np.sum(
        np.square(theta[1:])
    )

    grad = 1/m* np.dot(
            X.T,
            hypothesis - y
        )

    grad += lambda_ / m * np.insert(theta[1:], 0, 0)

    # ============================================================
    return J, grad


theta = np.array([1, 1])
J, _ = linearRegCostFunction(np.concatenate([np.ones((m, 1)), X], axis=1), y, theta, 1)

print('Cost at theta = [1, 1]:\t   %f ' % J)
print('This value should be about 303.993192)\n' % J)


theta = np.array([1, 1])
J, grad = linearRegCostFunction(np.concatenate([np.ones((m, 1)), X], axis=1), y, theta, 1)

print('Gradient at theta = [1, 1]:  [{:.6f}, {:.6f}] '.format(*grad))
print(' (this value should be about [-15.303016, 598.250744])\n')


# # add a columns of ones for the y-intercept
# X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)
# theta = utils.trainLinearReg(linearRegCostFunction, X_aug, y, lambda_=0)

# #  Plot fit over the data
# pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1.5)
# pyplot.xlabel('Change in water level (x)')
# pyplot.ylabel('Water flowing out of the dam (y)')
# pyplot.plot(X, np.dot(X_aug, theta), '--', lw=2);

# pyplot.show()

def learningCurve(X, y, Xval, yval, lambda_=0):
    """
    Generates the train and cross validation set errors needed to plot a learning curve
    returns the train and cross validation set errors for a learning curve. 
    
    In this function, you will compute the train and test errors for
    dataset sizes from 1 up to m. In practice, when working with larger
    datasets, you might want to do this in larger intervals.
    
    Parameters
    ----------
    X : array_like
        The training dataset. Matrix with shape (m x n + 1) where m is the 
        total number of examples, and n is the number of features 
        before adding the bias term.
    
    y : array_like
        The functions values at each training datapoint. A vector of
        shape (m, ).
    
    Xval : array_like
        The validation dataset. Matrix with shape (m_val x n + 1) where m is the 
        total number of examples, and n is the number of features 
        before adding the bias term.
    
    yval : array_like
        The functions values at each validation datapoint. A vector of
        shape (m_val, ).
    
    lambda_ : float, optional
        The regularization parameter.
    
    Returns
    -------
    error_train : array_like
        A vector of shape m. error_train[i] contains the training error for
        i examples.
    error_val : array_like
        A vecotr of shape m. error_val[i] contains the validation error for
        i training examples.
    
    Instructions
    ------------
    Fill in this function to return training errors in error_train and the
    cross validation errors in error_val. i.e., error_train[i] and 
    error_val[i] should give you the errors obtained after training on i examples.
    
    Notes
    -----
    - You should evaluate the training error on the first i training
      examples (i.e., X[:i, :] and y[:i]).
    
      For the cross-validation error, you should instead evaluate on
      the _entire_ cross validation set (Xval and yval).
    
    - If you are using your cost function (linearRegCostFunction) to compute
      the training and cross validation error, you should call the function with
      the lambda argument set to 0. Do note that you will still need to use
      lambda when running the training to obtain the theta parameters.
    
    Hint
    ----
    You can loop over the examples with the following:
     
           for i in range(1, m+1):
               # Compute train/cross validation errors using training examples 
               # X[:i, :] and y[:i], storing the result in 
               # error_train[i-1] and error_val[i-1]
               ....  
    """
    # Number of training examples
    m = y.size

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val   = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
         
    for i in range(1, m+1):
        theta = utils.trainLinearReg(linearRegCostFunction, X[:i], y[:i], lambda_)
        error_train[i-1] = linearRegCostFunction(X[:i], y[:i], theta)[0]
        error_val[i-1] = linearRegCostFunction(Xval, yval, theta)[0]
        
    # =============================================================
    return error_train, error_val


# X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)
# Xval_aug = np.concatenate([np.ones((yval.size, 1)), Xval], axis=1)
# error_train, error_val = learningCurve(X_aug, y, Xval_aug, yval, lambda_=0)

# pyplot.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2)
# pyplot.title('Learning curve for linear regression')
# pyplot.legend(['Train', 'Cross Validation'])
# pyplot.xlabel('Number of training examples')
# pyplot.ylabel('Error')
# pyplot.axis([0, 13, 0, 150])

# print('# Training Examples\tTrain Error\tCross Validation Error')
# for i in range(m):
#     print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

# pyplot.show()


def polyFeatures(X, p):
    """
    Maps X (1D vector) into the p-th power.
    
    Parameters
    ----------
    X : array_like
        A data vector of size m, where m is the number of examples.
    
    p : int
        The polynomial power to map the features. 
    
    Returns 
    -------
    X_poly : array_like
        A matrix of shape (m x p) where p is the polynomial 
        power and m is the number of examples. That is:
    
        X_poly[i, :] = [X[i], X[i]**2, X[i]**3 ...  X[i]**p]
    
    Instructions
    ------------
    Given a vector X, return a matrix X_poly where the p-th column of
    X contains the values of X to the p-th power.
    """
    # You need to return the following variables correctly.
    X_poly = np.zeros((X.shape[0], p))

    # ====================== YOUR CODE HERE ======================

    for i in range(p):
        X_poly[:, i] = X.T**(i+1)

    # ============================================================
    return X_poly


p = 8

# Map X onto Polynomial Features and Normalize
print(X[0])
X_poly = polyFeatures(X, p)
print(X_poly[0])
X_poly, mu, sigma = utils.featureNormalize(X_poly)
print(mu, sigma)
X_poly = np.concatenate([np.ones((m, 1)), X_poly], axis=1)

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.concatenate([np.ones((ytest.size, 1)), X_poly_test], axis=1)

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.concatenate([np.ones((yval.size, 1)), X_poly_val], axis=1)

# print('Normalized Training Example 1:')
# print(X_poly[0, :])


# lambda_ = 0
# theta = utils.trainLinearReg(linearRegCostFunction, X_poly, y,
#                              lambda_=lambda_, maxiter=55)

# # Plot training data and fit
# pyplot.plot(X, y, 'ro', ms=10, mew=1.5, mec='k')

# utils.plotFit(polyFeatures, np.min(X), np.max(X), mu, sigma, theta, p)

# pyplot.xlabel('Change in water level (x)')
# pyplot.ylabel('Water flowing out of the dam (y)')
# pyplot.title('Polynomial Regression Fit (lambda = %f)' % lambda_)
# pyplot.ylim([-20, 50])

# pyplot.figure()
# error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, lambda_)
# pyplot.plot(np.arange(1, 1+m), error_train, np.arange(1, 1+m), error_val)

# pyplot.title('Polynomial Regression Learning Curve (lambda = %f)' % lambda_)
# pyplot.xlabel('Number of training examples')
# pyplot.ylabel('Error')
# pyplot.axis([0, 13, 0, 100])
# pyplot.legend(['Train', 'Cross Validation'])

# print('Polynomial Regression (lambda = %f)\n' % lambda_)
# print('# Training Examples\tTrain Error\tCross Validation Error')
# for i in range(m):
#     print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

# pyplot.show()


def validationCurve(X, y, Xval, yval):
    """
    Generate the train and validation errors needed to plot a validation
    curve that we can use to select lambda_.
    
    Parameters
    ----------
    X : array_like
        The training dataset. Matrix with shape (m x n) where m is the 
        total number of training examples, and n is the number of features 
        including any polynomial features.
    
    y : array_like
        The functions values at each training datapoint. A vector of
        shape (m, ).
    
    Xval : array_like
        The validation dataset. Matrix with shape (m_val x n) where m is the 
        total number of validation examples, and n is the number of features 
        including any polynomial features.
    
    yval : array_like
        The functions values at each validation datapoint. A vector of
        shape (m_val, ).
    
    Returns
    -------
    lambda_vec : list
        The values of the regularization parameters which were used in 
        cross validation.
    
    error_train : list
        The training error computed at each value for the regularization
        parameter.
    
    error_val : list
        The validation error computed at each value for the regularization
        parameter.
    
    Instructions
    ------------
    Fill in this function to return training errors in `error_train` and
    the validation errors in `error_val`. The vector `lambda_vec` contains
    the different lambda parameters to use for each calculation of the
    errors, i.e, `error_train[i]`, and `error_val[i]` should give you the
    errors obtained after training with `lambda_ = lambda_vec[i]`.

    Note
    ----
    You can loop over lambda_vec with the following:
    
          for i in range(len(lambda_vec))
              lambda = lambda_vec[i]
              # Compute train / val errors when training linear 
              # regression with regularization parameter lambda_
              # You should store the result in error_train[i]
              # and error_val[i]
              ....
    """
    # Selected values of lambda (you should not change this)
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]

    # You need to return these variables correctly.
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))

    # ====================== YOUR CODE HERE ======================

    for i in range(len(lambda_vec)):
        lambda_ = lambda_vec[i]
        theta = utils.trainLinearReg(linearRegCostFunction, X, y, lambda_)
        error_train[i] = linearRegCostFunction(X, y, theta)[0]
        error_val[i] = linearRegCostFunction(Xval, yval, theta)[0]

    # ============================================================
    return lambda_vec, error_train, error_val





# lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

# pyplot.plot(lambda_vec, error_train, '-o', lambda_vec, error_val, '-o', lw=2)
# pyplot.legend(['Train', 'Cross Validation'])
# pyplot.xlabel('lambda')
# pyplot.ylabel('Error')

# print('lambda\t\tTrain Error\tValidation Error')
# for i in range(len(lambda_vec)):
#     print(' %f\t%f\t%f' % (lambda_vec[i], error_train[i], error_val[i]))

# pyplot.show()


theta = utils.trainLinearReg(linearRegCostFunction, X_poly, y, 3)
error_test, _ = linearRegCostFunction(X_poly_test, ytest, theta)

print(error_test)


def random_examples_error(X, y, Xval, Yval, lambda_=0, num_iter=50):
    m=X.shape[0]
    n=X.shape[1]

    error_train = np.zeros((num_iter, m))
    error_val = np.zeros((num_iter, m))
    for i in range(num_iter):
        train_data = np.c_[X,y.T]
        np.random.shuffle(train_data)
        X_ = train_data[:,:n]
        y_ = train_data[:,n:].T[0]

        val_data = np.c_[Xval,Yval.T]
        np.random.shuffle(val_data)
        Xval_ = val_data[:,:n]
        Yval_ = val_data[:,n:].T[0]

        error_train[i], error_val[i] =  learningCurve(X_,y_,Xval_,Yval_, lambda_=lambda_)
    print(error_train)
    avg_error_train = np.mean(error_train, axis=0)
    avg_error_val = np.mean(error_val, axis=0)

    return avg_error_train, avg_error_val
        


error_train, error_val = random_examples_error(X_poly, y, X_poly_val, yval, lambda_=0.01)

pyplot.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val, lw=2)
pyplot.title('Learning curve for linear regression')
pyplot.legend(['Train', 'Cross Validation'])
pyplot.xlabel('Number of training examples')
pyplot.ylabel('Error')
pyplot.axis([0, 13, 0, 150])

print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i+1, error_train[i], error_val[i]))

pyplot.show()