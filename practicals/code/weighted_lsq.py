import numpy as np

def weighted_lsq(X, y, d):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """
    W = np.diag(d)
    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    XTWX = np.dot(X.T, np.dot(W, X))
    XTWy = np.dot(X.T, np.dot(W, y))
    beta = np.dot(np.linalg.inv(np.dot(XTWX, X)), np.dot(XTWy, y))

    return beta
