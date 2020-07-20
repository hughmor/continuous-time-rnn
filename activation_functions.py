"""

    Author: Hugh Morison
    
    This file contains some common activation functions for use in ANNs.
    All functions operate on an input $x$ (scalar, list, or numpy array) and return the output in a numpy array.

"""
import numpy as np

def linear(x):
    """
    Linear Unit Function.

    :param x: input vector
    :return: output of x
    """
    x = np.atleast_1d(x)
    return x

def relu(x):
    """
    Rectified Linear Unit Function.

    :param x: input vector
    :return: output of max(0,x)
    """
    x = np.atleast_1d(x)
    return np.maximum(np.zeros_like(x), x)


def softplus(x):
    """
    Softplus Function.

    :param x: input vector
    :return: output of log(1+exp(x))
    """
    x = np.atleast_1d(x)
    return np.log(1+np.exp(x))


def sigmoid(x):
    """
    Sigmoid Function.

    :param x: input vector
    :return: output of 1/(1+exp(-x))
    """
    x = np.atleast_1d(x)
    return 1/(1+np.exp(-x))

