"""

    Author: Hugh Morison
    
    This file contains some common activation functions for use in ANNs.
    All functions operate on an input $x$ (scalar, list, or numpy array) and return the output in a numpy array.

    TODO: add input magnitude contraints? most functions are periodic or saturating...
    TODO: add log, exp -> these need reasonable bounds
"""
import numpy as np


def linear(x, a=1, x0=0):
    """
    Linear Unit Function.

    :param x: input vector
    :param a: linear slope
    :param x0: bias position (y(x0)=0)
    :return: output of a*(x-x0)
    """
    x = np.atleast_1d(x) - x0
    return a * x


def relu(x, a=1, x0=0):
    """
    Rectified Linear Unit Function.

    :param x: input vector
    :param a: linear slope
    :param x0: center position (bias)
    :return: output of a*max(0,x-x0)
    """
    x = np.atleast_1d(x) - x0
    return a * np.maximum(np.zeros_like(x), x)


def elu(x, a=1, x0=0):
    """
    Exponential Linear Unit Function.

    :param x: input vector
    :param a: linear slope
    :param x0: center position (bias)
    :return: output of a*min(max(0,x-x0),exp(x-x0)-1)
    """
    x = np.atleast_1d(x) - x0
    return a * np.minimum(np.maximum(np.zeros_like(x), x), np.exp(x) - 1)


def softplus(x, a=1, x0=0):
    """
    Softplus Function.

    :param x: input vector
    :param a: gain
    :param x0: center position (bias)
    :return: output of a*log(1+exp(x-x0))
    """
    x = np.atleast_1d(x) - x0
    return a * np.log(1+np.exp(x))


def sigmoid(x, a=1, b=1, x0=0):
    """
    Sigmoid Function.

    :param x: input vector
    :param a: gain
    :param b: rate
    :param x0: center position (bias)
    :return: output of a/(1+exp(-b*(x-x0)))
    """
    x = np.atleast_1d(x) - x0
    return a / (1+np.exp(-b*x))

def lorentzian(x, x0=0, width=1):
    """
    Lorentzian Lineshape Function.

    :param x: input vector
    :return: output of 1/(1+x^2)
    """
    x = np.atleast_1d(x)
    x = 2 * (x - x0) / width
    return 1-1/(1 + x*x)


def tanh(x, a=1, b=1, x0=0):
    """
    Hyperbolic Tangent Function.

    :param x: input vector
    :param a: gain
    :param b: rate
    :param x0: center position (bias)
    :return: output of a*tanh(b(x-x0))
    """
    x = np.atleast_1d(x) - x0
    return a * np.tanh(b*x)

def sin_sq(x, phi, frq):
    """
    Sinusoidal Function.

    :param x: input vector
    :return: output of sin(x)
    """
    x = np.atleast_1d(x)
    return 0.5 * (np.sin(frq*(x+phi)) + 1)

def sin(x, a=1, k=1, x0=0):
    """
    Sinusoidal Function.

    :param x: input vector
    :param a: gain
    :param k: frequency
    :param x0: bias
    :return: output of a*sin(k*(x-x0))
    """
    x = np.atleast_1d(x) - x0
    return a*np.sin(k*x)

# TODO: continue parameters from here onward

def clamp(x):
    """
    Clamped Function.

    :param x: input vector
    :return: output of max(-1, min(1, x))
    """
    x = np.atleast_1d(x)
    one = np.ones_like(x)
    return np.maximum(-1*one, np.minimum(one, x))


def square(x):
    """
    Square Function.

    :param x: input vector
    :return: output of x*x
    """
    x = np.atleast_1d(x)
    return x*x


def cube(x):
    """
    Cube Function.

    :param x: input vector
    :return: output of x*x*x
    """
    x = np.atleast_1d(x)
    return x*x*x


def absolute(x):
    """
    Absolute Value Function.

    :param x: input vector
    :return: output of |x|
    """
    x = np.atleast_1d(x)
    return np.abs(x)


def hat(x):
    """
    Hat Function.

    :param x: input vector
    :return: output of max(0, 1-|x|)
    """
    x = np.atleast_1d(x)
    return np.maximum(np.zeros_like(x), 1-np.abs(x))

