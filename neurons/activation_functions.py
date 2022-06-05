"""

    Author: Hugh Morison
    
    This file contains some common activation functions for use in ANNs.
    All functions operate on an input $x$ (scalar, list, or numpy array) and return the output in a numpy array.

    TODO: add input magnitude contraints? most functions are periodic or saturating...
    TODO: add log, exp -> these need reasonable bounds
"""
import numpy as np
from numpy import maximum as max


def linear(x, a=1, b=0):
    """
    Linear Unit Function.

    :param x: input vector
    :param a: linear slope
    :param b: y-intercept
    :return: output of a*x+b
    """
    x = np.atleast_1d(x)
    return a * x + b


def relu(x, a=1, x0=0):
    """
    Rectified Linear Unit Function.

    :param x: input vector
    :param a: linear slope
    :param x0: center position (bias)
    :return: output of a*max(0,x-x0)
    """
    x = np.atleast_1d(x) - x0
    return a * max(np.zeros_like(x), x)


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


class ActivationFunctions:
    functions = {
        "linear": linear,
        "linear unit": linear,

        "relu": relu,
        "rectified linear unit": relu,

        "elu": elu,
        "exponential linear unit": elu,
        
        "softplus": softplus,
        "soft plus": softplus,

        "sigmoid": sigmoid,

        "tanh": tanh,
        "hyperbolic tangent": tanh,

        "sin": sin,
        "sinusoid": sin,
        "sinusoidal": sin,

        "clamp": clamp,

        "square": square,
        "quadratic": square,
        "x^2": square,
        "cube": cube,
        "cubic": cube,
        "x^3": cube,
        
        "abs": absolute,
        "absolute value": absolute,

        "hat": hat,
    }
    allowed_modes = list(functions.keys())

    @classmethod
    def get(cls, mode):
        return cls.functions[mode.lower()]

    @classmethod
    def register(cls, func, name):
        cls.functions[name] = func #TODO: must check that the function matches the required signature and output type
        cls.allowed_modes.append(name)
        print(f'Registered custom function with name {name}.')
