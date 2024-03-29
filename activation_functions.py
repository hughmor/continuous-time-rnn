"""

    Author: Hugh Morison
    
    This file contains some common activation functions for use in ANNs.
    All functions operate on an input $x$ (scalar, list, or numpy array) and return the output in a numpy array.

    TODO: add input magnitude contraints? most functions are periodic or saturating...
    TODO: add log, exp -> these need reasonable bounds
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


def elu(x):
    """
    Exponential Linear Unit Function.

    :param x: input vector
    :return: output of min(max(0,x),exp(x)-1)
    """
    x = np.atleast_1d(x)
    return np.minimum(np.maximum(np.zeros_like(x), x), np.exp(x) - 1)


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
    return 1/(1+np.exp(-1*x))


def tanh(x):
    """
    Hyperbolic Tangent Function.

    :param x: input vector
    :return: output of tanh(x)
    """
    x = np.atleast_1d(x)
    return np.tanh(x)


def sin(x):
    """
    Sinusoidal Function.

    :param x: input vector
    :return: output of sin(x)
    """
    x = np.atleast_1d(x)
    return np.sin(x)


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
