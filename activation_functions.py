import numpy as np


def relu(x):
    return max(0, x)


def softplus(x):
    return np.log(1+np.exp(x))


def sigmoid(x):
    return 1/(1+np.exp(-x))

