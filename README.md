# continuous-time-rnn

# About
This is a package for emulating recurrent neural networks using continuous-time equations. The equation for a CTRNN is `ds/dt = -s/tau + W f(s) + W_in x_in`, where `s` is a vector representing neuron states, `x_in` is an input vector, `tau` is the decay constant of a neuron, `f` is a nonlinear activation function, and `W` and `W_in` are weight matrices.

The equations are solved as an initial value problem using numerical integration. Available solvers in this package are the Euler method, and 4-th order Runge-Kutta algorithms.

Many common ML activation functions are implemented, and you can also define custom activation functions.

# Example
A CTRNN instance can be created from a parameter dictionary.

```
import numpy as np
from ctrnn import CTRNN

n_neurons = 24
n_in = 3
params = {
    'number of neurons': n_neurons,
    'number of inputs': n_in,
    'number of outputs': n_neurons,
    'decay constant': 0.01,
    'weight matrix': np.random.random(size=(n_neurons,n_neurons)),
    'input weights': np.array([
        [1.,0.,0.],
        [0.,2.,0.],
        [0.,0.,3.]
    ]),
    'integration mode': 'RK4',
    'activation': 'ReLU',
    'initial state': np.ones(shape=n_neurons),
    'biases': np.zeros(shape=n_neurons),
}

nn = CTRNN(**params)

sim_time_seconds = 1.0
N = 100
freq = 5
input1 = [np.sin(2. * np.pi * freq * (x/T) ) for x in range(N)]
input2 = [np.cos(2. * np.pi * freq * (x/T) ) for x in range(N)]
input3 = [one + two for one,two in zip(input1, input2)]
inputs = [input1, input2, input3]

dt = sim_time_seconds / N
nn.simulate(inputs, dt)

```
