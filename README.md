# continuous-time-rnn

# About
This is a package for emulating recurrent neural networks using continuous-time equations. The equation for a CTRNN is `ds/dt = -s/tau + W f(s) + W_in x_in`, where `s` is a vector representing neuron states, `x_in` is an input vector, `tau` is the decay constant of a neuron, `f` is a nonlinear activation function, and `W` and `W_in` are weight matrices.

The equations are solved as an initial value problem using numerical integration. Available solvers in this package are the Euler method, and 4-th order Runge-Kutta algorithms.

Many common ML activation functions are implemented, and you can also define custom activation functions.
