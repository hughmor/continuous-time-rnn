import numpy as np
from collections import deque


class CTRNN:
    """
    Simulator for a continuous-time recurrent neural network (CTRNN). This class contains the parameters of the network and simulation,
    and has functionality to perform numerical integration of the network's differential equation.
    """
    def __init__(self, **kwargs):
        # Network parameters
        self.size = kwargs.get('number of neurons', 1)
        self.n_in = kwargs.get('number of inputs', 1)
        self.n_out = kwargs.get('number of outputs', 1)
        self.decay_constant = kwargs.get('decay constant', 1.0e-3) #TODO: Allow for array of taus (one for each neuron)
        if kwargs.get('randomize weights', True): #TODO: Allow for matrices to be provided
            self.weight_matrix = np.random.random(size=(self.size, self.size))
            self.input_weights = np.random.random(size=(self.n_in, self.n_in))
        else:
            self.weight_matrix = kwargs.get('weight matrix', np.ones(shape=(self.size, self.size)))
            self.input_weights = kwargs.get('input weights', np.ones(shape=(self.n_in, self.n_in)))

        # Integration
        self.t_seconds = 0.0
        self._step_solver = None
        self._int_mode = kwargs.get('integration mode', 'RK4')
        self._init_solver()
        
        # Activation functions
        self.activation_function = None
        self._act_mode = kwargs.get('activation', None)
        self._init_activation_function()

        # Initialize vectors
        self.neuron_vector = kwargs.get('initial state', np.zeros(shape=self.size))
        self.bias_vector = kwargs.get('biases', np.zeros(shape=self.size))
        self.state_vector = self.activation_function(self.neuron_vector + self.bias_vector)

        # Throw parameter errors
        self._enforce_parameter_constraints()

    @property
    def integration_mode(self):
        return self._int_mode

    @property
    def activation_mode(self):
        return self._act_mode

    @staticmethod
    def _ds_dt(s, tau, inpt):
        """Static Definition of the CTRNN equation"""
        return inpt - s / tau

    def ds_dt(self, t, s):
        """
        Evaluates the differential equation decribing the system with state s at time t
        
        Parameters:
            t (float): Evaluation time.
            s (ndarray): The state of the system.
        """
        return self._ds_dt(s, self.decay_constant, self._neuron_inputs)

    def reset(self):
        """Resets the state of the network (sets state and time to zeros)"""
        self.state_vector = np.zeros_like(self.state_vector)
        self.t_seconds = 0.0

    def simulate(self, x, dt):
        """
        Run the simulation over a given input vector with defined timestep between input samples

        Parameters:
            x (list): 2d input array with shape (number of inputs, number of samples)
            dt (float): timestep between each sample in seconds
        """
        self.reset()

        self.input_sequence = np.array(x)
        self.output_sequence = np.zeros(shape=(self.n_out, self.input_sequence.shape[1]))
        
        for i in range(self.input_sequence.shape[1]): # TODO: right now evaluation time is based on input length... change to time parameter as input
            next_output = self.advance(self.input_sequence[:,i], dt, dt)
            self.output_sequence[:,i] = next_output[-self.n_out:]
        
        return self.output_sequence

    def advance(self, inputs, evolution_time, dt):
        """
        Advance the simulation by a given amount of time with a constant input over this time (weight inputs, integrate system of equations, and apply activation function)
        """
        end_time = self.t_seconds + evolution_time
        n_in = len(inputs)
        if self.n_in != n_in:
            raise RuntimeError(f"Expected input vecor of length {self.n_in}, got {n_in}")

        while self.t_seconds < end_time:
            dt = min(dt, end_time - self.t_seconds)
            self._neuron_inputs = np.dot(self.weight_matrix, self.state_vector) # Recurrent connections from previous state
            self._neuron_inputs[:n_in] += np.dot(self.input_weights, np.atleast_1d(inputs)) # Input connections (non-recurrent)
            self.neuron_vector = self._step_solver(self.ds_dt, self.neuron_vector, dt, t=self.t_seconds) # Integrate system with chosen IVP solver
            self.state_vector = self.activation_function(self.neuron_vector + self.bias_vector) # Apply activation function (offset by bias)
            self.t_seconds += dt
        return self.state_vector

    def _init_activation_function(self):
        activation_mode = self.activation_mode.lower()
        if activation_mode == 'relu' or activation_mode == 'rectified linear unit':
            from activation_functions import relu
            self.activation_function = relu
        elif activation_mode == 'sigmoid':
            from activation_functions import sigmoid
            self.activation_function = sigmoid
        elif activation_mode == 'softplus':
            from activation_functions import softplus
            self.activation_function = softplus
        elif activation_mode == 'linear' or activation_mode is None:
            from activation_functions import linear
            self.activation_function = linear
        elif activation_mode == 'tanh' or activation_mode == 'hyperbolic tangent':
            from activation_functions import tanh
            self.activation_function = tanh
        else:
            raise ValueError('Invalid Activation Function: ' + str(self.activation_mode))

    def _init_solver(self):
        if self.integration_mode == 'RK4':
            from solvers import runge_kutta_step as rungekutta
            self._step_solver = rungekutta
        elif self.integration_mode == 'Euler':
            from solvers import euler_step as euler
            self._step_solver = euler
        else:
            raise ValueError('Invalid Solver: ' + str(self.integration_mode))

    def _enforce_parameter_constraints(self):
        assert self.n_in >= 0, 'Number of inputs must be 0 or greater'
        assert self.n_out >= 0, 'Number of outputs must be 0 or greater'
        assert self.n_in + self.n_out <= 2*self.size
