import numpy as np
from collections import deque
from solvers import IntegrationSolvers
from activation_functions import ActivationFunctions

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
        self.set_integration_mode(kwargs.get('integration mode', 'RK4'))
        self._init_solver()
        
        # Activation functions
        self.activation_function = None
        self.set_activation_mode(kwargs.get('activation', 'linear'))
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

    def set_integration_mode(self, mode):
        """
        Setter for integration method to use during simulation.
        """
        if mode not in IntegrationSolvers.allowed_modes:
            raise ValueError(f"Invalid integration solver {mode}. Must be one of: {IntegrationSolvers.allowed_modes}")
        self._int_mode = mode

    @property
    def activation_mode(self):
        return self._act_mode

    def set_activation_mode(self, mode):
        """
        Setter for activation funtion type to use during evaluation.
        """
        if mode not in ActivationFunctions.allowed_modes:
            raise ValueError(f"Invalid activation function {mode}. Must be one of: {ActivationFunctions.allowed_modes}")
        self._act_mode = mode

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
        self.activation_function = ActivationFunctions.get(self.activation_mode)

    def _init_solver(self):
        self._step_solver = IntegrationSolvers.get(self.integration_mode)

    def _enforce_parameter_constraints(self):
        assert self.n_in >= 0, 'Number of inputs must be 0 or greater'
        assert self.n_out >= 0, 'Number of outputs must be 0 or greater'
        assert self.n_in + self.n_out <= 2*self.size
