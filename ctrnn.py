import numpy as np
from collections import deque


class CTRNN_Layer:
    """
        Simulator for a continuous-time recurrent neural network (CTRNN). This class contains the parameters of the network and simulation,
        and has functionality to perform numerical integration of the network's differential equation.
    """
    def __init__(self, size, n_in=None, n_out=None, initial_state=None, bias=0.0, gain=1.0, tau=1.0, weights_rec=None, weights_in=None, **kwargs):
        self.size = size
        self.n_in = n_in
        self.n_out = n_in
        if n_in is None:
            self.n_in = size
        if n_out is None:
            self.n_out = size
        self.bias_vector = bias * np.ones(shape=self.size)
        self.gain_vector = gain * np.ones(shape=self.size)
        self.tau_vector = tau * np.ones(shape=self.size)
        self.state_vector = self.bias_vector.copy()
        if initial_state is not None:
            self.state_vector = initial_state
        
        self.weight_matrix = np.random.random(size=(self.size, self.size))
        self.input_weights = np.random.random(size=(self.size, self.n_in))
        if weights_rec is not None:
            if weights_rec.shape != (self.size, self.size):
                raise ValueError('Recurrent weights have incorrect shape: ' + str(weights_rec.shape))
            self.weight_matrix = weights_rec
        if weights_in is not None:
            if weights_in.shape != (self.size, self.n_in):
                raise ValueError('Input weights have incorrect shape: ' + str(weights_in.shape))
            self.input_weights = weights_in


        # Simulator time
        self.t_seconds = 0.0
        self.dt_solver = kwargs.get('time step', 1.0e-6)

        # Integration
        self._step_solver = None
        self.integration_mode = kwargs.get('integration mode', 'RK4')
        self._init_solver()
        
        # Activation functions
        self.activation_function = None
        self.activation_mode = kwargs.get('activation', None)
        self._init_activation_function()

        self.output_vector = self.activation_function(self.state_vector)
        self.input_sequence = None

        # Throw parameter errors
        self._enforce_parameter_constraints()

    def reset(self):
        self.values = np.zeros_like(self.state_vector)
        self.t_seconds = 0.0

    def advance(self, x):
        x = np.array(x)
        assert self.n_in in x.shape, 'Input vector has incorrect shape: ' + str(x.shape)
        if x.shape[1] == self.n_in:
            x = x.transpose()
        self.input_sequence = x
        self.output_sequence = np.zeros(shape=(self.n_out, self.input_sequence.shape[1]))
        for i in range(self.input_sequence.shape[1]): # TODO: right now evaluation time is based on input length... change to time parameter as input
            self.step()
            self.output_sequence[:,i] = self.state_vector[-self.n_out:]
        
    def step(self):
        self._neuron_inputs = self._get_in_vector() # Input comes from recurrent units plus the external input
        self.state_vector = self._step_solver(self.ds_dt, self.state_vector, self.dt_solver, t=self.t_seconds) # Integrate system with chosen IVP solver
        self.output_vector = self.activation_function(self.gain_vector * self.state_vector + self.bias_vector) # Apply activation function (offset by bias)
        self.t_seconds += self.dt_solver
        
    def _get_in_vector(self):
        neuron_in = np.dot(self.weight_matrix, self.output_vector)
        if self.input_sequence is not None:
            raise NotImplementedError('Input sequence not implemented yet. Shouldnt get here...')
            i = int(self.t_seconds/self.dt_solver)-1
            inp = np.dot(self.input_weights, self.input_sequence[:,i])
            neuron_in[:len(inp)] += inp
        return neuron_in

    def ds_dt(self, t, s):
        decay = -s
        weighted_update = self._neuron_inputs
        update = decay + weighted_update
        return update/self.decay_constant

    def _init_activation_function(self):
        activation_mode = self.activation_mode # .lower() TODO: PUT THIS BACK LATER
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
        elif type(activation_mode == list): #TODO: inspect function to check number of params or names (use kwargs?)
            if activation_mode[0].lower() == 'lorentzian':
                from activation_functions import lorentzian
                self.activation_parameters = activation_mode[1:]
                self.activation_function = lambda x: lorentzian(x, *self.activation_parameters)
            elif activation_mode[0].lower() == 'sin' or activation_mode[0].lower() == 'sinusoidal':
                from activation_functions import sin
                self.activation_parameters = activation_mode[1:]
                self.activation_function = lambda x: sin(x, *self.activation_parameters)
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
