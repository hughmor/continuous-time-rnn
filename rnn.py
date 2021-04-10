import numpy as np
from collections import deque


class CTRNN:
    """
        Simulator for a continuous-time recurrent neural network (CTRNN). This class contains the parameters of the network and simulation,
        and has functionality to perform numerical integration of the network's differential equation.
    """
    def __init__(self, **kwargs):
        # Neuron states
        self.size = kwargs.get('number of neurons', 1)
        self.n_in = kwargs.get('number of inputs', 1)
        self.n_out = kwargs.get('number of outputs', 1)
        self.state_vector = kwargs.get('initial state', np.zeros(shape=self.size))
        self.bias_vector = kwargs.get('biases', np.zeros(shape=self.size))
        
        # Network parameters
        self.decay_constant = kwargs.get('decay constant', 1.0e-3) #TODO: Allow for array of taus (one for each neuron)
        if kwargs.get('randomize weights', True): #TODO: Allow for matrices to be provided
            self.weight_matrix = np.random.random(size=(self.size, self.size))
            self.input_weights = np.random.random(size=(self.n_in, self.n_in))
        else:
            self.weight_matrix = kwargs.get('weight matrix', np.ones(shape=(self.size, self.size)))
            self.input_weights = kwargs.get('input weights', np.ones(shape=(self.n_in, self.n_in)))

        # Simulator time
        self.t = 0.0
        self.dt_solver = kwargs.get('time step', 1.0e-6)

        # Integration
        self.step_solver = None
        self.integration_mode = kwargs.get('integration mode', 'RK4')
        self._init_solver()
        
        # Activation functions
        self.activation_function = None
        self.activation_mode = kwargs.get('activation', None)
        self._init_activation_function()

        # Delay line
        self.delay_line = deque()
        self.delay_time = kwargs.get('delay iterations', 2) * self.dt_solver

        self._enforce_parameter_constraints()

    def forward(self, x):
        x = np.array(x)
        assert self.n_in in x.shape, 'Input vector has incorrect shape: ' + str(x.shape)
        if x.shape[1] == self.n_in:
            x = x.transpose()
        self.input_sequence = x
        self.output_sequence = np.zeros(shape=(self.n_out, self.input_sequence.shape[1]))
        self.take_from_delay_line = False
        for i in range(self.input_sequence.shape[1]):
            if not self.take_from_delay_line and self.t >= self.delay_time:
                self.take_from_delay_line = True
            self.step()
            self.output_sequence[:,i] = self.state_vector[-self.n_out:]
        
    def step(self):
        self.delay_line.append(self.activation_function(self.state_vector + self.bias_vector))
        self.neuron_inputs = self.calc_in_vector()
        self.state_vector = self.step_solver(self.ds_dt, self.state_vector, self.dt_solver, t=self.t)
        self.t += self.dt_solver
        
    def calc_in_vector(self):
        neuron_in = np.dot(self.weight_matrix, self.delay_line.popleft()) if self.take_from_delay_line else np.zeros_like(self.state_vector)
        if self.input_sequence is not None:
            i = int(self.t/self.dt_solver)-1
            inp = np.dot(self.input_weights, self.input_sequence[:,i])
            neuron_in[:len(inp)] += inp
        return neuron_in

    def ds_dt(self, t, s):
        decay = -s/self.decay_constant
        weighted_update = self.neuron_inputs
        update = decay + weighted_update
        return update

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
            self.step_solver = rungekutta
        elif self.integration_mode == 'Euler':
            from solvers import euler_step as euler
            self.step_solver = euler
        else:
            raise ValueError('Invalid Solver: ' + str(self.integration_mode))

    def _enforce_parameter_constraints(self):
        assert self.n_in + self.n_out <= 2*self.size
