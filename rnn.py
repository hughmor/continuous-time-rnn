import numpy as np
from solvers import runge_kutta_step as rungekutta, euler_step as euler
from activation_functions import relu, sigmoid, softplus


class ContinuousRecurrentNetwork:

    def __init__(self, *args, **kwargs):
        self.integration_mode = kwargs.get('integration_mode', 'RK4')
        self.dt_solver = kwargs.get('time_step', 1.0e-6)
        self.solver_step = None
        self.__init_solver__()
        self.tau = kwargs.get('decay_constant', 1.0e-3)
        self.size = kwargs.get('n', 1)
        self.state_vectors = np.zeros(shape=self.size)
        self.activation_function = None
        self.__init_activation_function__(kwargs.get('Activation', 'Relu'))
        if kwargs.get('randomize_weights', False):
            self.weight_matrix = np.random.random(size=(self.size, self.size))
        else:
            self.weight_matrix = np.zeros(shape=(self.size, self.size))

    def step(self, *args, **kwargs):
        t = kwargs.get('current_time', 0)
        if self.solver_step is None:
            print('Solver Undefined')
        self.state_vectors = self.solver_step(self.ds_dt, self.state_vectors, self.dt_solver, t=t)

    def ds_dt(self, s):
        if self.activation_function is None:
            print('Activation Function Undefined')
        return -s/self.tau + np.dot(self.weight_matrix, self.activation_function(s))

    def __init_activation_function__(self, f_str):
        if f_str is 'Relu':
            self.activation_function = relu
        elif f_str is 'Sigmoid':
            self.activation_function = sigmoid
        else:
            print('Invalid Activation Function')

    def __init_solver__(self):
        if self.integration_mode == 'RK4':
            self.solver_step = rungekutta
        elif self.integration_mode == 'Euler':
            self.solver_step = euler
        else:
            print('Invalid Solver')
