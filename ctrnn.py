import numpy as np
from collections import deque
from typing import Union, Any, Callable
import multiprocessing as mp
import concurrent.futures
from scipy.integrate import solve_ivp
import time

from sympy import dsolve
from neurons import NeuronType, Yamada, LeakyIntegrate

MULTIPROCESS=False # if changing to true, currently broken

class CTRNN_Layer:
    """
        Simulator for a continuous-time recurrent neural network (CTRNN). This class contains the parameters of the network and simulation,
        and has functionality to perform numerical integration of the network's differential equation.
    """
    def __init__(
            self,
            size:                   int,
            n_in:                   int                         = None,
            n_out:                  int                         = None,
            neuron_type:            NeuronType                  = LeakyIntegrate,
            dt:                     float                       = 0.01,
            delay:                  float                       = 0.0,
            weights_rec:            np.ndarray                  = None,
            weights_in:             np.ndarray                  = None,
            solver_mode:            str                         = None,
            **kwargs,
        ) -> None:
        """A recurrently connected continuous-time neural network (CTRNN) layer. This class contains the parameters of the network and simulation,

        Args:
            size (int): Number of neurons in the layer
            n_in (int, optional): Number of external inputs to the neurons. Defaults to size.
            n_out (int, optional): Number of output neurons. Defaults to size.
            neuron_type (NeuronType, optional): The type of neuron to use. Defaults to LeakyIntegrate.
            dt (float, optional): The IVP solver timestep. Defaults to 0.01.
            delay (float, optional): The recurrence delay of the layer. Defaults to 0.0.
            weights_rec (np.ndarray, optional): The recurrent weights of the layer. Must have shape (size, size). Defaults to a random matrix.
            weights_in (np.ndarray, optional): The input weights to the layer. Must have shape (size, n_in) Defaults to a random matrix.
            solver_mode (str, optional): Type of IVP solver to use. This is passed to `scipy.solve_ivp`; see scipy documentation for more info. Defaults to 'RK45'.
            **kwargs: Additional keyword arguments are passed to the neuron type constructor.
        """        
        # Network parameters
        self.size = size
        self.n_in = n_in
        self.n_out = n_in
        if n_in is None:
            self.n_in = size
        if n_out is None:
            self.n_out = size
                       
        self.weight_matrix = np.random.random(size=(self.size, self.size))
        if weights_rec is not None:
            if weights_rec.shape != (self.size, self.size):
                raise ValueError('Recurrent weights have incorrect shape: ' + str(weights_rec.shape))
            self.weight_matrix = weights_rec

        # Simulator time
        self.t_seconds = 0.0
        self.dt_solver = dt
        self.delay = delay

        # Integration
        self.integration_mode = solver_mode
        if solver_mode is None:
            self.integration_mode = 'RK45'
        self._procs = []
        
        # Activation functions
        self.neurons = [neuron_type(index=i, **kwargs) for i in range(self.size)]

        self._input_nodes = []

    def reset(self):
        self.t_seconds = 0.0
        for neuron in self.neurons:
            neuron.reset()

    def solve(self, duration):
        """
        Solve the CTRNN layer's differential equation.

        Args:
            duration (float): The duration of the simulation in seconds.
            input_sequence (np.ndarray, optional): The input sequence to the layer. Must have shape (n_in, n_steps). Defaults to None.
        """

        self.reset()
        while self.t_seconds < duration:
            self.step()

    def step(self):
        if MULTIPROCESS:
            pool = mp.Pool(4)
            curr_state = self.state_vector(self.t_seconds)
            self._procs = [
                pool.apply_async(
                        solve_ivp,
                        args=(neuron.ds_dt),
                        kwds={
                            "t_span": (self.t_seconds, self.t_seconds + self.dt_solver),
                            "y0": (curr_state[i]),
                            "method": self.integration_mode,
                            "args": ((self._neuron_inputs[i],)),
                            "jac": None, # TODO: compute jacobians in neuron classes
                            },
                        #name=f'neuron_{i+1}',
                    ) for i,neuron in enumerate(self.neurons)
                ]
            results = [p.get() for p in self._procs]
            pool.close()
        else:
            results = solve_ivp(
                    self.ds_dt, #neuron.ds_dt,
                    t_span=(self.t_seconds, self.t_seconds + self.dt_solver),
                    y0=self.state_vector(self.t_seconds),
                    method=self.integration_mode,
                    #args=((self._neuron_inputs[i],)),
                )

            if not results.success:
                raise ValueError('Integration failed: ' + results.message)

            for i,neuron in enumerate(self.neurons):
                neuron.record_history(results.t, results.y[neuron.N_DIMS*i : neuron.N_DIMS*i+neuron.N_DIMS, :])

        self.t_seconds += self.dt_solver
        
    def ds_dt(self, t, state):
        ds = np.concatenate( list(neuron.ds_dt(t, state[neuron.N_DIMS*i : neuron.N_DIMS*i+neuron.N_DIMS], inp)
            for i, neuron, inp in zip(range(self.size), self.neurons, self._get_in_vector())) )
        return ds 

    def state_vector(self, t):
        return np.array([neuron.state(t) for neuron in self.neurons])

    def out_vector(self, t):
        return np.array([neuron.output(t) for neuron in self.neurons])

    def _get_in_vector(self):
        recurrent_in = self.out_vector(self.t_seconds-self.delay)
        neuron_in = np.einsum('ij,j', self.weight_matrix, recurrent_in)
    
        for inp in self._input_nodes:
            if inp.shape != (self.size,):
                raise ValueError('Input shape does not match layer size')
            neuron_in += inp

        # TODO: add general framework for inputs from other layers or external sources to contribute to the input to the incoming signal
        return neuron_in

    def add_input(self, input_node):
        self._input_nodes.append(input_node)

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
        raise NotImplementedError('Removed custom solvers to use solve_ivp instead.')
        if self.integration_mode == 'RK4':
            from solvers import runge_kutta_step as rungekutta
            self._step_solver = rungekutta
        elif self.integration_mode == 'Euler':
            from solvers import euler_step as euler
            self._step_solver = euler
        else:
            raise ValueError('Invalid Solver: ' + str(self.integration_mode))