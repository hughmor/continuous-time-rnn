from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import activation_functions as activations

class NeuronType(ABC):
    @abstractmethod
    def ds_dt(self, t, s, inp):
        ...
    
    @abstractmethod
    def state(self, t):
        ...

    @abstractmethod
    def output(self, t):
        ...

    def record_history(self, t,y):
        if self.t is None and self.y is None:
            self.t = t
            self.y = y
        elif self.t is None or self.y is None:
            raise Exception('t or y is None but not the other. This should not happen.')
        else:
            self.t = np.concatenate((self.t, t[1:]))
            self.y = np.concatenate((self.y, y[:, 1:]), axis=1)

        #print(f'recording_history for neuron {self._idx}:\n\t{self.t.shape}\n\t{self.y.shape}')

class LeakyIntegrate(NeuronType):
    
    N_DIMS = 1
    
    def __init__(self, index, tau=1.0, bias=0.0, gain=1.0, power=1.0, initial_state=None, activation='relu', activation_params=None):
        self._idx = index

        if isinstance(tau, (int, float)):
            self.tau = tau
        elif isinstance(tau, (list, np.ndarray)):
            self.tau = tau[index]

        if isinstance(bias, (int, float)):
            self.bias = bias
        elif isinstance(bias, (list, np.ndarray)):
            self.bias = bias[index]

        if isinstance(gain, (int, float)):
            self.gain = gain
        elif isinstance(gain, (list, np.ndarray)):
            self.gain = gain[index]

        if isinstance(power, (int, float)):
            self.power = power
        elif isinstance(power, (list, np.ndarray)):
            self.power = power[index]

        if initial_state is None:
            initial_state = self.bias

        if isinstance(initial_state, (int, float)):
            self._initial_state = initial_state
        elif isinstance(initial_state, (list, np.ndarray)):
            self._initial_state = initial_state[index]
        
        if isinstance(activation, str):
            self.activation = activations.get_activation_function(activation)
        elif isinstance(activation, Callable):
            self.activation = activation
        self._activation_pars = activation_params
        if activation_params is None:
            self._activation_pars = ()

        self.reset()

    def reset(self):
        self.t = None
        self.y = None

    def ds_dt(self, t, s, inp):        
        f = -(s-self.bias) + self.gain*inp # tau \dot{s} = f(s)
        return f/self.tau

    def state(self, t):
        if self.t is None or t < self.t[0]:
            return self._initial_state
        # interpolate self.y with t and self.t        
        return np.interp(t, self.t, self.y[0,:])

    def output(self, t):
        return self.power * self.activation(self.state(t), *self._activation_pars)
    

class Yamada(NeuronType):

    N_DIMS = 3

    def __init__(
            self,
            index,
            bias_current=4.3,
            absorbtion=3.52,
            diff_absorption=1.8,
            gain_relax_rate=0.05,
            absorb_relax_rate=1.0,
            inv_photon_lifetime=1.0,
            input_gain_factor=1.0,
            initial_state=None
            ):
        self._idx = index

        self.G = 0.0
        self.Q = 0.0
        self.I = 0.0
        if initial_state is not None:
            self.G, self.Q, self.I = initial_state

        self.A = bias_current
        self.a = diff_absorption
        self.B = absorbtion
        self.g = input_gain_factor
        self.Gamma_G = gain_relax_rate
        self.Gamma_Q = absorb_relax_rate
        self.Gamma_I = inv_photon_lifetime

        self.reset()


    def ds_dt(self, t, s, inp):
        G,Q,I = s
        Gamma_G, Gamma_Q, Gamma_I = self.Gamma_G, self.Gamma_Q, self.Gamma_I
        A, a, B, g = self.A, self.a, self.B, self.g
        return (
            Gamma_G * (A - G - G*I) + g*inp,
            Gamma_Q * (B - Q - a*Q*I),
            Gamma_I * (G - Q - 1) * I,
        )

    def get_state(self, t):
        return self.I
