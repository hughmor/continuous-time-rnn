"""

    Author: Hugh Morison 
    
    This file contains functions for various numerical integration methods.
    All functions have the signature (f, y0, dt, t) and return the value of y(t+dt) given a function f=dy/dt, an initial value y0=y(t).

    TODO: Add different degrees of runge kutta method, other solvers

"""
import math
import numpy as np


def euler_step(f, y0, dt, t=0):
    """
    Custom Euler method with other inputs.

    Steps IVP $dy/dt=f(t,y)$ for one timestep using Euler method.

    :param f: Function with signature dy_dt = f(t, y, f_in[t])
    :param y0: Initial value of function.
    :param dt: Time interval of a step.
    :param t: Time at which to start IVP (for time dependant inputs/functions). Defaults to 0.
    :return: y(t+dt) the next value of y.
    """
    y = np.array(np.atleast_1d(y0))
    return y + dt*np.atleast_1d(f(t, y))


def runge_kutta_step(f, y0, dt, t=0):
    """
    Custom RK method with other inputs.

    Steps IVP $dy/dt=f(t,y)$ for one timestep using fourth-order Runge Kutta method.

    :param f: Function with signature dy_dt = f(t, y, f_in[t])
    :param y0: Initial value of function.
    :param dt: Time interval of a step.
    :param t: Time at which to start IVP (for time dependant inputs/functions). Defaults to 0.
    :return: y(t+dt) the next value of y.
    """
    y = np.array(np.atleast_1d(y0))
    k1 = np.atleast_1d(f(t, y))
    k2 = np.atleast_1d(f(t + dt/2.0, y + k1 * dt / 2.))
    k3 = np.atleast_1d(f(t + dt/2.0, y + k2 * dt / 2.))
    k4 = np.atleast_1d(f(t + dt, y + k3 * dt))

    return y + dt*(1./6.)*(k1+2.*k2+2.*k3+k4)


class IntegrationSolvers:
    solvers = {
        "rk4": runge_kutta_step,
        "runge kutta": runge_kutta_step,
        "euler": euler_step,
    }
    allowed_modes = list(solvers.keys())

    @classmethod
    def get(cls, mode):
        return cls.solvers[mode.lower()]
