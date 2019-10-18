import math
import numpy as np


def euler_step(f, y0, dt, t=0, **kwargs):
    """
    Custom Euler method with other inputs.

    Steps IVP $dy/dt=f(t,y)$ for one timestep using Euler method.

    :param f: Function with signature dy_dt = f(t, y, f_in[t])
    :param y0: Initial value of function.
    :param dt: Time interval of a step.
    :param t: Time at which to start IVP (for time dependant inputs/functions). Defaults to 0.
    :return: y(t+dt) the next value of y.
    """
    y0 = np.array(np.atleast_1d(y0))
    return y0 + dt*np.atleast_1d(f(t, y0))


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
    y0 = np.array(np.atleast_1d(y0))
    k1 = np.atleast_1d(f(t, y0))
    k2 = np.atleast_1d(f(t + dt/2.0, y0 + k1 * dt / 2.))
    k3 = np.atleast_1d(f(t + dt/2.0, y0 + k2 * dt / 2.))
    k4 = np.atleast_1d(f(t + dt, y0 + k3 * dt))

    return y0 + dt*(1./6.)*(k1+2.*k2+2.*k3+k4)
