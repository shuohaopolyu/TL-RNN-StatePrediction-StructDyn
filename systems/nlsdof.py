"""
Non-linear single degree of freedom system

Inputs:
    mass: float, mass of the system
    stiffness: float, stiffness of the system
    damping: float, damping of the system
    t_eval: array like, with shape (n_t, )
            time vector
    f_t: callable function of time that returns the force at time t
    init_cond: array like, with shape (2, ), initial conditions for the system
                init_cond[0] is the initial displacement
                init_cond[1] is the initial velocity
"""

import numpy as np
from scipy import integrate


class NonLinearSDOF:
    def __init__(
        self,
        mass,
        stiffness,
        damping,
    ):
        self.mass = mass
        self.stiffness = stiffness
        self.damping = damping

    def _ode(self, t, y, f_t):
        x, v = y
        dxdt = v
        dvdt = (f_t(t) - self.damping * v - self.stiffness**3 * x) / self.mass
        return dxdt, dvdt



    def integrate(self, t_eval, f_t, init_cond):
        """
        Integrate the system and return the displacement and velocity
        """
        # Initial conditions
        x0 = init_cond[0]
        v0 = init_cond[1]

        # Integrate
        sol = integrate.solve_ivp(
            fun=self._ode,
            t_span=(t_eval[0], t_eval[-1]),
            y0=(x0, v0),
            t_eval=t_eval,
            args=(f_t,),
        )
        return sol.y[0], sol.y[1]
