"""
Shear type structure under ground motion
Initial state is assumed to be zero

Input:
    mass_vec: array like, with shape (n, )
                    mass of the superstructure
    stiff_vec: array like, with shape (n, )
                    stiffness of the superstructure
    damp_type:  currently only support Rayleigh
    damp_params: tuple, tuple, (lower_mode_index, upper_mode_index, damping_ratio)
    t: array like, with shape (n_t, )
            time vector
    acc_g: array like, with shape (n_t, )
                ground acceleration

Output:
    resp_disp: array like, with shape (n, n_t)
                displacement of the structure, first element is the base displacement, the rest are superstructure displacement
    resp_vel: array like, with shape (n, n_t)
                velocity of the structure, first element is the base velocity, the rest are superstructure velocity
    resp_acc: array like, with shape (n, n_t)
                acceleration of the structure, first element is the base acceleration, the rest are superstructure acceleration

"""

import numpy as np
from systems.lsds import MultiDOF
from scipy import interpolate


class ShearTypeStructure(MultiDOF):
    def __init__(
        self,
        mass_vec,
        stiff_vec,
        damp_vec=None,
        damp_type="d_mtx",
        t=None,
        acc_g=None,
    ):
        self.mass_vec = mass_vec
        self.acc_g = acc_g
        self.acc_func = interpolate.interp1d(t, acc_g)
        mass_mtx = np.diag(mass_vec)
        stiff_mtx = (
            np.diag(stiff_vec)
            + np.diag(-stiff_vec[1:], 1)
            + np.diag(-stiff_vec[1:], -1)
            + np.diag(np.append(stiff_vec[1:], 0))
        )
        damp_mtx = (
            np.diag(damp_vec)
            + np.diag(-damp_vec[1:], 1)
            + np.diag(-damp_vec[1:], -1)
            + np.diag(np.append(damp_vec[1:], 0))
        )
        super().__init__(
            mass_mtx,
            stiff_mtx,
            damp_type=damp_type,
            damp_params=damp_mtx,
            f_dof=[i for i in range(len(mass_vec))],
            t_eval=t,
            f_t=[
                lambda t: -self.acc_func(t) * self.mass_vec[i]
                for i in range(len(mass_vec))
            ],
            init_cond=None,
        )

    def run(self, method="Radau"):
        full_resp = self.response(method=method, type="full")
        return (
            full_resp["acceleration"],
            full_resp["velocity"],
            full_resp["displacement"],
        )

    # def newmark_beta_int(self, delta, varp):
    #     """
    #     Newmark-beta integration method
    #     """
    #     # Newmark-beta parameters
    #     gamma = 1 / 2
    #     beta = 1 / 4
    #     # Newmark-beta coefficients
    #     a0 = 1 / (beta * delta ** 2)
    #     a1 = gamma / (beta * delta)
    #     a2 = 1 / (beta * delta)
    #     a3 = 1 / (2 * beta) - 1
    #     a4 = gamma / beta - 1
    #     a5 = delta / 2 * (gamma / beta - 2)
    #     a6 = delta * (1 - gamma)
    #     a7 = gamma * delta
    #     # Newmark-beta integration
    #     acc = (
    #         a0 * self.f_t[-1](varp)
    #         + a2 * self.f_t[-2](varp)
    #         + a3 * self.f_t[-3](varp)
    #         + a5 * self.f_t[-4](varp)
    #     ) / (a0 * self.m_mtx[-1, -1] + a1 * self.c_mtx[-1, -1] + a2 * self.k_mtx[-1, -1])
    #     velo = self.x_dot[-1] + a6 * self.x_ddot[-2] + a7 * self.x_ddot[-1]
    #     disp = self.x[-1] + delta * self.x_dot[-1] + a3 * delta ** 2 * self.x_ddot[-2] + a4 * delta ** 2 * self.x_ddot[-1]
    #     return acc, velo, disp
