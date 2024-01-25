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
        damp_type="Rayleigh",
        damp_params=(0, 4, 0.03),
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
        super().__init__(
            mass_mtx,
            stiff_mtx,
            damp_type=damp_type,
            damp_params=damp_params,
            f_dof=[i for i in range(len(mass_vec))],
            t_eval=t,
            f_t=[
                lambda t: -self.acc_func(t) * self.mass_vec[i]
                for i in range(len(mass_vec))
            ],
            init_cond=None,
        )
        print(stiff_mtx)

    def run(self, method="Radau"):
        full_resp = self.response(method=method, type="full")
        return full_resp["acceleration"], full_resp["velocity"], full_resp["displacement"]

