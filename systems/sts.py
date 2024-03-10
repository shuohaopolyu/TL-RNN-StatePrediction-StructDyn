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
import matplotlib.pyplot as plt


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
        self.acc_func = interpolate.interp1d(
            t, acc_g, kind="quadratic", fill_value="extrapolate"
        )
        mass_mtx = np.diag(mass_vec)
        stiff_mtx = (
            np.diag(stiff_vec)
            + np.diag(-stiff_vec[1:], 1)
            + np.diag(-stiff_vec[1:], -1)
            + np.diag(np.append(stiff_vec[1:], 0))
        )
        if damp_type == "d_mtx":
            damp_mtx = (
                np.diag(damp_vec)
                + np.diag(-damp_vec[1:], 1)
                + np.diag(-damp_vec[1:], -1)
                + np.diag(np.append(damp_vec[1:], 0))
            )
        elif damp_type == "Rayleigh":
            omega1, omega2 = damp_vec[2] * np.pi * 2, damp_vec[3] * np.pi * 2
            mtx = np.array(
                [[1 / (2 * omega1), omega1 / 2], [1 / (2 * omega2), omega2 / 2]]
            )
            alpha, beta = np.linalg.solve(mtx, [damp_vec[0], damp_vec[1]])
            damp_mtx = alpha * mass_mtx + beta * stiff_mtx

        super().__init__(
            mass_mtx,
            stiff_mtx,
            damp_type="d_mtx",
            damp_params=damp_mtx,
            f_dof=[i for i in range(len(mass_vec))],
            t_eval=t,
            f_t=[
                lambda x, i=i: self.mass_vec[i] * (-self.acc_func(x))
                for i in range(len(mass_vec))
            ],  # takes me three days to debug this line
            # The i=i in the lambda function is important
            # because it creates a new scope for i within
            # each lambda function. If you don't do this,
            # all lambda functions will use the latest
            # value of i from the loop due to late binding of the loop variable.
            init_cond=None,
        )

    def run(self, method="DOP853"):
        full_resp = self.response(method=method, type="full")
        return (
            full_resp["acceleration"],
            full_resp["velocity"],
            full_resp["displacement"],
        )

    def print_damping_ratio(self, num):
        assert (
            num <= self.DOF
        ), "The number of modes should be less or equal to the DOF."
        dr = self.damping_ratio()
        print("Damping ratio: ")
        for i in range(num):
            print("Mode {}: {}".format(i + 1, dr[i]))
        return dr

    def print_natural_frequency(self, num):
        assert (
            num <= self.DOF
        ), "The number of modes should be less or equal to the DOF."
        wn = self.freqs_modes()[0] * 2 * np.pi
        print("Natural frequency (/rad^2): ")
        for i in range(num):
            print("Mode {}: {}".format(i + 1, wn[i]))
        return wn
