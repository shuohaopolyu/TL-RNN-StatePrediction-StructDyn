"""
Base isolated shear type structure subjected to ground motion.

Inputs:
    mass_super_vec:  array like, with shape (n, )
                    mass of the superstructure from bottom to top
    stiff_super_vec: array like, with shape (n, )
                    stiffness of the superstructure from bottom to top
    damp_type: currently only support Rayleigh
    damp_params: tuple, (lower_mode_index, upper_mode_index, damping_ratio)
    mass_base: float, mass of the base
    isolator_params: dict, parameters of the nonlinear damper, keys are "c_b", "k_b", "F_y", "q", "A", "alpha", "beta", "gamma", "n", "z_0"
    init_disp: array like, with shape (1+n, )
                initial displacement of the structure, first element is the base displacement, the rest are superstructure displacement
    init_vel: array like, with shape (1+n, )
                initial velocity of the structure, first element is the base velocity, the rest are superstructure velocity
    t: array like, with shape (n_t, )
            time vector
    acc_g: array like, with shape (n_t, )
                ground acceleration

Output:
    resp_disp: array like, with shape (1+n, n_t)
                displacement of the structure, first element is the base displacement, the rest are superstructure displacement
    resp_vel: array like, with shape (1+n, n_t)
                velocity of the structure, first element is the base velocity, the rest are superstructure velocity
    resp_acc: array like, with shape (1+n, n_t)
                acceleration of the structure, first element is the base acceleration, the rest are superstructure acceleration
    z: array like, with shape (1, n_t)
                displacement of the nonlinear damper

Note:
    1. Superstructure displacement, velocity and acceleration are relative to the base
    2. Base displacement, velocity and acceleration are relative to the ground
    3. N-Z model is used for the nonlinear isolator. For details please refer to Matsagar, V. A., & Jangid, R. S. (2003). 
       Seismic response of base-isolated structures during impact with adjacent structures. Engineering Structures, 25(10), 1311-1323.
    4. Newmark-beta method is used for time integration

"""

import numpy as np
from numpy import linalg as LA


class BaseIsolatedStructure:
    def __init__(
        self,
        mass_super_vec=None,
        stiff_super_vec=None,
        damp_type="Rayleigh",
        damp_params=(0, 4, 0.03),
        mass_base=1,
        isolator_params=None,
        x_0=None,
        x_dot_0=None,
        t=None,
        acc_g=None,
    ):
        self.mass_super_mtx = np.diag(mass_super_vec)
        self.stiff_super_mtx = (
            np.diag(stiff_super_vec)
            + np.diag(-stiff_super_vec[1:], 1)
            + np.diag(-stiff_super_vec[1:], -1)
            + np.diag(np.append(stiff_super_vec[1:], 0))
        )
        self.damp_super_mtx = self.update_damp_super_mtx(damp_type, damp_params)
        self.mass_base = mass_base
        self.t = t
        self.acc_g = acc_g
        self.isolator_params = isolator_params
        self.dt = self.t[1] - self.t[0]
        self.nt = len(self.t)
        self.dof = len(mass_super_vec) + 1
        self.resp_disp = np.zeros((self.dof, self.nt))
        self.resp_vel = np.zeros((self.dof, self.nt))
        self.resp_acc = np.zeros((self.dof, self.nt))
        self.resp_disp[:, 0] = x_0
        self.resp_vel[:, 0] = x_dot_0
        self.c_1 = isolator_params["c_b"]*0.2
        self.k_1 = self.stiff_super_mtx[0, 0] + self.stiff_super_mtx[0, 1]

    def update_damp_super_mtx(self, damp_type, damp_params):
        if damp_type == "Rayleigh":
            omega_sq, _ = LA.eig(LA.inv(self.mass_super_mtx) @ self.stiff_super_mtx)
            idx = omega_sq.argsort()
            omega_sq = np.real(omega_sq[idx])
            omega = np.sqrt(omega_sq)
            lower_index, upper_index, damping_ratio = damp_params
            alpha = (
                2
                * damping_ratio
                * omega[lower_index]
                * omega[upper_index]
                / (omega[lower_index] + omega[upper_index])
            )
            beta = 2 * damping_ratio / (omega[lower_index] + omega[upper_index])
            return alpha * self.mass_super_mtx + beta * self.stiff_super_mtx
        else:
            raise NotImplementedError

    def _intg_mtx(self):
        c_1 = self.c_1  # you need more consideration on this coefficient!!!
        k_1 = self.k_1
        alpha = self.isolator_params["alpha"]
        k_b = self.isolator_params["k_b"]
        c_b = self.isolator_params["c_b"]
        r = np.ones((self.dof - 1, 1))
        intg__M = np.block(
            [
                [self.mass_base, np.zeros((1, self.dof - 1))],
                [self.mass_super_mtx @ r, self.mass_super_mtx],
            ]
        )
        intg__C = np.block(
            [
                [c_b, -c_1, np.zeros((1, self.dof - 2))],
                [np.zeros((self.dof - 1, 1)), self.damp_super_mtx],
            ]
        )
        intg__K = np.block(
            [
                [alpha * k_b, -k_1, np.zeros((1, self.dof - 2))],
                [np.zeros((self.dof - 1, 1)), self.stiff_super_mtx],
            ]
        )
        inv_intg_M = LA.inv(intg__M)
        C = inv_intg_M @ intg__C
        K = inv_intg_M @ intg__K
        return inv_intg_M, C, K

    def newmark_beta_int(self, delta, varp):
        # delta = 0, varp = 1/2: Explicit central difference scheme
        # delta = 1/4, varp = 1/2: Average acceleration method
        # delta = 1/6, varp = 1/2: Linear acceleration method
        # delta = 1/12, varp = 1/2: Fox-Goodwin method

        k_1 = self.k_1
        c_1 = self.c_1
        m_b = self.mass_base
        c_b = self.isolator_params["c_b"]
        k_b = self.isolator_params["k_b"]
        F_y = self.isolator_params["F_y"]
        q = self.isolator_params["q"]
        A = self.isolator_params["A"]
        alpha = self.isolator_params["alpha"]
        beta = self.isolator_params["beta"]
        gamma = self.isolator_params["gamma"]
        n = self.isolator_params["n"]
        z_0 = self.isolator_params["z_0"]
        inv_intg_M, C, K = self._intg_mtx()
        inv_mass_mtx = LA.inv(self.mass_super_mtx)
        dt = self.dt
        inv_ICK_mtx = LA.inv(np.eye(self.dof) + varp * dt * C + delta * dt**2 * K)
        r = np.ones(self.dof - 1)
        z = np.zeros(self.nt)
        z[0] = z_0
        z_dot = np.zeros(self.nt)

        # Step 1: compute acceleration at step 0 from superstructure displacement and velocity at step 0, ground acceleration at step 0, and z at step 0
        self.resp_acc[0, 0] = (
            -m_b * self.acc_g[0]
            - c_b * self.resp_vel[0, 0]
            - alpha * k_b * self.resp_disp[0, 0]
            - (1 - alpha) * F_y * z[0]
            + k_1 * self.resp_disp[1, 0]
            + c_1 * self.resp_vel[1, 0]
        ) / m_b
        self.resp_acc[1:, 0] = -r * (
            self.acc_g[0] + self.resp_acc[0, 0]
        ) - inv_mass_mtx @ (
            self.damp_super_mtx @ self.resp_vel[1:, 0]
            + self.stiff_super_mtx @ self.resp_disp[1:, 0]
        )

        for i in range(self.nt - 1):
            # Step 2: compute z at step i+1 from base displacement and velocity at step i, and z at step i
            z_dot[i] = (
                A * self.resp_vel[0, i]
                - beta * np.abs(self.resp_vel[0, i]) * np.abs(z[i]) ** (n - 1) * z[i]
                - gamma * self.resp_vel[0, i] * np.abs(z[i]) ** n
            ) / q
            z[i + 1] = z[i] + dt * z_dot[i]

            # Step 3: compute acceleration at step i+1 from displacement, velocity and acceleration of structure at step i, plus ground acceleration at step i+1 and z at step i+1
            intg_f = np.block(
                [
                    -m_b * self.acc_g[i + 1] - (1 - alpha) * F_y * z[i + 1],
                    -self.mass_super_mtx @ r * self.acc_g[i + 1],
                ]
            )
            f = inv_intg_M @ intg_f
            self.resp_acc[:, i + 1] = inv_ICK_mtx @ (
                -K @ self.resp_disp[:, i]
                - (C + dt * K) @ self.resp_vel[:, i]
                + ((varp - 1) * dt * C + (delta - 0.5) * dt**2 * K)
                @ self.resp_acc[:, i]
                + f
            )

            # Step 4: compute displacement at step i+1 from acceleration at step i and i+1, plus velocity and displacement at step i
            self.resp_disp[:, i + 1] = (
                self.resp_disp[:, i]
                + dt * self.resp_vel[:, i]
                + 0.5 * dt**2 * self.resp_acc[:, i]
                + delta * dt**2 * (self.resp_acc[:, i + 1] - self.resp_acc[:, i])
            )

            # Step 5: compute velocity at step i+1 from acceleration at step i and i+1, plus velocity at step i
            self.resp_vel[:, i + 1] = (
                self.resp_vel[:, i]
                + dt * self.resp_acc[:, i]
                + varp * dt * (self.resp_acc[:, i + 1] - self.resp_acc[:, i])
            )
        self.z = z.reshape(1, -1)
        self.solution = {"acceleration": self.resp_acc, "displacement": self.resp_disp, "velocity": self.resp_vel, "z": self.z}

    def run(self, delta=0.25, varp=0.5):
        self.newmark_beta_int(delta, varp)
        return self.resp_disp, self.resp_vel, self.resp_acc, self.z
