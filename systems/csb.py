"""
Finite element model of a 2-span continously supported beam.
Coordinate origin is at the left end of the beam.
The left span of the beam is 0.84 m and the right span is 0.42 m.
Left and right ends of the beam are transversely fixed, while the mid-span is supported by a translational spring.
All three supports are also supported by rotational springs.
The beam is discretized into 2-node beam elements, each node has 2 DOFs (translational and rotational).
Each element is 0.02 m long, therefore the total number of elements is 63.
Total number of nodes is 64 and total number of DOFs is 128.
The beam is made of aluminum with the following reference properties:
- Elastic modulus: 68.5 GPa
- Density: 2.7 g/cm^3
- Cross-sectional area: 0.0015 m^2
- Moment of inertia: 1.125e-8 m^4
The beam is subjected to a stochastic transverse force at coordinate 0.54 m.

To define the system, the following parameters are required:
- material_properties: dictionary containing the material properties of the beam
- geometry_properties: dictionary containing the geometry properties of the beam
- element_length: length of each element
- damping_params: tuple containing the damping parameters (alpha, beta, gamma)
- f_loc: location
- resp_dof: response DOF
- t_eval: time vector
- f_t: force vector
- init_cond: initial conditions
"""

import numpy as np
from systems.lsds import MultiDOF
import scipy.integrate as spi
import matplotlib.pyplot as plt


class ContinuousBeam01(MultiDOF):
    def __init__(
        self,
        material_properties={
            "elastic_modulus": 68.5e9,
            "density": 2.7e3,
            "mid_support_rotational_stiffness": 1e4,
            "mid_support_translational_stiffness": 1e1,
            "left_support_rotational_stiffness": 1e5,
            "right_support_rotational_stiffness": 1e4,
        },
        geometry_properties={"b": 0.03, "h": 0.005, "l_1": 0.84, "l_2": 0.42},
        element_length=0.02,
        damping_params=(0, 1, 0.03),
        f_loc=0.54,
        resp_dof="full",
        t_eval=np.linspace(0, 1 - 1 / 1000, 1000),
        f_t=None,
        init_cond=None,
    ):
        self.E = material_properties["elastic_modulus"]
        self.rho = material_properties["density"]
        self.I = 1 / 12 * geometry_properties["b"] * geometry_properties["h"] ** 3
        self.A = geometry_properties["b"] * geometry_properties["h"]
        self.L = element_length
        self.k_theta_1 = material_properties["left_support_rotational_stiffness"]
        self.k_s = material_properties["mid_support_translational_stiffness"]
        self.k_theta_2 = material_properties["mid_support_rotational_stiffness"]
        self.k_theta_3 = material_properties["right_support_rotational_stiffness"]
        self.k_theta = [self.k_theta_1, self.k_s, self.k_theta_2, self.k_theta_3]
        self.element_number = int(
            (geometry_properties["l_1"] + geometry_properties["l_2"]) / self.L
        )
        self.node_number = self.element_number + 1
        self.dof_number = 2 * self.node_number
        self.f_dof = [int(f_loc / self.L * 2)]
        self.fixed_dof = [
            0,
            -2,
        ]
        self.support_spring_dof = [
            1,
            int(geometry_properties["l_1"] / self.L * 2),
            int(geometry_properties["l_1"] / self.L * 2 + 1),
            -1,
        ]
        if init_cond is None:
            init_cond = np.zeros(self.dof_number * 2)
        self.mass_mtx = self._assemble_global_matrix(type="mass")
        self.stiff_mtx = self._assemble_global_matrix(type="stiff")

        super().__init__(
            mass_mtx=self.mass_mtx,
            stiff_mtx=self.stiff_mtx,
            damp_type="Rayleigh",
            damp_params=damping_params,
            f_dof=self.f_dof,
            resp_dof=resp_dof,
            t_eval=t_eval,
            f_t=f_t,
            init_cond=init_cond,
        )

    def _element_stiffness_matrix(self):
        # 2-node beam element stiffness matrix
        EIL12 = 12 * self.E * self.I / (self.L**3)
        EIL6 = 6 * self.E * self.I / (self.L**2)
        EIL4 = 4 * self.E * self.I / (self.L)
        EIL2 = 2 * self.E * self.I / (self.L)
        K = np.diag(np.array([EIL12, EIL4, EIL12, EIL4]))
        K_down = np.zeros((4, 4))
        K_down[1, 0] = EIL6
        K_down[2, 0] = -EIL12
        K_down[2, 1] = -EIL6
        K_down[3, 0] = EIL6
        K_down[3, 1] = EIL2
        K_down[3, 2] = -EIL6
        K = K + K_down + K_down.transpose()
        return K

    def _element_mass_matrix(self):
        # 2-node beam element mass matrix
        const = self.rho * self.A * self.L / (420)
        M = np.diag(np.array([156, 4 * self.L**2, 156, 4 * self.L**2]))
        M_down = np.zeros((4, 4))
        M_down[1, 0] = 22 * self.L
        M_down[2, 0] = 54
        M_down[2, 1] = 13 * self.L
        M_down[3, 0] = -13 * self.L
        M_down[3, 1] = -3 * self.L**2
        M_down[3, 2] = -22 * self.L
        M = const * (M + M_down + M_down.transpose())
        return M

    def _apply_support_conditions(self, matrix):
        for dof in self.fixed_dof:
            temp = matrix[dof, dof]
            matrix[dof, :] = 0
            matrix[:, dof] = 0
            matrix[dof, dof] = temp
        return matrix

    def _add_support_spring_stiffness(self, matrix):
        for i, dof in enumerate(self.support_spring_dof):
            matrix[dof, dof] += self.k_theta[i]
        return matrix

    def _assemble_global_matrix(self, type="stiff"):
        global_matrix = np.zeros((self.dof_number, self.dof_number))
        for i in range(self.element_number):
            if type == "stiff":
                element_matrix = self._element_stiffness_matrix()
            else:
                element_matrix = self._element_mass_matrix()
            global_matrix[2 * i : 2 * i + 4, 2 * i : 2 * i + 4] += element_matrix
        global_matrix = self._apply_support_conditions(global_matrix)
        if type == "stiff":
            global_matrix = self._add_support_spring_stiffness(global_matrix)
        return global_matrix

    def _newmark_beta_int(self, delta, varp):
        M = self.mass_mtx
        C = self.damp_mtx
        K = self.stiff_mtx
        f_mtx = self.f_mtx()
        inv_M = np.linalg.inv(M)
        C = inv_M @ C
        K = inv_M @ K
        r = np.zeros((self.dof_number, 1))
        r[self.f_dof, 0] = 1
        r = inv_M @ r
        f = r @ (f_mtx[0, 0].reshape(1, 1))
        # f = r * f_mtx[0, 0]
        f = f.reshape(-1)
        dt = self.t_eval[1] - self.t_eval[0]
        dt2 = dt**2
        inv_ICK_mtx = np.linalg.inv(
            np.eye(self.dof_number) + varp * dt * C + delta * dt2 * K
        )
        resp_disp = np.zeros((self.dof_number, self.n_t))
        resp_velo = np.zeros((self.dof_number, self.n_t))
        resp_acc = np.zeros((self.dof_number, self.n_t))
        resp_disp[:, 0] = self.init_cond[: self.dof_number]
        resp_velo[:, 0] = self.init_cond[self.dof_number :]
        resp_acc[:, 0] = f - C @ resp_velo[:, 0] - K @ resp_disp[:, 0]
        const_mtx_1 = C + dt * K
        const_mtx_2 = (varp - 1) * dt * C + (delta - 0.5) * dt2 * K

        for i in range(1, self.n_t):
            f = r @ (f_mtx[0, i].reshape(1, 1))
            # f = r * f_mtx[0, i]
            f = f.reshape(-1)
            resp_acc[:, i] = inv_ICK_mtx @ (
                -K @ resp_disp[:, i - 1]
                - const_mtx_1 @ resp_velo[:, i - 1]
                + const_mtx_2 @ resp_acc[:, i - 1]
                + f
            )
            resp_velo[:, i] = (
                resp_velo[:, i - 1]
                + dt * resp_acc[:, i - 1]
                + varp * dt * (resp_acc[:, i] - resp_acc[:, i - 1])
            )
            resp_disp[:, i] = (
                resp_disp[:, i - 1]
                + dt * resp_velo[:, i - 1]
                + 0.5 * dt2 * resp_acc[:, i - 1]
                + delta * dt2 * (resp_acc[:, i] - resp_acc[:, i - 1])
            )

        return resp_disp, resp_velo, resp_acc

    def run(self, delta=1 / 4, varp=0.5):
        # using newmark-beta method to solve the equation of motion
        fq = 5000
        resp_disp, resp_velo, resp_acc = self._newmark_beta_int(delta, varp)
        force = self.f_mtx()
        t_interp = np.linspace(0, self.t_eval[-1], int(fq * self.t_eval[-1]) + 1)
        resp_disp = resp_disp
        resp_velo = resp_velo
        resp_acc = resp_acc
        force = force.flatten()
        t_interp = t_interp.flatten()
        resp_disp_reduceed = np.zeros((self.dof_number, len(t_interp)))
        resp_velo_reduceed = np.zeros((self.dof_number, len(t_interp)))
        resp_acc_reduceed = np.zeros((self.dof_number, len(t_interp)))
        self.t_eval = self.t_eval.flatten()
        for i in range(self.dof_number):
            resp_disp_reduceed[i, :] = np.interp(t_interp, self.t_eval, resp_disp[i, :])
            resp_velo_reduceed[i, :] = np.interp(t_interp, self.t_eval, resp_velo[i, :])
            resp_acc_reduceed[i, :] = np.interp(t_interp, self.t_eval, resp_acc[i, :])
        force = np.interp(t_interp, self.t_eval, force)
        return {
            "displacement": resp_disp_reduceed,
            "velocity": resp_velo_reduceed,
            "acceleration": resp_acc_reduceed,
            "force": force,
            "time": t_interp,
        }

    def frf(self, resized=True):
        force_dof = [54]
        resp_dof = [24, 44, 98]
        omega = np.linspace(10, 100, 450) * 2 * np.pi
        if resized:
            idx = [*range(50, 100)] + [*range(300, 350)]
            omega = omega[idx]
        omega = omega.reshape(-1, 1)
        omegasq = omega**2
        omega_mtx = np.repeat(omegasq, len(resp_dof), axis=1)
        frf_mtx = self.frf_mtx(resp_dof, force_dof, omega)
        frf_mtx = frf_mtx * omega_mtx * 2  # single sided frequency response
        return frf_mtx
