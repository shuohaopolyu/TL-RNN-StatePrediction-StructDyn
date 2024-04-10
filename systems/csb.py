"""
Finite element model of a 2-span continously supported beam.
The beam is modelled as a Bernoulli-Euler beam.
"""

import numpy as np
from systems.lsds import MultiDOF


class ContinuousBeam01(MultiDOF):
    def __init__(
        self,
        material_properties={
            "elastic_modulus": 70e9,
            "density": 2.7e3,
            "support_rotational_stiffness": 0,
        },
        geometry_properties={"b": 0.03, "h": 0.005, "l_1": 0.84, "l_2": 0.42},
        element_length=0.02,
        damping_params=(0, 3, 0.01),
        f_loc=0.56,
        resp_dof="full",
        t_eval=np.linspace(0, 10 - 1 / 1000, 10000),
        f_t=None,
        init_cond=None,
    ):
        self.E = material_properties["elastic_modulus"]
        self.rho = material_properties["density"]
        self.I = 1 / 12 * geometry_properties["b"] * geometry_properties["h"] ** 3
        self.A = geometry_properties["b"] * geometry_properties["h"]
        self.L = element_length
        self.k_theta = material_properties["support_rotational_stiffness"]
        self.element_number = int(
            (geometry_properties["l_1"] + geometry_properties["l_2"]) / self.L
        )
        self.node_number = self.element_number + 1
        self.dof_number = 2 * self.node_number
        self.f_dof = [int(f_loc / self.L * 2)]
        self.fixed_dof = [
            0,
            1,
            int(geometry_properties["l_1"] / self.L * 2),
            self.dof_number - 1,
            self.dof_number - 2,
        ]
        self.support_rotational_dof = [int(geometry_properties["l_1"] / self.L * 2 + 1)]
        if init_cond is None:
            init_cond = np.zeros(self.dof_number * 2)
        super().__init__(
            mass_mtx=self._assemble_global_matrix(type="mass"),
            stiff_mtx=self._assemble_global_matrix(type="stiff"),
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
        subs_num = np.std(np.diag(matrix))
        for dof in self.fixed_dof:
            matrix[dof, :] = 0
            matrix[:, dof] = 0
            matrix[dof, dof] = 1
        return matrix

    def _add_support_rotational_stiffness(self, matrix):
        for dof in self.support_rotational_dof:
            matrix[dof, dof] += self.k_theta
        return matrix

    def _assemble_global_matrix(self, type="stiff"):
        global_matrix = np.zeros((self.dof_number, self.dof_number))
        for i in range(self.element_number):
            element_matrix = (
                self._element_stiffness_matrix()
                if type == "stiff"
                else self._element_mass_matrix()
            )
            global_matrix[2 * i : 2 * i + 4, 2 * i : 2 * i + 4] += element_matrix
        global_matrix = self._apply_support_conditions(global_matrix)
        if type == "stiff":
            global_matrix = self._add_support_rotational_stiffness(global_matrix)
        return global_matrix
