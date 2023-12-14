"""
A finite element model of a 3-span continuous bridge.
The bridge is 22 m long, and two side spans are 6 m long each.
Cross-section of the bridge is 0.1 m x 0.1 m.
44 identical elements are used to model the bridge based on the Bernoulli-Euler beam theory.
Each element has 2 nodes, and each node has 2 dofs, i.e., the vertical displacement and rotation.
There are 4 constrained nodes at the supports.
Hence, there are 45 nodes and 86 dofs in total.
The beam_fem class is used to assemble the global mass and stiffness matrices.
"""

import numpy as np
from numpy import linalg as LA
from systems.lsds import MultiDOF


class ContinuousBridge(MultiDOF):
    def __init__(
        self,
        rho=1e4,
        E=2e10,
        A=1,
        I=1 / 12,
        L=0.5,
        damp_params=(0, 3, 0.03),
        f_dof=[11],
        resp_dof="full",
        t_eval=np.linspace(0, 100-1/100, 10000),
        f_t=None,
        init_cond=None,
    ):
        # system parameters
        self.rho = rho
        self.E = E
        self.I = I
        self.L = L
        self.A = A
        # mass and stiffness matrices
        self.mass_mtx = self.assemble_glb_mtx(type="mass")
        self.stiff_mtx = self.assemble_glb_mtx(type="stiff")
        super().__init__(
            mass_mtx=self.mass_mtx,
            stiff_mtx=self.stiff_mtx,
            damp_type="Rayleigh",
            damp_params=damp_params,
            f_dof=f_dof,
            resp_dof=resp_dof,
            t_eval=t_eval,
            f_t=f_t,
            init_cond=init_cond,
        )

    def ele_K(self):
        # element stiffness matrix
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

    def ele_M(self):
        # element mass matrix
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

    def assemble_glb_mtx(self, type="stiff"):
        # global mass and stiffness matrices
        # there are 44 elements, 45 nodes, 86 dofs in total
        ele_num = np.linspace(0, 43, 44, dtype=int).reshape((-1, 1))
        dof_num = np.linspace(0, 89, 90, dtype=int).reshape((-1, 1))
        ele_node = np.hstack(
            (ele_num * 2, ele_num * 2 + 1, ele_num * 2 + 2, ele_num * 2 + 3)
        )
        glb_mtx = np.zeros((45 * 2, 45 * 2))
        if type == "stiff":
            K = self.ele_K()
            for i in range(44):
                bv, cv = np.meshgrid(ele_node[i, :], ele_node[i, :])
                glb_mtx[bv, cv] += K
        elif type == "mass":
            M = self.ele_M()
            for i in range(44):
                bv, cv = np.meshgrid(ele_node[i, :], ele_node[i, :])
                glb_mtx[bv, cv] += M
        con_node_number = np.array([0, 24, 64, 88]).reshape((-1, 1))
        uncon_node = np.delete(dof_num, con_node_number)
        bv, cv = np.meshgrid(uncon_node, uncon_node)
        glb_mtx = glb_mtx[bv, cv]
        return glb_mtx


class MovingLoad:
    def __init__(self, load, v, t_0=0):
        self.load = load
        self.v = v
        self.t_0 = t_0
        self._full_left_node_coord = [0.0 + i*0.5 for i in range(11)]+[6.0+i*0.5 for i in range(19)] + [16.0+i*0.5 for i in range(11)]
        self._full_right_node_coord = [1.0 + i*0.5 for i in range(11)]+[7.0+i*0.5 for i in range(19)] + [17.0+i*0.5 for i in range(11)]
        self._full_mid_node_coord = [0.5 + i*0.5 for i in range(11)]+[6.5+i*0.5 for i in range(19)] + [16.5+i*0.5 for i in range(11)]

    def _load_func(self, mid_node_coord, left_node_coord, right_node_coord, t):
        t_1 = left_node_coord / self.v + self.t_0
        t_2 = mid_node_coord / self.v + self.t_0
        t_3 = right_node_coord / self.v + self.t_0

        if isinstance(t, float):
            if t < t_1:
                return 0
            elif t >= t_3:
                return 0
            elif t_1 <= t < t_2:
                return self.load * (t - t_1) / (t_2 - t_1)
            else:
                return self.load * (t_3 - t) / (t_3 - t_2)
        else:
            # Create masks for each condition
            mask1 = t < t_1
            mask2 = t >= t_3
            mask3 = (t_1 <= t) & (t < t_2)
            mask4 = ~mask1 & ~mask2 & ~mask3

            result = np.zeros_like(t)

            # Apply conditions
            result[mask3] = self.load * (t[mask3] - t_1) / (t_2 - t_1)
            result[mask4] = self.load * (t_3 - t[mask4]) / (t_3 - t_2)

            return result
        
    def load_funcs(self):
        f_t = []
        for i in range(len(self._full_mid_node_coord)):
            f_t.append(lambda t, i=i: self._load_func(self._full_mid_node_coord[i], self._full_left_node_coord[i], self._full_right_node_coord[i], t))
        return f_t

        
    
