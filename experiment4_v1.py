import taichi as ti
import taichi.math as tm
import numpy as np
import tool.geometrytool as geo
from data.lv_rv import meshData


@ti.data_oriented
class Body:
    def __init__(self, vert_np: np.ndarray, tet_np: np.ndarray,
                 tet_fiber_np: np.ndarray, tet_sheet_np: np.ndarray, tet_normal_np: np.ndarray,
                 num_tet_set_np, tet_set_np: np.ndarray,
                 bou_base_np: np.ndarray, bou_epi_np: np.ndarray,
                 bou_endo_lv_np: np.ndarray, bou_endo_rv_np: np.ndarray
                 ) -> None:
        self.density = 1000.0
        self.num_vertex = len(vert_np)
        self.vertex = ti.Vector.field(3, dtype=float, shape=(self.num_vertex, ))
        self.vertex.from_numpy(vert_np)
        self.np_vertex = vert_np
        self.num_tet = len(tet_np)
        self.elements = ti.Vector.field(4, dtype=ti.i32, shape=(self.num_tet, ))
        self.elements.from_numpy(tet_np)
        self.np_elements = tet_np
        # tet_fiber
        self.tet_fiber = ti.Vector.field(3, dtype=float, shape=(self.num_tet,))
        self.tet_fiber.from_numpy(tet_fiber_np)
        # tet_sheet
        self.tet_sheet = ti.Vector.field(3, dtype=float, shape=(self.num_tet,))
        self.tet_sheet.from_numpy(tet_sheet_np)
        # tet_normal
        self.tet_normal = ti.Vector.field(3, dtype=float, shape=(self.num_tet,))
        self.tet_normal.from_numpy(tet_normal_np)
        # num_tet_set
        self.num_tet_set = ti.field(int, ())
        self.num_tet_set[None] = num_tet_set_np
        # tet_set
        self.tet_set = ti.field(int, shape=(self.num_tet,))
        self.tet_set.from_numpy(tet_set_np)

        # bou
        self.num_bou_base_face = len(bou_base_np)
        self.bou_base = ti.Vector.field(3, int, shape=(self.num_bou_base_face,))
        self.bou_base.from_numpy(bou_base_np)

        self.num_bou_epi_face = len(bou_epi_np)
        self.bou_epi = ti.Vector.field(3, int, shape=(self.num_bou_epi_face,))
        self.bou_epi.from_numpy(bou_epi_np)

        self.num_bou_endo_lv_face = len(bou_endo_lv_np)
        self.bou_endo_lv = ti.Vector.field(3, int, shape=(self.num_bou_endo_lv_face,))
        self.bou_endo_lv.from_numpy(bou_endo_lv_np)

        self.num_bou_endo_rv_face = len(bou_endo_rv_np)
        self.bou_endo_rv = ti.Vector.field(3, int, shape=(self.num_bou_endo_rv_face,))
        self.bou_endo_rv.from_numpy(bou_endo_rv_np)

        # variables for visualization
        surfaces = geo.get_surface_from_tet(vertex=vert_np, elements=tet_np)
        self.surfaces = ti.field(ti.i32, shape=(surfaces.shape[0] * surfaces.shape[1]))
        self.surfaces.from_numpy(surfaces.reshape(-1))

        self.Dm = ti.Matrix.field(3, 3, float, shape=(self.num_tet,))
        self.DmInv = ti.Matrix.field(3, 3, float, shape=(self.num_tet,))
        self.DmInvT = ti.Matrix.field(3, 3, float, shape=(self.num_tet,))
        self.init_DmInv()

        self.vel = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.init_vel()
        self.Vm = ti.field(float, shape=(self.num_vertex,))

        # volume
        self.volume = ti.field(float, self.num_tet)
        self.init_volume()
        # # 顶点fiber方向
        # self.vert_fiber = ti.Vector.field(3, float, shape=(self.num_vertex,))
        # self.vert_fiber.from_numpy(vert_fiber)

        self.tet_Ta = ti.field(float, shape=(self.num_tet,))
        self.ver_Ta = ti.field(float, shape=(self.num_vertex,))
        self.init_electrophysiology()

        self.vert_color = ti.Vector.field(3, float, shape=(self.num_vertex,))
        self.init_vert_color()

    @ti.kernel
    def init_vert_color(self):
        for i in self.vert_color:
            self.vert_color[i] = tm.vec3(0.5, 0.5, 0.5)

    @ti.kernel
    def init_electrophysiology(self):
        for i in self.elements:
            self.tet_Ta[i] = 60.0

        for i in self.vertex:
            self.ver_Ta[i] = 60.0

    @ti.kernel
    def set_Ta(self, value: float):
        for i in self.elements:
            self.tet_Ta[i] = value

        for i in self.vertex:
            self.ver_Ta[i] = value

    def get_min_y(self):
        min_y = 10000000.0
        # print("nv:", self.num_vertex)
        for i in range(self.num_vertex):
            # print("i:", i)
            if min_y > self.vertex[i][1]:
                min_y = self.vertex[i][1]
        print(min_y)
        return min_y

    @ti.kernel
    def scale_vertex(self, scale: float):
        for i in range(self.num_vertex):
            self.vertex[i] *= scale

    @ti.kernel
    def init_DmInv(self):
        Dm, vertex, tet = ti.static(self.Dm, self.vertex, self.elements)
        # 0412 -
        for i in range(self.num_tet):
            Dm[i][0, 0] = vertex[tet[i][1]][0] - vertex[tet[i][0]][0]
            Dm[i][1, 0] = vertex[tet[i][1]][1] - vertex[tet[i][0]][1]
            Dm[i][2, 0] = vertex[tet[i][1]][2] - vertex[tet[i][0]][2]
            Dm[i][0, 1] = vertex[tet[i][2]][0] - vertex[tet[i][0]][0]
            Dm[i][1, 1] = vertex[tet[i][2]][1] - vertex[tet[i][0]][1]
            Dm[i][2, 1] = vertex[tet[i][2]][2] - vertex[tet[i][0]][2]
            Dm[i][0, 2] = vertex[tet[i][3]][0] - vertex[tet[i][0]][0]
            Dm[i][1, 2] = vertex[tet[i][3]][1] - vertex[tet[i][0]][1]
            Dm[i][2, 2] = vertex[tet[i][3]][2] - vertex[tet[i][0]][2]

        for i in range(self.num_tet):
            self.DmInv[i] = self.Dm[i].inverse()
            self.DmInvT[i] = self.DmInv[i].transpose()

    @ti.kernel
    def init_vel(self):
        for i in self.vel:
            self.vel[i] = tm.vec3([0., 0., 0.])

    @ti.kernel
    def init_volume(self):
        for i in self.volume:
            self.volume[i] = ti.abs(self.Dm[i].determinant()) / 6.0

    @ti.kernel
    def sample_tet_fiber(self):
        for i in range(self.num_tet):
            self.tet_fiber[i] = self.vert_fiber[self.elements[i][0]] + self.vert_fiber[self.elements[i][1]] + \
                                self.vert_fiber[self.elements[i][2]] + self.vert_fiber[self.elements[i][3]]
            self.tet_fiber[i] /= 4.0
            self.tet_fiber[i] /= tm.length(self.tet_fiber[i])

    @ti.kernel
    def translation(self, x: float, y: float, z: float):
        for i in self.vertex:
            self.vertex[i][0] += x
            self.vertex[i][1] += y
            self.vertex[i][2] += z


@ti.data_oriented
class diffusion_reaction:
    """
    use 2nd-order Strang splitting: An intergrative smoothed particle hydrodynamics method for modeling cardiac function
    use Aliec-Panfilov model (single cell)
    """
    def __init__(self, body: Body):
        self.body = body
        # self.Vm = ti.field(float, shape=(body.num_vertex,))
        self.Vm = self.body.Vm
        self.w = ti.field(float, shape=(body.num_vertex,))
        self.I_ext = ti.field(float, shape=(body.num_vertex,))
        self.init_Vm_w_and_I()

        # parameter of Aliec-Panfilov model
        self.k = 8.0
        self.a = 0.01
        self.b = 0.15
        self.epsilon_0 = 0.04  # 0.002
        self.mu_1 = 0.2
        self.mu_2 = 0.3
        self.C_m = 1.0

        # parameter of diffusion model
        self.sigma_f = 1.1
        self.sigma_s = 1.0
        self.sigma_n = 1.0
        self.Dm = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.DmInv = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.Ds = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.F = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.Be = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.init_Ds_F_Be()
        self.fiber = ti.Vector.field(3, float, shape=(body.num_tet,))
        self.sheet = ti.Vector.field(3, float, shape=(body.num_tet,))
        self.normal = ti.Vector.field(3, float, shape=(body.num_tet,))
        self.init_fiber()
        self.DM = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))
        self.Me = ti.Matrix.field(4, 4, float, shape=(body.num_tet,))
        self.Ke = ti.Matrix.field(4, 4, float, shape=(body.num_tet,))
        self.D = ti.Matrix.field(3, 3, float, shape=(body.num_tet,))

        # For conjugate gradient method
        self.cg_x = ti.field(float, self.body.num_vertex)
        self.cg_Ax = ti.field(float, self.body.num_vertex)
        self.cg_b = ti.field(float, self.body.num_vertex)
        self.cg_r = ti.field(float, self.body.num_vertex)
        self.cg_d = ti.field(float, self.body.num_vertex)
        self.cg_Ad = ti.field(float, self.body.num_vertex)
        self.pcg_M = ti.field(float, self.body.num_vertex)
        self.pcg_s = ti.field(float, self.body.num_vertex)
        self.cg_epsilon = 1.0e-3
        # debug
        self.cg_A = ti.field(float, shape=(self.body.num_vertex, self.body.num_vertex))

        self.tag_s1 = ti.field(int, shape=(self.body.num_vertex,))
        self.tag_s2_1 = ti.field(int, shape=(self.body.num_vertex,))
        self.tag_s2_2 = ti.field(int, shape=(self.body.num_vertex,))
        self.init_sim_tag()

    @ti.kernel
    def init_sim_tag(self):
        pos = ti.static(self.body.vertex)
        for i in pos:
            if -24 <= pos[i].x <= -18.0 and -6.0 <= pos[i].y <= 0.0 and -3.0 <= pos[i].z <= 3.0:
                self.tag_s1[i] = 1
            else:
                self.tag_s1[i] = 0

            if 6.0 >= pos[i].x >= 0.0 >= pos[i].y >= -6.0 and 0.0 <= pos[i].z:
                self.tag_s2_1[i] = 1
            else:
                self.tag_s2_1[i] = 0

            # if -24 <= pos[i].x <= -18.0 and -6.0 <= pos[i].y <= 0.0 and -3.0 <= pos[i].z <= 3.0:
            #     self.tag_s1[i] = 1
            # else:
            #     self.tag_s1[i] = 0

    @ti.kernel
    def init_Vm_w_and_I(self):
        for i in self.Vm:
            self.Vm[i] = 0.0

        for i in self.w:
            self.w[i] = 0.0

        for i in self.I_ext:
            self.I_ext[i] = 0.0

    @ti.kernel
    def init_Ds_F_Be(self):
        for i in range(self.body.num_tet):
            self.Dm[i][0, 0] = self.body.vertex[self.body.elements[i][0]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Dm[i][1, 0] = self.body.vertex[self.body.elements[i][0]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Dm[i][2, 0] = self.body.vertex[self.body.elements[i][0]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]
            self.Dm[i][0, 1] = self.body.vertex[self.body.elements[i][1]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Dm[i][1, 1] = self.body.vertex[self.body.elements[i][1]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Dm[i][2, 1] = self.body.vertex[self.body.elements[i][1]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]
            self.Dm[i][0, 2] = self.body.vertex[self.body.elements[i][2]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Dm[i][1, 2] = self.body.vertex[self.body.elements[i][2]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Dm[i][2, 2] = self.body.vertex[self.body.elements[i][2]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]

        for i in range(self.body.num_tet):
            self.DmInv[i] = self.Dm[i].inverse()

        for i in range(self.body.num_tet):
            self.Ds[i][0, 0] = self.body.vertex[self.body.elements[i][0]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Ds[i][1, 0] = self.body.vertex[self.body.elements[i][0]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Ds[i][2, 0] = self.body.vertex[self.body.elements[i][0]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]
            self.Ds[i][0, 1] = self.body.vertex[self.body.elements[i][1]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Ds[i][1, 1] = self.body.vertex[self.body.elements[i][1]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Ds[i][2, 1] = self.body.vertex[self.body.elements[i][1]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]
            self.Ds[i][0, 2] = self.body.vertex[self.body.elements[i][2]][0] - \
                               self.body.vertex[self.body.elements[i][3]][0]
            self.Ds[i][1, 2] = self.body.vertex[self.body.elements[i][2]][1] - \
                               self.body.vertex[self.body.elements[i][3]][1]
            self.Ds[i][2, 2] = self.body.vertex[self.body.elements[i][2]][2] - \
                               self.body.vertex[self.body.elements[i][3]][2]

        for i in range(self.body.num_tet):
            self.Be[i][0, 0] = self.body.vertex[self.body.elements[i][1]][0] - \
                               self.body.vertex[self.body.elements[i][0]][0]
            self.Be[i][1, 0] = self.body.vertex[self.body.elements[i][1]][1] - \
                               self.body.vertex[self.body.elements[i][0]][1]
            self.Be[i][2, 0] = self.body.vertex[self.body.elements[i][1]][2] - \
                               self.body.vertex[self.body.elements[i][0]][2]
            self.Be[i][0, 1] = self.body.vertex[self.body.elements[i][2]][0] - \
                               self.body.vertex[self.body.elements[i][0]][0]
            self.Be[i][1, 1] = self.body.vertex[self.body.elements[i][2]][1] - \
                               self.body.vertex[self.body.elements[i][0]][1]
            self.Be[i][2, 1] = self.body.vertex[self.body.elements[i][2]][2] - \
                               self.body.vertex[self.body.elements[i][0]][2]
            self.Be[i][0, 2] = self.body.vertex[self.body.elements[i][3]][0] - \
                               self.body.vertex[self.body.elements[i][0]][0]
            self.Be[i][1, 2] = self.body.vertex[self.body.elements[i][3]][1] - \
                               self.body.vertex[self.body.elements[i][0]][1]
            self.Be[i][2, 2] = self.body.vertex[self.body.elements[i][3]][2] - \
                               self.body.vertex[self.body.elements[i][0]][2]

        for i in range(self.body.num_tet):
            self.F[i] = self.Ds[i] @ self.DmInv[i]

    @ti.kernel
    def init_fiber(self):
        for i in range(self.body.num_tet):
            self.fiber[i] = self.F[i] @ self.body.tet_fiber[i]
            self.sheet[i] = self.F[i] @ self.body.tet_sheet[i]
            self.normal[i] = self.F[i] @ tm.cross(self.body.tet_fiber[i], self.body.tet_sheet[i])

    @ti.kernel
    def calculate_reaction(self, dt: float):
        """
        2 ODEs:
        Rv: C_m dV_m/dt = I_ion(V_m, w)
        Rw: dw/dt = g(V_m, w)
        use Reaction-by-Reaction splitting method to decouple ODEs,
        update: R(t+dt)=Rv(dt/2){Rw(dt/2){Rw(dt/2){Rv(dt/2){R(t)}}}}
        rewrite a single reaction equation in a general form as
        dy/dt = q(y,t) - p(y,t)y
        use QSS method, a linearly approximated exact solution, which is unconditionally stable:
        y(t+dt) = y(t)exp(-p(y(t),t)dt) + q(y(t),t)/p(y(t),t) * (1-exp(-p(y(t),t)dt))
        """
        for i in self.Vm:
            self.calculate_Rv(i, dt * 0.5)
            self.calculate_Rw(i, dt * 0.5)
            self.calculate_Rw(i, dt * 0.5)
            self.calculate_Rv(i, dt * 0.5)

    @ti.func
    def calculate_Rv(self, i, dt):
        """
        dV_m/dt = kV_m/C_m (V_m + aV_m -V_m^2) + I_ext - (ka+w)/C_m * V_m
        y = V_m, q(y,t) = kV_m/C_m * (V_m + aV_m -V_m^2) + I_ext, p(y,t) = (ka+w)/C_m
        """
        self.Vm[i] = self.Vm[i] * tm.exp(-1.0 * dt * ((self.k * self.a + self.w[i]) / self.C_m)) + (
                self.k * self.Vm[i] / self.C_m * (self.Vm[i] * (1.0 + self.a - self.Vm[i])) + self.I_ext[i]) / (
                                 (self.k * self.a + self.w[i]) / self.C_m) * (
                                 1.0 - tm.exp(-1.0 * dt * ((self.k * self.a + self.w[i]) / self.C_m)))

    @ti.func
    def calculate_Rw(self, i, dt):
        """
        dw/dt = epsilon(V_m, w) * k * V_m * (1 + b - V_m) - epsilon(V_m, w) * w
        epsilon(V_m, w) = epsilon_0 + mu_1 * w / (mu_2 + V_m)
        y = w, q(y,t) = epsilon(V_m, w) * k * V_m * (1 + b - V_m), p(y,t) = epsilon(V_m, w)
        """
        epsilon_Vm_w = self.epsilon_0 + self.mu_1 * self.w[i] / (self.mu_2 + self.Vm[i])
        self.w[i] = self.w[i] * tm.exp(-1.0 * dt * epsilon_Vm_w) + (
                self.k * self.Vm[i] * (1.0 + self.b - self.Vm[i])) * (
                               1.0 - tm.exp(-1.0 * dt * epsilon_Vm_w))

    @ti.kernel
    def calculate_M_and_K(self):
        fiber, sheet, normal = ti.static(self.fiber, self.sheet, self.normal)
        for i in range(self.body.num_tet):
            self.Me[i] = 0.25 / 6.0 * ti.abs(self.Be[i].determinant()) * \
                         tm.mat4([[1.0, 0.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]]) * 6.0
            J_phi = ti.Matrix([[-1., -1., -1.],
                               [1., 0., 0.],
                               [0., 1., 0.],
                               [0., 0., 1.]], float)
            D_f = ti.Matrix([[fiber[i][0] * fiber[i][0],
                              fiber[i][1] * fiber[i][0],
                              fiber[i][2] * fiber[i][0]],
                             [fiber[i][0] * fiber[i][1],
                              fiber[i][1] * fiber[i][1],
                              fiber[i][2] * fiber[i][1]],
                             [fiber[i][0] * fiber[i][2],
                              fiber[i][1] * fiber[i][2],
                              fiber[i][2] * fiber[i][2]]], float)
            D_s = ti.Matrix([[sheet[i][0] * sheet[i][0],
                              sheet[i][1] * sheet[i][0],
                              sheet[i][2] * sheet[i][0]],
                             [sheet[i][0] * sheet[i][1],
                              sheet[i][1] * sheet[i][1],
                              sheet[i][2] * sheet[i][1]],
                             [sheet[i][0] * sheet[i][2],
                              sheet[i][1] * sheet[i][2],
                              sheet[i][2] * sheet[i][2]]], float)
            D_n = ti.Matrix([[normal[i][0] * normal[i][0],
                              normal[i][1] * normal[i][0],
                              normal[i][2] * normal[i][0]],
                             [normal[i][0] * normal[i][1],
                              normal[i][1] * normal[i][1],
                              normal[i][2] * normal[i][1]],
                             [normal[i][0] * normal[i][2],
                              normal[i][1] * normal[i][2],
                              normal[i][2] * normal[i][2]]], float)
            norm_f = fiber[i][0] * fiber[i][0] + fiber[i][1] * fiber[i][1] + fiber[i][2] * fiber[i][2]
            norm_s = fiber[i][0] * fiber[i][0] + fiber[i][1] * fiber[i][1] + fiber[i][2] * fiber[i][2]
            norm_n = fiber[i][0] * fiber[i][0] + fiber[i][1] * fiber[i][1] + fiber[i][2] * fiber[i][2]
            self.DM[i] = self.sigma_f / norm_f * D_f + self.sigma_s / norm_s * D_s + self.sigma_n / norm_n * D_n
            J = self.F[i].determinant()
            A = J_phi @ self.Be[i].inverse() @ self.F[i].inverse()
            self.Ke[i] = 1.0 / 6.0 * J * ti.abs(self.Be[i].determinant()) * A @ self.DM[i] @ A.transpose() * 6.0
            self.D[i] = J * self.F[i].inverse() @ self.DM[i] @ self.F[i].inverse().transpose()

    @ti.kernel
    def compute_RHS(self):
        for i in self.cg_b:
            self.cg_b[i] = 0.0
        # rhs = b = f * dt + M * u(t), here, f = 0
        for i in range(self.body.num_tet):
            id0, id1, id2, id3 = (self.body.elements[i][0], self.body.elements[i][1],
                                  self.body.elements[i][2], self.body.elements[i][3])
            self.cg_b[id0] += (self.Me[i][0, 0] * self.Vm[id0] + self.Me[i][0, 1] * self.Vm[id1] +
                               self.Me[i][0, 2] * self.Vm[id2] + self.Me[i][0, 3] * self.Vm[id3])
            self.cg_b[id1] += (self.Me[i][1, 0] * self.Vm[id0] + self.Me[i][1, 1] * self.Vm[id1] +
                               self.Me[i][1, 2] * self.Vm[id2] + self.Me[i][1, 3] * self.Vm[id3])
            self.cg_b[id2] += (self.Me[i][2, 0] * self.Vm[id0] + self.Me[i][2, 1] * self.Vm[id1] +
                               self.Me[i][2, 2] * self.Vm[id2] + self.Me[i][2, 3] * self.Vm[id3])
            self.cg_b[id3] += (self.Me[i][3, 0] * self.Vm[id0] + self.Me[i][3, 1] * self.Vm[id1] +
                               self.Me[i][3, 2] * self.Vm[id2] + self.Me[i][3, 3] * self.Vm[id3])

    @ti.func
    def A_mult_x(self, dt, dst, src):
        # lhs = Ax = (M + K * dt) * u(t+1)
        for i in range(self.body.num_vertex):
            dst[i] = 0.0

        for i in range(self.body.num_tet):
            id0, id1, id2, id3 = (self.body.elements[i][0], self.body.elements[i][1],
                                  self.body.elements[i][2], self.body.elements[i][3])
            dst[id0] += (self.Me[i][0, 0] * src[id0] + self.Me[i][0, 1] * src[id1] +
                         self.Me[i][0, 2] * src[id2] + self.Me[i][0, 3] * src[id3])
            dst[id1] += (self.Me[i][1, 0] * src[id0] + self.Me[i][1, 1] * src[id1] +
                         self.Me[i][1, 2] * src[id2] + self.Me[i][1, 3] * src[id3])
            dst[id2] += (self.Me[i][2, 0] * src[id0] + self.Me[i][2, 1] * src[id1] +
                         self.Me[i][2, 2] * src[id2] + self.Me[i][2, 3] * src[id3])
            dst[id3] += (self.Me[i][3, 0] * src[id0] + self.Me[i][3, 1] * src[id1] +
                         self.Me[i][3, 2] * src[id2] + self.Me[i][3, 3] * src[id3])
            dst[id0] += (self.Ke[i][0, 0] * src[id0] + self.Ke[i][0, 1] * src[id1] +
                         self.Ke[i][0, 2] * src[id2] + self.Ke[i][0, 3] * src[id3]) * dt
            dst[id1] += (self.Ke[i][1, 0] * src[id0] + self.Ke[i][1, 1] * src[id1] +
                         self.Ke[i][1, 2] * src[id2] + self.Ke[i][1, 3] * src[id3]) * dt
            dst[id2] += (self.Ke[i][2, 0] * src[id0] + self.Ke[i][2, 1] * src[id1] +
                         self.Ke[i][2, 2] * src[id2] + self.Ke[i][2, 3] * src[id3]) * dt
            dst[id3] += (self.Ke[i][3, 0] * src[id0] + self.Ke[i][3, 1] * src[id1] +
                         self.Ke[i][3, 2] * src[id2] + self.Ke[i][3, 3] * src[id3]) * dt

    @ti.func
    def dot(self, v1, v2):
        result = 0.0
        for i in range(self.body.num_vertex):
            result += v1[i] * v2[i]
        return result

    # cg #
    # --------------------------------------------------------------------------------------------------- #
    @ti.kernel
    def cg_before_ite(self, dt: float) -> float:
        for i in range(self.body.num_vertex):
            self.cg_x[i] = self.Vm[i]
        self.A_mult_x(dt, self.cg_Ax, self.cg_x)

        for i in range(self.body.num_vertex):
            # r = b - A @ x
            self.cg_r[i] = self.cg_b[i] - self.cg_Ax[i]
            # d = r
            self.cg_d[i] = self.cg_r[i]

        delta_new = self.dot(self.cg_r, self.cg_r)
        return delta_new

    @ti.kernel
    def cg_run_iteration(self, dt: float, delta: float) -> float:
        delta_new = delta
        # q = A @ d
        self.A_mult_x(dt, self.cg_Ad, self.cg_d)
        # alpha = delta_new / d.dot(q)
        alpha = delta_new / self.dot(self.cg_d, self.cg_Ad)

        for i in range(self.body.num_vertex):
            # x = x + alpha * d
            self.cg_x[i] += alpha * self.cg_d[i]
            # r = b - A @ x || r = r - alpha * q
            self.cg_r[i] -= alpha * self.cg_Ad[i]
        delta_old = delta_new
        delta_new = self.dot(self.cg_r, self.cg_r)
        beta = delta_new / delta_old
        for i in range(self.body.num_vertex):
            # d = r + beta * d
            self.cg_d[i] = self.cg_r[i] + beta * self.cg_d[i]
        return delta_new

    def cg(self, dt: float):
        delta_new = self.cg_before_ite(dt)
        delta_0 = delta_new
        ite, iteMax = 0, 100
        while ite < iteMax and delta_new > (self.cg_epsilon**2) * delta_0:
            delta_new = self.cg_run_iteration(dt, delta_new)
            ite += 1

    @ti.kernel
    def cgUpdateVm(self):
        for i in self.Vm:
            self.Vm[i] = self.cg_x[i]

    def calculate_diffusion(self, dt):
        self.calculate_M_and_K()
        self.compute_RHS()
        self.cg(dt)
        self.cgUpdateVm()

    def update_Vm(self, dt):
        self.calculate_reaction(dt * 0.5)
        self.calculate_diffusion(dt)
        self.calculate_reaction(dt * 0.5)

    @ti.kernel
    def apply_stimulation_s1(self):
        vert = ti.static(self.body.vertex)
        for i in vert:
            if self.tag_s1[i] == 1:
                self.Vm[i] = 1.52

    @ti.kernel
    def apply_stimulation_s2_1(self):
        vert = ti.static(self.body.vertex)
        for i in vert:
            if self.tag_s2_1[i] == 1:
                self.Vm[i] = 1.55

    def apply_stimulation_s2_2(self):
        vert = ti.static(self.body.vertex)
        for i in vert:
            if self.tag_s2_2[i] == 1:
                self.Vm[i] = 1.55

    @ti.kernel
    def get_near_vertex_index(self, x: float, y: float, z: float) -> int:
        vert = ti.static(self.body.vertex)
        res = 0
        for i in vert:
            if (vert[i][0] - x)**2 + (vert[i][1] - y)**2 + (vert[i][2] - z)**2 < 1e-2:
                res = i
                print(i)
        return res

    def update(self, sub_steps):
        dt = 1. / 1.29 / 6. / sub_steps
        # dt = 1. / 60. / sub_steps
        for _ in range(sub_steps):
            self.update_Vm(dt)

        self.update_color()

    @ti.kernel
    def update_color(self):
        color = ti.static(self.body.vert_color)
        for i in color:
            color[i] = tm.vec3([self.Vm[i], self.Vm[i], 1])

    @ti.kernel
    def calculate_A(self, dt: float):
        for i, j in self.cg_A:
            self.cg_A[i, j] = 0.0

        for i in range(self.body.num_tet):
            id0, id1, id2, id3 = (self.body.elements[i][0], self.body.elements[i][1],
                                  self.body.elements[i][2], self.body.elements[i][3])
            self.cg_A[id0, id0] += self.Me[i][0, 0] + self.Ke[i][0, 0] * dt
            self.cg_A[id0, id1] += self.Me[i][0, 1] + self.Ke[i][0, 1] * dt
            self.cg_A[id0, id2] += self.Me[i][0, 2] + self.Ke[i][0, 2] * dt
            self.cg_A[id0, id3] += self.Me[i][0, 3] + self.Ke[i][0, 3] * dt

            self.cg_A[id1, id0] += self.Me[i][1, 0] + self.Ke[i][1, 0] * dt
            self.cg_A[id1, id1] += self.Me[i][1, 1] + self.Ke[i][1, 1] * dt
            self.cg_A[id1, id2] += self.Me[i][1, 2] + self.Ke[i][1, 2] * dt
            self.cg_A[id1, id3] += self.Me[i][1, 3] + self.Ke[i][1, 3] * dt

            self.cg_A[id2, id0] += self.Me[i][2, 0] + self.Ke[i][2, 0] * dt
            self.cg_A[id2, id1] += self.Me[i][2, 1] + self.Ke[i][2, 1] * dt
            self.cg_A[id2, id2] += self.Me[i][2, 2] + self.Ke[i][2, 2] * dt
            self.cg_A[id2, id3] += self.Me[i][2, 3] + self.Ke[i][2, 3] * dt

            self.cg_A[id3, id0] += self.Me[i][3, 0] + self.Ke[i][3, 0] * dt
            self.cg_A[id3, id1] += self.Me[i][3, 1] + self.Ke[i][3, 1] * dt
            self.cg_A[id3, id2] += self.Me[i][3, 2] + self.Ke[i][3, 2] * dt
            self.cg_A[id3, id3] += self.Me[i][3, 3] + self.Ke[i][3, 3] * dt


def read_data():
    # 顶点位置
    pos_np = np.array(meshData['verts'], dtype=float)
    pos_np = pos_np.reshape((-1, 3))
    # 四面体顶点索引
    tet_np = np.array(meshData['tetIds'], dtype=int)
    tet_np = tet_np.reshape((-1, 4))
    # tet_fiber方向
    fiber_tet_np = np.array(meshData['fiberDirection'], dtype=float)
    fiber_tet_np = fiber_tet_np.reshape((-1, 3))
    # tet_sheet方向
    sheet_tet_np = np.array(meshData['sheetDirection'], dtype=float)
    sheet_tet_np = sheet_tet_np.reshape((-1, 3))
    # tet_normal方向
    normal_tet_np = np.array(meshData['normalDirection'], dtype=float)
    normal_tet_np = normal_tet_np.reshape((-1, 3))
    # num_tet_set
    num_tet_set_np = np.array(meshData['num_tet_set'], dtype=int)[0]
    # tet_set
    tet_set_np = np.array(meshData['tet_set'], dtype=int)
    # bou_tag
    bou_base_face_np = np.array(meshData['bou_base_face'], dtype=int)
    bou_base_face_np = bou_base_face_np.reshape((-1, 3))
    bou_endo_lv_face_np = np.array(meshData['bou_endo_lv_face'], dtype=int)
    bou_endo_lv_face_np = bou_endo_lv_face_np.reshape((-1, 3))
    bou_endo_rv_face_np = np.array(meshData['bou_endo_rv_face'], dtype=int)
    bou_endo_rv_face_np = bou_endo_rv_face_np.reshape((-1, 3))
    bou_epi_face_np = np.array(meshData['bou_epi_face'], dtype=int)
    bou_epi_face_np = bou_epi_face_np.reshape((-1, 3))

    Body_ = Body(vert_np=pos_np,
                 tet_np=tet_np,
                 tet_fiber_np=fiber_tet_np,
                 tet_sheet_np=sheet_tet_np,
                 tet_normal_np=normal_tet_np,
                 num_tet_set_np=num_tet_set_np,
                 tet_set_np=tet_set_np,
                 bou_base_np=bou_base_face_np,
                 bou_epi_np=bou_epi_face_np,
                 bou_endo_lv_np=bou_endo_lv_face_np,
                 bou_endo_rv_np=bou_endo_rv_face_np
                 )

    return Body_


# @ti.kernel
# def get_vert_fiber_field_LV1():
#     for i in body.vertex:
#         vert_fiber_field[2 * i] = body.vertex[i]
#         # vert_fiber = vert_fiber_taichi[i]
#         vert_fiber = tm.vec3(vert_fiber_taichi[i][1], vert_fiber_taichi[i][0], vert_fiber_taichi[i][2])
#         vert_fiber_field[2 * i + 1] = body.vertex[i] + vert_fiber
    # for i in body.elements:
    #     id0, id1, id2, id3 = body.elements[i][0], body.elements[i][1], body.elements[i][2], body.elements[i][3]
    #     fiber_field_vertex[2 * i] = body.vertex[id0] + body.vertex[id1] + body.vertex[id2] + body.vertex[id3]
    #     fiber_field_vertex[2 * i] /= 4.0
    #     fiber = body.tet_fiber[i]
    #     fiber_field_vertex[2 * i + 1] = fiber_field_vertex[2 * i] + fiber * 0.1

# @ti.kernel
# def get_bou_endo_normal_field_LV1():
#     for i in dynamics_sys.bou_endo_face:
#         id0, id1, id2 = dynamics_sys.bou_endo_face[i][0], dynamics_sys.bou_endo_face[i][1], dynamics_sys.bou_endo_face[i][2]
#         bou_endo_normal_field[2 * i] = (dynamics_sys.pos[id0] + dynamics_sys.pos[id1] + dynamics_sys.pos[id2]) / 3.0
#         bou_endo_normal_field_color[2 * i] = tm.vec3(1.0, 0., 0.)
#         bou_endo_normal_field[2 * i + 1] = bou_endo_normal_field[2 * i] + dynamics_sys.normal_bou_endo_face[i] * 0.5
#         bou_endo_normal_field_color[2 * i] = tm.vec3(0.0, 1., 0.)


def experiment4_1():
    body = read_data()

    ep_sys = diffusion_reaction(body=body)
    # num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
    # dynamics_sys = xpbd.XPBD_SNH_with_active(body=body, num_pts_np=num_per_tet_set_np)
    #
    # # ep_sys.apply_stimulation()
    # # print(body.tet_fiber)
    # body.set_Ta(60)
    # ep_sys.update_color()

    # draw fiber field
    # vert_fiber_field = ti.Vector.field(3, dtype=float, shape=(2 * body.num_vertex))
    # vert_fiber_np = np.array(meshData['vert_fiber'], dtype=float)
    # vert_fiber_np = vert_fiber_np.reshape((-1, 3))
    # vert_fiber_taichi = ti.Vector.field(3, float, shape=(body.num_vertex,))
    # vert_fiber_taichi.from_numpy(vert_fiber_np)
    # get_vert_fiber_field_LV1()

    # draw bou endo normal
    # bou_endo_normal_field = ti.Vector.field(3, float, shape=(2 * dynamics_sys.num_bou_endo_face,))
    # bou_endo_normal_field_color = ti.Vector.field(3, float, shape=(2 * dynamics_sys.num_bou_endo_face,))
    # get_bou_endo_normal_field_LV1()

    # ---------------------------------------------------------------------------- #
    #                                      gui                                     #
    # ---------------------------------------------------------------------------- #

    open_gui = True
    # set parameter
    windowLength = 1024
    lengthScale = min(windowLength, 512)
    light_distance = lengthScale / 25.

    if open_gui:
        # init the window, canvas, scene and camera
        window = ti.ui.Window("body show", (windowLength, windowLength), vsync=True)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        # initial camera position
        camera.position(-10.50518621, 40.032794, 117.36565294)
        camera.lookat(-10.47637129, 39.56587387, 116.48182304)
        camera.up(0., 1., 0.)

        iter_time = 0
        while window.running:
            if iter_time == 0 or iter_time == 1000:
                ep_sys.apply_stimulation_s1()
                iter_time += 1
            elif iter_time == 300 or iter_time == 1300:  # 106
                ep_sys.apply_stimulation_s2_1()
                iter_time += 1
            else:
                iter_time += 1

            ep_sys.update(1)
            # dynamics_sys.update()
            # print(body.tet_Ta)

            # set the camera, you can move around by pressing 'wasdeq'
            camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.LMB)
            scene.set_camera(camera)

            # set the light
            scene.point_light(pos=(-light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.point_light(pos=(light_distance, 0., light_distance), color=(0.5, 0.5, 0.5))
            scene.ambient_light(color=(0.5, 0.5, 0.5))

            # draw
            # scene.particles(body.vertex, radius=0.02, color=(0, 1, 1))
            # scene.mesh(body.vertex, indices=body.surfaces, per_vertex_color=vert_color, two_sided=False)
            scene.mesh(body.vertex, indices=body.surfaces, two_sided=False, per_vertex_color=ep_sys.body.vert_color)
            # scene.lines(vert_fiber_field, color=(0., 0.0, 1.), width=2.0)
            # scene.lines(bou_endo_normal_field, per_vertex_color=bou_endo_normal_field_color, width=2.0)

            # show the frame
            canvas.scene(scene)
            window.show()
        
        # print(camera.curr_position)
        # print(camera.curr_lookat)
        # print(camera.curr_up)
        # print(ep_sys.Vm)


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f32, kernel_profiler=True, device_memory_fraction=0.9, device_memory_GB=4)
    experiment4_1()

