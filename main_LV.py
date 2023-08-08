import taichi as ti
import taichi.math as tm
import numpy as np
from Geometry.body_LV import Body
import Electrophysiology.ep_LV as ep
import Dynamics.XPBD.XPBD_SNH_LV as xpbd
from data.LV1 import meshData
# from data.cube import meshData
# from data.heart import meshData


def read_data():
    # 顶点位置
    pos_np = np.array(meshData['verts'], dtype=float)
    pos_np = pos_np.reshape((-1, 3))
    # 四面体顶点索引
    tet_np = np.array(meshData['tetIds'], dtype=int)
    tet_np = tet_np.reshape((-1, 4))
    # edge
    edge_np = np.array(meshData['tetEdgeIds'], dtype=int)
    edge_np = edge_np.reshape((-1, 2))
    # surface tri index
    # surf_tri_np = np.array(meshData['tetSurfaceTriIds'], dtype=int)
    # surf_tri_np = surf_tri_np.reshape((-1, 3))
    # tet_fiber方向
    fiber_tet_np = np.array(meshData['fiberDirection'], dtype=float)
    fiber_tet_np = fiber_tet_np.reshape((-1, 3))

    # tet_sheet方向
    sheet_tet_np = np.array(meshData['sheetDirection'], dtype=float)
    sheet_tet_np = sheet_tet_np.reshape((-1, 3))
    # num_edge_set
    num_edge_set_np = np.array(meshData['num_edge_set'], dtype=int)[0]
    # edge_set
    edge_set_np = np.array(meshData['edge_set'], dtype=int)
    # num_tet_set
    num_tet_set_np = np.array(meshData['num_tet_set'], dtype=int)[0]
    # tet_set
    tet_set_np = np.array(meshData['tet_set'], dtype=int)
    # bou_tag
    bou_tag_dirichlet_np = np.array(meshData['bou_tag_dirichlet'], dtype=int)
    bou_tag_neumann_np = np.array(meshData['bou_tag_neumann'], dtype=int)

    bou_endo_np = np.array(meshData['bou_endo_lv_face'], dtype=int)
    bou_endo_np = bou_endo_np.reshape((-1, 3))
    bou_epi_np = np.array(meshData['bou_epi_lv_face'], dtype=int)
    bou_epi_np = bou_epi_np.reshape((-1, 3))

    Body_ = Body(pos_np, tet_np, edge_np, fiber_tet_np, sheet_tet_np, num_edge_set_np, edge_set_np, num_tet_set_np,
                tet_set_np, bou_tag_dirichlet_np, bou_tag_neumann_np, bou_endo_np, bou_epi_np)
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


if __name__ == "__main__":
    ti.init(arch=ti.cuda, default_fp=ti.f32, kernel_profiler=True)

    body = read_data()
    # body.translation(0., 20.5, 0.)
    num_per_tet_set_np = np.array(meshData['sum_tet_set'], dtype=int)
    dynamics_sys = xpbd.XPBD_SNH_with_active(body=body, num_pts_np=num_per_tet_set_np)
    ep_sys = ep.diffusion_reaction(body=body)
    # ep_sys.apply_stimulation()
    # print(body.tet_fiber)
    body.set_Ta(60)
    ep_sys.update_color()

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

    # for i in range(body.vertex.shape[0]):
    #     body.bou_tag_dirichlet[i] = 0

    vert_color = ti.Vector.field(3, float, shape=(body.num_vertex,))
    for i in range(vert_color.shape[0]):
        vert_color[i] = tm.vec3(1.0, 0., 0.)

    if open_gui:
        # init the window, canvas, scene and camera
        window = ti.ui.Window("body show", (windowLength, windowLength), vsync=True)
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        # initial camera position
        camera.position(2.28, 22.6, 34.89)
        camera.lookat(2.24, 22, 34)
        camera.up(0., 1., 0.)

        sigma_para = 5e-1
        ep_sys.sigma_f = sigma_para
        ep_sys.sigma_s = sigma_para
        ep_sys.sigma_n = sigma_para
        iter_time = 0
        while window.running:
            if iter_time == 0:
                ep_sys.apply_stimulation()
                iter_time += 1
            elif iter_time == 10:
                ep_sys.cancel_stimulation()
                iter_time += 1
            elif iter_time < 600:
                iter_time += 1
            elif iter_time == 600:
                iter_time = 0

            ep_sys.update(1)
            dynamics_sys.update()
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
            scene.mesh(body.vertex, indices=body.surfaces, two_sided=False, per_vertex_color=ep_sys.vertex_color)
            # scene.lines(vert_fiber_field, color=(0., 0.0, 1.), width=2.0)
            # scene.lines(bou_endo_normal_field, per_vertex_color=bou_endo_normal_field_color, width=2.0)

            # show the frame
            canvas.scene(scene)
            window.show()

        # print(camera.curr_position)
        # print(camera.curr_lookat)
        # print(camera.curr_up)
        print(ep_sys.Vm)
