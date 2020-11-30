#http://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html
import open3d as o3d
# import open3d_tutorial as o3dtut
import numpy as np
import matplotlib.pyplot as plt
from  datetime import datetime as dt
# mesh = o3dtut.get_bunny_mesh()
# pcd = mesh.sample_points_poisson_disk(750)
# o3d.visualization.draw_geometries([pcd])
def add_color_normal(pcd): # in-place coloring and adding normal
    # pcd.paint_uniform_color(np.random.rand(3))
    size = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 30
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    pcd.estimate_normals( kdt_n)
    pass

# alpha = 0.03
# print(f"alpha={alpha:.3f}")
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
def surface_reconstruction(pcd):
    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
    # print(mesh)
    # o3d.visualization.draw_geometries([mesh],
    #                                   zoom=0.664,
    #                                   front=[-0.4761, -0.4698, -0.7434],
    #                                   lookat=[1.8900, 3.2596, 0.9284],
    #                                   up=[0.2304, -0.8825, 0.4101])
    return mesh, densities
def visualize_densities(densities):
    print('visualize densities')
    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    o3d.visualization.draw_geometries([density_mesh],
                                      zoom=0.664,
                                      front=[-0.4761, -0.4698, -0.7434],
                                      lookat=[1.8900, 3.2596, 0.9284],
                                      up=[0.2304, -0.8825, 0.4101])
def remove_low_density_vertices(mesh,densities):
    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(mesh)
    o3d.visualization.draw_geometries([mesh],
                                      zoom=0.664,
                                      front=[-0.4761, -0.4698, -0.7434],
                                      lookat=[1.8900, 3.2596, 0.9284],
                                      up=[0.2304, -0.8825, 0.4101])
def Normal_Estimation(pcd):
    # gt_mesh = o3dtut.get_bunny_mesh()
    # pcd = gt_mesh.sample_points_poisson_disk(5000)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros(
        (1, 3)))  # invalidate existing normals

    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
fn='reconstruction-000001.cloud-ZED_21888201.ply'
pcd = o3d.io.read_point_cloud(fn)
add_color_normal(pcd)
# mesh, densities=surface_reconstruction(pcd)
# visualize_densities(densities)
# remove_low_density_vertices(mesh,densities)
Normal_Estimation(pcd)