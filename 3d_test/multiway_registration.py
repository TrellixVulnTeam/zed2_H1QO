import open3d as o3d
import numpy as np
from  datetime import datetime as dt
#http://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html

def add_color_normal(pcd): # in-place coloring and adding normal
    # pcd.paint_uniform_color(np.random.rand(3))
    size = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 30
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    pcd.estimate_normals( kdt_n)
    pass
def load_point_clouds(fns,voxel_size=0.0):
    pcds = []
    for fn in fns:
        pcd = o3d.io.read_point_cloud(fn)
        # pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        add_color_normal(pcd)
        pcds.append(pcd)
    return pcds
def view_org_point_cloud(fns,voxel_size):
    pcds_down = load_point_clouds(fns,voxel_size)
    # o3d.visualization.draw_geometries(pcds_down,
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])
    return pcds_down
def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def pcds_full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph
def fns_full_registration(pcds_down,max_correspondence_distance_coarse,max_correspondence_distance_fine):
    # pcds_down = load_point_clouds(fns,voxel_size)
        # o3d.visualization.draw_geometries(pcds_down,
        #                                   zoom=0.3412,
        #                                   front=[0.4257, -0.2125, -0.8795],
        #                                   lookat=[2.6172, 2.0475, 1.532],
        #                                   up=[-0.0694, -0.9768, 0.2024])
    print("Full registration ...")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = pcds_full_registration(pcds_down,
                                       max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine)
    return pose_graph
def Optimizing_PoseGraph(pose_graph,max_correspondence_distance_fine):
    print("Optimizing PoseGraph ...")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
    return pose_graph
if __name__ == "__main__":
    fns = ['reconstruction-000001.cloud-ZED_21888201.ply',
           # 'reconstruction-000001.cloud-ZED_22378008.ply',
           # 'reconstruction-000000.cloud-ZED_22378008.ply',
           'reconstruction-000002.cloud-ZED_21888201.ply']

    voxel_size = 0.02
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    pcds_down = view_org_point_cloud(fns,voxel_size)
    pose_graph=fns_full_registration(pcds_down, max_correspondence_distance_coarse, max_correspondence_distance_fine)

    # o3d.io.write_point_cloud("Full_registration.ply", pose_graph)
    pose_graph=Optimizing_PoseGraph(pose_graph, max_correspondence_distance_fine)
    # o3d.io.write_point_cloud("Optimizing_PoseGraph.ply", pose_graph)
    print("Transform points and display")
    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)
    # o3d.visualization.draw_geometries(pcds_down,
    #                                   zoom=0.3412,
    #                                   front=[0.4257, -0.2125, -0.8795],
    #                                   lookat=[2.6172, 2.0475, 1.532],
    #                                   up=[-0.0694, -0.9768, 0.2024])
    pcds = load_point_clouds(fns,voxel_size)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]
    # str_dt=dt.utcnow().strftime('%y%m%d_%H%M%S')
    # o3d.io.write_point_cloud(f"multiway_registration_{str_dt}.ply", pcd_combined)
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    str_dt=dt.utcnow().strftime('%y%m%d_%H%M%S')
    o3d.io.write_point_cloud(f"multiway_registration_{str_dt}.ply", pcd_combined_down)
    o3d.visualization.draw_geometries([pcd_combined_down],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
