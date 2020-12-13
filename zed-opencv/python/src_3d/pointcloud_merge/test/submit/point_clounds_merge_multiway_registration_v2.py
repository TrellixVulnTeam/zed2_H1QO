import open3d as o3d
import numpy as np
from  datetime import datetime as dt
from point_clounds_remove_noise_v3 import guided_filter,remove_statistical_outlier,remove_radius_outlier
import copy as cp
import os
#http://www.open3d.org/docs/latest/tutorial/Advanced/multiway_registration.html

def add_color_normal(pcd,size):
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    pcd.estimate_normals( kdt_n)
    return pcd
def load_point_clouds(fns):
    pcds = []
    for fn in fns:
        pcd = o3d.io.read_point_cloud(fn)
        pcds.append(pcd)
    return pcds
def pairwise_registration(source, target, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    print("Apply point-to-plane ICP")
    # print(max_correspondence_distance_coarse,max_correspondence_distance_fine)
    icp_coarse = o3d.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


def pcds_full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse,
                      max_correspondence_distance_fine)
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph
def fns_full_registration(pcds_down,max_correspondence_distance_coarse,max_correspondence_distance_fine):

    print("Full registration ...")
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = pcds_full_registration(pcds_down,
                                       max_correspondence_distance_coarse,
                                       max_correspondence_distance_fine)
    return pose_graph
def Optimizing_PoseGraph(pose_graph,max_correspondence_distance_fine):
    print("Optimizing PoseGraph ...")
    option = o3d.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.registration.global_optimization(
            pose_graph,
            o3d.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.registration.GlobalOptimizationConvergenceCriteria(),
            option)
    return pose_graph
def get_absolute_file_paths(directory,fn='image.npy'):
   fils_list=[]
   fn_list=[]
   dir_list=[]
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           if  f.find(fn)>=0:
               fils_list.append( os.path.abspath(os.path.join(dirpath, f)))
               fn_list.append( f)
               dir_list.append(dirpath)
   return fils_list,fn_list,dir_list
def get_merge_fns(basep,i):
    fn_merge='pcd_mask_mergeid_%04d.ply'%(i)#0000
    files_list, fn_list, dir_list = get_absolute_file_paths(basep,  fn=fn_merge)
    return files_list
def main_ransac_icp(id,fuchi,pose):
    basep = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_{fuchi}/{pose}'
    baseout = f'C:/00_work/05_src/data/20201202_shiyoko_6f/merge_{fuchi}/{pose}'
    os.makedirs(baseout,exist_ok=True)
    files_list=get_merge_fns(basep, id)
    pcds_load = load_point_clouds(files_list)
    voxel_size_lst=[]
    for i ,pcd in enumerate(pcds_load):
        voxel_sizet = np.abs((pcd.get_max_bound() -pcd.get_min_bound())).max() / 100
        voxel_size_lst.append(voxel_sizet)
    voxel_size=max(voxel_size_lst)
    max_correspondence_distance_coarse = voxel_size * 30
    max_correspondence_distance_fine = voxel_size * 1.2
    print('voxel_size:', voxel_size)
    pcdsn = []
    for pcd in pcds_load:
        pcdr=remove_radius_outlier(pcd)
        pcdn = add_color_normal(pcdr, voxel_size)
        pcdsn.append(pcdn)
    pcds_down = pcdsn
    # pcds_down = []
    # for pcd in pcdsn:
    #     pcd_filter = guided_filter(pcd, radius=0.01, epsilon=0.1)
    #     pcds_down.append(pcd_filter)
    pcds = cp.deepcopy(pcds_down)
    pose_graph = fns_full_registration(pcds_down, max_correspondence_distance_coarse, max_correspondence_distance_fine)
    pose_graph = Optimizing_PoseGraph(pose_graph, max_correspondence_distance_fine)

    for point_id in range(len(pcds_down)):
        print(pose_graph.nodes[point_id].pose)
        pcds_down[point_id].transform(pose_graph.nodes[point_id].pose)

    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]

    # str_dt = dt.utcnow().strftime('%y%m%d_%H%M%S')
    ft_pcd_m = f'{baseout}/mwr_pcd_aligned_mergeid_%04d.ply' % (id)
    o3d.io.write_point_cloud(ft_pcd_m, pcd_combined)
if __name__ == "__main__":
    main_ransac_icp(0, 'org', 'd2_a15_h1.5')