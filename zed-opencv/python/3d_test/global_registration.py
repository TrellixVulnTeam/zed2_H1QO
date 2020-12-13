#http://www.open3d.org/docs/latest/tutorial/Advanced/global_registration.html
import open3d as o3d
import copy
import numpy as np
from point_clounds_remove_noise_v3 import guided_filter,remove_statistical_outlier,remove_radius_outlier
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      # zoom=0.4559,
                                      # front=[0.6452, -0.3036, -0.7011],
                                      # lookat=[1.9892, 2.0208, 1.8945],
                                      # up=[-0.2779, -0.9482, 0.1556]
                                      )
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    # pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd = remove_radius_outlier(pcd)
    pcd_down = pcd.uniform_down_sample(every_k_points=5)
    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 30
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh
def prepare_dataset(voxel_size):
    id = '0000'
    ply0 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/d2_a15_h1/cam0/b/20201202155214/pcd_mask_mergeid_{id}.ply'
    ply1 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/d2_a15_h1/cam1/b/20201202155216/pcd_mask_mergeid_{id}.ply'

    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(ply0)
    target = o3d.io.read_point_cloud(ply1)
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    # draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(40000,500))
    return result
def refine_registration(source, target,result_ransac, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result
def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result_ransac = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    result=refine_registration(source_down, target_down, result_ransac, source_fpfh, target_fpfh, voxel_size)
    return result

#----prepare data
voxel_size = 0.01  # means 5cm for this dataset
def fast_merge():
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size)


    result_ransac = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   voxel_size)

    # print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)


def global_merge():
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size)
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    draw_registration_result(source_down, target_down, result_ransac.transformation)


def refine_merge():
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(
        voxel_size)
    result_icp = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                     voxel_size)
    print(result_icp)
    # draw_registration_result(source, target, result_icp.transformation)

    # print(result_fast)
    draw_registration_result(source_down, target_down, result_icp.transformation)

if __name__ == "__main__":
    global_merge()