#https://qiita.com/n_chiba_/items/fc9605cde5c19a8c7dad#%E3%83%A1%E3%82%A4%E3%83%B3
import open3d
import numpy as np
import copy
def preprocess_point_cloud(pointcloud, voxel_size):
    # Keypoint を Voxel Down Sample で生成
    keypoints = pointcloud.voxel_down_sample( voxel_size)

    # 法線推定
    radius_normal = voxel_size * 2
    view_point = np.array([0., 10., 10.], dtype="float64")
    keypoints.estimate_normals(
        search_param = open3d.geometry.KDTreeSearchParamHybrid(radius = radius_normal, max_nn = 30))
    keypoints.orient_normals_towards_camera_location( camera_location = view_point)

    #　FPFH特徴量計算
    radius_feature = voxel_size * 5
    fpfh = open3d.registration.compute_fpfh_feature(
        keypoints,
        search_param = open3d.geometry.KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))

    return keypoints, fpfh
def execute_global_registration(kp1, kp2, fpfh1, fpfh2, voxel_size):
    distance_threshold = voxel_size * 2.5
    result = open3d.registration.registration_ransac_based_on_feature_matching(
        kp1, kp2, fpfh1, fpfh2, distance_threshold,
        open3d.registration.TransformationEstimationPointToPoint(False), 4,
        [open3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         open3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        open3d.registration.RANSACConvergenceCriteria(500000, 1000))
    return result
def refine_registration(scene1, scene2, trans, voxel_size):
    distance_threshold = voxel_size * 0.4
    result = open3d.registration.registration_icp(
        scene1, scene2, distance_threshold, trans,
        open3d.registration.TransformationEstimationPointToPoint())
    return result
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # open3d.visualization.draw_geometries([source_temp, target_temp])
    return source, target

def make_pcd(points, colors):
  pcd = open3d.geometry.PointCloud()
  pcd.points = open3d.utility.Vector3dVector(points)
  pcd.colors = open3d.utility.Vector3dVector(colors)
  return pcd

if __name__ == "__main__":
    id = '0000'
    ply0 = f'C:/00_work/05_src/data/20201124/202011161_sample/ply/cam0/20201116133246_027942/pcd_mask_mergeid_{id}.ply'
    ply1 = f'C:/00_work/05_src/data/20201124/202011161_sample/ply/cam1/20201116133246_072897/pcd_mask_mergeid_{id}.ply'


    id = '0047'
    ply0=f'C:/00_work/05_src/data/20201124/202011161_sample/ply/cam0/20201116133925_495618/pcd_mask_mergeid_{id}.ply'
    ply1=f'C:/00_work/05_src/data/20201124/202011161_sample/ply/cam1/20201116133925_659685/pcd_mask_mergeid_{id}.ply'
    scene1 = open3d.io.read_point_cloud(ply0)
    scene2 = open3d.io.read_point_cloud(ply1)

    # scene2 を適当に回転・並進
    transform_matrix = np.asarray([
        [1., 0., 0., -0.1],
        [0., 0., -1., 0.1],
        [0., 1., 0., -0.1],
        [0., 0., 0., 1.]], dtype="float64")
    scene2.transform(transform_matrix)

    # 位置合わせ前の点群の表示
    draw_registration_result(scene1, scene2, np.eye(4))

    voxel_size = 0.01

    # RANSAC による Global Registration
    scene1_kp, scene1_fpfh = preprocess_point_cloud(scene1, voxel_size)
    scene2_kp, scene2_fpfh = preprocess_point_cloud(scene2, voxel_size)
    result_ransac = execute_global_registration(scene1_kp, scene2_kp, scene1_fpfh, scene2_fpfh, voxel_size)
    draw_registration_result(scene1, scene2, result_ransac.transformation)

    # ICP による refine
    result = refine_registration(scene1, scene2, result_ransac.transformation, voxel_size)
    source_temp, target_temp=draw_registration_result(scene1, scene2, result.transformation)
    all_points = []
    all_colors = []
    pcd_aligned = [source_temp, target_temp]
    for i, pcd in enumerate(pcd_aligned):
        all_points.append(np.asarray(pcd.points))
        all_colors.append(np.asarray(pcd.colors))
    pcd_a_merge = make_pcd(np.vstack(all_points), np.vstack(all_colors))
    ft_pcd_m = f'C:/00_work/05_src/data/20201124/202011161_sample/ply/ranzac_icp_pcd_aligned_mergeid_{id}.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd_a_merge)