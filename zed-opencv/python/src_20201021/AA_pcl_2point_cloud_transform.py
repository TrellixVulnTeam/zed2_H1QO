import open3d
import numpy as np
import copy
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    open3d.visualization.draw_geometries([source_temp, target_temp])
    return source_temp
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
def main():
    # 読み込み
    b='C:/00_work/05_src/data/frm_t/'

    pa=b+'20201015155835/'
    pb=b+'20201015155844/'
    pcd20201015155835 = open3d.io.read_point_cloud(pa+'pcd_extracted.ply')
    pcd20201015155844 = open3d.io.read_point_cloud(pb+'pcd_extracted.ply')
    # pcd20201015155835 = open3d.io.read_point_cloud('C:/00_work/05_src/data/frame_20201015155835_mv.ply')
    # pcd20201015155844 = open3d.io.read_point_cloud('C:/00_work/05_src/data/frame_20201015155844_mv.ply')

    # scene2 を適当に回転・並進
    transform_matrix = np.asarray([
        [1., 0., 0., -0.1],
        [0., 0., -1., 0.1],
        [0., 1., 0., -0.1],
        [0., 0., 0., 1.]], dtype="float64")
    # pcd20201015155844.transform(transform_matrix)
    # pcd20201015155835.transform(transform_matrix)

    # 位置合わせ前の点群の表示
    source_temp_pcd20201015155835=draw_registration_result(pcd20201015155835, pcd20201015155844, np.eye(4))
    open3d.io.write_point_cloud(f'{pa}/pcd_extract_plane_pos.ply', source_temp_pcd20201015155835)
    voxel_size = 0.01

    # RANSAC による Global Registration
    scene1_kp, scene1_fpfh = preprocess_point_cloud(pcd20201015155835, voxel_size)
    scene2_kp, scene2_fpfh = preprocess_point_cloud(pcd20201015155844, voxel_size)
    result_ransac = execute_global_registration(scene1_kp, scene2_kp, scene1_fpfh, scene2_fpfh, voxel_size)
    source_temp_pcd20201015155835=draw_registration_result(pcd20201015155835, pcd20201015155844, result_ransac.transformation)
    open3d.io.write_point_cloud(f'{pa}/pcd_extract_plane_RANSAC.ply', source_temp_pcd20201015155835)
    print("RANSAC:",result_ransac.transformation)

    # ICP による refine
    result = refine_registration(pcd20201015155835, pcd20201015155844, result_ransac.transformation, voxel_size)
    source_temp_pcd20201015155835=draw_registration_result(pcd20201015155835, pcd20201015155844, result.transformation)
    open3d.io.write_point_cloud(f'{pa}/pcd_extract_plane_icp.ply', source_temp_pcd20201015155835)
    print('ICP',result.transformation)

    pcd20201015155835 = open3d.io.read_point_cloud('C:/00_work/05_src/data/frame_20201015155835_mv.ply')
    pcd20201015155835_ransac=pcd20201015155835.transform(result_ransac.transformation)
    pcd20201015155835_icp=pcd20201015155835.transform(result.transformation)

    open3d.io.write_point_cloud(f'C:/00_work/05_src/data/frame_20201015155835.ply', pcd20201015155835_ransac)
    # pcd20201015155844 = open3d.io.read_point_cloud('C:/00_work/05_src/data/frame_20201015155844_mv.ply')

if __name__ == "__main__":
    main()