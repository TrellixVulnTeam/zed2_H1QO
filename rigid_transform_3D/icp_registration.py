# examples/Python/Basic/icp_registration.py

import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # o3d.visualization.draw_geometries([source_temp, target_temp])
    return source_temp


if __name__ == "__main__":
    source = o3d.io.read_point_cloud("C:/00_work/05_src/zed2/zed-opencv/python/data_20201021_1/reconstruction-000003.cloud-ZED_22378008.ply")
    target = o3d.io.read_point_cloud("C:/00_work/05_src/zed2/zed-opencv/python/data_20201021_1/reconstruction-000001.cloud-ZED_22378008.ply")
    threshold = 0.02
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    # trans_init = np.asarray([[1, 0, 0, 0],
    #                          [0, 1, 0, 0],
    #                          [0, 0, 1, 0],
    #                          [0, 0, 0, 1]])
    draw_registration_result(source, target, trans_init)
    print("Initial alignment")
    evaluation = o3d.registration.evaluate_registration(source, target,
                                                        threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")

    source_temp_p2p=draw_registration_result(source, target, reg_p2p.transformation)
    o3d.io.write_point_cloud("source_temp_p2p.ply",source_temp_p2p)
    print("Apply point-to-plane ICP")
    reg_p2l = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    print("")
    source_temp_p2plance=draw_registration_result(source, target, reg_p2l.transformation)
    o3d.io.write_point_cloud("source_temp_p2plance.ply",source_temp_p2plance)