# examples/Python/Basic/icp_registration.py
#http://www.open3d.org/docs/0.7.0/tutorial/Basic/icp_registration.html
#伊藤さん案③
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
    return source_temp,target_temp
def add_color_normal(pcd): # in-place coloring and adding normal
    # pcd.paint_uniform_color(np.random.rand(3))
    size = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 30
    kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    pcd.estimate_normals( kdt_n)
    pass
def make_pcd(points, colors):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  pcd.colors = o3d.utility.Vector3dVector(colors)
  return pcd
if __name__ == "__main__":
    id = '0000'
    ply0 = f'C:/00_work/05_src/data/20201124/202011161_sample/ply/cam0/20201116133246_027942/pcd_mask_mergeid_{id}.ply'
    ply1 = f'C:/00_work/05_src/data/20201124/202011161_sample/ply/cam1/20201116133246_072897/pcd_mask_mergeid_{id}.ply'


    id = '0047'
    ply0=f'C:/00_work/05_src/data/20201124/202011161_sample/ply/cam0/20201116133925_495618/pcd_mask_mergeid_{id}.ply'
    ply1=f'C:/00_work/05_src/data/20201124/202011161_sample/ply/cam1/20201116133925_659685/pcd_mask_mergeid_{id}.ply'
    source = o3d.io.read_point_cloud(ply0)
    target = o3d.io.read_point_cloud(ply1)
    add_color_normal(source)
    add_color_normal(target)
    threshold = 0.02
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                             [-0.139, 0.967, -0.215, 0.7],
                             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
    source_temp,target_temp=draw_registration_result(source, target, trans_init)
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
    source_temp,target_temp=draw_registration_result(source, target, reg_p2p.transformation)

    print("Apply point-to-plane ICP")
    reg_p2l = o3d.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    print("")
    source_temp,target_temp=draw_registration_result(source, target, reg_p2l.transformation)
    all_points = []
    all_colors = []
    pcd_aligned=[source_temp,target_temp]
    for i, pcd in enumerate(pcd_aligned):
        all_points.append(np.asarray(pcd.points))
        all_colors.append(np.asarray(pcd.colors))
    pcd_a_merge = make_pcd(np.vstack(all_points), np.vstack(all_colors))
    ft_pcd_m = f'C:/00_work/05_src/data/20201124/202011161_sample/ply/icp_registration_pcd_aligned_mergeid_{id}.ply'
    o3d.io.write_point_cloud(ft_pcd_m, pcd_a_merge)