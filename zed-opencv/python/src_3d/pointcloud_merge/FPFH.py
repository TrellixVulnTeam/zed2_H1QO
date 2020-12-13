#https://qiita.com/tttamaki/items/14ac652f4030751ad2f1
import sys
sys.path.append("../..") # Open3D/build/lib/ へのパス
import copy
import numpy as np
import open3d as py3d
RANSAC=py3d.registration.registration_ransac_based_on_feature_matching
ICP= py3d.registration.registration_icp
FPFH= py3d.registration.compute_fpfh_feature


def show(model, scene, model_to_scene_trans=np.identity(4)):
    model_t = copy.deepcopy(model)
    scene_t = copy.deepcopy(scene)

    model_t.paint_uniform_color([1, 0, 0])
    scene_t.paint_uniform_color([0, 0, 1])

    model_t.transform(model_to_scene_trans)

    py3d.visualization.draw_geometries([model_t, scene_t])

id = '0000'
ply0 = f'C:/00_work/05_src/data/20201124/202011161_sample/ply/cam0/20201116133246_027942/pcd_mask_mergeid_{id}.ply'
ply1 = f'C:/00_work/05_src/data/20201124/202011161_sample/ply/cam1/20201116133246_072897/pcd_mask_mergeid_{id}.ply'


id = '0047'
ply0=f'C:/00_work/05_src/data/20201124/202011161_sample/ply/cam0/20201116133925_495618/pcd_mask_mergeid_{id}.ply'
ply1=f'C:/00_work/05_src/data/20201124/202011161_sample/ply/cam1/20201116133925_659685/pcd_mask_mergeid_{id}.ply'

model = py3d.io.read_point_cloud(ply0)
scene = py3d.io.read_point_cloud(ply1)
## PCLモデルを使うならこちら
#model = py3d.read_point_cloud("milk.pcd")
#scene = py3d.read_point_cloud("milk_cartoon_all_small_clorox.pcd")

# いろいろなサイズの元： model点群の1/10を基本サイズsizeにする
size = np.abs((model.get_max_bound() - model.get_min_bound())).max() / 10
kdt_n = py3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
kdt_f = py3d.geometry.KDTreeSearchParamHybrid(radius=size * 50, max_nn=50)

model.estimate_normals( kdt_n)
scene.estimate_normals( kdt_n)
show(model, scene)

# ダウンサンプリング
model_d = model.voxel_down_sample( size)
scene_d = scene.voxel_down_sample(size)
model_d.estimate_normals(kdt_n)
scene_d.estimate_normals(kdt_n)
show(model_d, scene_d)

# 特徴量計算
model_f = FPFH(model_d, kdt_f)
scene_f = FPFH(scene_d, kdt_f)

# 準備
checker = [py3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
           py3d.registration.CorrespondenceCheckerBasedOnDistance(size * 2)]

est_ptp = py3d.registration.TransformationEstimationPointToPoint()
est_ptpln = py3d.registration.TransformationEstimationPointToPlane()

criteria = py3d.registration.RANSACConvergenceCriteria(max_iteration=40000,
                                          max_validation=500)
# RANSACマッチング
result1 = RANSAC(model_d, scene_d,
                 model_f, scene_f,
                 max_correspondence_distance=size * 2,
                 estimation_method=est_ptp,
                 ransac_n=4,
                 checkers=checker,
                 criteria=criteria)
show(model_d, scene_d, result1.transformation)

# ICPで微修正
result2 = ICP(model, scene, size, result1.transformation, est_ptpln)
show(model, scene, result2.transformation)