#https://qiita.com/tttamaki/items/648422860869bbccc72d
import sys
sys.path.append("../..") # Open3D/build/lib/ へのパス
import numpy as np
import open3d as py3d
RANSAC=py3d.registration.registration_ransac_based_on_feature_matching
ICP=py3d.registration.registration_icp
FPFH=py3d.registration.compute_fpfh_feature
GET_GTG=py3d.registration.get_information_matrix_from_point_clouds

def register(pcd1, pcd2, size):
    # ペアの点群を位置合わせ

    kdt_n = py3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    kdt_f = py3d.geometry.KDTreeSearchParamHybrid(radius=size * 10, max_nn=50)

    # ダウンサンプリング
    pcd1_d = pcd1.voxel_down_sample( size)
    pcd2_d = pcd2.voxel_down_sample( size)
    pcd1_d.estimate_normals( kdt_n)
    pcd2_d.estimate_normals( kdt_n)

    # 特徴量計算
    pcd1_f = FPFH(pcd1_d, kdt_f)
    pcd2_f = FPFH(pcd2_d, kdt_f)

    # 準備
    checker = [py3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
               py3d.registration.CorrespondenceCheckerBasedOnDistance(size * 2)]

    est_ptp = py3d.registration.TransformationEstimationPointToPoint()
    est_ptpln = py3d.registration.TransformationEstimationPointToPlane()

    criteria = py3d.registration.RANSACConvergenceCriteria(max_iteration=400000,
                                              max_validation=500)
    # RANSACマッチング
    result1 = RANSAC(pcd1_d, pcd2_d,
                     pcd1_f, pcd2_f,
                     max_correspondence_distance=size * 2,
                     estimation_method=est_ptp,
                     ransac_n=4,
                     checkers=checker,
                     criteria=criteria)
    # ICPで微修正
    result2 = ICP(pcd1, pcd2, size, result1.transformation, est_ptpln)

    return result2.transformation


def merge(pcds):
    # 複数の点群を1つの点群にマージする

    all_points = []
    all_colors = []
    for pcd in pcds:
        all_points.append(np.asarray(pcd.points))
        all_colors.append(np.asarray(pcd.colors))

    merged_pcd = py3d.geometry.PointCloud()
    merged_pcd.points = py3d.utility.Vector3dVector(np.vstack(all_points))
    merged_pcd.colors = py3d.utility.Vector3dVector(np.vstack(all_points))

    return merged_pcd


def add_color_normal(pcd): # in-place coloring and adding normal
    # pcd.paint_uniform_color(np.random.rand(3))
    size = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 30
    kdt_n = py3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    pcd.estimate_normals( kdt_n)
    pass


def load_pcds(pcd_files):

    pcds = []
    for f in pcd_files:
        pcd = py3d.io.read_point_cloud(f)
        add_color_normal(pcd)
        pcds.append(pcd)


    return pcds


def align_pcds(pcds, size):
    # 複数の点群を位置合わせ

    pose_graph = py3d.registration.PoseGraph()
    accum_pose = np.identity(4) # id0から各ノードへの累積姿勢
    pose_graph.nodes.append(py3d.registration.PoseGraphNode(accum_pose))

    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            source = pcds[source_id]
            target = pcds[target_id]

            trans = register(source, target, size)
            GTG_mat = GET_GTG(source, target, size, trans) # これが点の情報を含む

            if target_id == source_id + 1: # 次のidの点群ならaccum_poseにposeを積算
                accum_pose = trans @ accum_pose
                pose_graph.nodes.append(py3d.registration.PoseGraphNode(np.linalg.inv(accum_pose))) # 各ノードは，このノードのidからid0への変換姿勢を持つので，invする
                # そうでないならnodeは作らない
            pose_graph.edges.append(py3d.registration.PoseGraphEdge(source_id,
                                                       target_id,
                                                       trans,
                                                       GTG_mat,
                                                       uncertain=True)) # bunnyの場合，隣でも怪しいので全部True


    # 設定
    solver = py3d.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = py3d.registration.GlobalOptimizationConvergenceCriteria()
    option = py3d.registration.GlobalOptimizationOption(
             max_correspondence_distance=size / 10,
             edge_prune_threshold=size / 10,
             reference_node=0)

    # 最適化
    py3d.registration.global_optimization(pose_graph,
                            method=solver,
                            criteria=criteria,
                            option=option)

    # 推定した姿勢で点群を変換
    for pcd_id in range(n_pcds):
        trans = pose_graph.nodes[pcd_id].pose
        pcds[pcd_id].transform(trans)


    return pcds

def make_pcd(points, colors):
  pcd = py3d.geometry.PointCloud()
  pcd.points = py3d.utility.Vector3dVector(points)
  pcd.colors = py3d.utility.Vector3dVector(colors)
  return pcd
def merge(pcds):
    # 複数の点群を1つの点群にマージする

    all_points = []
    for pcd in pcds:
        all_points.append(np.asarray(pcd.points))

    merged_pcd = py3d.PointCloud()
    merged_pcd.points = py3d.Vector3dVector(np.vstack(all_points))

    return merged_pcd
ply1='C:/00_work/05_src/data/fromWATA/cam1/AB_ON/20201030155144/pcd.ply'
ply0='C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/pcd.ply'
ply0='C:/00_work/05_src/data/20201124/202011161_sample/ply/cam0/20201116133246_027942/pcd_mask_mergeid_0000.ply'
ply1='C:/00_work/05_src/data/20201124/202011161_sample/ply/cam1/20201116133246_072897/pcd_mask_mergeid_0000.ply'
ply2='C:/00_work/05_src/data/20201124/202011161_sample/ply/cam2/20201116133246_079488/pcd_mask_mergeid_0000.ply'
pcds = load_pcds([ply0,ply1,ply2])
# pcds = load_pcds([ply0,ply1])
# py3d.visualization.draw_geometries(pcds, "input pcds")

size = np.abs((pcds[0].get_max_bound() - pcds[0].get_min_bound())).max() / 30

pcd_aligned = align_pcds(pcds, size)
all_points = []
all_colors = []
for i, pcd in enumerate(pcd_aligned):
    all_points.append(np.asarray(pcd.points))
    all_colors.append(np.asarray(pcd.colors))
    print("pcd:",i)
    ply = make_pcd(np.vstack(all_points), np.vstack(all_colors))
    ft_pcd_m = f'C:/00_work/05_src/data/20201124/202011161_sample/ply/pcd_aligned_mergeid_0000_%d.ply'% (i)
    py3d.io.write_point_cloud(ft_pcd_m, pcd)
pcd_a_merge=make_pcd(np.vstack(all_points),np.vstack(all_colors))
ft_pcd_m='C:/00_work/05_src/data/20201124/202011161_sample/ply/pcd_aligned_01_mergeid_0000.ply'
py3d.io.write_point_cloud(ft_pcd_m, pcd_a_merge)
