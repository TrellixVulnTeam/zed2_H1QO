#https://qiita.com/tttamaki/items/648422860869bbccc72d
import numpy as np
import open3d as py3d
import cv2
import os, itertools
import datetime
#pip install open3d==0.10.0.1
# import pcl
import matplotlib.pyplot as plt
class ransac_icp:
    def __init__(self):
        self.RANSAC = py3d.registration.registration_ransac_based_on_feature_matching
        self.ICP = py3d.registration.registration_icp
        self.FPFH = py3d.registration.compute_fpfh_feature
        self.GET_GTG = py3d.registration.get_information_matrix_from_point_clouds

    def register(self,pcd1, pcd2, size):
        # ペアの点群を位置合わせ

        kdt_n = py3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
        kdt_f = py3d.geometry.KDTreeSearchParamHybrid(radius=size *2, max_nn=50)

        # ダウンサンプリング
        pcd1_d = pcd1.voxel_down_sample( size)
        pcd2_d = pcd2.voxel_down_sample( size)
        pcd1_d.estimate_normals( kdt_n)
        pcd2_d.estimate_normals( kdt_n)

        # 特徴量計算
        pcd1_f = self.FPFH(pcd1_d, kdt_f)
        pcd2_f = self.FPFH(pcd2_d, kdt_f)

        # 準備
        checker = [py3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                   py3d.registration.CorrespondenceCheckerBasedOnDistance(size * 2)]

        est_ptp = py3d.registration.TransformationEstimationPointToPoint()
        est_ptpln = py3d.registration.TransformationEstimationPointToPlane()

        criteria = py3d.registration.RANSACConvergenceCriteria(max_iteration=400000,
                                                  max_validation=500)
        # RANSACマッチング
        result1 = self.RANSAC(pcd1_d, pcd2_d,
                         pcd1_f, pcd2_f,
                         max_correspondence_distance=size * 2,
                         estimation_method=est_ptp,
                         ransac_n=4,
                         checkers=checker,
                         criteria=criteria)
        # ICPで微修正
        result2 = self.ICP(pcd1, pcd2, size, result1.transformation, est_ptpln)

        return result2.transformation


    def merge(self,pcds):
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


    def add_color_normal(self,pcd,size): # in-place coloring and adding normal
        # pcd.paint_uniform_color(np.random.rand(3))
        kdt_n = py3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
        pcd.estimate_normals( kdt_n)
        return pcd

    def load_pcds(self,pcd_files):

        pcds = []
        for f in pcd_files:
            pcd = py3d.io.read_point_cloud(f)
            pcd_tem=self.remove_noise_ply(pcd)
            cnt_org=np.array(pcd.points).shape[0]
            cnt_noise=np.array(pcd_tem.points).shape[0]
            if cnt_noise/cnt_org>0.5:
                pcd=pcd_tem
            pcds.append(pcd)
        return pcds


    def align_pcds(self,pcds, size):
        # 複数の点群を位置合わせ

        pose_graph = py3d.registration.PoseGraph()
        accum_pose = np.identity(4) # id0から各ノードへの累積姿勢
        pose_graph.nodes.append(py3d.registration.PoseGraphNode(accum_pose))

        n_pcds = len(pcds)
        for source_id in range(n_pcds):
            for target_id in range(source_id + 1, n_pcds):
                source = pcds[source_id]
                target = pcds[target_id]

                trans = self.register(source, target, size)
                GTG_mat = self.GET_GTG(source, target, size, trans) # これが点の情報を含む

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
        translst=[]
        for pcd_id in range(n_pcds):
            trans = pose_graph.nodes[pcd_id].pose
            pcds[pcd_id].transform(trans)
            translst.append(trans)


        return pcds,translst

    def make_pcd(self,points, colors):
      pcd = py3d.geometry.PointCloud()
      pcd.points = py3d.utility.Vector3dVector(points)
      pcd.colors = py3d.utility.Vector3dVector(colors)
      return pcd
    def merge(self,pcds):
        # 複数の点群を1つの点群にマージする

        all_points = []
        for pcd in pcds:
            all_points.append(np.asarray(pcd.points))

        merged_pcd = py3d.PointCloud()
        merged_pcd.points = py3d.Vector3dVector(np.vstack(all_points))

        return merged_pcd
    #https://github.com/aipiano/guided-filter-point-cloud-denoise/blob/master/main.py
    #https://github.com/sakizuki/SSII2018_Tutorial_Open3D/blob/master/Python/kdtree.py
    def guided_filter(self,pcd, radius=0.01, epsilon=0.1):
        kdtree = py3d.geometry.KDTreeFlann(pcd)
        points_copy = np.array(pcd.points)
        points = np.asarray(pcd.points)
        num_points = len(pcd.points)

        for i in range(num_points):
            # 1.RNNの方法
            k, idx, _ = kdtree.search_radius_vector_3d(pcd.points[i], radius)
            # 2.KNNの方法
            # k, idx, _ = kdtree.search_knn_vector_3d(pcd.points[i], 200)
            # 3.RKNNの方法
            # k, idx, _ = kdtree.search_hybrid_vector_3d(pcd.points[i], radius=0.1, max_nn=100)
            if k < 3:
                continue

            neighbors = points[idx, :]
            mean = np.mean(neighbors, 0)
            cov = np.cov(neighbors.T)
            e = np.linalg.inv(cov + epsilon * np.eye(3))

            A = cov @ e
            b = mean - A @ mean

            points_copy[i] = A @ points[i] + b

        pcd.points = py3d.utility.Vector3dVector(points_copy)
        # guided_filter(pcd, 0.01, 0.1)
        return pcd

    def remove_noise_ply(self,pcd):
        # (1) Load a ply point cloud, print it, and render it
        print("Load a ply point cloud, print it, and render it")
        # pcd = py3d.io.read_point_cloud(fn_pcd)
        # py3d.visualization.draw_geometries([pcd])
        # (2) Downsample the point cloud with a voxel
        print("Downsample the point cloud with a voxel of 0.02")
        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
        # py3d.visualization.draw_geometries([voxel_down_pcd])
        # (3) Every 5th points are selected
        print("Every 5th points are selected")
        uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
        # py3d.visualization.draw_geometries([uni_down_pcd])
        # (4) Statistical oulier removal
        print("Statistical oulier removal")
        cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                            std_ratio=2.0)
        # display_inlier_outlier(voxel_down_pcd, ind)
        # (5) Radius oulier removal
        print("Radius oulier removal")
        # cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
        cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=2.5)
        inlier_cloud = voxel_down_pcd.select_by_index(ind)
        outlier_cloud = voxel_down_pcd.select_by_index(ind, invert=True)
        # inlier_cloud, outlier_cloud = display_inlier_outlier(voxel_down_pcd, ind)
        return inlier_cloud
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
    ri=ransac_icp()
    basep = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_{fuchi}/{pose}'
    baseout = f'C:/00_work/05_src/data/20201202_shiyoko_6f/merge_{fuchi}/{pose}'
    os.makedirs(baseout,exist_ok=True)
    files_list=get_merge_fns(basep, id)
    pcds = ri.load_pcds(files_list)

    size = np.abs((pcds[0].get_max_bound() - pcds[0].get_min_bound())).max() / 30
    pcdsn=[]
    for pcd in pcds:
        pcdn=ri.add_color_normal(pcd,size)
        pcdsn.append(pcdn)

    pcd_aligned,translst = ri.align_pcds(pcdsn, size)
    all_points = []
    all_colors = []
    for i, pcd in enumerate(pcd_aligned):
        all_points.append(np.asarray(pcd.points))
        all_colors.append(np.asarray(pcd.colors))
        print("pcd:",i)
        ft_pcd_m = f'{baseout}/pcd_aligned_mergeid_%04d_cam%d.ply'% (id,i)
        py3d.io.write_point_cloud(ft_pcd_m, pcd)
    pcd_a_merge=ri.make_pcd(np.vstack(all_points),np.vstack(all_colors))
    ft_pcd_m=f'{baseout}/pcd_aligned_mergeid_%04d.ply'% (id)
    py3d.io.write_point_cloud(ft_pcd_m, pcd_a_merge)
#点群ノイズの除外処理http://whitewell.sakura.ne.jp/Open3D/PointCloudOutlierRemoval.html
def display_inlier_outlier(cloud, ind):
    # inlier_cloud = py3d.geometry.PointCloud.select_down_sample(ind)
    # outlier_cloud = py3d.geometry.PointCloud.select_down_sample(cloud,ind, invert=True)

    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)


    print("Showing outliers (red) and inliers (gray): ")
    # outlier_cloud.paint_uniform_color([1, 0, 0])
    # inlier_cloud.paint_uniform_color([0, 0, 1])
    # py3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    # py3d.visualization.draw_geometries([inlier_cloud])
    return inlier_cloud,outlier_cloud
def remove_noise_ply(fn_pcd):
    # (1) Load a ply point cloud, print it, and render it
    print("Load a ply point cloud, print it, and render it")
    pcd = py3d.io.read_point_cloud(fn_pcd)
    # py3d.visualization.draw_geometries([pcd])
    # (2) Downsample the point cloud with a voxel
    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    # py3d.visualization.draw_geometries([voxel_down_pcd])
    # (3) Every 5th points are selected
    print("Every 5th points are selected")
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    # py3d.visualization.draw_geometries([uni_down_pcd])
    # (4) Statistical oulier removal
    print("Statistical oulier removal")
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                        std_ratio=2.0)
    display_inlier_outlier(voxel_down_pcd, ind)
    # (5) Radius oulier removal
    print("Radius oulier removal")
    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    inlier_cloud,outlier_cloud=display_inlier_outlier(voxel_down_pcd, ind)
    return inlier_cloud
def remove_noise_mask():
    fn='C:/00_work/05_src/data/20201124/202011161_sample/ply/cam0/20201116133925_495618/moving_detection_mask_mergeid_0047.png'
    img=cv2.imread(fn)
    ksize = 11
    # 中央値フィルタ
    img_mask = cv2.medianBlur(img, ksize)
    plt.imshow(img_mask)
    plt.show()
def main():
    # p = pcl.load("D:/tests/tutorials/table_scene_lms400.pcd")
    fn_pcd='C:/00_work/05_src/data/20201124/202011161_sample/ply/cam0/20201116133246_027942/pcd_mask_mergeid_0000.ply'
    p=py3d.io.read_point_cloud(fn_pcd)
    # 使用K近邻的50个点计算标准距离，距离超过标准距离1倍的被认为是离群点

    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k(50)
    fil.set_std_dev_mul_thresh(1.0)

    # pcl.save(fil.filter(),
    #          "D:/tests/tutorials/table_scene_lms400_inliers.pcd")

    fil.set_negative(True) #保存不满足条件的，即离群点文件
    # pcl.save(fil.filter(),
    #          "D:/tests/tutorials/table_scene_lms400_outliers.pcd")
def main_merge_point_cloud():
    fuchi='fuchi00'
    # fuchi='fuchi10'
    dt_ids = ['d2_a15_h1', 'd2_a30_h1', 'd2_a45_h1',
              'd2_a15_h1.5', 'd2_a30_h1.5', 'd2_a45_h1.5',
              'd4_a15_h1', 'd4_a30_h1', 'd4_a45_h1',
              'd4_a15_h1.5', 'd4_a30_h1.5', 'd4_a45_h1.5'
              ]
    for dt_id in dt_ids :
        for i in range(4):
            main_ransac_icp(i,fuchi,dt_id)
            # main_ransac_icp(i,fuchi,'d4_a30_h1.5')
            # main_ransac_icp(i,fuchi,'d4_a45_h1.5')
def main_noise_point_cloud():
    ply = 'fuchi_ply_size15'
    ri = ransac_icp()
    id = '0001'
    ply0 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/finished/d2_a15_h1/cam0/b/20201202155244/pcd_mask_mergeid_{id}.ply'
    ply1 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/finished/d2_a15_h1/cam1/b/20201202155216/pcd_mask_mergeid_{id}.ply'
    ply2 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/finished/d2_a15_h1/cam2/b/20201202155218/pcd_mask_mergeid_{id}.ply'

    pcds = ri.load_pcds([ply0, ply1, ply2])
    pcd_filter=ri.guided_filter(pcds[0], radius=0.01, epsilon=0.1)
    dt_s= datetime.datetime.now().strftime('%Y%m%d%H%M%S_%f')
    ft_pcd_m=f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/pcd_mergeid_{id}_{dt_s}.ply'
    py3d.io.write_point_cloud(ft_pcd_m, pcd_filter)
if __name__ == "__main__":
    # main_noise_point_cloud()
    main_merge_point_cloud()
    pass