
import numpy as np
import open3d as py3d
import datetime
import cv2
# import pcl
# https://github.com/aipiano/guided-filter-point-cloud-denoise/blob/master/main.py
# https://github.com/sakizuki/SSII2018_Tutorial_Open3D/blob/master/Python/kdtree.py
def guided_filter( pcd, radius=0.01, epsilon=0.1):
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
#http://whitewell.sakura.ne.jp/Open3D/PointCloudOutlierRemoval.html
def remove_noise_ply(pcd):
    # (1) Load a ply point cloud, print it, and render it
    print("Load a ply point cloud, print it, and render it")
    # pcd = py3d.io.read_point_cloud(fn_pcd)
    # py3d.visualization.draw_geometries([pcd])
    # (2) Downsample the point cloud with a voxel
    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)
    # py3d.visualization.draw_geometries([voxel_down_pcd])
    # (3) Every 5th points are selected
    print("Every 5th points are selected")
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=2)
    # py3d.visualization.draw_geometries([uni_down_pcd])
    # (4) Statistical oulier removal
    print("Statistical oulier removal")
    cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=100,std_ratio=2.0)
    inlier_statistical_outlier= uni_down_pcd.select_by_index(ind)

    # py3d.visualization.draw_geometries([inlier_statistical_outlier])
    # display_inlier_outlier(voxel_down_pcd, ind)
    # (5) Radius oulier removal
    print("Radius oulier removal")
    # cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    cl, ind = uni_down_pcd.remove_radius_outlier(nb_points=100, radius=0.05)
    inlier_remove_radius_outlier = uni_down_pcd.select_by_index(ind)
    # outlier_cloud = inlier_statistical_outlier.select_by_index(ind, invert=True)
    # inlier_cloud, outlier_cloud = display_inlier_outlier(voxel_down_pcd, ind)

    # cl, ind = inlier_remove_radius_outlier.remove_statistical_outlier(nb_neighbors=30,std_ratio=2.0)
    # inlier_remove_statistical_outlier= inlier_remove_radius_outlier.select_by_index(ind)

    # py3d.visualization.draw_geometries([inlier_cloud])
    return inlier_statistical_outlier,inlier_remove_radius_outlier

def remove_statistical_outlier(pcd):
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=2)
    cl, ind = uni_down_pcd.remove_statistical_outlier(nb_neighbors=200,std_ratio=2.0)
    inlier_statistical_outlier= uni_down_pcd.select_by_index(ind)
    return inlier_statistical_outlier
def remove_radius_outlier(pcd):
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=2)
    n_point_base=129018
    n_point=np.array(pcd.points).shape[0]
    nb_points=int(800*n_point/n_point_base)
    cl, ind = uni_down_pcd.remove_radius_outlier(nb_points=nb_points, radius=0.1)
    inlier_remove_radius_outlier = uni_down_pcd.select_by_index(ind)
    return inlier_remove_radius_outlier
def remove_noise_mask(img,ksize):
    # 中央値フィルタ
    img=np.float32(img)
    img_mask = cv2.medianBlur(img, ksize)
    return img_mask


def check_fuchi(img, y, x,size):

    #point is black
    if np.any((img[y,x,0] == 0) & (img[y,x,1] == 0) & (img[y,x,2] == 0)):
        return False
    #image size
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    #search area
    x0 = (x - size) if x > size else 0
    y0 = (y - size) if y > size else 0
    x1 = (x + size + 1) if (x + size + 1) < w else w
    y1 = (y + size + 1) if (y + size + 1) < h else h

    img_check = img[y0:y1, x0:x1]
    #search area including black
    return np.any((img_check[:,:,0] == 0) & (img_check[:,:,1] == 0) & (img_check[:,:,2] == 0))
def load_pcds(pcd_files):

    pcds = []
    for f in pcd_files:
        pcd = py3d.io.read_point_cloud(f)
        pcds.append(pcd)
    return pcds
def main_noise_point_cloud():
    id = '0001'
    ply0 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/finished/d2_a15_h1/cam0/b/20201202155244/pcd_mask_mergeid_{id}.ply'
    ply1 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/finished/d2_a15_h1/cam1/b/20201202155216/pcd_mask_mergeid_{id}.ply'
    ply2 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/finished/d2_a15_h1/cam2/b/20201202155218/pcd_mask_mergeid_{id}.ply'

    pcds = load_pcds([ply0, ply1, ply2])
    pcd_filter=guided_filter(pcds[0], radius=0.01, epsilon=0.1)
    dt_s= datetime.datetime.now().strftime('%Y%m%d%H%M%S_%f')
    ft_pcd_m=f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/pcd_mergeid_{id}_{dt_s}.ply'
    py3d.io.write_point_cloud(ft_pcd_m, pcd_filter)
if __name__ == "__main__":
    # main_noise_point_cloud()
    id = '0001'
    ply0 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/d2_a15_h1/cam0/b/20201202155244/pcd_mask_mergeid_{id}.ply'
    ply1 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/d2_a15_h1/cam1/b/20201202155216/pcd_mask_mergeid_{id}.ply'
    ply2 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/d2_a15_h1/cam2/b/20201202155218/pcd_mask_mergeid_{id}.ply'

    id = '0002'
    ply0 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/d2_a15_h1/cam0/w/20201202155315/pcd_mask_mergeid_{id}.ply'
    ply1 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/d2_a15_h1/cam1/b/20201202155216/pcd_mask_mergeid_{id}.ply'
    ply2 = f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/d2_a15_h1/cam2/b/20201202155218/pcd_mask_mergeid_{id}.ply'

    pcds = load_pcds([ply0, ply1, ply2])
    inlier_statistical_outlier=remove_statistical_outlier(pcds[0])
    inlier_remove_radius_outlier=remove_radius_outlier(pcds[0])
    dt_s= datetime.datetime.now().strftime('%Y%m%d%H%M%S_%f')
    ft_pcd_m=f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/inlier_statistical_outlier_{id}_{dt_s}.ply'
    py3d.io.write_point_cloud(ft_pcd_m, inlier_statistical_outlier)
    ft_pcd_m=f'C:/00_work/05_src/data/20201202_shiyoko_6f/pointcloud_org/inlier_remove_radius_outlier_{id}_{dt_s}.ply'
    py3d.io.write_point_cloud(ft_pcd_m, inlier_remove_radius_outlier)

    pass

