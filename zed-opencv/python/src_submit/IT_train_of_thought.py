import numpy as np
import open3d


p = 'C:/00_work/05_src/data/frm_t/20201015155835'
# p = "data/toYOU_20201021_/20201015155835" pcd_extract_plane_rotation.ply
f = f"{p}/pcd_extracted.ply"
pcd = open3d.io.read_point_cloud(f)
def get_reg_linear(points,colors,X,Y,Z,pos_reg=2,color=[255,0,0]):
    # Zについて線形回帰実施(入力はXとY)。赤色で平面出力
    # XY=np.hstack([X, Y])
    XY=np.vstack([X, Y]).T
    XY=np.hstack((np.ones((XY.shape[0],1)).astype(XY.dtype), XY))

    # http://reliawiki.org/index.php/Multiple_Linear_Regression_Analysis
    # "Estimating Regression Models Using Least Squares"
    # （最小二乗法で一気にパラメータを計算する手法。）
    XY1=XY.T @ XY
    XY2=XY.T@Z
    bhat = np.linalg.inv(XY1)@XY2
    print(bhat)
    Zhat = (XY @ bhat).reshape(-1,1)
    if pos_reg==0:
        points2=np.hstack([Zhat,points[:,1].reshape(-1,1), points[:,2].reshape(-1,1)])
    elif pos_reg==1:
        points2=np.hstack([points[:,0].reshape(-1,1), Zhat,points[:,2].reshape(-1,1)])
    else:
        points2=np.hstack([points[:,:2], Zhat])
    colors2 = np.zeros(colors.shape).astype(colors.dtype)
    colors2[:] = np.array(color).astype(colors.dtype)
    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(points2)
    pcd2.colors = open3d.utility.Vector3dVector(colors2)

    # open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_z.ply', pcd2)
    return pcd2
def main_get_reg_linear():
    pcd = open3d.io.read_point_cloud(f)
    points = np.copy(np.array(pcd.points))
    colors = np.copy(np.array(pcd.colors))
    x, y, z=points[:,0], points[:,1], points[:,2]
    pcd_z=get_reg_linear(points,colors,x,y,z,pos_reg=2,color=[255,0,0])
    open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_z.ply', pcd_z)
    # 于さんtrial: Yについて回帰実施。緑色で平面出力
    # XZ=np.hstack([X, Z])
    # XZ=np.hstack([np.ones((XZ.shape[0],1)).astype(XZ.dtype), XZ])
    pcd = open3d.io.read_point_cloud(f)
    points = np.copy(np.array(pcd.points))
    colors = np.copy(np.array(pcd.colors))
    x, y, z=points[:,0], points[:,1], points[:,2]
    pcd_y=get_reg_linear(points,colors,x,z,y,pos_reg=1,color=[0,0,255])
    open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_y.ply', pcd_y)
    # 続きを書いて、実際にデータを出力してみてください

    # open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_y.ply', pcd3)

    # 于さんtrial: Xについて回帰実施。緑色で平面出力
    #YZ...

    pcd = open3d.io.read_point_cloud(f)
    points = np.copy(np.array(pcd.points))
    colors = np.copy(np.array(pcd.colors))
    x, y, z=points[:,0], points[:,1], points[:,2]
    pcd_x=get_reg_linear(points,colors,y,z,x,pos_reg=0,color=[0,255,0])
    open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_x.ply', pcd_x)
    # open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_x.ply', pcd4)

main_get_reg_linear()