import numpy as np
import open3d


p = 'C:/00_work/05_src/data/frm_t/20201015155844'
# p = 'C:/00_work/05_src/data/frm_t/20201015155835'
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
    bhat = np.linalg.inv(XY.T @ XY) @ XY.T@Z
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
import itertools
from sklearn.linear_model import LinearRegression
def reg_linear_xyz():
    pcd = open3d.io.read_point_cloud(f)
    points = np.array(pcd.points)
    colors = np.array(pcd.colors)
    # ws,hs=points.shape[:,:2]
    train_x=points
    train_y=np.ones(points.shape[0])
    lr = LinearRegression()
    lr.fit(train_x, train_y)
    train_y_p=lr.predict(train_x)

    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(train_y_p)
    pcd2.colors = open3d.utility.Vector3dVector(colors)
    open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_xyz.ply', pcd2)
# reg_linear_xyz()
main_get_reg_linear()
from numpy import zeros
def fit_pointcloud_plane(A,B):
    """
    Estimate parameters from sensor readings in the Cartesian frame.
    Each row in the P matrix contains a single 3D point measurement;
    the matrix P has size n x 3 (for n points). The format is:

    P = [[x1, y1, z1],
         [x2, x2, z2], ...]

    where all coordinate values are in metres. Three parameters are
    required to fit the plane, a, b, and c, according to the equation

    z = a + bx + cy

    The function should retrn the parameters as a NumPy array of size
    three, in the order [a, b, c].
    """
    # param_est = zeros(3)

    # A = P[:, 0:2]
    # B = P[:, 2]

    row_num = (A.shape)[0]
    one_array = np.ones(row_num)
    A = np.column_stack((one_array, A))

    a = A.T @ A
    b = A.T @ B

    param_est = np.linalg.solve(a, b)

    return param_est,A

def get_pcd_reg(points_r,colors_r,color=[255,0,0]):
    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(points_r)
    colors2 = np.zeros(colors_r.shape).astype(colors_r.dtype)
    colors2[:] = np.array(color).astype(colors_r.dtype)

    pcd2.colors = open3d.utility.Vector3dVector()
    pcd2.points = open3d.utility.Vector3dVector(points_r)
    pcd2.colors = open3d.utility.Vector3dVector(colors2)
    # open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_z_t.ply', pcd2)
    return pcd2

pcd = open3d.io.read_point_cloud(f)
points = np.array(pcd.points)
colors = np.array(pcd.colors)

points_c=np.copy(points)
pa=points_c[:, 0:2]
pb=points_c[:, 2]
param,x_fit=fit_pointcloud_plane(pa,pb)
print(param)
y_fit=param@x_fit.T
points_c[:,2]=y_fit
pcd_o=get_pcd_reg(points_c,colors,color=[255,0,0])
open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_z_t.ply', pcd_o)

points_c=np.copy(points)
pa=np.vstack([points_c[:, 0], points_c[:, 2]]).T
pb=points_c[:, 1]
param,x_fit=fit_pointcloud_plane(pa,pb)
print(param)
y_fit=param@x_fit.T
points_c[:,1]=y_fit
pcd_o=get_pcd_reg(points_c,colors,color=[0,255,0])
open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_y_t.ply', pcd_o)



points_c=np.copy(points)
# pa=points_c[:, 0:2]
pa=np.vstack([points_c[:, 1], points_c[:, 2]]).T
pb=points_c[:, 0]
param,x_fit=fit_pointcloud_plane(pa,pb)
print(param)
y_fit=param@x_fit.T
points_c[:,0]=y_fit
pcd_o=get_pcd_reg(points_c,colors,color=[0,0,255])
open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_x_t.ply', pcd_o)