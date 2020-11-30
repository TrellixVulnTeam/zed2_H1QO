import os, json, quaternion, open3d
import numpy as np
from easydict import EasyDict
from AB_point_cloud_perspective import convert_zed2pcd_to_ply
basedir='C:/00_work/05_src/data/fromWATA'
cam=['cam0','cam1']
mode=['A_ON','AB_OFF','AB_ON','B_ON']

def make_merge(pcd0,pcd1):
    points_pcd=np.append(np.array(pcd0.points),np.array(pcd1.points),axis = 0)
    colors_pcd=np.append(np.array(pcd0.colors),np.array(pcd1.colors),axis = 0)
    pcd_merge=make_pcd(points_pcd,colors_pcd)
    return pcd_merge
#https://ja.wikipedia.org/wiki/回転行列
def get_rotation_matrix(x,axis_r='y'): #0:Rx, 1:Ry, 2:Rz
    rotation_matrix = EasyDict({})
    rotation_matrix.x = [
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ]
    rotation_matrix.y = [
        [np.cos(x), 0, np.sin(x)],
        [0, 1, 0],
        [-np.sin(x), 0, np.cos(x)]
    ]
    rotation_matrix.z = [
        [np.cos(x), -np.sin(x), 0],
        [np.sin(x), np.cos(x), 0],
        [0, 0, 1]
    ]
    return np.array(rotation_matrix[axis_r])
def npy_2_ply(d):
    zed2pcd = np.load(f"{d}/pcd.npy")
    ply=convert_zed2pcd_to_ply(zed2pcd)
    open3d.io.write_point_cloud(f"{d}/pcd.ply", ply)
def npy_2_ply_all():
    for ca in cam:
        for m in mode:
            p_cm=f"{basedir}/{ca}/{m}"
            p_lds =os.listdir(p_cm)
            for p_ld in p_lds:
                p = f"{p_cm}/{p_ld}"
                if os.path.isdir(p):
                    print(p)
                    if os.path.exists(f"{p}/pcd.ply"):
                        continue
                    npy_2_ply(p)

def make_pcd(points, colors):
  pcd = open3d.geometry.PointCloud()
  pcd.points = open3d.utility.Vector3dVector(points)
  pcd.colors = open3d.utility.Vector3dVector(colors)
  return pcd
ply0='C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/pcd.ply'
ply1='C:/00_work/05_src/data/fromWATA/cam1/AB_ON/20201030155144/pcd.ply'
tran0='C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/transform.npy'
tran1='C:/00_work/05_src/data/fromWATA/cam1/AB_ON/20201030155144/transform.npy'
pcd0 = open3d.io.read_point_cloud(ply0)
pcd1 = open3d.io.read_point_cloud(ply1)
# transform0 = np.load(tran0)
# transform1 = np.load(tran1)
tx,ty,tz=2.987055406248771, 5.17373172864616, 2.0
transform0 = np.array([
    [1, 0, 0, -tx],
    [0, 1, 0, -ty],
    [0, 0, 1, -tz],
    [0, 0, 0, 1]])
tx,ty,tz=2.5416530054277664, 4.402272140611028, 1.5
transform1 = np.array([
    [1, 0, 0, -tx],
    [0, 1, 0, -ty],
    [0, 0, 1, -tz],
    [0, 0, 0, 1]])
pcd0t=pcd0.transform(transform0)
pcd1t=pcd1.transform(transform1)
# pcd0t=pcd0
# pcd1t=pcd1
def rotate_ply(pcd,rotate_m):
    points=np.array(pcd.points)@rotate_m
    colors=np.array(pcd.colors)
    return make_pcd(points,colors)

rotate_my0=get_rotation_matrix(np.pi*240/180,axis_r='y')
rotate_my1=get_rotation_matrix(np.pi*120/180,axis_r='y')
pcd0t_r=rotate_ply(pcd0t,rotate_my0)
pcd1t_r=rotate_ply(pcd1t,rotate_my1)

ft_pcd_m='C:/00_work/05_src/data/fromWATA/cam01_pcd_trans_merge_rotation_y.ply'
open3d.io.write_point_cloud(ft_pcd_m, make_merge(pcd0t_r,pcd1t_r))
ft_pcd0='C:/00_work/05_src/data/fromWATA/cam0_pcd_trans.ply'
ft_pcd1='C:/00_work/05_src/data/fromWATA/cam1_pcd_trans.ply'
# open3d.io.write_point_cloud(ft_pcd0, pcd0t_r)
# open3d.io.write_point_cloud(ft_pcd1, pcd1t_r)

angz0=-np.arctan(transform0[2,3]/transform0[0,3])/2
angz1=np.arctan(transform1[2,3]/transform1[0,3])/2
print(angz0,angz1)
rotate_mz0=get_rotation_matrix(angz0,axis_r='z')
rotate_mz1=get_rotation_matrix(angz1,axis_r='z')
pcd0t_rz=rotate_ply(pcd0t_r,rotate_mz0)
pcd1t_rz=rotate_ply(pcd1t_r,rotate_mz1)
ft_pcd_m='C:/00_work/05_src/data/fromWATA/cam01_pcd_trans_merge_rotation_yz.ply'
open3d.io.write_point_cloud(ft_pcd_m, make_merge(pcd0t_rz,pcd1t_rz))

