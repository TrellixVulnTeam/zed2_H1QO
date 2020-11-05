import numpy as np
import os, json, quaternion, open3d
import numpy as np
from AB_point_cloud_perspective import convert_zed2pcd_to_ply
basedir='C:/00_work/05_src/data/fromWATA'
cam=['cam0','cam1']
mode=['A_ON','AB_OFF','AB_ON','B_ON']
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

def get_p(R,t,K):
    world2cam = np.hstack((R, np.dot(-R, t).reshape(3,-1)))
    P = np.dot(K, world2cam)
    return P
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
fR0='C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/rotation.npy'
fR1='C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030155144/rotation.npy'
ft0='C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/translation.npy'
ft1='C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030155144/translation.npy'
R0=np.expand_dims(np.load(fR0),axis=1).T
R1=np.expand_dims(np.load(fR1),axis=1).T
t0=np.expand_dims(np.load(ft0),axis=1)
t1=np.expand_dims(np.load(ft1),axis=1)
K0=np.loadtxt('C:/00_work/05_src/data/fromWATA/ZED_21888201-camera-intrinsics.txt')
K1=np.loadtxt('C:/00_work/05_src/data/fromWATA/ZED_22378008-camera-intrinsics.txt')
transform0=get_p(R0,t0,K0)
transform1=get_p(R1,t1,K1)
pcd1t=pcd1.transform(transform1)
pcd0t=pcd0.transform(transform0)

ft_pcd0='C:/00_work/05_src/data/fromWATA/cam0_pcd_trans.ply'
ft_pcd1='C:/00_work/05_src/data/fromWATA/cam1_pcd_trans.ply'
open3d.io.write_point_cloud(ft_pcd0, pcd0t)
open3d.io.write_point_cloud(ft_pcd1, pcd1t)
points_pcd=np.append(np.array(pcd0t.points),np.array(pcd1t.points),axis = 0)
colors_pcd=np.append(np.array(pcd0t.colors),np.array(pcd1t.colors),axis = 0)
pcd_merge=make_pcd(points_pcd,colors_pcd)
ft_pcd_m='C:/00_work/05_src/data/fromWATA/cam01_pcd_trans_p0.ply'
open3d.io.write_point_cloud(ft_pcd_m, pcd_merge)
