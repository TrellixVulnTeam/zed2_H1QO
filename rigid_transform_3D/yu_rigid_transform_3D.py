#!/usr/bin/env python3
import numpy as np
import open3d
from rigid_transform_3D import rigid_transform_3D

plyA=open3d.io.read_point_cloud(filename='C:/00_work/05_src/data/frm_t/20201015155844_extracted_frame.ply')
plyB=open3d.io.read_point_cloud(filename='C:/00_work/05_src/data/frm_t/20201015155835_extracted_frame.ply')

A = np.array(plyA.points)
Ac = np.array(plyA.colors)

B = np.array(plyB.points)
Bc = np.array(plyB.colors)
rows=B.shape[0]
As=A[:rows]
Asc=Ac[:rows]
As=As.reshape(3,-1)

Bs=B[:rows]
Bsc=Bc[:rows]
Bs=Bs.reshape(3,-1)

# At=np.vstack((A[:2,:],np.zeros(A.shape[1])))
# # Ar=Ar.reshape(3,-1)
# Art=np.vstack((Ar[:2,:],np.zeros(Ar.shape[1])))

# Recover R and t
ret_R, ret_t = rigid_transform_3D(Bs,As)

Bst = (ret_R@Bs) + ret_t
def make_pcd(points, colors):
 pcd = open3d.geometry.PointCloud()
 pcd.points = open3d.utility.Vector3dVector(points)
 pcd.colors = open3d.utility.Vector3dVector(colors)
 return pcd

ply = make_pcd(np.array(Bst.reshape(-1,3)), Bsc)
fn='C:/00_work/05_src/data/frm_t/20201015155835_extracted_frame_rigit2.ply'
open3d.io.write_point_cloud(fn, ply)


tran=np.hstack((ret_R,ret_t))
transform=np.vstack((tran,np.array([0,0,0,1])))
# transform=np.multiply(transform,2)
transform=-transform
plyt = plyB.transform(transform)
fn='C:/00_work/05_src/data/frm_t/20201015155835_extracted_frame_tran_1.ply'
open3d.io.write_point_cloud(fn, plyt)

# ply = plyA.transform(transform)
# fn='C:/00_work/05_src/data/A_pcd_extracted_trans.ply'
# open3d.io.write_point_cloud(fn, ply)


# ply = make_pcd(np.array(At.reshape(-1,3)), Ac)
# fn='C:/00_work/05_src/data/20201015155835/A_pcd_extracted_trans.ply'
# open3d.io.write_point_cloud(fn, ply)