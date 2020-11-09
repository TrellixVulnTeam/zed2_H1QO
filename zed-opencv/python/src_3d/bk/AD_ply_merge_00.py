import os, json, quaternion, open3d
import numpy as np
from easydict import EasyDict
basedir='C:/00_work/05_src/data/fromWATA'
cam=['cam0','cam1']
mode=['A_ON','AB_OFF','AB_ON','B_ON']
#カメラセット（左）5.30 			120			10			1.50
#カメラセット（右）6.30 			240			10			2.00

def coordinate_xyz(r,z,angy):
    AngQOP=np.arcsin(z/r)
    QO=r*np.cos(AngQOP) #ポイントからXY平面の投影から原点の距離
    AngXY=np.pi-angy
    x=np.abs(QO*np.cos(AngXY))
    y=np.abs(QO*np.sin(AngXY))
    return [x,y,z]
def main_coordinate_xyz():
    r,z,angy=5.3,1.5,(2*np.pi)/3.0 #120/180
    cam_left=coordinate_xyz(r,z,angy)

    r,z,angy=6.3,2.0,(4*np.pi)/3.0 #240
    cam_right=coordinate_xyz(r,z,angy)

    print(cam_left)
    print(cam_right)

def make_merge(pcd0,pcd1):
    points_pcd=np.append(np.array(pcd0.points),np.array(pcd1.points),axis = 0)
    colors_pcd=np.append(np.array(pcd0.colors),np.array(pcd1.colors),axis = 0)
    pcd_merge=make_pcd(points_pcd,colors_pcd)
    return pcd_merge

def make_pcd(points, colors):
  pcd = open3d.geometry.PointCloud()
  pcd.points = open3d.utility.Vector3dVector(points)
  pcd.colors = open3d.utility.Vector3dVector(colors)
  return pcd

#x y z座標を取得
main_coordinate_xyz()

ply0='C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/pcd.ply'
ply1='C:/00_work/05_src/data/fromWATA/cam1/AB_ON/20201030155144/pcd.ply'
tran0='C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/transform.npy'
tran1='C:/00_work/05_src/data/fromWATA/cam1/AB_ON/20201030155144/transform.npy'

#############cam01_pcd_trans_merge
pcd0 = open3d.io.read_point_cloud(ply0)
pcd1 = open3d.io.read_point_cloud(ply1)
transform0 = np.load(tran0)
transform1 = np.load(tran1)
pcd0t=pcd0.transform(transform0)
pcd1t=pcd1.transform(transform1)
ft_pcd_m='C:/00_work/05_src/data/fromWATA/cam01_pcd_trans_merge.ply'
open3d.io.write_point_cloud(ft_pcd_m, make_merge(pcd0t,pcd1t))

#############cam01_pcd_trans_merge1
tx,ty,tz=-2.987055406248771, -5.17373172864616, 2.0
transform0 = np.array([
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0, 1]])
tx,ty,tz=2.5416530054277664, -4.402272140611028, 1.5
transform1 = np.array([
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0, 1]])

pcd0 = open3d.io.read_point_cloud(ply0)
pcd1 = open3d.io.read_point_cloud(ply1)
pcd0t=pcd0.transform(transform0)
pcd1t=pcd1.transform(transform1)
ft_pcd_m='C:/00_work/05_src/data/fromWATA/cam01_pcd_trans_merge1.ply'
open3d.io.write_point_cloud(ft_pcd_m, make_merge(pcd0t,pcd1t))

#############cam01_pcd_trans_merge2
tx,ty,tz=-2.987055406248771, 5.17373172864616, 2.0
transform0 = np.array([
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0, 1]])
tx,ty,tz=2.5416530054277664, -4.402272140611028, 1.5
transform1 = np.array([
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0, 1]])
pcd0 = open3d.io.read_point_cloud(ply0)
pcd1 = open3d.io.read_point_cloud(ply1)
pcd0t=pcd0.transform(transform0)
pcd1t=pcd1.transform(transform1)
ft_pcd_m='C:/00_work/05_src/data/fromWATA/cam01_pcd_trans_merge2.ply'
open3d.io.write_point_cloud(ft_pcd_m, make_merge(pcd0t,pcd1t))

#############cam01_pcd_trans_merge3
tx,ty,tz=2.987055406248771, 5.17373172864616, 2.0
transform0 = np.array([
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0, 1]])
tx,ty,tz=2.5416530054277664, 4.402272140611028, 1.5
transform1 = np.array([
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0, 1]])
pcd0 = open3d.io.read_point_cloud(ply0)
pcd1 = open3d.io.read_point_cloud(ply1)
pcd0t=pcd0.transform(transform0)
pcd1t=pcd1.transform(transform1)
ft_pcd_m='C:/00_work/05_src/data/fromWATA/cam01_pcd_trans_merge3.ply'
open3d.io.write_point_cloud(ft_pcd_m, make_merge(pcd0t,pcd1t))


#############cam01_pcd_trans_merge4
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
pcd0 = open3d.io.read_point_cloud(ply0)
pcd1 = open3d.io.read_point_cloud(ply1)
pcd0t=pcd0.transform(transform0)
pcd1t=pcd1.transform(transform1)
ft_pcd_m='C:/00_work/05_src/data/fromWATA/cam01_pcd_trans_merge4.ply'
open3d.io.write_point_cloud(ft_pcd_m, make_merge(pcd0t,pcd1t))



#############cam01_pcd_trans_merge5
tx,ty,tz=2.987055406248771, -5.17373172864616, 2.0
transform0 = np.array([
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0, 1]])
tx,ty,tz=-2.5416530054277664, 4.402272140611028, 1.5
transform1 = np.array([
    [1, 0, 0, tx],
    [0, 1, 0, ty],
    [0, 0, 1, tz],
    [0, 0, 0, 1]])
pcd0 = open3d.io.read_point_cloud(ply0)
pcd1 = open3d.io.read_point_cloud(ply1)
pcd0t=pcd0.transform(transform0)
pcd1t=pcd1.transform(transform1)
ft_pcd_m='C:/00_work/05_src/data/fromWATA/cam01_pcd_trans_merge5.ply'
open3d.io.write_point_cloud(ft_pcd_m, make_merge(pcd0t,pcd1t))