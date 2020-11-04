# https://www.geeksforgeeks.org/angle-between-two-planes-in-3d/

# Python program to find the Angle between
# two Planes in 3 D.
'''
Approach: Consider the below equations of given two planes:

P1 : a1 * x + b1 * y + c1 * z + d1 = 0 and,
P2 : a2 * x + b2 * y + c2 * z + d2 = 0,
'''
import math
from  transformations import rotation_matrix,angle_between_vectors, vector_product,translation_matrix,affine_matrix_from_points
# Function to find Angle
def distance(v1, v2):
    a1, b1, c1=v1
    a2, b2, c2=v2
    d = (a1 * a2 + b1 * b2 + c1 * c2)
    e1 = math.sqrt(a1 * a1 + b1 * b1 + c1 * c1)
    e2 = math.sqrt(a2 * a2 + b2 * b2 + c2 * c2)
    d = d / (e1 * e2)
    A = math.degrees(math.acos(d))
    print("Angle is", A, "degree")
    return -A


# Driver Code
import numpy
import open3d
def make_pcd(points, colors):
  pcd = open3d.geometry.PointCloud()
  pcd.points = open3d.utility.Vector3dVector(points)
  pcd.colors = open3d.utility.Vector3dVector(colors)
  return pcd
# https://developer.rhino3d.com/guides/rhinopython/python-rhinoscriptsyntax-plane/
#https://developer.rhino3d.com/guides/rhinopython/
#https://ja.wikipedia.org/wiki/%E5%9B%9E%E8%BB%A2%E8%A1%8C%E5%88%97
import numpy as np
from easydict import EasyDict

def get_rotation_matrix(x,axis_r='y'): #0:Rx, 1:Ry, 2:Rz
    rotation_matrix = EasyDict({})
    rotation_matrix.x = [
        [1, 0, 0],
        [0, np.cos(x), np.sin(x)],
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

vec20201015155844_z=[-0.0889259,-2.69011117,-1]
vec20201015155844_y=[-0.0336802,-1,-0.3684832]
vec20201015155844_x=[-1,-18.56481766,-6.71416052]

vec20201015155835_z=[-0.06199354,4.22205205,-1]
vec20201015155835_y=[0.03323493,-1,0.00610331]
vec20201015155835_x=[-1,19.77586931,-0.05332476]
# plane_y: 1*x+0*y+1*z+0=0
# vec20201015155835_z=[0,0,1]
def rotaton_ply(p,v0,v1):
    rotation_m = rotation_matrix(angle_between_vectors(v0, v1), vector_product(v0, v1))
    f = f"{p}/pcd_extracted.ply"
    pcd = open3d.io.read_point_cloud(f)
    pcd_r=pcd.transform(rotation_m)
    translate_m = translation_matrix(vector_product(v0, v1))
    pcd_r=pcd_r.transform(translate_m)
    open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_matrix_2plane_test.ply', pcd_r)
v_dst=[1,0,0]
# v_dst=vec20201015155835_y
rotaton_ply('C:/00_work/05_src/data/frm_t/20201015155844',vec20201015155844_z,v_dst)
rotaton_ply('C:/00_work/05_src/data/frm_t/20201015155835',vec20201015155835_y,v_dst)
pcd20201015155835 = open3d.io.read_point_cloud('C:/00_work/05_src/data/frm_t/20201015155835/pcd_extracted.ply')
pcd20201015155844 = open3d.io.read_point_cloud('C:/00_work/05_src/data/frm_t/20201015155844/pcd_extracted.ply')
points20201015155835 = np.array(pcd20201015155835.points)
colors20201015155835 = np.array(pcd20201015155835.colors)
points20201015155844 = np.array(pcd20201015155844.points)
colors20201015155844 = np.array(pcd20201015155844.colors)
cnt=points20201015155835.shape[0]
points20201015155844=points20201015155844[:cnt,:]
translate_m=affine_matrix_from_points(points20201015155844.T, points20201015155835.T, shear=True, scale=True, usesvd=True)

pcd_r = pcd20201015155835.transform(translate_m)
open3d.io.write_point_cloud(f'C:/00_work/05_src/data/frm_t/20201015155835/pcd_extract_plane_matrix_2plane_test_affine.ply', pcd_r)
pcd_r = pcd20201015155844.transform(translate_m)
open3d.io.write_point_cloud(f'C:/00_work/05_src/data/frm_t/20201015155844/pcd_extract_plane_matrix_2plane_test_affine.ply', pcd_r)

# k=unit_vector(vec20201015155844_z)
vec_z=[0,0,1]
vec_x=[1,0,0]
vec_y=[0,1,0]
ang_20201015155844_z=distance([-0.0889259,-2.69011117,0],vec_z)
ang_20201015155844_x=distance([0,-2.69011117,-1],vec_z)
rotation_20201015155844_x=get_rotation_matrix(ang_20201015155844_x,'x')
rotation_20201015155844_z=get_rotation_matrix(ang_20201015155844_z,'z')
rotation_20201015155844=rotation_20201015155844_x @ rotation_20201015155844_z




trans_matrix=np.vstack((rotation_20201015155844_z,np.array([0,0,0])))
trans_matrix=np.hstack((trans_matrix,np.expand_dims(np.array([0,0,0,1]),axis=1)))
# pcd_r_20201015155844=pcd.transform(trans_matrix)

# rotation_20201015155844=get_rotation_matrix(ang_20201015155844,'x')
# rotation_20201015155835=get_rotation_matrix(ang_20201015155835,'x')

# print(rotation_20201015155844)
# print(rotation_20201015155835)

# trans_matrix=np.vstack((rotation_20201015155844,np.array([0,0,0])))
# trans_matrix=np.hstack((trans_matrix,np.expand_dims(np.array([0,0,0,1]),axis=1)))
# pcd_r_20201015155844=pcd.transform(trans_matrix)
# open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_matrix_y.ply', pcd_r_20201015155844)

import numpy as np
import vg
from pytransform3d.rotations import matrix_from_axis_angle
from scipy.spatial.transform import Rotation as Rotation

def _rotmat(vector, points):
    """
    Rotates a 3xn array of 3D coordinates from the +z normal to an
    arbitrary new normal vector.
    """

    vector = vg.normalize(vector)
    axis = vg.perpendicular(vg.basis.z, vector)
    angle = vg.angle(vg.basis.z, vector, units='rad')

    a = np.hstack((axis, (angle,)))
    R = matrix_from_axis_angle(a)

    r = Rotation.from_matrix(R)
    rotmat = r.apply(points)

    return rotmat

# p = 'C:/00_work/05_src/data/frm_t/20201015155844'
# f = f"{p}/pcd_extracted.ply"
# pcd = open3d.io.read_point_cloud(f)
# points = np.array(pcd.points)
# colors = np.array(pcd.colors)
# vector=np.array([-0.0889259,-2.69011117,-1])
# rotmat=_rotmat(vector, points)
# pcdn=make_pcd(rotmat,colors)
# open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_matrix_testdddddd.ply', pcdn)
# p = 'C:/00_work/05_src/data/frm_t/20201015155835'
# f = f"{p}/pcd_extracted.ply"
# pcd = open3d.io.read_point_cloud(f)
# trans_matrix=np.vstack((rotation_20201015155835,np.array([0,0,0])))
# trans_matrix=np.hstack((trans_matrix,np.expand_dims(np.array([0,0,0,1]),axis=1)))
# pcd_r_20201015155835=pcd.transform(trans_matrix)
# open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_matrix_y.ply', pcd_r_20201015155835)



# rotation_20201015155844=get_rotation_matrix(ang_20201015155844,'z')
# rotation_20201015155835=get_rotation_matrix(ang_20201015155835,'z')
#
# p = 'C:/00_work/05_src/data/frm_t/20201015155844'
# trans_matrix=np.vstack((rotation_20201015155844,np.array([0,0,0])))
# trans_matrix=np.hstack((trans_matrix,np.expand_dims(np.array([0,0,0,1]),axis=1)))
# pcd_r_20201015155844_z=pcd_r_20201015155844.transform(trans_matrix)
# open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_matrix_y_z.ply', pcd_r_20201015155844_z)
#
# p = 'C:/00_work/05_src/data/frm_t/20201015155835'
# trans_matrix=np.vstack((rotation_20201015155835,np.array([0,0,0])))
# trans_matrix=np.vstack((rotation_20201015155835,np.array([0,0,0])))
# trans_matrix=np.hstack((trans_matrix,np.expand_dims(np.array([0,0,0,1]),axis=1)))
# pcd_r_20201015155835_z=pcd_r_20201015155835.transform(trans_matrix)
# open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_matrix_y_z.ply', pcd_r_20201015155835_z)