import cv2
import numpy as np
import cv2
import quaternion
# import  pyquaternion as  quaternion
from transforms3d.quaternions import quat2mat, mat2quat
from testply import convert_xyzrgb_to_ply,make_pcd
import open3d
# import pcl
#python -m pip install numpy-quaternion
#pip install pyquaternion
#conda install -c conda-forge quaternion
# camera intrinsics
cx = 325.5
cy = 253.5
fx = 518.0
fy = 519.0
depthScale = 1000.0

colorImgs, depthImgs = [], []

bp='C:/00_work/05_src/zed2/zed-opencv/python/dt/data_test/'
# read pose.txt
# fn_d='reconstruction-000005.depth-ZED_22378008.png'
# img=cv2.imread(f'{bp}/{fn_d}')
pose = []

def translations_quaternions_to_transform(pose):
    t = pose[:3]
    q = pose[3:]

    T = np.eye(4)
    T[:3, :3] = quat2mat(q)
    T[:3, 3] = t
    return T
with open(f'{bp}pose.txt', 'r') as f:
    for line in f.readlines():
        line = line.replace('\n', '')  # remove returns
        line = line.split(' ')  # split into 7 items
        vector = []
        for num in line:
            vector.append(float(num))
        vector = np.array(vector)
        # q=[vector[0], vector[1], vector[2],vector[6], vector[3], vector[4], vector[5]]
        q=[vector[0], vector[1], vector[2],vector[3], vector[4], vector[5],vector[6]]
        T=translations_quaternions_to_transform(q)
        pose.append(T)

# read color and depth images
view = []
colors_all=[]
points_all=[]
for i in range(3):
    fn=f'{bp}reconstruction-00000{i}.cloud-ZED_22378008.ply'
    ply=open3d.io.read_point_cloud(fn)
    # fn=f'{bp}reconstruction-00000{i}.pose-ZED_22378008.csv'
    # trans_intro=np.loadtxt(fn)
    # ply=ply_t.transform(trans_intro)
    points=np.array(ply.points)
    colors=np.array(ply.colors)
    points_1=np.hstack((points,np.ones((points.shape[0],1))))
    point_worlds=[]
    for point in points_1:
        point_world = np.dot( pose[i],point)#np.dot(pose[i], points_1)
        point_worlds.append(point_world[:3])
    colors_all.extend(colors)
    points_all.extend(point_worlds)
    print(len(points_all))
    pcd=make_pcd(point_worlds, colors)
    open3d.io.write_point_cloud(f'pcd_merge_{i}.ply', pcd)

pcd=make_pcd(points_all, colors_all)
open3d.io.write_point_cloud('pcd_merge.ply', pcd)