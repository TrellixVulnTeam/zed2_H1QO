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