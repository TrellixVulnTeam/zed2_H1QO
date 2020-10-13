from PIL import Image
from easydict import EasyDict
import numpy as np
import zed2pcl, json, open3d



# VGAでの撮影。
res = "RESOLUTION.VGA"
config ="config/zed2cam.json"
# intr = EasyDict(json.load(open(config))[res])

files = [
 "data_1013_1/reconstruction-000000.color-ZED_21888201.jpg",
 "data_1013_1/reconstruction-000000.depth-ZED_21888201.png",
 "data_1013_1/ZED_21888201-camera-intrinsics.txt"]
color = np.array(Image.open(files[0]))
depth = np.array(Image.open(files[1]))
k=np.loadtxt(files[2])
intr=EasyDict({})
intr.fx=k[0,0]
intr.fy=k[1,1]
intr.cx=k[0,2]
intr.cy=k[1,2]
pcd = zed2pcl.calcurate_xyz(color, depth, intr)
open3d.io.write_point_cloud('reconstruction-000000.pcd-ZED_21888201.ply', pcd)
