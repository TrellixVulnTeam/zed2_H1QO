import numpy as np
import zed2cam_bk, zed2pcl, open3d
menu = zed2cam_bk.init(zed2cam_bk.menu)
menu = zed2cam_bk.switch_take_mode(menu, zed2cam_bk.TakeMode.RGBD)
menu = zed2cam_bk.take(menu)

save_dir = 'data/20201012171626'

color = np.load(f'{save_dir}/image.npy')
depth = np.load(f'{save_dir}//depth.npy')
transform = np.load(f'{save_dir}/pose_transform.npy')

res = str(zed2cam_bk.mode.resolution[menu.cam.resolution]).split('.')[1]
intr = zed2cam_bk.cam_reso[res]

color = zed2pcl.convert_bgra2rgba(color)
pcd = zed2pcl.calcurate_xyz(color, depth, intr)
open3d.io.write_point_cloud('test.ply', pcd)
