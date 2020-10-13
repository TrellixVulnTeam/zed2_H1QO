import numpy as np
import zed2cam, zed2pcl, open3d
menu = zed2cam.init(zed2cam.menu)
menu = zed2cam.switch_take_mode(menu, zed2cam.TakeMode.RGBD)
menu = zed2cam.take(menu)


save_dir = 'data/20201013095439'

color = np.load(f'{save_dir}/image.npy')
depth = np.load(f'{save_dir}/depth.npy')
transform = np.load(f'{save_dir}/transform.npy')

res = str(zed2cam.mode.resolution[menu.cam.resolution]).split('.')[1]
intr = zed2cam.cam_reso[res]

color = zed2pcl.convert_bgra2rgba(color)
pcd = zed2pcl.calcurate_xyz(color, depth, intr)
open3d.io.write_point_cloud('test.ply', pcd)
