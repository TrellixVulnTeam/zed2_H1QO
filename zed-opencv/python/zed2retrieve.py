import numpy as np
import os, json, open3d, quaternion, math, itertools
import vectormath as vmath
from easydict import EasyDict
from zed2pcl import *
#pip install easydict
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# base settings
base_dir = "G:/マイドライブ/00_work/zed_dt/Zed2/"
data_dir = os.path.join(base_dir, 'data/20200928')

# config
cam_id, reso = 'A', '1280x720'
menu = EasyDict({
    'reso' : reso,
    'option' : {
      'with_frame' : True,
      'with_arrow' : True,
      'correction' : True
    },
    'config'  : EasyDict(json.load(
                    open(os.path.join(base_dir, 'zed2cam.json')))),
    'frames'  : None,
    'pcds'    : None,
    'sensors' : None,
})

def get_custom_data(menu):
  menu.frames  = { cam_id: json.load(open(os.path.join(base_dir,
                  'camera{}_{}_frame.json'.format(cam_id, reso))))
                   for cam_id in ['A', 'B', 'C']}
  menu.pcds    = get_custom_pcds(menu)
  menu.sensors = { cam_id: json.load(open(os.path.join(base_dir,
                  'camera{}_{}_sensor.json'.format(cam_id, reso))))
                  for cam_id in ['A', 'B', 'C']}
  return menu

def get_custom_filename_by_menu(menu, cam_id):
  menu_name = ''
  if any([menu.option[m] for m in menu.option.keys()]):
    menu_name = '_with'
    menu_name += '-frame' if menu.option.with_frame else ''
    menu_name += '-arrow' if menu.option.with_arrow else ''
    menu_name += '-correction' if menu.option.with_arrow else ''
  target_dir = os.path.join(data_dir, 'output_ply')
  os.makedirs(target_dir, exist_ok=True)
  return os.path.join(target_dir, 'camera{}_{}{}.ply'.format(cam_id, menu.reso, menu_name))

def get_custom_pcds(menu=menu):
  # get point cloud data
  pcds = {}
  for cam_id in ['A', 'B', 'C']:
    filename = get_custom_filename_by_menu(menu, cam_id)
    menu.cam_id = cam_id
    filename='camera-'+cam_id+'-1280x720_with-frame-arrow-correction.ply'
    if not os.path.exists(filename):
      pcds[menu.cam_id] = create_pcd_from_float_depth(menu) 
      open3d.io.write_point_cloud(filename, pcds[cam_id])
    else:
      open3d.io.read_point_cloud(filename)
  menu.pcds = pcds
  return menu

def get_corrected_pcd_by_imu(cam_id, pcds, sensors, with_arrow=False):
  points = np.array(pcds[cam_id].points)
  colors = np.array(pcds[cam_id].colors)
  acc = sensors[cam_id]['imu']['acceleration']
  acc_normalized = np.array(vmath.Vector3(acc).normalize())
  # coordinate system convert
  # Zed2 imu system:[x, y, z]
  # Open3d: (-y,x,-z)
  # meshlab: 
  acc_converted = np.array([-acc_normalized[1], acc_normalized[0], -acc_normalized[2]])
  if with_arrow:
    add_points=np.array([[acc_converted[0]*0.001*e, acc_converted[1]*0.001*e, acc_converted[2]*0.001*e] for e in range(2000)])
    add_colors=np.array([[1.,0.,0.] for e in range(2000)])
    points = np.vstack((points, add_points))
    colors = np.vstack((colors, add_colors))
    filename = os.path.join(data_dir, 'camera{}_{}_with_acc-arrow.ply'.format(cam_id, reso))
    open3d.io.write_point_cloud(filename, make_pcd(points, colors))
  q=get_quaternion_from_vectors(source=np.array([1.,0.,0.]), target=acc_converted)
  points = quaternion.rotate_vectors(q, points)
  pcd = make_pcd(points, colors)
  filename = os.path.join(data_dir, 'camera{}_{}{}_rot.ply'.format(cam_id, reso, '_with_arrow' if with_arrow else ''))
  open3d.io.write_point_cloud(filename, pcd)
  return pcd
print("start get_custom_data")
menu = get_custom_data(menu)
print("end get_custom_data")

print("start get_corrected_pcd_by_imu")
pcd_b = get_corrected_pcd_by_imu('B', menu.pcds, menu.sensors, with_arrow=True)
print("end get_corrected_pcd_by_imu")
pcd_a = get_corrected_pcd_by_imu('A', menu.pcds, menu.sensors, with_arrow=True)

img=np.load(os.path.join(data_dir, 'camera{}_{}_{}.npy'.format('C', reso, 'color')))
img = convert_bgra2rgba(img)
plt.imshow(img)

frameonly_pcds = {}
pcds = {}
for cam_id in ['A', 'B', 'C']:
  color, depth = [
      np.load(os.path.join(data_dir, 'camera{}_{}_{}.npy'.format(cam_id, menu.reso, c_or_d))) \
      for c_or_d in ['color', 'depth']]
  color = convert_bgra2rgba(color)
  menu.cam_id = cam_id
  menu.window = menu.frames[cam_id]['window']
  pcds[cam_id] = create_pcd_from_float_depth(menu)
  frameonly_pcds[cam_id] = extract_red_point(pcds[cam_id])
  filename = os.path.join(data_dir, 'camera{}_{}_frameonly.ply'.format(menu.cam_id, menu.reso))
  open3d.io.write_point_cloud(filename, pcds[cam_id])

ps=quaternion.rotate_vectors(q, points_add_a)
open3d.io.write_point_cloud(os.path.join(data_dir, 'arrow_rot.ply'), make_pcd(ps, colors_add_a))
