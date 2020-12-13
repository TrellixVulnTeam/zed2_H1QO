import os, glob, shutil, zed2pcl,argparse
import numpy as np
from easydict import EasyDict
from scipy.spatial.transform import Rotation as R
parser = argparse.ArgumentParser()
parser.add_argument("--target_dir", help="file search start directory", type=str)
parser.add_argument("--output_dir", help="save directory", type=str)
args = parser.parse_args()
def make_transform_menu(cam, p0):
  assert isinstance(cam, EasyDict)
  assert cam.keys() >= {'distance_ref_point_on_floor', 'rot_y_axis', 'rot_zenkei', 'height'}
  menu = EasyDict({
    'transforms':[{
      # Ry rot (correction of zenkei)
      "p" : p0,
      "matrix" : R.from_rotvec([[0, -cam.rot_zenkei, 0]]).as_matrix(), # Ry
      "out_name" : f"Ry{int(np.round(np.rad2deg(cam.rot_zenkei)))}deg"
    },
    {
      # Rz rot (correction of kakudo)
      "p" : f"{p0}/pcd_Ry{int(np.round(np.rad2deg(cam.rot_zenkei)))}deg.ply",
      "matrix" : R.from_rotvec([[0, 0, -cam.rot_y_axis]]).as_matrix(), # Rz
      "out_name" : f"Rz{int(np.round(np.rad2deg(cam.rot_y_axis)))}deg"
    },
    {
      # translation (set reference point to zero point)
      "p" : f"{p0}/pcd_Ry{int(np.round(np.rad2deg(cam.rot_zenkei)))}deg_Rz{int(np.round(np.rad2deg(cam.rot_y_axis)))}deg.ply",
      "vector" : np.array([
        -cam.distance_ref_point_on_floor * np.cos(cam.rot_y_axis), 
        -cam.distance_ref_point_on_floor * np.sin(cam.rot_y_axis), 
        cam.height]), # Translation
      "out_name" : "Tr"
    }]
  })
  return menu

def execute_transform(menu):
  for f in [f for f in glob.glob(f'{menu.transforms[0].p}/**', 
              recursive=True) if not 'pcd_original.ply' in f and 'ply' in f]:
    os.remove(f) # 既存ファイルを削除
  for conf in menu.transforms:
    new_pcd = zed2pcl.apply_transformation(
      p=conf.p,       matrix=conf.matrix if 'matrix' in conf.keys() else None,
      vector=conf.vector if 'vector' in conf.keys() else None, 
      out_name=conf.out_name)

if __name__ == '__main__':
  location = EasyDict({
    'cam0':{
      'distance_ref_point_on_floor' : 6.3,
      'rot_y_axis' : 4 * np.pi/3, # 240deg
      'rot_zenkei' : np.pi/12,    # 15deg
      'height' : 2.0
    },
    'cam1':{
      'distance_ref_point_on_floor' : 5.3,
      'rot_y_axis' : 2 * np.pi/3, # 120deg
      'rot_zenkei' : np.pi/12,    # 15deg
      'height' : 1.5
    },
    'cam2':{
      'distance_ref_point_on_floor' : 5.3,
      'rot_y_axis' : 7 * np.pi/18, # 70deg
      'rot_zenkei' : np.pi/12,    # 15deg
      'height' : 1.7
    }
  }
  )
  args.target_dir='fuchi_test_size10'
  fs = [f.replace('\\', '/') for f in glob.glob(
    f'{args.target_dir}/**', recursive=True) \
    if 'pcd_original.ply' in f]
  ps = set([os.path.dirname(f) for f in fs])
  for p0 in ps:
    print(f'p0:[{p0}]')
    if 'cam0' in p0:
      cam = location.cam0
    elif 'cam1' in p0:
      cam = location.cam1
    else:
      cam = location.cam2
    menu = make_transform_menu(cam, p0)
    execute_transform(menu)
