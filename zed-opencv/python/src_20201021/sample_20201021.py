import os, zed2pcl

base_dir = './data/toYOU_20201021'

def get_base_outputs(d):
  zed2pcl.save_image_as_png(d)
  zed2pcl.apply_rotation(d)
  zed2pcl.extract_in_4point_erea(d)

for p in os.listdir(base_dir):
  d = f'{base_dir}/{p}'
  get_base_outputs(d)

