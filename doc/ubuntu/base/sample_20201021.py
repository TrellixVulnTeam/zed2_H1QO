import os #, zed2pcl
import json,itertools,open3d,struct
import numpy as np
from easydict import EasyDict
from sklearn.linear_model import LinearRegression
base_dir = 'C:/00_work/05_src/data/frm_t'


def make_pcd(points, colors):
  pcd = open3d.geometry.PointCloud()
  pcd.points = open3d.utility.Vector3dVector(points)
  pcd.colors = open3d.utility.Vector3dVector(colors)
  return pcd
def zed_depthfloat_to_abgr(f):
  """
    ZED pcd data format:
      ----------------------------------------------------------
      https://www.stereolabs.com/docs/depth-sensing/using-depth/
      ----------------------------------------------------------
      The point cloud stores its data on 4 channels using 32-bit
      float for each channel.
      The last float is used to store color information, where
      R, G, B, and alpha channels (4 x 8-bit) are concatenated
      into a single 32-bit float.
  """
  # https://stackoverflow.com/questions/23624212/how-to-convert-a-float-into-hex/38879403
  if f == 0.:
    return [0,0,0,0]
  else:
    h = hex(struct.unpack('<I', struct.pack('<f', f))[0])
    return [eval('0x'+a+b) for a, b in zip(h[::2], h[1::2]) if a+b != '0x']

def convert_zed2pcd_to_ply(zed_pcd):
  zed_points = zed_pcd[:,:,:3]
  zed_colors = zed_pcd[:,:,3]
  points, colors = [], []
  for x, y in itertools.product(
    [a for a in range(zed_colors.shape[0])],
    [a for a in range(zed_colors.shape[1])]):
    if zed_points[x, y].sum() == 0. and zed_points[x, y].max() == 0.:
      continue
    if np.isinf(zed_points[x, y]).any() or  np.isnan(zed_points[x, y]).any():
      continue
    # color
    tmp = zed_depthfloat_to_abgr(zed_colors[x, y])
    tmp = [tmp[3], tmp[2], tmp[1]] # ABGR to RGB
    tmp = np.array(tmp).astype(np.float64) / 255. # for ply (color is double)
    colors.append(tmp)
    # point
    tmp = np.array(zed_points[x, y]).astype(np.float64)
    points.append(tmp)
  return make_pcd(points, colors)
def extract_in_4point_erea(d, is_wh=True):
  def make_ps(p0,p1,p2,p3):
    (a, b) = ('w', 'h') if is_wh == True else ('h', 'w')
    return [EasyDict({a:p[0], b:p[1]}) \
        for p in [p0, p1, p2, p3]]
  def get_center(ps):
    center = EasyDict({})
    center.w = np.array([p.w for p in ps]).mean()
    center.h = np.array([p.h for p in ps]).mean()
    return center
  def get_slope_intercention(p0, p1):
    slope = (p0.h - p1.h) / (p0.w - p1.w)
    intercept = p0.h - p0.w * slope
    return slope, intercept
  def check_in_line(p0, p1, p2):
    slope, intercept = get_slope_intercention(p0, p1)
    return p2.h == int(p2.w * slope + intercept)
  def check_mode(ps):
    mode, ignore, res = 'square', None, []
    for p0, p1, p2 in itertools.combinations(ps,3):
      if check_in_line(p0, p1, p2):
        px = [p0,p1,p2]
        idx=np.array([p.w for p in px]).argsort()[1]
        ignore = px[idx]
        res.append([True, px, ignore])
      else:
        res.append([False, None, None])
    check = [x[0] for x in res]
    if all(check):
      raise Exception("all in 1 line!!!")
    elif any(check):
      mode = 'triangle'
      ignore = res[check.index(True)][2]
    return mode, ignore
  def check_pos(ps):
    res = EasyDict({}) # {right_up, left_up, left_down, right_down}
    center = get_center(ps)
    mode, ignore = check_mode(ps)
    if mode == 'square':
      w_indices = list(np.array([p.w for p in ps]).argsort())
      lefts = [ps[i] for i in w_indices[:2]]
      rights = [ps[i] for i in w_indices[2:]]
      res.left_up = lefts[lefts[0].h > lefts[1].h]
      res.left_down = lefts[not lefts[0].h > lefts[1].h]
      res.right_up = rights[rights[0].h > rights[1].h]
      res.right_down = rights[not rights[0].h > rights[1].h]
    elif mode == 'triangle':
      px = ps
      w_indices = list(np.array([p.w for p in ps]).argsort())
      res.start = ps[w_indices[0]]
      res.middle = ps[w_indices[1]]
      res.end = ps[w_indices[2]]
    assert (set(res.keys()) & set([f'{x}_{y}'
      for x,y in itertools.product(
        ['right', 'left'], ['up', 'down'])]) == set(res.keys())) \
        or (set(res.keys()) == set(['start', 'middle', 'end']))
    for k in res.keys():
      if len(res[k]) == 1:
        res[k] = res[k][0]
    return EasyDict(res), mode
  def get_area_of_pos(pos):
    ws, hs = sorted([pos[k].w for k in pos.keys()]), sorted([pos[k].h for k in pos.keys()])
    area = EasyDict({'w': {'min':ws[0], 'max':ws[-1]},
                     'h': {'min':hs[0], 'max':hs[-1]}})
    return area
  zed2pcd = np.load(f"{d}/pcd.npy")
  p0,p1,p2,p3 = json.load(open(f'{d}/frame.json'))['extract_frame_wh']
  ps = make_ps(p0,p1,p2,p3)
  assert all([len(p) == 2 and
   all([p[x] in [a for a in range(zed2pcd.shape[n])] for x, n in zip(['h','w'],[0,1])])
     for p in ps])
  extracted_area = np.zeros(zed2pcd.shape).astype(zed2pcd.dtype)
  pos, mode = check_pos(ps)
  area = get_area_of_pos(pos)
  # vertical line scan algorythm
  if mode == 'square':
    start_p = 'left_up' if pos.left_up.w <= pos.left_down.w \
                else 'left_down'
    upper_p = 'right_up' if 'up' in start_p else 'left_up'
    lower_p = 'left_down' if 'up' in start_p else 'right_down'
    end_p   = 'right_down' if 'right' in upper_p else 'right_up'
    for w in range(area.w.min, area.w.max+1):
      if w == area.w.min or w == area.w.max:
        p = start_p if w == area.w.min else end_p
        w, h = pos[p].w, pos[p].h
        extracted_area[h, w, :] = zed2pcd[h, w, :]
      else:
        # edges
        upleft = start_p if w < pos[upper_p].w else upper_p
        upright = end_p if w >= pos[upper_p].w else upper_p
        lowleft = start_p if w < pos[lower_p].w else lower_p
        lowright = end_p if w >= pos[lower_p].w else lower_p
        # check slope
        up_slope = (pos[upright].h - pos[upleft].h) / \
                  (pos[upright].w - pos[upleft].w)
        up_h = int(up_slope * (w - pos[upleft].w) + pos[upleft].h)
        low_slope = (pos[lowright].h - pos[lowleft].h) / \
                  (pos[lowright].w - pos[lowleft].w)
        low_h = int(low_slope * (w - pos[lowleft].w) + pos[lowleft].h)
        extracted_area[up_h:low_h, w, :] = zed2pcd[up_h:low_h, w, :]
  else:
    print('*** WARN ***\n \
           Triangle mode is not implemented!!!\n \
           Please check frame.json and correct \n \
           position NOT to be triangle.')
    return
  pcd = convert_zed2pcd_to_ply(extracted_area)
  open3d.io.write_point_cloud(f'{d}/pcd_extracted.ply', pcd)
  return extracted_area

def train_dep(train_points):
  train_points = np.array(train_points)
  lr = LinearRegression()
  train_x = train_points[:, :2]
  train_y = train_points[:, 2]
  lr.fit(train_x, train_y)
  return lr
for p in os.listdir(base_dir):
  d = f'{base_dir}/{p}'
  if os.path.isdir(d):
    extracted_pcd=extract_in_4point_erea(d)

    # train_dt=extracted_pcd[:,:,:3].reshape(-1,3)
    # lrz=train_dep(train_dt)
    # ply = convert_zed2pcd_to_ply(extracted_pcd)
    # open3d.io.write_point_cloud(f'{base_dir}/'+d+'_pcd_extracted.ply', ply)

