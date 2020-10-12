import numpy as np
# import os, open3d, quaternion, functools, itertools, struct, time
import os, open3d,  functools, itertools, struct, time
import vectormath as vmath

verbose=True

def stop_watch(func) :
  @functools.wraps(func)
  def wrapper(*args, **kargs) :
    start = time.time()
    result = func(*args,**kargs)
    if verbose:
      print(f"{func.__name__}() elaplsed: {time.time() - start:.03f} [sec]")
    return result
  return wrapper

def make_pcd(points, colors):
  pcd = open3d.geometry.PointCloud()
  pcd.points = open3d.utility.Vector3dVector(points)
  pcd.colors = open3d.utility.Vector3dVector(colors)
  return pcd

@stop_watch
def calcurate_xyz(color, depth, intr):
  """
  camera. Given depth value d at (u, v) image coordinate, the corresponding 3d point is:
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
  
  for example, depth value of ZED 2 is float32 and its value indicates meter directly.
  so we need to ignore "depth_scale" like open3d's RGBDImage style.
  """
  #intr = menu.config[menu.config['table'][menu.reso]]
  ws, hs = ([a for a in range(depth.shape[n])] for n in [0,1])
  points, colors = [], []
  for u, v in itertools.product(ws, hs):
    z = float(depth[u, v])
    if np.isnan(z) or np.isinf(z):
      continue
    x = float((u - intr.cx) * z / intr.fx)
    y = float((v - intr.cy) * z / intr.fy)
    point = np.asarray([x,y,z])
    points.append(point)
    rgb = (color[u, v][:3]/255).astype(np.float64)
    colors.append(rgb)
    #colors.append((color[u, v][:3]).astype(np.uint8))
  pcd = make_pcd(np.array(points), colors)
  return pcd #if not verbose else pcd, points, colors

def create_pcd_from_float_depth(menu):
  # ZED2 cam returns depth map with float32, 
  #   and open3d's create_from_depth_image() does not support float.
  color, depth = [
      np.load(os.path.join(data_dir, 'camera{}_{}_{}.npy'.format(menu.cam_id, menu.reso, c_or_d))) \
      for c_or_d in ['color', 'depth']]
  window = menu.frames[menu.cam_id]['window']
  color = convert_bgra2rgba(color)
  if menu.option.with_frame and (np.array(window) != None).all():
    color, depth, depth_med = add_frame2img(menu, color, depth)
    if menu.frames != None:
      menu.frames[menu.cam_id].depth_med = depth_med
  pcd = calcurate_xyz(menu, color, depth)
  if menu.option.correction:
    pass
  return pcd

def add_frame2img(color, depth, window_frame):
  h0,w0,h1,w1 = window_frame
  # prepare data
  img = np.zeros(color.shape).astype(color.dtype)
  img[:] = color[:]
  dep = np.zeros(depth.shape).astype(depth.dtype)
  dep[:] = depth[:]
  # write frame line with red 
  red = [255,0,0,255]
  img[h0:h1, w0] = red
  img[h0:h1, w1] = red
  img[h0, w0:w1] = red
  img[h1, w0:w1] = red
  # calcurate median of depth value inside of frame line
  target_depth = depth[h0:h1,  w0:w1]
  ws, hs = [[a for a in range(n)] for n in target_depth.shape]
  depth_vals = np.array([target_depth[w,h] for w, h in itertools.product(ws,hs) if not (np.isnan(target_depth[w,h]) or np.isinf(target_depth[w,h]))])
  depth_med = np.median(depth_vals)
  print('depth_med:', depth_med)
  # fill depth_med value in frame line
  dep[h0:h1, w0] = depth_med
  dep[h0:h1, w1] = depth_med
  dep[h0, w0:w1] = depth_med
  dep[h1, w0:w1] = depth_med
  return img, dep, depth_med # depth_med is needed for merge some cam's pcd data.

def convert_bgra2rgba(img):
  # ZED2 numpy color is BGRA.
  rgba = np.zeros(img.shape).astype(np.uint8)
  rgba[:,:,0] = img[:,:,2] # R
  rgba[:,:,1] = img[:,:,1] # G
  rgba[:,:,2] = img[:,:,0] # B
  rgba[:,:,3] = img[:,:,3] # A
  return rgba

def get_quaternion_from_vectors(target=None, source=None):
  """
    参考URL
    https://knowledge.shade3d.jp/knowledgebase/2%E3%81%A4%E3%81%AE%E3%83%99%E3%82%AF%E3%83%88%E3%83%AB%E3%81%8C%E4%BD%9C%E3%82%8B%E5%9B%9E%E8%BB%A2%E8%A1%8C%E5%88%97%E3%82%92%E8%A8%88%E7%AE%97-%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%97%E3%83%88
    ベクトルaをbに向かせるquaternionを求める.
  """
  qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
  normalize = lambda x: np.array(vmath.Vector3(x).normalize())
  src_vec = normalize(source)
  target_vec = normalize(target)
  cross_vec = np.cross(target_vec, src_vec)
  cross_vec_norm = -np.linalg.norm(cross_vec)
  cross_vec = normalize(cross_vec)
  epsilon = 0.0002
  inner_vec = np.dot(src_vec, target_vec)
  if -epsilon < cross_vec_norm or 1.0 < inner_vec:
    if inner_vec < (epsilon - 1.0):
      trans_axis_src = np.array([-src_vec[1], src_vec[2], src_vec[0]])
      c = normalize(np.cross(trans_axis_src, src_vec))
      qw = 0.0
      qx = c[0]
      qy = c[1]
      qz = c[2]
  else:
    e = cross_vec * math.sqrt(0.5 * (1.0 - inner_vec))
    qw = math.sqrt(0.5 * (1.0 + inner_vec))
    qx = e[0]
    qy = e[1]
    qz = e[2]
  return np.quaternion(qw,qx,qy,qz)

def extract_red_point(pcd):
  ext = EasyDict({})
  ext.p, ext.c = [], []
  points, colors = np.array(pcd.points), np.array(pcd.colors)
  for i in range(points.shape[0]):
    p, c = points[i], colors[i]
    if not (c == np.array([1.,0.,0.]).astype(c.dtype)).all():
      continue
    else:
      ext.p.append(p)
      ext.c.append(c)
  return make_pcd(ext.p, ext.c)

def find_plane_from_pcd(pcd):
  # 参考URL: https://qiita.com/sage-git/items/f64620d18eeff8a11308
  r = np.array(pcd.points)
  c = np.mean(r, axis=0)
  r0 = r - c
  u, s, v = np.linalg.svd(r0)
  nv = v[-1, :]      # 最小のsに対応するベクトル
  ds = np.dot(r, nv) # サンプル点群の平面と原点との距離
  param = np.r_[nv, -np.mean(ds)]
  return param

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
  h = hex(struct.unpack('<I', struct.pack('<f', f))[0])
  return [eval('0x'+a+b) for a, b in zip(h[::2], h[1::2]) if a+b != '0x']

# https://takala.tokyo/takala_wp/2018/11/28/736/
np_zed_depthfloat_to_abgr = np.frompyfunc(zed_depthfloat_to_abgr, 1, 1)

@stop_watch
def convert_zed2pcd_to_ply(zed_pcd):
  zed_points = zed_pcd[:,:,:3]
  zed_colors = zed_pcd[:,:,3]
  points, colors = [], []
  for x, y in itertools.product(
    [a for a in range(zed_colors.shape[0])],
    [a for a in range(zed_colors.shape[1])]):
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
