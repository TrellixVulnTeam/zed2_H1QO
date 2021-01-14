import numpy as np
import cupy as cp
import os, glob, struct, time, functools

verbose = True

get_rgb_from_zed2pcd = \
  cp.ElementwiseKernel(
  in_params='raw float32 x, int32 hs, int32 ws, int32 chs',
  out_params='T res',
  operation=\
  '''
  // set RGBA(each 8bit in float 32bit) value in x[y0, x0] \
  //   to data res[y0, x0, 0:4]
  int x0, y0, idx, ch0;
  x0 = i /  chs % ws;
  y0 = i / (chs * ws) % hs;
  idx = int(y0 * ws + x0);
  ch0 = i % chs;
  res = ((unsigned char *) &x[idx])[ch0];
  ''',
  name='get_rgb_from_zed2pcd')

def stop_watch(func) :
  @functools.wraps(func)
  def wrapper(*args, **kargs) :
    start = time.time()
    result = func(*args,**kargs)
    if verbose:
      print(f"{func.__name__}() elaplsed: {time.time() - start:.03f} [sec]")
    return result
  return wrapper

@stop_watch
def convert_zed2pcd_to_ply(zed2pcd=None, p=None):
  assert (zed2pcd == None) ^ (p == None)
  if p != None:
    print(f"{p}/pcd.npy loading...")
    zed2pcd = np.load(f"{p}/pcd.npy")
  zed_points = zed2pcd[:,:,:3]
  zed_colors = zed2pcd[:,:,3]
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
  if p == None:
    return make_pcd(points, colors)
  else:
    pcd = make_pcd(points, colors)
    open3d.io.write_point_cloud(f'{p}/pcd_original.ply', pcd)

@stop_watch
def convert_zed2pcd_to_ply_faster(p=None, zed2pcd=None):
  if isinstance(zed2pcd, cp.ndarray):
    assert len(zed2pcd.shape) == 3
  if [p, zed2pcd] == [None, None]:
    zed2pcd = cp.array([[
      [0x12345678, 0x98765432, 0xfeedbeef],
      [0x12345678, 0x98765432, 0xdeadbeef]]], 
      dtype=cp.float32)
  if p != None:
    print(f"{p}/pcd.npy loading...")
    zed2pcd = cp.array(np.load(f"{p}/pcd.npy"))
  chs = 4 # (R,G,B,A)  
  zed_points = zed2pcd[:,:,:3]
  encoded_color = zed2pcd[:,:,3]
  decoded = cp.zeros((*encoded_color.shape[:2], chs)).astype(cp.uint32)
  (hs, ws) = encoded_color.shape
  return get_rgb_from_zed2pcd(encoded_color, hs, ws, chs, decoded)

def convert_zed2pcd_to_abgr(f):
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
    return [eval('0x'+a+b) for a, b in zip(h[::2], h[1::2]) if a+b != '0x'], struct.unpack('<I', struct.pack('<f', f))[0]

if __name__ == '__main__':
  res1 = convert_zed2pcd_to_ply_faster(p='./dt/output/cam0/20201110150842_494663')
  res2 = convert_zed2pcd_to_ply_faster(p='./dt/output/cam0/20201110150846_477187')
  res3 = convert_zed2pcd_to_ply_faster(p='./dt/output/cam0/20201110150847_471665')
