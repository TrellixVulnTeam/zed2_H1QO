import numpy as np
import struct, itertools,open3d
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
        return [0, 0, 0, 0]
    else:
        h = hex(struct.unpack('<I', struct.pack('<f', f))[0])
        return [eval('0x' + a + b) for a, b in zip(h[::2], h[1::2]) if a + b != '0x']


def make_pcd(points, colors):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    return pcd
def convert_zed2pcd_to_ply(zed_pcd):
    zed_points = zed_pcd[:, :, :3]
    zed_colors = zed_pcd[:, :, 3]
    points, colors = [], []
    for x, y in itertools.product(
            [a for a in range(zed_colors.shape[0])],
            [a for a in range(zed_colors.shape[1])]):
        if zed_points[x, y].sum() == 0. and zed_points[x, y].max() == 0.:
            continue
        if np.isinf(zed_points[x, y]).any() or np.isnan(zed_points[x, y]).any():
            continue
        # color
        tmp = zed_depthfloat_to_abgr(zed_colors[x, y])
        tmp = [tmp[3], tmp[2], tmp[1]]  # ABGR to RGB
        tmp = np.array(tmp).astype(np.float64) / 255.  # for ply (color is double)
        colors.append(tmp)
        # point
        tmp = np.array(zed_points[x, y]).astype(np.float64)
        points.append(tmp)
    return make_pcd(points, colors)

def convert_xyzrgb_to_ply(zed_pcd):
    # zed_points = zed_pcd[:, :, :3]
    # zed_colors = zed_pcd[:, :, 3]
    points, colors = [], []
    for  point in zed_pcd:
        if sum(point[:3]) == 0.:
            continue
        #color
        tmp = zed_depthfloat_to_abgr(point[3])
        tmp = [tmp[3], tmp[2], tmp[1]]
        tmp = np.array(tmp).astype(np.float64) / 255
        colors.append(tmp)
        # point
        tmp = np.array(point[:3]).astype(np.float64)
        points.append(tmp)
    return make_pcd(points, colors)
filename='D:/02_AIPJ/004_ISB/slambook/ch13/dense_RGBD/map.pcd'
# with  open(filename) as f:
#   foo = f.readlines()
# zed2pcd = np.load(filename,encoding='latin1',allow_pickle=True)
# zed2pcd = np.loadtxt(filename,encoding="utf-8")
zed2pcd = open3d.io.read_point_cloud(filename)

# pcd = convert_zed2pcd_to_ply(zed2pcd)
open3d.io.write_point_cloud('pcd_original.ply', zed2pcd)