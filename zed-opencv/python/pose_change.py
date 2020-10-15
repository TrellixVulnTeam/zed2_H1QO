import numpy as np
import open3d
from numba import njit, prange

def cam2pix(cam_pts, intr):
  """Convert camera coordinates to pixel coordinates.
  """
  intr = intr.astype(np.float32)
  fx, fy = intr[0, 0], intr[1, 1]
  cx, cy = intr[0, 2], intr[1, 2]
  pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
  for i in prange(cam_pts.shape[0]):
    pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
    pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
  return pix
def pcwrite(filename, xyzrgb):
  """Save a point cloud to a polygon .ply file.
  """
  xyz = xyzrgb[:, :3]
  rgb = xyzrgb[:, 3:].astype(np.uint8)

  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(xyz.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(xyz.shape[0]):
    ply_file.write("%f %f %f %d %d %d\n"%(
      xyz[i, 0], xyz[i, 1], xyz[i, 2],
      rgb[i, 0], rgb[i, 1], rgb[i, 2],
    ))

def rigid_transform(xyz, transform):
  """Applies a rigid transform to an (N, 3) pointcloud.
  """
  xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
  xyz_t_h = np.dot(transform, xyz_h.T).T
  return xyz_t_h[:, :3]

ply=open3d.io.read_point_cloud(filename='reconstruction-000000.pcd-ZED_21888201.ply')
transform=np.loadtxt('data_1013_1/reconstruction-000000.pose-ZED_21888201.txt')
pyln=ply.transform(transform)

open3d.io.write_point_cloud('reconstruction-000000.pcd-ZED_21888201_tra.ply', pyln)
# points = np.asarray(ply.points)
# old_color=np.asarray(ply.colors)
# transform=np.loadtxt('data_1013_1/reconstruction-000000.pose-ZED_21888201.txt')
# cam_intr=np.loadtxt('data_1013_1/ZED_21888201-camera-intrinsics.txt')
# cam_pts=rigid_transform(points,np.linalg.inv(transform))
# pix_z = cam_pts[:, 2]
# pix = cam2pix(cam_pts, cam_intr)
# pix_x, pix_y = pix[:, 0], pix[:, 1]
#
# new_color = old_color[pix_y,pix_x]
# pcwrite("pc.ply", points_n)
k=0

