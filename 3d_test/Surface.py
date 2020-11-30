import numpy as np
import open3d as o3d
import pyvista as pv
#https://stackoverflow.com/questions/54898657/i-want-to-generate-a-mesh-from-a-point-cloud-in-python
fn='reconstruction-000001.cloud-ZED_21888201.ply'
# points is a 3D numpy array (n_points, 3) coordinates of a sphere
pcd = o3d.io.read_point_cloud(fn)
points=np.array(pcd.points)
cloud = pv.PolyData(points)
cloud.plot()

volume = cloud.delaunay_3d(alpha=2.)
shell = volume.extract_geometry()
shell.plot()