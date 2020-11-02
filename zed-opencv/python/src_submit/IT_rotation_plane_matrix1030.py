import open3d,math as mt,numpy as np

p = 'C:/00_work/05_src/data/frm_t/20201015155835'
f = f"{p}/pcd_extracted.ply"
pcd = open3d.io.read_point_cloud(f)
points = np.copy(np.array(pcd.points))
colors = np.copy(np.array(pcd.colors))
def test_matrix(rotation_m,i):
    pointsn=points@rotation_m
    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(pointsn)
    pcd2.colors = open3d.utility.Vector3dVector(colors)
    open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_rotation_m%3d.ply'%(i), pcd2)

rotation_m= [
    [1,0.1,0.1],
    [0,1,0],
    [0,0.1,1]]
test_matrix(rotation_m,1)
rotation_m= [
    [1,0.1,0.1],
    [0,1,0.2],
    [0,0.2,1]]

test_matrix(rotation_m,2)
rotation_m= [
    [1,0.1,0.1],
    [0,1,-0.2],
    [0,0.2,1]]
test_matrix(rotation_m,3)
rotation_m= [
    [1,0.1,0.1],
    [0,1,0],
    [0,0.2,1]]
test_matrix(rotation_m,4)