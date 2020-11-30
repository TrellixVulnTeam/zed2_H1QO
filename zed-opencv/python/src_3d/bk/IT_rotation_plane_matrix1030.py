import open3d,math as mt,numpy as np

p = 'C:/00_work/05_src/data/frm_t/20201015155835'
f = f"{p}/pcd_extracted.ply"
pcd = open3d.io.read_point_cloud(f)
points = np.copy(np.array(pcd.points))
colors = np.copy(np.array(pcd.colors))
def test_matrix(rotation_m,i):
    pointsn=np.matmul(points,rotation_m)
    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(pointsn)
    pcd2.colors = open3d.utility.Vector3dVector(colors)
    open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_rotation_m%03d.ply'%(i), pcd2)

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
# [0,0,1][[-2.8966267e-03  3.1407675e-04  1.0005289e+00]]
rotation_m=[[-0.05013368 ,-0.073052 ,  -0.04055914],
 [-0.2624987,  -0.12672341 ,-0.09770957],
 [-0.06854045, -0.09249231, -0.04825318]]
test_matrix(rotation_m,3)


# [1,0,0][[ 0.78576213 -0.07347012 -0.03006236]]
rotation_m= [[-0.02223897 , 0.00164234 ,-0.01287944],
 [ 0.25433427,  0.01291677 , 0.7181615 ],
 [ 0.4049464 , -0.00235477, -0.03856532]]

test_matrix(rotation_m,4)

# [0,1,0]
rotation_m=[[-1.7241530e-04 ,-5.0471473e-02 , 1.1375386e-01],
 [ 2.0281222e-01 , 1.4182341e-01 , 2.0452596e-01],
 [ 3.7447242e-03, -9.1163673e-02, -2.9816501e-02]]
test_matrix(rotation_m,5)