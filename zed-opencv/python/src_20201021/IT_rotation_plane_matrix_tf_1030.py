import open3d,math as mt,numpy as np

vec20201015155844_z=[-0.0889259,-2.69011117,-1]
vec20201015155844_y=[-0.0336802,-1,-0.3684832]
vec20201015155844_x=[-1,-18.56481766,-6.71416052]

vec20201015155835_z=[-0.06199354,4.22205205,-1]
vec20201015155835_y=[0.03323493,-1,0.00610331]
vec20201015155835_x=[-1,19.77586931,-0.05332476]

def get_reg_linear(X,Y,Z):
    XY=np.vstack([X, Y]).T
    XY=np.hstack((np.ones((XY.shape[0],1)).astype(XY.dtype), XY))
    bhat = np.linalg.inv(XY.T @ XY) @ XY.T@Z
    # print(bhat)
    # Zhat = (XY @ bhat).reshape(-1,1)


    # open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_z.ply', pcd2)
    return bhat

def rotaton_ply(p,rotation_m):
    f = f"{p}/pcd_extracted.ply"
    pcd = open3d.io.read_point_cloud(f)
    points = np.array(pcd.points)
    colors = np.array(pcd.colors)
    rotation_mt=np.array(rotation_m)[:3,:3]
    pointsn=np.matmul(points,rotation_mt)
    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(pointsn)
    pcd2.colors = open3d.utility.Vector3dVector(colors)
    open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_rotation3.ply', pcd2)
def rotaton_ply2(p,rotation_m):
    f = f"{p}/pcd_extracted.ply"
    pcd = open3d.io.read_point_cloud(f)
    pcd_r=pcd.transform(rotation_m)

    open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_rotation2.ply', pcd_r)
v_dst=[0,1,0]
rotation_m=[
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1]
        ]
rotation_m=[[0.98480366,0.03097015,-0.1708877,0.],
[-0.03097015,-0.93688282,-0.34826918,0.],
[-0.1708877,0.34826918,-0.92168648,0.],
[0.,0.,0.,1.]]
p = 'C:/00_work/05_src/data/frm_t/20201015155835'
f = f"{p}/pcd_extracted.ply"
pcd = open3d.io.read_point_cloud(f)
points = np.copy(np.array(pcd.points))
colors = np.copy(np.array(pcd.colors))
def test_matrix(rotation_m):
    pointsn=points@rotation_m
    pcd2 = open3d.geometry.PointCloud()
    pcd2.points = open3d.utility.Vector3dVector(pointsn)
    pcd2.colors = open3d.utility.Vector3dVector(colors)
    open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_rotation_m.ply', pcd2)

rotation_m=[[ 0.09173882,0.12425363,-0.82724282],
     [ 0.33314376,0.1891932,0.24174654],
     [ 0.65176417,0.74904716,-0.85179793]
     ]
# test_matrix(rotation_m)
v_dst=np.array([0,1,0])
loss=1
for i in range(1000000):
    rotation_m =np.random.uniform(low=-1.0, high=1.0, size=(3,3))
    points = np.copy(np.array(pcd.points))
    pointsn=points@rotation_m
    X,Y,Z=pointsn[:3]
    v_src=get_reg_linear(X, Y, Z)
    losst=np.sum((v_src-v_dst)**2)
    if i%1000==0:
        print(i,losst)
    if losst<loss:
        loss=losst
        print(loss,rotation_m)
