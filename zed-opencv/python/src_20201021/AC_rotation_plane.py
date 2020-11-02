import open3d,math as mt,numpy as np
def vector_compute(v0, v1, axis=0):
    return np.cross(v0, v1, axis=axis)

def vector_normalize(data):
    data = np.array(data, dtype=np.float64, copy=True)
    return mt.sqrt(np.dot(data, data))
def ang_two_vectors(v0, v1, directed=True, axis=0):
    v0 = np.array(v0, dtype=np.float64, copy=False)
    v1 = np.array(v1, dtype=np.float64, copy=False)
    dot = np.sum(v0 * v1, axis=axis)
    dot /= vector_normalize(v0) * vector_normalize(v1)
    return np.arccos(dot if directed else np.fabs(dot))

def unit_ch_vector(data, axis=None, out=None):
    data = np.array(data, dtype=np.float64, copy=True)
    data /= mt.sqrt(np.dot(data, data))
    return data
#https://ja.wikipedia.org/wiki/%E5%9B%9E%E8%BB%A2%E8%A1%8C%E5%88%97
def get_rotation_matrix(ang, direct, point=None):
    sina = mt.sin(ang)
    cosa = mt.cos(ang)
    direct = unit_ch_vector(direct[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direct, direct) * (1.0 - cosa)
    direct *= sina
    R += np.array([[ 0.0,         -direct[2],  direct[1]],
                      [ direct[2], 0.0,          -direct[0]],
                      [-direct[1], direct[0],  0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

vec20201015155844_z=[-0.0889259,-2.69011117,-1]
vec20201015155844_y=[-0.0336802,-1,-0.3684832]
vec20201015155844_x=[-1,-18.56481766,-6.71416052]

vec20201015155835_z=[-0.06199354,4.22205205,-1]
vec20201015155835_y=[0.03323493,-1,0.00610331]
vec20201015155835_x=[-1,19.77586931,-0.05332476]
def rotaton_ply(p,v0,v1):
    rotation_m = get_rotation_matrix(ang_two_vectors(v0, v1), vector_compute(v0, v1))
    f = f"{p}/pcd_extracted.ply"
    pcd = open3d.io.read_point_cloud(f)
    print(rotation_m)
    pcd_r=pcd.transform(rotation_m)
    open3d.io.write_point_cloud(f'{p}/pcd_extract_plane_rotation.ply', pcd_r)
v_dst=[0,1,0]
# v_dst=vec20201015155835_y
rotaton_ply('C:/00_work/05_src/data/frm_t/20201015155844',vec20201015155844_z,v_dst)
rotaton_ply('C:/00_work/05_src/data/frm_t/20201015155835',vec20201015155835_y,v_dst)

