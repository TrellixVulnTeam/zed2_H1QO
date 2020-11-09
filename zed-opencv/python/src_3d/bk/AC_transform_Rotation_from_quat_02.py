import os, json, quaternion, open3d
import numpy as np,math as mt
from easydict import EasyDict
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
from scipy.spatial.transform import Rotation as R

def lms_regression_for_3d_pcd(pcd, out_idx=2):
  axis = [0, 1, 2]
  assert out_idx in axis
  points = np.array(pcd.points)
  colors = np.array(pcd.colors)
  # outについて線形回帰実施(入力はin0とin1)。
  axis.remove(out_idx)
  in_idx0, in_idx1 = axis[0], axis[1]
  # データ準備
  out, in0, in1 = points[:, out_idx], points[:, in_idx0], points[:, in_idx1]
  # http://reliawiki.org/index.php/Multiple_Linear_Regression_Analysis
  # "Estimating Regression Models Using Least Squares"
  # （最小二乗法で一気にパラメータを計算する手法。）
  # http://www.banbeichadexiaojiubei.com/index.php/2019/12/24/3d%E5%B9%B3%E9%9D%A2%E6%8B%9F%E5%90%88%E7%AE%97%E6%B3%95%E6%8B%9F%E5%90%88lidar%E7%82%B9%E4%BA%91/
  in0in1 = np.vstack([in0, in1]).T  # vstack + transposeにするのは回帰座標の調整上必要
  in0in1 = np.hstack([np.ones((in0in1.shape[0], 1)).astype(in0in1.dtype), in0in1]) # [1, β0, β1]
  bhat = np.linalg.inv(in0in1.T @ in0in1) @ in0in1.T @ out
  ohat = (in0in1 @ bhat).reshape(-1, 1)
  # out_idx毎に出力順序が異なるので、それに対応させ、points2を作成
  res = {in_idx0:in0.reshape(-1, 1), in_idx1:in1.reshape(-1, 1), out_idx:ohat}
  points2 = np.hstack([res[i] for i in [0,1,2]])
  # out_idxの値毎に色を変え, colors2を作成
  color = np.array([255 if i == out_idx else 0 for i in range(3)]).astype(colors.dtype)
  colors2 = np.zeros(colors.shape).astype(colors.dtype)
  colors2[:] = color
  plane_parameter = {'const':bhat[0], in_idx0:bhat[1], in_idx1:bhat[2], out_idx:1.}
  pcd2 = open3d.geometry.PointCloud()
  pcd2.points = open3d.utility.Vector3dVector(points2)
  pcd2.colors = open3d.utility.Vector3dVector(colors2)
  return pcd2, plane_parameter

def get_plane_parameters(p):
  f = f"{p}/pcd_extracted.ply"
  pcd = open3d.io.read_point_cloud(f)
  pcds, plane_parameters = {}, {}
  for i, ax in enumerate(['x', 'y', 'z']):
    pcds[ax], plane_parameters[ax] = lms_regression_for_3d_pcd(pcd, out_idx=i)
    open3d.io.write_point_cloud(
      f"{f.replace('.ply','')}_plane_{ax}.ply", pcds[ax])
    json.dump(plane_parameters[ax],
      open(f"{f.replace('.ply','')}_plane_{ax}.json", 'w'), indent=2)
  return pcds, plane_parameters

def mul_quaternion(q0, q1):
  right = np.array([q1.w, q1.x, q1.y, q1.z]).reshape(-1,1)
  left = np.array([
    [q0.w, -q0.x, -q0.y, -q0.z],
    [q0.x,  q0.w, -q0.z,  q0.y],
    [q0.y,  q0.z,  q0.w, -q0.x],
    [q0.z, -q0.y,  q0.x,  q0.w]])
  return np.quaternion(*(left @ right).reshape(-1))

def get_rotation_quaternion(alpha, axis=np.array([1., 1., 1.]), is_degree=True):
  # https://showa-yojyo.github.io/notebook/python-quaternion.html#id13
  # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
  # https://en.wikipedia.org/wiki/File:Euler_AxisAngle.png
  assert isinstance(alpha, float)
  assert isinstance(axis, np.ndarray)
  assert axis.shape == (3,)
  # alpha should be radian
  alpha_half = np.deg2rad(alpha / 2) if is_degree else alpha / 2
  cosa = np.cos(alpha_half)
  sina = np.sin(alpha_half)
  norm = np.linalg.norm(axis)
  v = sina * (axis / norm)
  return np.quaternion(cosa, *v)

def get_quaternion_from_vector(vec):
  return np.quaternion(0., *vec)

def get_normalized_vector(json_path=None, array=None):
  if not json_path is None:
    source = json.load(open(json_path))
    assert len([k for k in source.keys() if k in ['0','1','2']]) == 3
    vector = np.array([source[a] for a in ['0','1','2']])
    vector /= np.linalg.norm(vector)
    return vector
  elif not array is None:
    assert isinstance(array, np.ndarray)
    assert array.shape == (3,)
    vector = array / np.linalg.norm(array)
    return vector
  else:
    print('please input json_path or numpy array')
    return None

def get_rot_quaternion_from_vectors(source_vector, target_vector):
  """
    source_vectorをtarget_vectorに回転する
    quaternionの取得

    回転元ベクトルと回転先ベクトルの
    「平均ベクトル」を中心に180°回転させて
    回転のquaternionを取得する。
  """
  assert isinstance(source_vector, np.ndarray)
  assert isinstance(target_vector, np.ndarray)
  src_vec = get_normalized_vector(array=source_vector)
  tgt_vec = get_normalized_vector(array=target_vector)
  mean_vec= get_normalized_vector(array=(tgt_vec + src_vec)/2)
  q = get_rotation_quaternion(180., mean_vec)
  return q


def verify_apply_rotation(p0):
  p = f'{p0}/pcd_extracted_plane_x.json'
  src_vec = get_normalized_vector(json_path=p)
  tgt_vec = np.array([0., 1., 0.])
  q = get_rot_quaternion_from_vectors(src_vec, tgt_vec)
  p_q = np.quaternion(*([0.] + list(src_vec)))
  # if q is correctly difined,
  # src_vec will be tgt_vec after applying rotation by q.
  q_tgt = q * get_quaternion_from_vector(src_vec) * q.conj()
  # ASSERTION:
  #   q_tgt is quaternion, and tgt.vec is its vector.
  #   if q is collectly defined,
  #     q_tgt.vec - tgt_vec will be near zero.
  assert abs(q_tgt.vec - tgt_vec).max() < 1e-15
def make_pcd(points, colors):
  pcd = open3d.geometry.PointCloud()
  pcd.points = open3d.utility.Vector3dVector(points)
  pcd.colors = open3d.utility.Vector3dVector(colors)
  return pcd


def apply_rotation_quaternion(q, positions):
  assert isinstance(q, quaternion.quaternion)
  assert isinstance(positions, np.ndarray)
  """
  https://kamino.hatenablog.com/entry/rotation_expressions#sec3_2
  クォータニオンを使って点 p=(x,y,z) を回転させるときは、
  点pを p_q = 0 + xi + yj + zk と読み替え、以下のように計算する。
    p_q_rot = q * p_q * q.conj()

  * positionsはplyから読み込んだpcd.points。
  * forループだと効率が悪いので、行列で一気に計算するのもあり。
    mul_quaternion()がクォータニオンの乗算の原理。
    right変数の列数を増やせば1回の行列積で演算可能。
  """
  p_q_lst = []
  for p in list(positions):
    p_q = np.quaternion(0., *p)
    p_q_lst.append(p_q)
  p_q_m = np.array(p_q_lst)
  p_q_rot = q * p_q_m * q.conj()
  points_n = []
  for p in list(p_q_rot):
    points_n.append([p.x, p.y, p.z])
  return np.array(points_n)
def main_apply_rotation_quaternion():
    # root_dir = 'C:/Users/003420/Desktop/Works/NICT/predevelopment/Zed2'
    base = 'C:/00_work/05_src/data/frm_t'
    # os.chdir(root_dir)
    src_vec_lst = EasyDict({})
    dst_vec_lst = EasyDict({})
    src_vec_lst['20201015155835'] = [0.03323493, -1, 0.00610331]
    src_vec_lst['20201015155844'] = [-0.0336802, -1, -0.3684832]
    # dst_vec_lst['20201015155835']=[0.03323493,-1,0.00610331]
    # dst_vec_lst['20201015155844']=[-0.0336802,-1,-0.3684832]
    for p in ["20201015155835", "20201015155844"]:
        p0 = f"{base}/{p}"
        # _ = get_plane_parameters(p0)

        src_vec = np.array(src_vec_lst[p])
        dst_vec = np.array([0, 1, 0])
        q = get_rot_quaternion_from_vectors(src_vec, dst_vec)
        f = f"{p0}/pcd_extracted.ply"
        pcd = open3d.io.read_point_cloud(f)
        points = np.array(pcd.points)
        colors = np.array(pcd.colors)
        points_r = apply_rotation_quaternion(q, points)
        pcd_n = make_pcd(points_r, colors)
        filename = f"{p0}/pcd_extracted_quaternion.ply"
        open3d.io.write_point_cloud(filename, pcd_n)

def unit_ch_vector(data):
    data = np.array(data, dtype=np.float64, copy=True)
    data /= mt.sqrt(np.dot(data, data))
    return data
def rotate_ply(pcd,rotate_m):
    points=np.array(pcd.points)@rotate_m
    colors=np.array(pcd.colors)
    return make_pcd(points,colors)
def main_r_from_quat_001():
    src_vec0 = np.array([-2.987055406 , 5.17373172864616 ,   2.00])
    src_vec1 = np.array([2.541653005 ,- 4.402272140611028 ,   1.50])
    tgt_vec0 =  np.array([0., 1.0, 0.])
    tgt_vec1 =  np.array([0., 1.0, 0.])
    def q_2_m(src_vec, tgt_vec):
        q = get_rot_quaternion_from_vectors(src_vec, tgt_vec)
        r = R.from_quat([q.w,q.x,q.y,q.z])
        m=r.as_matrix()
        return m
    R0=q_2_m(src_vec0, tgt_vec0)
    R1=q_2_m(src_vec1, tgt_vec1)

    ply0 = 'C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/pcd.ply'
    ply1 = 'C:/00_work/05_src/data/fromWATA/cam1/AB_ON/20201030155144/pcd.ply'
    pcd0 = open3d.io.read_point_cloud(ply0)
    pcd1 = open3d.io.read_point_cloud(ply1)
    pcd0t=rotate_ply(pcd0,R0)
    pcd1t=rotate_ply(pcd1,R1)
    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_001_cam0_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd0t)
    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_001_cam1_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd1t)
    # def r_2_t(r):
    #     t0=np.hstack((r,np.expand_dims([0,0,0],axis=1)))
    #     t1=np.vstack((t0,np.expand_dims([0,0,0,1],axis=1).T))
    #     return t1
    # t0=r_2_t(R0)
    # t1=r_2_t(R1)
    # pcd0t=pcd0.transform(t0)
    # pcd1t=pcd1.transform(t1)

def main_r_from_quat_002():
    src_vec0 = np.array([-2.987055406 , 5.17373172864616 ,   2.00])
    src_vec1 = np.array([2.541653005 ,- 4.402272140611028 ,   1.50])
    tgt_vec0 =  np.array([0., 5.17373172864616, 0.])
    tgt_vec1 =  np.array([0., 4.402272140611028, 0.])
    # tgt_vec0 =  np.array([0., 1.0, 0.])
    # tgt_vec1 =  np.array([0., 1.0, 0.])
    def q_2_m(src_vec, tgt_vec):
        q = get_rot_quaternion_from_vectors(src_vec, tgt_vec)
        r = R.from_quat([q.w,q.x,q.y,q.z])
        m=r.as_matrix()
        return m
    R0=q_2_m(src_vec0, tgt_vec0)
    R1=q_2_m(src_vec1, tgt_vec1)

    ply0 = 'C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/pcd.ply'
    ply1 = 'C:/00_work/05_src/data/fromWATA/cam1/AB_ON/20201030155144/pcd.ply'
    pcd0 = open3d.io.read_point_cloud(ply0)
    pcd1 = open3d.io.read_point_cloud(ply1)
    pcd0t=rotate_ply(pcd0,R0)
    pcd1t=rotate_ply(pcd1,R1)
    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_002_cam0_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd0t)
    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_002_cam1_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd1t)
    # def r_2_t(r):
    #     t0=np.hstack((r,np.expand_dims([0,0,0],axis=1)))
    #     t1=np.vstack((t0,np.expand_dims([0,0,0,1],axis=1).T))
    #     return t1
    # t0=r_2_t(R0)
    # t1=r_2_t(R1)
    # pcd0t=pcd0.transform(t0)
    # pcd1t=pcd1.transform(t1)

def main_r_from_quat_103():
    src_vec0 = np.array([-2.987055406 , 5.17373172864616 ,   2.00])
    src_vec1 = np.array([2.541653005 ,- 4.402272140611028 ,   1.50])
    def q_2_m(src_vec, tgt_vec):
        q = get_rot_quaternion_from_vectors(src_vec, tgt_vec)
        r = R.from_quat([q.w,q.x,q.y,q.z])
        m=r.as_matrix()
        return m
    R0=q_2_m(src_vec0, src_vec1)
    # R1=q_2_m(src_vec1, tgt_vec1)

    ply0 = 'C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/pcd.ply'
    ply1 = 'C:/00_work/05_src/data/fromWATA/cam1/AB_ON/20201030155144/pcd.ply'
    pcd0 = open3d.io.read_point_cloud(ply0)
    pcd1 = open3d.io.read_point_cloud(ply1)
    pcd0t=rotate_ply(pcd0,R0)
    pcd1t=pcd1#rotate_ply(pcd1,R1)
    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_103_cam0_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd0t)
    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_103_cam1_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd1t)

def main_r_from_quat_002_1():
    src_vec0 = np.array([-2.987055406 , 5.17373172864616 ,   2.00])
    src_vec1 = np.array([2.541653005 ,- 4.402272140611028 ,   1.50])
    tgt_vec0 =  np.array([0., 5.17373172864616, 0.])
    tgt_vec1 =  np.array([0., -4.402272140611028, 0.])
    # tgt_vec0 =  np.array([0., 1.0, 0.])
    # tgt_vec1 =  np.array([0., 1.0, 0.])
    def q_2_m(src_vec, tgt_vec):
        q = get_rot_quaternion_from_vectors(src_vec, tgt_vec)
        r = R.from_quat([q.w,q.x,q.y,q.z])
        m=r.as_matrix()
        return m
    R0=q_2_m(src_vec0, tgt_vec0)
    R1=q_2_m(src_vec1, tgt_vec1)

    ply0 = 'C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/pcd.ply'
    ply1 = 'C:/00_work/05_src/data/fromWATA/cam1/AB_ON/20201030155144/pcd.ply'
    pcd0 = open3d.io.read_point_cloud(ply0)
    pcd1 = open3d.io.read_point_cloud(ply1)
    pcd0t=rotate_ply(pcd0,R0)
    pcd1t=rotate_ply(pcd1,R1)
    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_002_1_cam0_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd0t)
    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_002_1_cam1_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd1t)
    # def r_2_t(r):
    #     t0=np.hstack((r,np.expand_dims([0,0,0],axis=1)))
    #     t1=np.vstack((t0,np.expand_dims([0,0,0,1],axis=1).T))
    #     return t1
    # t0=r_2_t(R0)
    # t1=r_2_t(R1)
    # pcd0t=pcd0.transform(t0)
    # pcd1t=pcd1.transform(t1)
# 伊藤さん案を実施
def main_r_from_quat_003():
    src_vec0 = np.array([2.987055406 , -5.17373172864616 ,   2.00])
    src_vec1 = np.array([-2.541653005 ,4.402272140611028 ,   1.50])

    def q_2_m(src_vec, tgt_vec):
        q = get_rot_quaternion_from_vectors(src_vec, tgt_vec)
        r = R.from_quat([q.w,q.x,q.y,q.z])
        m=r.as_matrix()
        return m
    def vec_step1(src_vec):
        dst_vec=np.array([src_vec[0]*-1,src_vec[1]*-1,0])
        return dst_vec
    def vec_step2(src_vec):
        dst_vec=np.array([0,src_vec[1],0])
        return dst_vec
    R0_step1=q_2_m(src_vec0*-1, vec_step1(src_vec0))
    R0_step2=q_2_m(vec_step1(src_vec0), vec_step2(src_vec0))

    R1_step1=q_2_m(src_vec1*-1, vec_step1(src_vec1))
    R1_step2=q_2_m(vec_step1(src_vec1), vec_step2(src_vec1))

    ply0 = 'C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/pcd.ply'
    ply1 = 'C:/00_work/05_src/data/fromWATA/cam1/AB_ON/20201030155144/pcd.ply'
    pcd0 = open3d.io.read_point_cloud(ply0)
    pcd1 = open3d.io.read_point_cloud(ply1)
    pcd0t_s1=rotate_ply(pcd0,R0_step1)
    pcd0t=rotate_ply(pcd0t_s1,R0_step2)

    pcd1t_s1=rotate_ply(pcd1,R1_step1)
    pcd1t=rotate_ply(pcd1t_s1,R1_step2)
    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_003_cam0_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd0t)
    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_003_cam1_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd1t)

def main_r_from_quat_004():
    src_vec0 = np.array([2.987055406 , -5.17373172864616 ,   -2.00])
    src_vec1 = np.array([-2.541653005 ,4.402272140611028 ,   -1.50])
    tgt_vec0 =  np.array([ 0,5.17373172864616, 0.])
    tgt_vec1 =  np.array([0,-4.402272140611028, 0.])
    def q_2_m(src_vec, tgt_vec):
        q = get_rot_quaternion_from_vectors(src_vec, tgt_vec)
        r = R.from_quat([q.w,q.x,q.y,q.z])
        m=r.as_matrix()
        return m
    R0=q_2_m(src_vec0,tgt_vec0)
    R1=q_2_m(src_vec1,tgt_vec1)

    ply0 = 'C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/pcd.ply'
    ply1 = 'C:/00_work/05_src/data/fromWATA/cam1/AB_ON/20201030155144/pcd.ply'
    pcd0 = open3d.io.read_point_cloud(ply0)
    pcd1 = open3d.io.read_point_cloud(ply1)
    pcd0t=rotate_ply(pcd0,R0)
    pcd1t=rotate_ply(pcd1,R1)


    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_004_cam0_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd0t)
    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_004_cam1_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd1t)

def main_r_from_quat_004_x():
    src_vec0 = np.array([2.987055406 , -5.17373172864616 ,   -2.00])
    src_vec1 = np.array([-2.541653005 ,4.402272140611028 ,   -1.50])
    tgt_vec0 =  np.array([ -2.987055406 ,0, 0.])
    tgt_vec1 =  np.array([2.541653005,0, 0.])
    def q_2_m(src_vec, tgt_vec):
        q = get_rot_quaternion_from_vectors(src_vec, tgt_vec)
        r = R.from_quat([q.w,q.x,q.y,q.z])
        m=r.as_matrix()
        return m
    R0=q_2_m(src_vec0,tgt_vec0)
    R1=q_2_m(src_vec1,tgt_vec1)

    ply0 = 'C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/pcd.ply'
    ply1 = 'C:/00_work/05_src/data/fromWATA/cam1/AB_ON/20201030155144/pcd.ply'
    pcd0 = open3d.io.read_point_cloud(ply0)
    pcd1 = open3d.io.read_point_cloud(ply1)
    pcd0t=rotate_ply(pcd0,R0)
    pcd1t=rotate_ply(pcd1,R1)


    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_004_x_cam0_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd0t)
    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_004_x_cam1_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd1t)

def main_r_from_quat_004_z():
    src_vec0 = np.array([2.987055406 , -5.17373172864616 ,   -2.00])
    src_vec1 = np.array([-2.541653005 ,4.402272140611028 ,   -1.50])
    tgt_vec0 =  np.array([0,0, 2.00])
    tgt_vec1 =  np.array([0,0, 1.50])
    def q_2_m(src_vec, tgt_vec):
        q = get_rot_quaternion_from_vectors(src_vec, tgt_vec)
        r = R.from_quat([q.w,q.x,q.y,q.z])
        m=r.as_matrix()
        return m
    R0=q_2_m(src_vec0,tgt_vec0)
    R1=q_2_m(src_vec1,tgt_vec1)

    ply0 = 'C:/00_work/05_src/data/fromWATA/cam0/AB_ON/20201030154916/pcd.ply'
    ply1 = 'C:/00_work/05_src/data/fromWATA/cam1/AB_ON/20201030155144/pcd.ply'
    pcd0 = open3d.io.read_point_cloud(ply0)
    pcd1 = open3d.io.read_point_cloud(ply1)
    pcd0t=rotate_ply(pcd0,R0)
    pcd1t=rotate_ply(pcd1,R1)


    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_004_z_cam0_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd0t)
    ft_pcd_m='C:/00_work/05_src/data/fromWATA/q_004_z_cam1_pcd_trans.ply'
    open3d.io.write_point_cloud(ft_pcd_m, pcd1t)

if __name__ == '__main__':
    # main_r_from_quat_002()
    # main_r_from_quat_003()
    main_r_from_quat_004()
    main_r_from_quat_004_x()
    main_r_from_quat_004_z()