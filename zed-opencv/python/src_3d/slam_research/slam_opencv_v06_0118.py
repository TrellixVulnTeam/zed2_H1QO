'''
https://github.com/tmako123/nextage_jupyter/blob/master/ICP_3d_bunny.ipynb
1.add icp adjustment
2.ソースの整理
https://github.com/uoip/g2opy
budlel adjustment
'''


# cams_para=[cam_zed2_22378008_left_hd1080,cam_zed2_22115402_left_hd1080]
# for i,para in enumerate(cams_para):
#     ply = image2PointCloud(imgdic.rgb1, imgdic.dpt1, para)
#     fn_ply = f'{bp}cam1/pcd_{i}.ply'
#     print(fn_ply)
#     open3d.io.write_point_cloud(fn_ply, ply)


# cams=['cam0','cam1','cam2']
# for cam in cams:
#     fn_pcd=f'{bp}{cam}/pcd.npy'
#     pcd= np.load(fn_pcd)
#     ply=convert_zed2pcd_to_ply(pcd)
#     fn_ply=f'{bp}{cam}/pcd.ply'
#     print(fn_ply)
#     open3d.io.write_point_cloud(fn_ply,ply)



# ply2=open3d.io.read_point_cloud(f'{bp}cam1/pcd.ply')
# points=np.array(ply2.points)/1000
# colors=np.array(ply2.colors)
# ply3=make_pcd(points,colors)
# open3d.io.write_point_cloud( f'{bp}cam1/pcd_org.ply',ply3)


'''
 sudo apt-get install python-pip
sudo pip install scikit-image
 sudo apt-get install python3-pandas python3-skimage python3-matplotlib python3-numpy
 cam0 : 22378008
   "StereoLabs_ZED2_22378008_LEFT_HD1080": {
      "fx": 1055.26,
      "fy": 1054.92,
      "cx": 962.91,
      "cy": 567.182,
      "k1": -0.044316,
      "k2": 0.013421,
      "k3": -0.00587738,
      "p1": -5.95855e-05,
      "p2": -0.000620575
   },
   "StereoLabs_ZED2_22378008_RIGHT_HD1080": {
      "fx": 1058.41,
      "fy": 1057.92,
      "cx": 975.7,
      "cy": 564.898,
      "k1": -0.0449042,
      "k2": 0.0132566,
      "k3": -0.0055767,
      "p1": 0.000154823,
      "p2": -0.000235482
   },

cam1 : 21888201
   "StereoLabs_ZED2_21888201_LEFT_HD1080": {
      "fx": 1058.51,
      "fy": 1057.56,
      "cx": 972.89,
      "cy": 570.658,
      "k1": -0.0420924,
      "k2": 0.0125523,
      "k3": -0.00594646,
      "p1": -0.000225192,
      "p2": 0.000656782
   },
   "StereoLabs_ZED2_21888201_RIGHT_HD1080": {
      "fx": 1060.16,
      "fy": 1059.39,
      "cx": 963.09,
      "cy": 573.715,
      "k1": -0.0427442,
      "k2": 0.0123999,
      "k3": -0.00566246,
      "p1": -0.000234326,
      "p2": -0.000346455
   },

cam2 : 22115402
   "StereoLabs_ZED2_22115402_LEFT_HD1080": {
      "fx": 1058.82,
      "fy": 1058.16,
      "cx": 909.73,
      "cy": 560.099,
      "k1": -0.0342197,
      "k2": 0.00251121,
      "k3": -0.00254083,
      "p1": -0.000790116,
      "p2": -0.000526767
   },
   "StereoLabs_ZED2_22115402_RIGHT_HD1080": {
      "fx": 1057.29,
      "fy": 1056.79,
      "cx": 899.54,
      "cy": 564.198,
      "k1": -0.0374644,
      "k2": 0.00519864,
      "k3": -0.0033058,
      "p1": -4.99033e-05,
      "p2": 0.000179592
   },
'''

import open3d
import cv2,itertools,struct
import numpy as np
import math
from easydict import EasyDict
import copy as cp

# import pcl
#python -m pip install numpy-quaternion
#pip install pyquaternion
#conda install -c conda-forge quaternion
# camera intrinsics

'''
https://pystyle.info/opencv-feature-matching/
https://www.366service.com/jp/qa/ed8c0298cc30a02ee80c3d9ecef63a69
https://programtalk.com/python-examples/cv2.solvePnP/
'''


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
# ヤコビアン計算
def calcJacob(obj, obs, deltaPose, JtJ, JtE):
    if obj.shape[1] != obs.shape[1]:
        print("ERROR data size is not the same!")
    # 座標変換
    # obj_h = makeHomogeneous(obj)
    # obs_h = deltaPose.dot(obj_h)
    # est = delHomogeneous(obs_h)
    est = obj

    for (p, q) in zip(est.transpose(), obs.transpose()):
        X = p[0]
        Y = p[1]
        Z = p[2]
        ex = q[0] - p[0]
        ey = q[1] - p[1]
        ez = q[2] - p[2]

        JtJ[0, 0] += 1.0
        JtJ[0, 4] += Z
        JtJ[0, 5] += -Y
        JtJ[1, 1] += 1.0
        JtJ[1, 3] += -Z
        JtJ[1, 5] += X
        JtJ[2, 2] += 1.0
        JtJ[2, 3] += Y
        JtJ[2, 4] += -Z
        JtJ[3, 3] += Y * Y + Z * Z
        JtJ[3, 4] += -X * Y
        JtJ[3, 5] += -X * Z
        JtJ[4, 4] += X * X + Z * Z
        JtJ[4, 5] += -Y * Z
        JtJ[5, 5] += X * X + Y * Y

        JtE[0, 0] += ex
        JtE[1, 0] += ey
        JtE[2, 0] += ez
        JtE[3, 0] += -Z * ey + Y * ez
        JtE[4, 0] += Z * ex - X * ez
        JtE[5, 0] += -Y * ex + X * ey

    # fill
    JtJ[1, 0] = JtJ[0, 1]
    JtJ[2, 0] = JtJ[0, 2]
    JtJ[2, 1] = JtJ[1, 2]
    JtJ[3, 0] = JtJ[0, 3]
    JtJ[3, 1] = JtJ[1, 3]
    JtJ[3, 2] = JtJ[2, 3]
    JtJ[4, 0] = JtJ[0, 4]
    JtJ[4, 1] = JtJ[1, 4]
    JtJ[4, 2] = JtJ[2, 4]
    JtJ[4, 3] = JtJ[3, 4]
    JtJ[5, 0] = JtJ[0, 5]
    JtJ[5, 1] = JtJ[1, 5]
    JtJ[5, 2] = JtJ[2, 5]
    JtJ[5, 3] = JtJ[3, 5]
    JtJ[5, 4] = JtJ[4, 5]
#objと最近傍のobs点を算出する
def sampling3d(point3d, samples):
    indexes = np.random.randint(0, point3d.shape[0], samples)
    return point3d[indexes]


#参照資料

def get_good_match_knn(bf,desc1,desc2):
    # マッチング器を作成する。
    matches = bf.knnMatch(desc1, desc2, k=2)
    # レシオテストを行う。
    good_matches = []
    thresh = 0.99
    for first, second in matches:
        if first.distance < second.distance * thresh:
            good_matches.append(first)
    print("good_matches:",len(good_matches))
    return good_matches
def get_good_match(bf,desc1,desc2):
    matches = bf.match(desc1, desc2)
    mindst=100
    maxdst=0
    for match in matches:
        dst=match.distance
        if dst < mindst:mindst=dst
        if dst>maxdst:maxdst=dst
    # レシオテストを行う。
    good_matches=[]
    for match in matches:
        if match.distance<max(2*mindst,30.0):
            good_matches.append(match)
    print("good_matches:",len(good_matches))
    return matches
def computeKeyPointsAndMaches_akaze(rgb0,rgb1):
    # OBR 特徴量検出器を作成する。
    img1 = cv2.cvtColor(rgb0, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)

    #https://greenhornprofessional.hatenablog.com/entry/2020/04/03/005128
    # AKAZE検出器の生成
    detector = cv2.AKAZE_create()
    # 特徴点を検出する。
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)

    # マッチング器を作成する。
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # good_matches= get_good_match(bf,desc1, desc2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good_matches=get_good_match_knn(bf,desc1,desc2)

    dst = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
    return kp1,kp2,good_matches,dst
def computeKeyPointsAndMaches_orb(rgb0,rgb1):
    # OBR 特徴量検出器を作成する。
    detector = cv2.ORB_create()
    # 特徴点を検出する。
    kp1, desc1 = detector.detectAndCompute(rgb0, None)
    kp2, desc2 = detector.detectAndCompute(rgb1, None)

    # マッチングを行う。
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good_matches=get_good_match_knn(bf,desc1,desc2)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # good_matches= get_good_match(bf,desc1, desc2)
    # マッチング結果を描画する。
    dst = cv2.drawMatches(rgb0, kp1, rgb1, kp2, good_matches, None)
    # cv2.imwrite("data/dst2.jpg",dst)
    return kp1,kp2,good_matches,dst
def computeKeyPointsAndMaches_sift(rgb0,rgb1):
    # OBR 特徴量検出器を作成する。
    # detector = cv2.ORB_create()
    #http://whitewell.sakura.ne.jp/OpenCV/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    #http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    # SIFT検出器を始める
    detector = cv2.xfeatures2d.SIFT_create()

    # 特徴点を検出する。
    kp1, desc1 = detector.detectAndCompute(rgb0, None)
    kp2, desc2 = detector.detectAndCompute(rgb1, None)


    # FLANNのパラメータ
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    bf = cv2.FlannBasedMatcher(index_params, search_params)

    # マッチングを行う。
    good_matches=get_good_match_knn(bf,desc1,desc2)
    # マッチング結果を描画する。
    dst = cv2.drawMatches(imgdic.rgb0, kp1, imgdic.rgb1, kp2, good_matches, None)
    # cv2.imwrite("data/dst2.jpg",dst)
    return kp1,kp2,good_matches,dst
def make_pcd(points, colors):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    return pcd
def image2PointCloud(rgb, depth,cam):
   points=[]
   colors=[]
   row,col =rgb.shape[:2]
   for i in range(row):
       for j in range(col):
           d=depth[i,j]
           if d==0:
               continue
           pd=point2dTo3d([j,i,d],cam)
           # pd=point2dTo3d([i,j,d],cam)

           if np.isnan(np.mean(pd)) or np.isinf(np.mean(pd)) or np.isneginf(np.mean(pd)) :
               continue
           points.append(pd)
           b,g,r=rgb[i,j]/255.0
           colors.append([r,g,b])
   return make_pcd(points, colors)


def merge_points_cloud(clouds):
    points=[]
    colors=[]
    for ply in clouds:
        ps=np.array(ply.points)
        cs=np.array(ply.colors)
        points.extend(ps)
        colors.extend(cs)
    return make_pcd(points,colors)
def point2dTo3d(point,cam):
    z=point[2]/cam.depthScale
    x=(point[0]-cam.cx)*z/cam.fx
    y=(point[1]-cam.cy)*z/cam.fy
    return [x,y,z]


def generateRotationMatrix(r):
    px = r[0]
    py = r[1]
    pz = r[2]

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(px), np.sin(px)],
                   [0, -np.sin(px), np.cos(px)]])
    Ry = np.array([[np.cos(py), 0, -np.sin(py)],
                   [0, 1, 0],
                   [np.sin(py), 0, np.cos(py)]])
    Rz = np.array([[np.cos(pz), np.sin(pz), 0],
                   [-np.sin(pz), np.cos(pz), 0],
                   [0, 0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    return R


def generateTransMatrix(t, r):
    R = generateRotationMatrix(r)
    transMatrix = np.eye(4, 4)
    for y in range(0, 3):
        for x in range(0, 3):
            transMatrix[y, x] = R[y, x]
        transMatrix[y, 3] = t[y]
    return transMatrix
def generateTransMatrixRt(t, R):
    transMatrix = np.eye(4, 4)
    for y in range(0, 3):
        for x in range(0, 3):
            transMatrix[y, x] = R[y, x]
        transMatrix[y, 3] = t[y]
    return transMatrix
def get_good_matches_pts(goodMatches,kp1,kp2,imgdic,cams):
    pts_obj1=[]
    pts_obj2=[]
    pts_img1=[]
    pts_img2=[]
    def make_obj(p,d,cam):
        pt=[p[1],p[0],d]
        pd= point2dTo3d(pt,cam)
        return pd
    for i, goodMatch in enumerate(goodMatches):
        p1=np.array(kp1[goodMatch.queryIdx].pt,dtype=np.int)
        p2=np.array(kp2[goodMatch.trainIdx].pt,dtype=np.int)
        d1=imgdic.dpt0[p1[1],p1[0]]
        d2=imgdic.dpt1[p2[1],p2[0]]
        if d1 == 0 or d2 == 0 :
            continue;
        pts1=make_obj(p1, d1,cams[0])
        pts2=make_obj(p2,d2,cams[1])
        if np.isnan(np.mean(pts1))  or np.isnan(np.mean(pts2)) or np.isinf(np.mean(pts1)) or np.isinf(np.mean(pts2)):
            continue
        pts_obj1.append(pts1)
        pts_obj2.append(pts2)
        pts_img1.append(p1)
        pts_img2.append(p2)

    if len(pts_obj1)==0 or len(pts_img2)==0:
        return -1
    print(f"pts_obj1:{len(pts_obj1)},pts_obj2:{len(pts_obj2)}")
    id=0
    cameraMatrix1=[
        [cams[id].fx,0,cams[id].cx],
        [0,cams[id].fy,cams[id].cy],
        [0,0,1]
        ]
    id=1
    cameraMatrix2=[
        [cams[id].fx,0,cams[id].cx],
        [0,cams[id].fy,cams[id].cy],
        [0,0,1]
        ]
    distCoeffs=[0, 0, 0, 0, 0]
    pts_img1=np.array(pts_img1,dtype=np.float32)
    pts_img2=np.array(pts_img2,dtype=np.float32)
    pts_obj1 = np.array(pts_obj1, dtype=np.float32)
    pts_obj2 = np.array(pts_obj2, dtype=np.float32)
    cameraMatrix1 = np.array(cameraMatrix1, dtype=np.float64)
    cameraMatrix2 = np.array(cameraMatrix2, dtype=np.float64)
    distCoeffs = np.array(distCoeffs, dtype=np.float64)

    _, rVec1, tVec1=cv2.solvePnP(pts_obj1, pts_img1, cameraMatrix1, distCoeffs=distCoeffs)
    _, rVec2, tVec2=cv2.solvePnP(pts_obj2, pts_img2, cameraMatrix2, distCoeffs=distCoeffs)
    R1=cv2.Rodrigues(rVec1)[0]
    R2=cv2.Rodrigues(rVec2)[0]
    R_12 = R2 @ R1.T
    # t_12= R2 @ (-R1.T @ tVec1) + tVec2
    t_12= tVec2 - R_12@tVec1
    # trans12=generateTransMatrixRt(t_12,R_12)
    trans12 = np.eye(4)
    trans12[:3, :3] = R_12
    trans12[:3, 3] = t_12.flatten()
    return trans12,pts_obj1,pts_obj2


class compute_pose_points:
    def __init__(self):
        pass

    def calcPose3D(self,obj, obs, deltaPose):
        # ヤコビ行列生成
        JtJ = np.zeros((6, 6))
        JtE = np.zeros((6, 1))
        calcJacob(obj, obs, deltaPose, JtJ, JtE)

        # solve
        x = np.linalg.solve(JtJ, JtE)
        t = x[0:3]
        omega = x[3:6]

        # 補正量計算
        angle = math.sqrt(omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2])
        dR = np.eye(3, 3)
        if angle < 1.0e-12:
            # print("rot small")
            pass
        else:
            dR[0, 1] = -omega[2]
            dR[0, 2] = omega[1]
            dR[1, 0] = omega[2]
            dR[1, 2] = -omega[0]
            dR[2, 0] = -omega[1]
            dR[2, 1] = omega[0]

        # 行列生成
        transMatrix = np.eye(4, 4)
        for y in range(0, 3):
            for x in range(0, 3):
                transMatrix[y, x] = dR[y, x]
            transMatrix[y, 3] = t[y]
        return transMatrix
    def findCrsp(self,obj, obs):
        query = np.empty((3, 0))
        target = np.empty((3, 0))
        for i in range(obj.shape[1]):
            dist_ = np.linalg.norm(obs.T - obj.T[i], axis=1)
            minId = np.argmin(dist_)
            query = np.concatenate((query, obj.T[i].reshape(3, 1)), axis=1)
            target = np.concatenate((target, obs.T[minId].reshape(3, 1)), axis=1)
        return query, target
    # 同次座標系に関する関数
    def makeHomogeneous(self,vector):
        rows, cols = vector.shape[:2]
        ones = [np.ones(cols)]
        return np.r_[vector, ones]

    def delHomogeneous(self,vector):
        rows, cols = vector.shape[:2]
        val = vector[:rows - 1]
        dim = vector[rows - 1:]
        return val / dim

    def warp3d(self,point3d, M):
        h_point3d = self.makeHomogeneous(point3d)
        h_w_point3d = M.dot(h_point3d)
        return self.delHomogeneous(h_w_point3d)
    def keypoint_tm(self,points1s,points2s):
        points1s=np.array(points1s)
        points2s=np.array(points2s)

        estPose = np.eye(4, 4)
        obj=points1s.T
        obs=points2s.T
        var_min=0.0004
        for i in range(1000):
            estPoseOld=cp.deepcopy(estPose)
            w_obj  = self.warp3d(obj, estPose)
            query, target = self.findCrsp(w_obj , obs)
            deltaPose = self.calcPose3D(query, target, estPose)
            estPose = deltaPose.dot(estPose)
            print("estPose var:",np.var(estPose-estPoseOld))
            if var_min >np.var(estPose-estPoseOld):
                break
            # print("estPose:",estPose)
        return estPose
def move_2_center(cloud):
        colors=np.array(cloud.colors)
        points=np.array(cloud.points)
        points_m = points - np.mean(points ,axis=0)
        cloud_t = make_pcd(points_m, colors)
        return cloud_t


def main_func(keymatchfun,ext,cams,imgdic):
    pc=compute_pose_points()
    kp1,kp2,goodMatches,img_dst=keymatchfun(imgdic.rgb0,imgdic.rgb1)
    pose_mat,pts_obj1,pts_obj2=get_good_matches_pts(goodMatches,kp1,kp2,imgdic,cams)
    estPose=pc.keypoint_tm(pts_obj1,pts_obj2)

    cloud0=open3d.io.read_point_cloud(f'{bpd}cam0/pcd_mask.ply')
    cloud1=open3d.io.read_point_cloud(f'{bpd}cam1/pcd_mask.ply')
    cloud0_t=cp.deepcopy(cloud0)
    cloud0_gm=cloud0_t.transform(pose_mat)

    cv2.imwrite(f"{bp}{ext}_img_dst.jpg",img_dst)

    cloud0_t=cp.deepcopy(cloud0)
    cloud0t2=cloud0_t.transform(estPose)
    open3d.io.write_point_cloud(f'{bpd}{ext}_pcd_trans_0_solvePnP.ply', cloud0_gm)
    open3d.io.write_point_cloud(f'{bpd}{ext}_pcd_trans_0_calcPose3D.ply', cloud0t2)

    pcd=merge_points_cloud([cloud0_gm,cloud1])
    open3d.io.write_point_cloud(f'{bpd}{ext}_pcd_merge_solvePnP.ply', pcd)


    pcd=merge_points_cloud([cloud0t2,cloud1])
    open3d.io.write_point_cloud(f'{bpd}{ext}_pcd_merge_calcPose3D.ply', pcd)

keyfuns=[
    [computeKeyPointsAndMaches_akaze,"akaze"],
    # [computeKeyPointsAndMaches_orb,"orb"],
    [computeKeyPointsAndMaches_sift,"sift"],
         ]

cam_zed2_22378008_left_hd1080 = EasyDict({})
cam_zed2_22378008_left_hd1080.fx=1055.26
cam_zed2_22378008_left_hd1080.fy=1054.92
cam_zed2_22378008_left_hd1080.cx=962.91
cam_zed2_22378008_left_hd1080.cy=567.182
cam_zed2_22378008_left_hd1080.depthScale=1000.0

'''

      "fx": 1058.51,
      "fy": 1057.56,
      "cx": 972.89,
      "cy": 570.658,
'''
cam_zed2_21888201_left_hd1080 = EasyDict({})
cam_zed2_21888201_left_hd1080.fx=1058.51
cam_zed2_21888201_left_hd1080.fy=1057.56
cam_zed2_21888201_left_hd1080.cx=972.89
cam_zed2_21888201_left_hd1080.cy=570.658
cam_zed2_21888201_left_hd1080.depthScale=1000.0

'''

   "StereoLabs_ZED2_22115402_LEFT_HD1080": {
      "fx": 1058.82,
      "fy": 1058.16,
      "cx": 909.73,
      "cy": 560.099,
'''
cam_zed2_22115402_left_hd1080 = EasyDict({})
cam_zed2_22115402_left_hd1080.fx=1058.82
cam_zed2_22115402_left_hd1080.fy=1058.16
cam_zed2_22115402_left_hd1080.cx=909.73
cam_zed2_22115402_left_hd1080.cy=560.099
cam_zed2_22115402_left_hd1080.depthScale=1000.0

bp='data/d5_c/'
bpd = 'pointcloud/d5_c/'

imgdic = EasyDict({})
imgdic.rgb0=cv2.imread(f"{bp}/cam0/image.png")
imgdic.rgb1=cv2.imread(f"{bp}/cam1/image.png")
imgdic.rgb2=cv2.imread(f"{bp}/cam2/image.png")
imgdic.dpt0=np.load(f"{bp}/cam0/depth.npy")
imgdic.dpt1=np.load(f"{bp}/cam1/depth.npy")
imgdic.dpt2=np.load(f"{bp}/cam2/depth.npy")
'''
cam_zed2_21888201_left_hd1080:cam0
cam_zed2_22378008_left_hd1080:cam1
cam_zed2_22115402_left_hd1080:cam2
'''
cams_para=[cam_zed2_21888201_left_hd1080,cam_zed2_22378008_left_hd1080,cam_zed2_22115402_left_hd1080]
for matchfun, ext in keyfuns:
    main_func(matchfun, ext,cams_para,imgdic)
