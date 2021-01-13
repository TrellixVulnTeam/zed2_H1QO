'''
https://github.com/tmako123/nextage_jupyter/blob/master/ICP_3d_bunny.ipynb
1.add icp adjustment
2.ソースの整理
https://github.com/uoip/g2opy
budlel adjustment
'''

import numpy as np
import cv2
import cv2 as cv
# import quaternion
import  pyquaternion as  quaternion
from pyntcloud import PyntCloud
import open3d
import cv2
import numpy as np
import math 
import random
from easydict import EasyDict


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
def calcPose3D(obj, obs, deltaPose):
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
#objと最近傍のobs点を算出する
def sampling3d(point3d, samples):
    indexes = np.random.randint(0, point3d.shape[0], samples)
    return point3d[indexes]

def findCrsp(obj, obs):
    query = np.empty((3,0))
    target = np.empty((3,0))
    for i in range(obj.shape[1]):
        dist_ = np.linalg.norm(obs.T - obj.T[i], axis = 1)
        minId = np.argmin(dist_)
        query = np.concatenate((query, obj.T[i].reshape(3,1)), axis = 1)
        target = np.concatenate((target, obs.T[minId].reshape(3,1)), axis = 1)
    return query, target

# 同次座標系に関する関数
def makeHomogeneous(vector):
    rows, cols = vector.shape[:2]
    ones = [np.ones(cols)]
    return np.r_[vector, ones]


def delHomogeneous(vector):
    rows, cols = vector.shape[:2]
    val = vector[:rows - 1]
    dim = vector[rows - 1:]
    return val / dim
def warp3d(point3d, M):
    h_point3d = makeHomogeneous(point3d)
    h_w_point3d = M.dot(h_point3d)
    return delHomogeneous(h_w_point3d)
#参照資料

def get_good_match_knn(bf,desc1,desc2):
    # マッチング器を作成する。
    matches = bf.knnMatch(desc1, desc2, k=2)
    # レシオテストを行う。
    good_matches = []
    thresh = 0.7
    for first, second in matches:
        if first.distance < second.distance * thresh:
            good_matches.append(first)
    print("good_matches:",len(good_matches))
    return good_matches
def get_good_match(bf,desc1,desc2):
    matches = bf.match(desc1, desc2)
    mindst=10000
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
def computeKeyPointsAndMaches_akaze(imgdic):
    # OBR 特徴量検出器を作成する。
    img1 = cv2.cvtColor(imgdic.rgb1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(imgdic.rgb2, cv2.COLOR_BGR2GRAY)

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
def computeKeyPointsAndMaches_orb(imgdic):
    # OBR 特徴量検出器を作成する。
    detector = cv2.ORB_create()
    # 特徴点を検出する。
    kp1, desc1 = detector.detectAndCompute(imgdic.rgb1, None)
    kp2, desc2 = detector.detectAndCompute(imgdic.rgb2, None)

    # マッチングを行う。
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good_matches=get_good_match_knn(bf,desc1,desc2)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # good_matches= get_good_match(bf,desc1, desc2)
    # マッチング結果を描画する。
    dst = cv2.drawMatches(imgdic.rgb1, kp1, imgdic.rgb2, kp2, good_matches, None)
    # cv2.imwrite("data/dst2.jpg",dst)
    return kp1,kp2,good_matches,dst
def computeKeyPointsAndMaches_sift(imgdic):
    # OBR 特徴量検出器を作成する。
    # detector = cv2.ORB_create()
    #http://whitewell.sakura.ne.jp/OpenCV/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    #http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    # SIFT検出器を始める
    detector = cv2.xfeatures2d.SIFT_create()

    # 特徴点を検出する。
    kp1, desc1 = detector.detectAndCompute(imgdic.rgb1, None)
    kp2, desc2 = detector.detectAndCompute(imgdic.rgb2, None)


    # FLANNのパラメータ
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    bf = cv2.FlannBasedMatcher(index_params, search_params)

    # マッチングを行う。
    good_matches=get_good_match_knn(bf,desc1,desc2)
    # マッチング結果を描画する。
    dst = cv2.drawMatches(imgdic.rgb1, kp1, imgdic.rgb2, kp2, good_matches, None)
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
   r,c =rgb.shape[:2]
   for i in range(r):
       for j in range(c):
           d=depth[i,j]
           if d==0:
               continue
           pd=point2dTo3d([i,j,d],cam)
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
def get_good_matches_pts(goodMatches,kp1,kp2,imgdic,cam):
    pts_obj1=[]
    pts_obj2=[]
    pts_img=[]
    def make_obj(p,d):
        pt=[p[0],p[1],d]
        pd= point2dTo3d(pt,cam)
        return pd
    for i, goodMatch in enumerate(goodMatches):
        p1=np.array(kp1[goodMatch.queryIdx].pt,dtype=np.int)
        p2=np.array(kp2[goodMatch.trainIdx].pt,dtype=np.int)
        d1=imgdic.dpt1[p1[1],p1[0]]
        d2=imgdic.dpt2[p2[1],p2[0]]
        if d1 == 0 or d2 == 0 :
            continue;
        pts_img.append(kp2[goodMatch.trainIdx].pt)

        pts_obj1.append( make_obj(p1,d1))
        pts_obj2.append(make_obj(p2,d2))

    if len(pts_obj1)==0 or len(pts_img)==0:
        return -1
    print(f"pts_obj1:{len(pts_obj1)},{len(pts_obj2)}")
    cameraMatrix=[
        [cam.fx,0,cam.cx],
        [0,cam.fy,cam.cy],
        [0,0,1]
        ]
    distCoeffs=[0, 0, 0, 0, 0]
    pts_img=np.array(pts_img,dtype=np.float32)
    pts_obj = np.array(pts_obj1, dtype=np.float32)
    cameraMatrix = np.array(cameraMatrix, dtype=np.float64)
    distCoeffs = np.array(distCoeffs, dtype=np.float64)

    # _, rvec0, T0 = cv2.solvePnP(pts_obj, pts_img, cameraMatrix, distCoeffs=distCoeffs)
    # _, rVec, tVec = cv2.solvePnP(pts_obj, pts_img, cameraMatrix, distCoeffs=None,
    #                              flags=cv.SOLVEPNP_ITERATIVE, useExtrinsicGuess=False, rvec=rvec0, tvec=T0)
    # _, rVec, tVec = cv2.solvePnP(pts_obj, pts_img, cameraMatrix, distCoeffs=distCoeffs)
    #https://sites.google.com/site/lifeslash7830/home/hua-xiang-chu-li/opencvwoshittaposeestimation
    _, rVec, tVec ,inliers = cv2.solvePnPRansac(pts_obj, pts_img, cameraMatrix, distCoeffs=distCoeffs, flags=cv2.SOLVEPNP_P3P,
        iterationsCount=1000)
    print("inliers:",len(inliers))

    rotation_matrix=cv2.Rodrigues(rVec)[0]
    #https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/box_dimensioner_multicam/calibration_kabsch.py
    query_T_w = np.eye(4)
    query_T_w[:3, :3] = rotation_matrix
    query_T_w[:3, 3] = tVec.flatten() #-rot.T.dot(tvec)
    pose_mat = np.linalg.inv(query_T_w)

    rot = np.transpose(rotation_matrix)
    tVec_n = - np.matmul(rot,  tVec.flatten())
    query_T_r = np.eye(4)
    query_T_r[:3, :3] =rot
    query_T_r[:3, 3] = tVec_n.flatten()

    rot, _ = cv2.Rodrigues(rVec)
    tvec = -rot.T.dot(tVec)
    query_2 = np.eye(4)
    query_2[:3, :3] = rotation_matrix.T
    query_2[:3, 3] = tvec.flatten()
    TT=generateTransMatrix(tVec,rVec)
    return TT,pts_obj1,pts_obj2



def keypoint_tm(points1s,points2s):
    points1s=np.array(points1s)
    points2s=np.array(points2s)
    points1s = points1s - np.mean(points1s, axis=0)
    points2s = points2s - np.mean(points2s, axis=0)
    estPose = np.eye(4, 4)
    obj=points1s.T
    obs=points2s.T
    for i in range(100):
        w_obj  = warp3d(obj, estPose)
        query, target = findCrsp(w_obj , obs)
        deltaPose = calcPose3D(query, target, estPose)
        estPose = deltaPose.dot(estPose)
        # print("estPose:",estPose)
    return estPose
def move_2_center(cloud):
        colors=np.array(cloud.colors)
        points=np.array(cloud.points)
        points_m = points - np.mean(points ,axis=0)
        cloud_t = make_pcd(points_m, colors)
        return cloud_t

bp='data_v4/'

imgdic = EasyDict({})
imgdic.rgb1=cv2.imread(f"./{bp}/color/4.png")
imgdic.rgb2=cv2.imread(f"./{bp}/color/5.png")
imgdic.dpt1=cv2.imread(f"./{bp}/depth/4.pgm", cv2.IMREAD_UNCHANGED)
imgdic.dpt2=cv2.imread(f"./{bp}/depth/5.pgm", cv2.IMREAD_UNCHANGED)

def main_func(keymatchfun,ext,cam):

    kp1,kp2,goodMatches,img_dst=keymatchfun(imgdic)
    pose_mat,pts_obj1,pts_obj2=get_good_matches_pts(goodMatches,kp1,kp2,imgdic,cam)


    pts_obj1_t=warp3d(np.array(pts_obj1).T, pose_mat)
    estPose=keypoint_tm(pts_obj1_t.T,pts_obj2)

    cloud1=image2PointCloud(imgdic.rgb1,imgdic.dpt1,cam)
    cloud2=image2PointCloud(imgdic.rgb2,imgdic.dpt2,cam)
    cloud1=move_2_center(cloud1)
    cloud2=move_2_center(cloud2)
    cloud1t=cloud1.transform(pose_mat)
    open3d.io.write_point_cloud(f'{bp}pcd_org2.ply', cloud2)

    cv2.imwrite(f"./{bp}{ext}_img_dst.jpg",img_dst)
    # open3d.io.write_point_cloud(f'{bp}pcd_org1.ply', cloud1)
    points1=np.array(cloud1t.points)
    colors1=np.array(cloud1t.colors)
    points1t = warp3d(points1.T, estPose)

    cloud1t2 = make_pcd(points1t.T, colors1)
    # cloud2t=cloud2.transform(pose_mat)
    open3d.io.write_point_cloud(f'{bp}{ext}_pcd_trans_1t.ply', cloud1t)
    open3d.io.write_point_cloud(f'{bp}{ext}_pcd_trans_1t2.ply', cloud1t2)
    # # open3d.io.write_point_cloud('pcd_trans_2t.ply', cloud2t)
    pcd=merge_points_cloud([cloud1t,cloud2])
    open3d.io.write_point_cloud(f'{bp}{ext}_pcd_merge_t.ply', pcd)
keyfuns=[
    [computeKeyPointsAndMaches_akaze,"akaze"],
    [computeKeyPointsAndMaches_orb,"orb"],
    [computeKeyPointsAndMaches_sift,"sift"],
         ]

cam = EasyDict({})
cam.cx=325.5
cam.cy=253.5
cam.fx=518.0
cam.fy=519.0
cam.depthScale=1000.0

for matchfun, ext in keyfuns:
    main_func(matchfun, ext,cam)

