import numpy as np
import cv2
import cv2 as cv
import quaternion
# import  pyquaternion as  quaternion
import open3d
# import pcl
#python -m pip install numpy-quaternion
#pip install pyquaternion
#conda install -c conda-forge quaternion
# camera intrinsics
cx = 325.5
cy = 253.5
fx = 518.0
fy = 519.0
depthScale = 1000.0

colorImgs, depthImgs = [], []

bp='data/'
rgb1=cv2.imread(f"./{bp}/rgb1.png")
rgb2=cv2.imread(f"./{bp}/rgb2.png")
depth1=cv2.imread(f"./{bp}/depth1.png")[:,:,0]
depth2=cv2.imread(f"./{bp}/depth2.png")[:,:,0]
'''
https://pystyle.info/opencv-feature-matching/
https://www.366service.com/jp/qa/ed8c0298cc30a02ee80c3d9ecef63a69
https://programtalk.com/python-examples/cv2.solvePnP/
'''
#参照資料
def computeKeyPointsAndMaches(img1,img2):
    # OBR 特徴量検出器を作成する。
    detector = cv2.ORB_create()

    # 特徴点を検出する。
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)

    # マッチング器を作成する。
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # マッチングを行う。
    matches = bf.knnMatch(desc1, desc2, k=2)

    # レシオテストを行う。
    good_matches = []
    thresh = 0.7
    for first, second in matches:
        if first.distance < second.distance * thresh:
            good_matches.append(first)
    print("good_matches:",len(good_matches))
    # マッチング結果を描画する。
    dst = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
    cv2.imwrite("data/dst2.jpg",dst)
    return kp1,kp2,good_matches
def make_pcd(points, colors):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)
    return pcd
def image2PointCloud(rgb, depth):
   points=[]
   colors=[]
   r,c =rgb.shape[:2]
   for i in range(r):
       for j in range(c):
           d=depth[i,j]
           if d==0:
               continue
           p=point2dTo3d([i,j,d])
           points.append(p)
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
def point2dTo3d(point):
    z=point[2]/depthScale
    x=(point[0]-cx)*z/fx
    y=(point[1]-cy)*z/fy
    return [x,y,z]
def get_good_matches_pts(goodMatches,kp1,kp2):
    pts_obj1=[]
    pts_img=[]
    for i, goodMatch in enumerate(goodMatches):
        p=np.array(kp1[goodMatch.queryIdx].pt,dtype=np.int)
        d=depth1[p[1],p[0]]
        if d == 0 :
            continue;
        pts_img.append(kp2[goodMatch.trainIdx].pt)
        pt=[p[0],p[1],d]
        pd = point2dTo3d(pt)
        # 将(u,v,d)转成(x,y,z)
        pts_obj1.append(pd)
    if len(pts_obj1)==0 or len(pts_img)==0:
        return -1
    print(f"pts_obj1:{len(pts_obj1)}")
    cameraMatrix=[
        [fx,0,cx],
        [0,fy,cy],
        [0,0,1]
        ]
    distCoeffs=[0, 0, 0, 0, 0]
    pts_img=np.array(pts_img,dtype=np.float32)
    pts_obj = np.array(pts_obj1, dtype=np.float32)
    cameraMatrix = np.array(cameraMatrix, dtype=np.float64)
    distCoeffs = np.array(distCoeffs, dtype=np.float64)

    _, rvec0, T0 = cv2.solvePnP(pts_obj, pts_img, cameraMatrix, distCoeffs=distCoeffs)
    _, rVec, tVec = cv2.solvePnP(pts_obj, pts_img, cameraMatrix, distCoeffs=None,
                                 flags=cv.SOLVEPNP_ITERATIVE, useExtrinsicGuess=False, rvec=rvec0, tvec=T0)

    rotM = cv2.Rodrigues(rVec)[0]
    pose_mat = cv2.hconcat((rotM, tVec))
    pose_mat=cv2.vconcat((pose_mat,np.array([[0.0,0.0,0.0,1.0]])))
    return pose_mat
kp1,kp2,goodMatches=computeKeyPointsAndMaches(rgb1,rgb2)
pose_mat=get_good_matches_pts(goodMatches,kp1,kp2)
cloud1=image2PointCloud(rgb1,depth1)
cloud2=image2PointCloud(rgb2,depth2)

open3d.io.write_point_cloud('pcd_org1.ply', cloud1)
open3d.io.write_point_cloud('pcd_org2.ply', cloud2)

cloud1t=cloud1.transform(pose_mat)
# cloud2t=cloud2.transform(pose_mat)
open3d.io.write_point_cloud('pcd_trans_1.ply', cloud1t)
pcd=merge_points_cloud([cloud1t,cloud2])
open3d.io.write_point_cloud('pcd_merge.ply', pcd)