import numpy as np
import cv2
import cv2 as cv
# import quaternion
import  pyquaternion as  quaternion
from transforms3d.quaternions import quat2mat, mat2quat
# from testply import convert_xyzrgb_to_ply,make_pcd
import open3d as py3d
from IPython.display import Image, display
from ransac_icp_pointcloud_merge_ob_sag1202 import ransac_icp
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
#-- Step 1: Detect the keypoints using SURF Detector
# https://pystyle.info/opencv-feature-matching/
def computeKeyPointsAndDesp(src):
    minHessian = 400
    detector = cv.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints = detector.detect(src)
    #-- Draw keypoints
    img_keypoints = np.empty((src.shape[0], src.shape[1], 3), dtype=np.uint8)
    cv.drawKeypoints(src, keypoints, img_keypoints)
    #-- Show detected (drawn) keypoints
    cv.imshow('SURF Keypoints', img_keypoints)
    cv.show()
def imshow(img):
    """ndarray 配列をインラインで Notebook 上に表示する。
    """
    ret, encoded = cv2.imencode(".jpg", img)
    display(Image(encoded))
def computeKeyPointsAndDesp1(img):
    # 特徴点を検出する。
    # OBR 特徴検出器を作成する。
    detector = cv2.ORB_create()
    kp = detector.detect(img)
    # 特徴点を描画する。
    dst = cv2.drawKeypoints(img, kp, None)
    kp, desc = detector.compute(img, kp)
    print(len(kp), desc.shape)
    # 特徴点を検出する。
    kp, desc = detector.detectAndCompute(img, None)
    print(len(kp), desc.shape)
    # imshow(dst)
    # cv.show()
#https://pystyle.info/opencv-feature-matching/
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
    thresh = 0.99
    for first, second in matches:
        if first.distance < second.distance * thresh:
            good_matches.append(first)
    print("good_matches:",len(good_matches))
    # マッチング結果を描画する。
    dst = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
    # cv2.imwrite("data/dst2.jpg",dst)
    return kp1,kp2,good_matches,dst
def get_pose_ransac_icp(pcds):
    ri=ransac_icp()
    sizes=[]
    for pcd in pcds:
        size =np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max()
        sizes.append(size)
    size=np.max(sizes)*2
    pcdsn=[]
    for pcd in pcds:
        pcdn=ri.add_color_normal(pcd,size)
        pcdsn.append(pcdn)
    pcd_aligned,translst = ri.align_pcds(pcdsn, size)
    return translst
def make_pcd(points, colors):
    pcd = py3d.geometry.PointCloud()
    pcd.points = py3d.utility.Vector3dVector(points)
    pcd.colors = py3d.utility.Vector3dVector(colors)
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
def get_good_matches_pts(depth1,depth2,rgb1,rgb2,goodMatches,kp1,kp2):
    pts_obj1=[]
    pts_obj2=[]
    colors_obj1=[]
    colors_obj2=[]
    pts_img=[]
    for i, goodMatch in enumerate(goodMatches):
        p=np.array(kp1[goodMatch.queryIdx].pt,dtype=np.int)
        d=depth1[p[1],p[0]]
        p2=np.array(kp2[goodMatch.trainIdx].pt,dtype=np.int)
        d2=depth2[p2[1],p2[0]]
        if d == 0 or d2 ==0:
            continue;
        pts_img.append(kp2[goodMatch.trainIdx].pt)
        pt=[p[0],p[1],d]
        pd = point2dTo3d(pt)
        # 将(u,v,d)转成(x,y,z)
        pts_obj1.append(pd)
        pt2=[p2[0],p2[1],d2]
        pd2 = point2dTo3d(pt2)
        pts_obj2.append(pd2)

        b, g, r = rgb1[p[1],p[0]] / 255.0
        colors_obj1.append([r, g, b])
        b, g, r = rgb2[p2[1],p2[0]] / 255.0
        colors_obj2.append([r, g, b])
    if len(pts_obj1)==0 or len(pts_img)==0:
        return -1
    print(f"pts_obj1:{len(pts_obj1)}",f"pts_obj2:{len(pts_obj2)}")
    pcd1=make_pcd(pts_obj1,colors_obj1)
    pcd2=make_pcd(pts_obj2,colors_obj2)
    pcds=[pcd1,pcd2]
    translst=get_pose_ransac_icp(pcds)
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
    # rvec0 = np.array(ini_pose[i, 0:3].cpu().view(3, 1))
    # T0 = np.array(ini_pose[i, 3:6].cpu().view(3, 1))
    _, rVec, tVec = cv2.solvePnP(pts_obj, pts_img, cameraMatrix, distCoeffs=None,
                                 flags=cv.SOLVEPNP_ITERATIVE, useExtrinsicGuess=False, rvec=rvec0, tvec=T0)
    #https://www.366service.com/jp/qa/ed8c0298cc30a02ee80c3d9ecef63a69
    rotM = cv2.Rodrigues(rVec)[0]
    # rotM=np.array(rotM).T
    # rotM = -np.array(rotM).T * np.array(tVec)
    # rotation_mat, _ = cv2.Rodrigues(rVec)
    pose_mat = cv2.hconcat((rotM, tVec))
    #https://programtalk.com/python-examples/cv2.solvePnP/
    pose_mat=cv2.vconcat((pose_mat,np.array([[0.0,0.0,0.0,1.0]])))
    return pose_mat,translst
def main_sample():
    bp='data/'
    rgb1=cv2.imread(f"./{bp}/rgb1.png")
    rgb2=cv2.imread(f"./{bp}/rgb2.png")
    depth1=cv2.imread(f"./{bp}/depth1.png")[:,:,0]
    depth2=cv2.imread(f"./{bp}/depth2.png")[:,:,0]

    kp1, kp2, goodMatches ,dst= computeKeyPointsAndMaches(rgb1, rgb2)
    cv2.imwrite("data/dst2.jpg", dst)
    pose_mat, translst = get_good_matches_pts(depth1,depth2,rgb1,rgb2,goodMatches,kp1,kp2)
    cloud1 = image2PointCloud(rgb1, depth1)
    cloud2 = image2PointCloud(rgb2, depth2)

    py3d.io.write_point_cloud('pcd_merge_org1.ply', cloud1)
    py3d.io.write_point_cloud('pcd_merge_org2.ply', cloud2)

    cloud1t_r = cloud1.transform(translst[0])
    cloud2t_r = cloud2.transform(translst[1])
    py3d.io.write_point_cloud('pcd_merge_1t_ransac.ply', cloud1t_r)
    py3d.io.write_point_cloud('pcd_merge_2t_ransac.ply', cloud2t_r)

    cloud2t = cloud2.transform(pose_mat)
    cloud1t = cloud1.transform(pose_mat) # good quanlity
    py3d.io.write_point_cloud('pcd_merge_1t.ply', cloud1t)
    py3d.io.write_point_cloud('pcd_merge_2t.ply', cloud2t)
    # py3d.io.write_point_cloud('pcd_merge.ply', pcd)
import os

def get_absolute_file_paths(directory,fn='image.npy'):
   fils_list=[]
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           if  f.find(fn)>=0:
               fils_list.append( os.path.abspath(os.path.join(dirpath, f)))
   fils_list=sorted(fils_list)
   return fils_list
def get_merge_fns(basep,i):
    fn_merge='pcd_mask_mergeid_%04d.ply'%(i)
    fn_rgb='image_org_mergeid_%04d.png'%(i)
    fn_msk='moving_detection_mask_mergeid_%04d.npy'%(i)
    fn_rgb_msk='rgb_mask_mergeid_%04d.png'%(i)
    ply_list = get_absolute_file_paths(basep,  fn=fn_merge)
    rgb_list = get_absolute_file_paths(basep,  fn=fn_rgb)
    msk_list = get_absolute_file_paths(basep,  fn=fn_msk)
    rgb_msk_list = get_absolute_file_paths(basep,  fn=fn_rgb_msk)
    return ply_list,rgb_list,msk_list,rgb_msk_list
def get_msk_fns(basep,i):
    fn_merge='pcd_mask_mergeid_%04d.ply'%(i)
    fn_rgb='image_org_mergeid_%04d.png'%(i)
    fn_msk='moving_detection_mask_mergeid_%04d.npy'%(i)
    ply_list = get_absolute_file_paths(basep,  fn=fn_merge)
    rgb_list = get_absolute_file_paths(basep,  fn=fn_rgb)
    msk_list = get_absolute_file_paths(basep,  fn=fn_msk)
    return ply_list,rgb_list,msk_list
def cut_out(image, mask):
    if type(image) != np.ndarray:
        raise TypeError("image must be a Numpy array")
    elif type(mask) != np.ndarray:
        raise TypeError("mask must be a Numpy array")
    elif image.shape != mask.shape:
        raise ValueError("image and mask must have the same shape")

    return np.where(mask==0, 0, image)
def getbbox_msk(img_msk):
    r,c = img_msk.shape[:2]
    rs,re=0,0
    cs,ce=0,0
    for i in range(r):
        if sum(img_msk[i,:])>0 and re==0 and rs==0:
            rs=i
        if sum(img_msk[i,:]) == 0 and rs>0 and re==0:
            re=i
            break
    for i in range(c):
        if sum(img_msk[:,i])>0 and cs==0 and ce == 0:
            cs=i
        if sum(img_msk[:,i]) ==0 and cs>0 and ce==0:
            ce=i
            break
    return rs,re,cs,ce
def get_rgb_img_msk(id,bp,po,pose):
    ri=ransac_icp()
    basep = f'{bp}/{pose}'
    baseout = f'{po}/{pose}'
    os.makedirs(baseout,exist_ok=True)
    ply_list,rgb_list,msk_list=get_msk_fns(basep, id)
    for p, r, m in zip(ply_list, rgb_list, msk_list):
        rgb=cv2.imread(r)
        img_msk=np.load(m)
        msk1=np.expand_dims(img_msk,axis=2)
        img_msk3=np.concatenate([msk1,msk1,msk1],axis=2)
        rgb_msk=rgb*img_msk3
        rs,re,cs,ce=getbbox_msk(img_msk)
        img_crop=rgb_msk[rs:re,cs:ce]
        fn=os.path.basename(os.path.normpath(m))
        fn_rgb=m.replace(fn,'rgb_mask_mergeid_%04d.png'%(id))
        # img_rgb=cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        cv2.imwrite(fn_rgb,img_crop)
        print(fn_rgb)
def merge_rgb_img_msk(id,bp,po,pose):
    basep = f'{bp}/{pose}'
    baseout = f'{po}/{pose}'
    os.makedirs(baseout,exist_ok=True)
    ply_list,rgb_list,msk_list,rgb_msk_list=get_merge_fns(basep, id)
    fn_rgb1,fn_rgb2,_=rgb_msk_list

    rgb1=cv2.imread(fn_rgb1)
    rgb2=cv2.imread(fn_rgb2)
    # rgb1 = cv2.medianBlur(rgb1, 3)
    # rgb2 = cv2.medianBlur(rgb2, 3)
    # rgb2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2GRAY)
    # rgb1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)
    kp1, kp2, goodMatches ,dst= computeKeyPointsAndMaches(rgb1, rgb2)
    cv2.imwrite(f"{basep}/dsp_%04d.jpg"%(id),dst)

    # pcds = ri.load_pcds(files_list)

    # size = np.abs((pcds[0].get_max_bound() - pcds[0].get_min_bound())).max() / 30
    # pcdsn=[]
    # for pcd in pcds:
    #     pcdn=ri.add_color_normal(pcd,size)
    #     pcdsn.append(pcdn)
    #
    # pcd_aligned,translst = ri.align_pcds(pcdsn, size)
    # all_points = []
    # all_colors = []
    # for i, pcd in enumerate(pcd_aligned):
    #     all_points.append(np.asarray(pcd.points))
    #     all_colors.append(np.asarray(pcd.colors))
    #     print("pcd:",i)
    #     ft_pcd_m = f'{baseout}/pcd_aligned_mergeid_%04d_cam%d.ply'% (id,i)
    #     py3d.io.write_point_cloud(ft_pcd_m, pcd)
    # pcd_a_merge=ri.make_pcd(np.vstack(all_points),np.vstack(all_colors))
    # ft_pcd_m=f'{baseout}/pcd_aligned_mergeid_%04d.ply'% (id)
    # py3d.io.write_point_cloud(ft_pcd_m, pcd_a_merge)
def preprocess_dt():
    bp='D:/02_AIPJ/004_ISB/pointclouds_dt/'
    po='D:/02_AIPJ/004_ISB/pointclouds_dt/'
    dt_ids = ['d2_a30_h1.5', 'd2_a45_h1.5']
    for dt_id in dt_ids :
        for i in range(4):
            get_rgb_img_msk(i,bp,po,dt_id)
def merge_dt():
    bp='D:/02_AIPJ/004_ISB/pointclouds_dt/'
    po='D:/02_AIPJ/004_ISB/pointclouds_dt/'
    dt_ids = ['d2_a30_h1.5', 'd2_a45_h1.5']
    for dt_id in dt_ids :
        for i in range(4):
            merge_rgb_img_msk(i,bp,po,dt_id)
merge_dt()

'''
https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
camera pose
https://www.stereolabs.com/docs/positional-tracking/
https://www.stereolabs.com/docs/positional-tracking/coordinate-frames/

spatial mapping
https://www.stereolabs.com/docs/spatial-mapping/using-mapping/
'''