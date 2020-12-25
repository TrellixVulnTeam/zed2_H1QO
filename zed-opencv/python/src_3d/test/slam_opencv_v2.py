import numpy as np
import cv2
import cv2 as cv
import itertools
import os
import struct
import quaternion
# import  pyquaternion as  quaternion
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
depthScale = 1000.0

cx = 1111.735
cy = 650.270
fx = 1046.1599
fy = 1046.1599
#-- Step 1: Detect the keypoints using SURF Detector
# https://pystyle.info/opencv-feature-matching/

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
def computeKeyPointsAndMaches_ORB(img1,img2):
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # OBR 特徴量検出器を作成する。
    detector = cv2.ORB_create(nfeatures=2000)
    # detector = cv2.xfeatures2d.SURF_create()

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

def merge_points_cloud(clouds):
    points=[]
    colors=[]
    for ply in clouds:
        ps=np.array(ply.points)
        cs=np.array(ply.colors)
        points.extend(ps)
        colors.extend(cs)
    return make_pcd(points,colors)

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
    #depth_mask_mergeid_0000
    fn_depth='depth_mask_mergeid_%04d.npy'%(i)
    fn_pcd_msk='pcd_mask_mergeid_%04d.npy'%(i)
    ply_list = get_absolute_file_paths(basep,  fn=fn_merge)
    rgb_list = get_absolute_file_paths(basep,  fn=fn_rgb)
    msk_list = get_absolute_file_paths(basep,  fn=fn_msk)
    rgb_msk_list = get_absolute_file_paths(basep,  fn=fn_rgb_msk)
    depth_list = get_absolute_file_paths(basep,  fn=fn_depth)
    pcd_list = get_absolute_file_paths(basep,  fn=fn_pcd_msk)

    return ply_list,rgb_list,msk_list,rgb_msk_list,depth_list,pcd_list
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
def convert_bgra2rgba(img):
    # ZED2 numpy color is BGRA.
    rgba = np.zeros(img.shape).astype(np.uint8)
    rgba[:, :, 0] = img[:, :, 2]  # R
    rgba[:, :, 1] = img[:, :, 1]  # G
    rgba[:, :, 2] = img[:, :, 0]  # B
    rgba[:, :, 3] = img[:, :, 3]  # A
    return rgba

def load_image_from_npy(p, convert=True):
    color_org = np.load(p)
    color = convert_bgra2rgba(color_org) if convert else color_org
    # pil_img = Image.fromarray(color.astype(np.uint8))
    # pil_img.save(f'{p}/image.png')
    return color[:, :, :3], color_org

def get_rgb_img_msk(id,bp,po,pose):
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

        fn_depth=m.replace(fn,'depth.npy')
        img_dep=np.load(fn_depth)
        img_dep[img_dep == -np.inf] = 0
        img_dep[img_dep == np.inf] = 0
        dep_msk=img_dep*img_msk

        dep_crop=dep_msk[rs:re,cs:ce]
        fn_rgb=m.replace(fn,'depth_mask_mergeid_%04d.npy'%(id))
        np.save(fn_rgb,dep_crop)


        fn_pcd=m.replace(fn,'pcd.npy')
        pcd=np.load(fn_pcd)
        # img_image[img_image == -np.inf] = 0
        # img_image[img_image == np.inf] = 0
        img_msk4=np.concatenate([msk1,msk1,msk1,msk1],axis=2)
        pcd_msk=pcd*img_msk4

        pcd_crop=pcd_msk[rs:re,cs:ce]
        fn_rgb=m.replace(fn,'pcd_mask_mergeid_%04d.npy'%(id))
        np.save(fn_rgb,pcd_crop)

        # ply1 = convert_zed2pcd_to_ply(pcd_crop)

        # py3d.io.write_point_cloud(f"data/pcd_org1_%04d.ply" % (id), ply1)
        #image.npy
        print(fn_rgb)

def get_good_matches_pts_ply(pcd1,pcd2,goodMatches,kp1,kp2):
    zed_points1 = pcd1[:, :, :3]
    zed_colors1 = pcd1[:, :, 3]
    zed_points2 = pcd2[:, :, :3]
    zed_colors2 = pcd2[:, :, 3]
    pts_obj1=[]
    pts_obj2=[]
    colors_obj1=[]
    colors_obj2=[]
    pts_img=[]
    for i, goodMatch in enumerate(goodMatches):
        p1=np.array(kp1[goodMatch.queryIdx].pt,dtype=np.int)
        d1=zed_points1[p1[1],p1[0]][2]
        p2=np.array(kp2[goodMatch.trainIdx].pt,dtype=np.int)
        d2=zed_points2[p2[1],p2[0]][2]
        if d1 == 0 or d2 ==0:
            continue;
        pts_img.append(kp2[goodMatch.trainIdx].pt)

        pts_obj1.append(zed_points1[p1[1],p1[0]])
        pts_obj2.append(zed_points2[p2[1],p2[0]])

        tmp = zed_depthfloat_to_abgr(zed_colors1[p1[1],p1[0]])
        tmp = [tmp[3], tmp[2], tmp[1]]  # ABGR to RGB
        tmp = np.array(tmp).astype(np.float64) / 255.  # for ply (color is double)
        colors_obj1.append(tmp)

        tmp = zed_depthfloat_to_abgr(zed_colors2[p2[1],p2[0]])
        tmp = [tmp[3], tmp[2], tmp[1]]  # ABGR to RGB
        tmp = np.array(tmp).astype(np.float64) / 255.  # for ply (color is double)
        colors_obj2.append(tmp)

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

    _, rVec, tVec = cv2.solvePnP(pts_obj, pts_img, cameraMatrix, distCoeffs=None,
                                 flags=cv.SOLVEPNP_ITERATIVE, useExtrinsicGuess=False, rvec=rvec0, tvec=T0)
    #https://www.366service.com/jp/qa/ed8c0298cc30a02ee80c3d9ecef63a69
    rotM = cv2.Rodrigues(rVec)[0]
    pose_mat = cv2.hconcat((rotM, tVec))
    #https://programtalk.com/python-examples/cv2.solvePnP/
    pose_mat=cv2.vconcat((pose_mat,np.array([[0.0,0.0,0.0,1.0]])))
    return pose_mat,translst
#https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
def computeKeyPointsAndMachesSift(img1,img2):
    # OBR 特徴量検出器を作成する。
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # sift
    detector = cv2.xfeatures2d.SIFT_create()
    # 特徴点を検出する。
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    # マッチング器を作成する。
    # bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    bf = cv.BFMatcher()
    # マッチングを行う。
    matches = bf.knnMatch(desc1, desc2, k=2)
    # matches = bf.match(desc1, desc2)
    # レシオテストを行う。
    # matches = sorted(matches, key=lambda x: x.distance)
    good_matches = []
    thresh = 0.9
    # good_matches=matches[:50]
    for first, second in matches:
        if first.distance < second.distance * thresh:
            good_matches.append(first)
    print("good_matches:",len(good_matches))
    # マッチング結果を描画する。
    dst = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
    # cv2.imwrite("data/dst2.jpg",dst)
    return kp1,kp2,good_matches,dst
def merge_rgb_img_msk():
    hext='test_' #'data\d2_a30_h1.5\cam0\b\20201202170506'
    basep = 'data_v2'
    fn_rgb1=f'{basep}/reconstruction-000000.color-ZED_22378008.jpg'
    fn_rgb2=f'{basep}/reconstruction-000002.color-ZED_22378008.jpg'
    # fn_rgb1=f'data/d2_a30_h1.5/cam0/b/20201202170506/image_org_mergeid_0000.png'
    # fn_rgb2=f'data/d2_a30_h1.5/cam1/b/20201202170508/image_org_mergeid_0000.png'
    rgb1=cv2.imread(fn_rgb1)
    rgb2=cv2.imread(fn_rgb2)

    # rgb_grey_1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)
    # rgb_grey_2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2GRAY)
    # ret, rgb_grey_1 = cv2.threshold(rgb_grey_1, 0, 255, cv2.THRESH_OTSU)
    # ret, rgb_grey_2 = cv2.threshold(rgb_grey_2, 0, 255, cv2.THRESH_OTSU)

    kp1, kp2, goodMatches ,dst= computeKeyPointsAndMaches_ORB(rgb1, rgb2)
    # kp1, kp2, goodMatches ,dst= computeKeyPointsAndMachesSift(rgb1, rgb2)
    cv2.imwrite(f"{basep}/{hext}drawMatches-0_2.jpg",dst)
    '''
    pcd1='reconstruction-000000.cloud-ZED_22378008.ply'
    pcd2='reconstruction-000002.cloud-ZED_22378008.ply'
    pcd1=np.load(pcd1)
    pcd2=np.load(pcd2)
    # ply1=convert_zed2pcd_to_ply(pcd1)
    # py3d.io.write_point_cloud(f"data/pcd_org1_%04d.ply"%(id), ply1)
    pose_mat, translst = get_good_matches_pts_ply(pcd1,pcd2,goodMatches,kp1,kp2)
    ply1='reconstruction-000000.cloud-ZED_22378008.ply'
    ply2='reconstruction-000002.cloud-ZED_22378008.ply'
    cloud1 = py3d.io.read_point_cloud(ply1)
    cloud2 = py3d.io.read_point_cloud(ply2)

    py3d.io.write_point_cloud(f"{basep}/pcd_org1_%04d.ply"%(id), cloud1)
    py3d.io.write_point_cloud(f"{basep}/pcd_org2_%04d.ply"%(id), cloud2)

    cloud1t_r = cloud1.transform(translst[0])
    cloud2t_r = cloud2.transform(translst[1])
    # cloud2t = cloud2.transform(pose_mat)
    cloud1t = cloud1.transform(pose_mat) # good quanlity
    py3d.io.write_point_cloud(f"{basep}/pcd_trans1_%04d.ply"%(id), cloud1t)
    # py3d.io.write_point_cloud(f"{basep}/pcd_trans2_%04d.ply"%(id), cloud2t)
    pcd_merge=merge_points_cloud([cloud1t,cloud2])
    py3d.io.write_point_cloud(f"{basep}/pcd_merge_%04d.ply"%(id), pcd_merge)

    pcd_merge=merge_points_cloud([cloud1t_r,cloud2t_r])
    py3d.io.write_point_cloud(f"{basep}/pcd_ransac_merge_%04d.ply"%(id), pcd_merge)
    
    '''
merge_rgb_img_msk()