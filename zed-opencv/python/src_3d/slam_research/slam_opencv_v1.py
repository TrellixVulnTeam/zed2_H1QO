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
# 1050.3172607421875 0.0 1112.1431884765625
# 0.0 1050.3172607421875 656.2730102539062
# 0.0 0.0 1.0
depthScale = 1000.0

cx = 1112.1
cy = 656.2
fx = 1050.3
fy = 1050.3
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
    thresh = 0.8
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
def computeKeyPointsAndMachesSift_2(img1,img2):
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
def merge_rgb_img_msk(id,bp,po,pose):
    basep = f'{bp}/{pose}'
    baseout = f'{po}/{pose}'
    os.makedirs(baseout,exist_ok=True)
    ply_list,rgb_list,msk_list,rgb_msk_list,depth_list,pcd_list=get_merge_fns(basep, id)
    fn_rgb1,fn_rgb2,_=rgb_msk_list

    rgb1=cv2.imread(fn_rgb1)
    rgb2=cv2.imread(fn_rgb2)
    # rgb_grey_1 = cv2.cvtColor(rgb1, cv2.COLOR_BGR2GRAY)
    # rgb_grey_2 = cv2.cvtColor(rgb2, cv2.COLOR_BGR2GRAY)
    # ret, rgb_grey_1 = cv2.threshold(rgb_grey_1, 0, 255, cv2.THRESH_OTSU)
    # ret, rgb_grey_2 = cv2.threshold(rgb_grey_2, 0, 255, cv2.THRESH_OTSU)

    # kp1, kp2, goodMatches ,dst= computeKeyPointsAndMaches_ORB(rgb1, rgb2)
    kp1, kp2, goodMatches ,dst= computeKeyPointsAndMachesSift(rgb1, rgb2)
    cv2.imwrite(f"{basep}/dsp_%04d.jpg"%(id),dst)
def preprocess_dt():
    bp='data/'
    po='data/'
    dt_ids = ['d2_a30_h1.5']#, 'd2_a45_h1.5']
    for dt_id in dt_ids :
        for i in range(4):
            get_rgb_img_msk(i,bp,po,dt_id)
def merge_dt():
    bp='data/'
    po='data/'
    dt_ids = ['d2_a30_h1.5']#, 'd2_a45_h1.5']
    for dt_id in dt_ids :
        for i in range(4):
            merge_rgb_img_msk(i,bp,po,dt_id)
# preprocess_dt()
merge_dt()