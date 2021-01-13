'''
④人のいない時間帯の画像群の平均画像を取得
⑤（人のいる画像の画素値－⑤の平均画像の画素値）を計算。どうなるかを見て、動体検出として利用できるかを考える。


【④人のいない時間帯の画像群の平均画像を取得】の設計
************************************
①画像と同じサイズのndarrayをnp.zeros
②（object detection)モデルを使って、人がいない画像を洗い出して
③人がいない全て画像を読み込みまして、画像のpixels値の合計値を計算して、画像枚数を割りして、平均画像を計算します。

【⑤（人のいる画像の画素値－⑤の平均画像の画素値）を計算。どうなるかを見て、動体検出として利用できるかを考える。】の設計
************************************
①【④人のいない時間帯の画像群の平均画像を取得】の処理により、平均画像を計算
②【人のいる画像の画素値】－【平均画像の画素値】計算して、画像を出力します。
③【人のいる画像の画素値】と【平均画像の画素値】を入力して、
　【https://qiita.com/K_M95/items/4eed79a7da6b3dafa96d】の方法で結果を確認します。
/media/user1/Data/data/Data_out_20201116/

'''
import cv2,numpy as np,datetime
import pandas as pd
from PIL import Image
from move_detection_v2 import detect_frame
import os,itertools,struct
from skimage.measure import compare_ssim
# from zed2pcl import zed_depthfloat_to_abgr,make_pcd
import open3d
import copy as cp
from object_detection_draw_v7 import load_model_obj,human_bbox_on_image_inferenced_result,get_img_bboxes,object_detection_from_npy_all_file,load_image_from_npy
from segmentation_DeepLabV3_v5 import load_model_seg,load_image_small,run_inference
#image1,image2,difference=compute_difference_cv2(image1,image2,color=[255, 0, 0])

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
    return [0,0,0,0]
  else:
    h = hex(struct.unpack('<I', struct.pack('<f', f))[0])
    return [eval('0x'+a+b) for a, b in zip(h[::2], h[1::2]) if a+b != '0x']

def make_pcd(points, colors):
  pcd = open3d.geometry.PointCloud()
  pcd.points = open3d.utility.Vector3dVector(points)
  pcd.colors = open3d.utility.Vector3dVector(colors)
  return pcd
def compute_difference_cv2(image1,image2,color=[0, 0, 255]):
    # compute difference
    #print(image1.shape,image2.shape)
    difference = cv2.subtract(image1, image2)
    # color the mask red 
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    difference[mask != 255] = color

    # add the red mask to the images to make the differences obvious
    image1[mask != 255] = color
    image2[mask != 255] = color
    return difference,image2
def compute_difference_ssim(before,after,color=[0, 0, 255]):
    # Convert images to grayscale
    before = cv2.GaussianBlur(before, (3, 3), 0)
    after = cv2.GaussianBlur(after, (3, 3), 0)
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = compare_ssim(before_gray, after_gray, full=True)
    #print("Image similarity", score)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, color, -1)
            cv2.drawContours(filled_after, [c], 0, color, -1)
    return mask,filled_after
# image2,_ =load_image_from_npy(rightImage)
# image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
# before,after,diff,mask,filled_after=compute_difference_ssim(before,after,color=[0, 0, 255])
def get_human_lst():
    path = '/media/user1/Data/data/20201116/'
    patho = '/media/user1/Data/data/Data_out_20201116/'
    path_category_index = '../dt/category_index.npy'
    # "https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1"
    # "https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1"

    # category_index = np.load(path_category_index, allow_pickle='TRUE').item()
    path_mod = "https://tfhub.dev/tensorflow/efficientdet/d4/1"
    human_dt_lst = object_detection_from_npy_all_file(path, patho, path_mod, path_category_index)
    pd.DataFrame(human_dt_lst).to_csv(f'{patho}/human_detect.csv', header=None, index=None)
    human_dt_lst = pd.read_csv(f'{patho}/human_detect.csv', header=None)
    image_sum = None
    for i, dt in enumerate(human_dt_lst.values.tolist()):
        path_img, flg = dt
        if i == 0:
            img, color_org = load_image_from_npy(path_img)
            image_sum = np.zeros(img.shape)
        if flg == 0:
            img, color_org = load_image_from_npy(path_img)
            image_sum = image_sum + img
    image_average = image_sum / (i + 1)
    image_average = np.array(image_average, dtype=np.uint8)
    image_average_fp = f'{patho}/image_average.npy'
    np.save(image_average_fp, image_average)

    image_average = np.load(image_average_fp)
    for i, dt in enumerate(human_dt_lst.values.tolist()):
        path_img, flg = dt
        if flg == 1:
            pathoful = path_img.replace(path.replace('/', '\\'), patho)
            img, color_org = load_image_from_npy(path_img)
            image_diff = img - image_average
            image_diff_fp = pathoful.replace('image.npy', 'moving_detection_diff.png')
            # image_diff_fp=f'{pathoful}/moving_detection_diff.png'
            cv2.imwrite(image_diff_fp, image_diff)

            gray = cv2.cvtColor(image_average, cv2.COLOR_BGR2GRAY)
            avg = gray.copy().astype("float")
            img_detect, img_detect_concat, avg = detect_frame(img, avg)
            pil_img = Image.fromarray(img_detect.astype(np.uint8))

            image_diff_fp = pathoful.replace('image.npy', 'moving_detection.png')
            pil_img.save(image_diff_fp)
def get_average_image(df_image):
    image_sum = None
    cnt=len(df_image.values.tolist())
    for i, dt in enumerate(df_image.values.tolist()):
        path_img, flg = dt
        if i%100==0:
          print(i,i/cnt,path_img)
        if i == 0:
            img, color_org = load_image_from_npy(path_img)
            image_sum = np.zeros(img.shape)
        if flg == 0:
            img, color_org = load_image_from_npy(path_img)
            image_sum = image_sum + img
    image_average = image_sum / (i + 1)
    image_average = np.array(image_average, dtype=np.uint8)
    return image_average

'''
伊藤さんの設計
# x.shape => (30, 1242, 2208, 3)
# xはpersonを検知していない画像
# y = np.mean(x, axis=0) # yは元の画像とshapeが同じ、かつxの平均。
# y.shape => (1242, 2208, 3) 
'''
def get_average_image_2(df_image):
    cnt=len(df_image.values.tolist())
    img_exp=None
    df2=df_image.copy(deep=True)
    df2.columns = ['fn', 'flg']
    df2['del']=0
    print(df_image.shape)
    for i, dt in enumerate(df_image.values.tolist()):
        path_img, flg = dt
        path_img=path_img.replace('/media/user1/Data/data/20201116/','/home/user1/yu_develop/202011161_sample/image/')
        
        if os.path.exists(path_img)==False:
            df2.iloc[i,2]=1
            continue
        if i%10==0:
          print(i,i/cnt,path_img)
        print(path_img)
        if img_exp is None:
            img, color_org = load_image_from_npy(path_img)
            img_exp=np.expand_dims(img,axis=0)
        if flg == 0 and i>0:
            img, color_org = load_image_from_npy(path_img)
            img_exp_i=np.expand_dims(img,axis=0)
            img_exp=np.concatenate([img_exp,img_exp_i],axis=0)
    print("img_exp shape",img_exp.shape)
    img_avg = np.mean(img_exp, axis=0)
    df3=df2[df2['del']==0]
    df3=df3.loc[:,['fn', 'flg']]
    return np.array(img_avg, dtype=np.uint8),df3

'''
伊藤さんの設計
# ((image - y) > 1.1 * (image - y)) 
# ((image - y) < 0.1 * (image - y)) => (True,False)のndarrayがreturn
# ((image - y) < 1.1 * (image - y)) 
# ((image - y) > 0.9 * (image - y)) 
# condition1 = ((image - y) < 1.1 * (image - y)) 
# condition2 = ((image - y) > 0.9 * (image - y))  
# (condition1 & condition2) => 両方が成立する要素がTrueになる

20201118午後仕様　伊藤さん
（背景になる：平均画像画素値の±10%以内）
# (image <= (1.0+r) * y) 
# (image >= (1.0-r) * y) => (True,False)のndarrayがreturn
（背景をTrueにするndarray）
# condition1 = (image <= (1.0+r) * y) 
# condition2 = (image >= (1.0-r) * y)  
# (condition1 & condition2) => 両方が成立する要素がTrueで、背景となる。
'''
def get_diff_of_two_image_2(image,y,r=0.1):
    img_bk=np.zeros(image.shape)
    condition1 = (image <= (1.0+r) * y)
    condition2 = (image >= (1.0-r) * y)
    img_out_b=(condition1 & condition2)
    condition3 = (image > (1.0+r) * y)
    condition4 = (image < (1.0-r) * y)
    img_out_h=(condition3 & condition4)
    #img_out=img_out_b+img_out_h
    return img_out_b.astype(np.int)*255,img_out_h.astype(np.int)*255
def get_diff_of_two_image(image,y,r=0.1):
    img_i1 = ((image - y) > (1+r) * (image - y))
    img_i0 = ((image - y) < r * (image - y))
    img_s1 = ((image - y) < (1+r)* (image - y))
    img_s0 = ((image - y) > (1-r) * (image - y))
    img_out=img_i1&img_i0+img_s1&img_s0
    return img_out.astype(np.int)

def get_human_lst_each_diff_2():
    path = '/home/user1/yu_develop/202011161_sample/image/'
    patho = '/home/user1/yu_develop/202011161_sample/out/'
    camlst=['cam0','cam1','cam2']
    for cam in camlst:
        df = pd.read_csv(f'{patho}/human_detect_{cam}_s.csv', header=None)
        df_human=df[df[1]==1]
        image_average_fp = f'{patho}/image_average_{cam}_s.npy'
        image_average=np.load(image_average_fp)

        image_average = cv2.GaussianBlur(image_average, (5, 5), 0)
        image_average_gray = cv2.cvtColor(image_average, cv2.COLOR_BGR2GRAY)
        cnt=len(df_human.values.tolist())
        for i, dt in enumerate(df_human.values.tolist()):
            path_img, flg = dt
            print(i,i/cnt,path_img)
            path_img=path_img.replace('/media/user1/Data/data/20201116/','/home/user1/yu_develop/202011161_sample/image/')
            pathoful = path_img.replace(path, patho)
            os.makedirs(pathoful,exist_ok=True)
            img, color_org = load_image_from_npy(path_img)
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image_diff,image_diff2=get_diff_of_two_image_2(img_gray,image_average_gray)
            image_diff_fp = pathoful.replace('image.npy', 'moving_detection_diff_gaus_gray.png')
            cv2.imwrite(image_diff_fp, image_diff)
            # image_diff_fp = pathoful.replace('image.npy', 'moving_detection_diff.npy')
            #np.save(image_diff_fp, image_diff)

            
            image_diff_fp = pathoful.replace('image.npy', 'moving_detection_diff_2_gaus_gray.png')
            cv2.imwrite(image_diff_fp, image_diff2)
            # image_diff_fp = pathoful.replace('image.npy', 'moving_detection_diff_2.npy')
            #np.save(image_diff_fp, image_diff2)

            gray = cv2.cvtColor(image_average, cv2.COLOR_BGR2GRAY)
            avg = gray.copy().astype("float")
            img_detect, img_detect_concat, avg = detect_frame(img, avg)
            pil_img = Image.fromarray(img_detect.astype(np.uint8))

            image_diff_fp = pathoful.replace('image.npy', 'moving_detection_gaus.png')
            #print(image_diff_fp) 
            pil_img.save(image_diff_fp)


def get_human_lst_each_diff_fun(path,patho,func=None,ext='cv2'):
    # path = '/home/user1/yu_develop/202011161_sample/image/'
    # patho = '/home/user1/yu_develop/202011161_sample/out/'
    camlst = ['cam0', 'cam1', 'cam2']
    if func is None:
        return None,None
    for cam in camlst:
        df = pd.read_csv(f'{patho}/human_detect_{cam}_s.csv', header=None)
        df_human = df[df[1] == 1]
        image_average_fp = f'{patho}/image_average_{cam}_s.npy'
        image_average = np.load(image_average_fp)

        #image_average = cv2.GaussianBlur(image_average, (5, 5), 0)
        image_average_gray = cv2.cvtColor(image_average, cv2.COLOR_BGR2GRAY)
        image_average = cv2.cvtColor(image_average, cv2.COLOR_BGR2RGB)
        cnt = len(df_human.values.tolist())
        for i, dt in enumerate(df_human.values.tolist()):
            path_img, flg = dt
            print(i, i / cnt, path_img)
            path_img = path_img.replace('/media/user1/Data/data/20201116/',
                                        '/home/user1/yu_develop/202011161_sample/image/')
            pathoful = path_img.replace(path, patho)
            pathoful_fd=pathoful.replace('image.npy', '')
            os.makedirs(pathoful_fd, exist_ok=True)
            img, color_org = load_image_from_npy(path_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # image_diff, image_diff2 = get_diff_of_two_image_2(img_gray, image_average_gray)
            # image1, image_diff2, image_diff = compute_difference_cv2(image_average_gray, img, color=[255, 0, 0])
            image_diff2, image_diff = func(image_average, img, color=[255, 0, 0])
            image_diff_fp = pathoful.replace('image.npy', f'image_org_plus_{ext}.png')
            cv2.imwrite(image_diff_fp, image_diff)
            
            image_diff_fp = pathoful.replace('image.npy', 'image.png')
            cv2.imwrite(image_diff_fp, img)

            image_diff_fp = pathoful.replace('image.npy', f'{ext}_moving_detection.png')
            cv2.imwrite(image_diff_fp, image_diff2)

            gray = cv2.cvtColor(image_average, cv2.COLOR_BGR2GRAY)
            avg = gray.copy().astype("float")
            img_detect, img_detect_concat, avg = detect_frame(img, avg)
            pil_img = Image.fromarray(img_detect.astype(np.uint8))

            image_diff_fp = pathoful.replace('image.npy', f'opencv_moving_detection.png')
            # print(image_diff_fp)
            pil_img.save(image_diff_fp)
def get_human_lst_each_cam():
    path = '/media/user1/Data/data/20201116/'
    patho = '/media/user1/Data/data/Data_out_20201116/'
    camlst=['cam0','cam1','cam2']
    df_detection = pd.read_csv(f'{patho}/human_detect.csv', header=None)
    cnt=len(camlst)
    for i,cam in enumerate(camlst):
        df=df_detection[df_detection[0].str.contains(cam)]
        image_average=get_average_image(df)
        # image_average,df_n=get_average_image_2(df)
        image_average_fp = f'{patho}/image_average_{cam}.npy'
        np.save(image_average_fp, image_average)
        image_average_fp = f'{patho}/image_average_{cam}.png'
        cv2.imwrite(image_average_fp, image_average)
        df_n.columns = ['fn', 'flg']
        df_n = df_n.sort_values('fn')
        df_n.to_csv(f'{patho}/human_detect_{cam}_s.csv', header=None, index=None)

def get_human_lst_each_cam_2():
    patho = '/home/user1/yu_develop/202011161_sample/out/'
    camlst=['cam0','cam1','cam2']
    df_detection = pd.read_csv(f'{patho}/human_detect.csv', header=None)
    
    df_detection=df_detection.replace('/media/user1/Data/data/20201116/','/home/user1/yu_develop/202011161_sample/image/')
    for i,cam in enumerate(camlst):
        df=df_detection[df_detection[0].str.contains(cam)]
        # image_average=get_average_image(df)
        image_average,df_n=get_average_image_2(df)
        image_average_fp = f'{patho}/image_average_{cam}_s.npy'
        np.save(image_average_fp, image_average)
        image_average_fp = f'{patho}/image_average_{cam}_s.png'
        cv2.imwrite(image_average_fp, image_average)
        df_n.columns = ['fn', 'flg']
        df_n = df_n.sort_values('fn')
        df_n.to_csv(f'{patho}/human_detect_{cam}_s.csv', header=None, index=None)
def get_human_lst_each_diff():
    path = '/media/user1/Data/data/20201116/'
    patho = '/media/user1/Data/data/Data_out_20201116/'
    camlst=['cam0','cam1','cam2']
    for cam in camlst:
        df = pd.read_csv(f'{patho}/human_detect_{cam}.csv', header=None)
        df_human=df[df[1]==1]
        image_average_fp = f'{patho}/image_average_{cam}.npy'
        image_average=np.load(image_average_fp)
        cnt=len(df_human.values.tolist())
        for i, dt in enumerate(df_human.values.tolist()):
            path_img, flg = dt
            print(i,i/cnt,path_img)
            pathoful = path_img.replace(path, patho)
            img, color_org = load_image_from_npy(path_img)
            image_diff = img - image_average
            image_diff_fp = pathoful.replace('image.npy', 'moving_detection_diff.png')
            #print(image_diff_fp)
            cv2.imwrite(image_diff_fp, image_diff)
            gray = cv2.cvtColor(image_average, cv2.COLOR_BGR2GRAY)
            avg = gray.copy().astype("float")
            img_detect, img_detect_concat, avg = detect_frame(img, avg)
            pil_img = Image.fromarray(img_detect.astype(np.uint8))

            image_diff_fp = pathoful.replace('image.npy', 'moving_detection.png')
            #print(image_diff_fp)
            pil_img.save(image_diff_fp)


class moving_object_merge:
    def __init__(self,mod_path_obj,path_category_index,mod_path_seg,flg=None):
        self.mod_path_obj = mod_path_obj
        self.path_category_index = path_category_index
        self.mod_path_seg = mod_path_seg
        self.flg=flg
        if self.flg=='AI':
            self.load_model_segmentation_object_detection()

    def convert_zed2pcd_to_ply_moving(self,zed_pcd):
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
            # point
            tmp = np.array(zed_points[x, y]).astype(np.float64)
            if tmp[0] == 0 and tmp[1] == 0 and tmp[2] == 0:
                continue
            points.append(tmp)
            # color
            tmp = zed_depthfloat_to_abgr(zed_colors[x, y])
            tmp = [tmp[3], tmp[2], tmp[1]]  # ABGR to RGB
            tmp = np.array(tmp).astype(np.float64) / 255.  # for ply (color is double)
            colors.append(tmp)
        return make_pcd(points, colors)
    def get_moving_object_mask(self,img_path_moving,img_path_average):
        img, color_org = load_image_from_npy(img_path_moving)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # image_average_fp = f'{patho}/image_average_{cam}_s.npy'
        image_average = np.load(img_path_average)
        # image_diff, image_diff2 = get_diff_of_two_image_2(img_gray, image_average_gray)
        # image1, image_diff2, image_diff = compute_difference_cv2(image_average_gray, img, color=[255, 0, 0])
        image_msk, image_org_msk = compute_difference_ssim(image_average, img, color=[1, 1, 1])
        return image_msk,img
    def get_pcd_by_mask(self,pcd_path,img_msk):
        zed_pcd = np.load(pcd_path)
        msk1=np.expand_dims(img_msk[:, :, 0],axis=2)
        img_msk4=np.concatenate([msk1,msk1,msk1,msk1],axis=2)
        zed_pcd_out=zed_pcd*img_msk4
        ply=self.convert_zed2pcd_to_ply_moving(zed_pcd_out)
        return ply
    def get_moving_object_ply_by_mask(self,img_path_moving,img_path_average,pcd_path):
        if self.flg == 'AI':
            img_msk,img=self.get_mask_by_segmentation_object_detection(img_path_moving)
        else:
            img_msk,img=self.get_moving_object_mask(img_path_moving,img_path_average)
        print(img_msk.shape)
        #img_msk = remove_noise_mask(img_msk, 3)
        img_msk = fuchi_deal_msk(img_msk,self.fuchi_size)
        ply=self.get_pcd_by_mask(pcd_path,img_msk)
        return ply,img_msk,img

    def get_absolute_file_paths(self,directory,ext=".npy",fn='image.npy') -> object:
       fils_list=[]
       fn_list=[]
       dir_list=[]
       for dirpath,_,filenames in os.walk(directory):
           for f in filenames:
               if  f.find(fn)>=0:
                   fils_list.append( os.path.abspath(os.path.join(dirpath, f)))
                   fn_list.append( f)
                   dir_list.append(dirpath)
       return fils_list,fn_list,dir_list
    def get_same_time_image_from_cameras(self,pathi, camlst=['cam0', 'cam1', 'cam2'], th=200000):
        files_list, fn_list, dir_list = self.get_absolute_file_paths(pathi, ext=".npy", fn='image.npy')
        # fpdir=np.concatenate([np.expand_dims(files_list,axis=1),np.expand_dims(dir_list,axis=1)],axis=1)
        lst_file = []
        for fn_full, fn, fd in zip(files_list, fn_list, dir_list):
            time_str = os.path.basename(os.path.normpath(fd))
            path_cam = fd.replace(time_str, '')
            cam = os.path.basename(os.path.normpath(path_cam))
            lst_file.append([fn_full, fn, cam, time_str])
        df_all = pd.DataFrame(lst_file, columns=['file_full', 'file_name', 'camera', 'times'])
        df_cam0 = df_all[df_all['file_full'].str.contains(camlst[0])].reset_index(drop=True)
        same_time_cam_lst = []
        for dt in df_cam0.values.tolist():
            fn_full, fn, cam, time_str = dt
            time_0 = int(time_str[6:])
            subtract_lst = []
            # subtract_lst.append([fn_full, fn, cam, time_str ,0])
            subtract_lst.append(time_str)
            flg_same_time = True
            for cam in camlst[1:]:
                df_cami = df_all[df_all['file_full'].str.contains(cam)].reset_index(drop=True)
                # df_cami['times'].replace('_', '', inplace=True)

                df_cami["times_c"] = df_cami["times"].str.replace('_', '')
                df_cami["times_t"] = df_cami["times_c"].str[6:]

                df_cami[["times_t"]] = df_cami[["times_t"]].apply(pd.to_numeric)
                i = np.argmin(abs(df_cami.loc[:, 'times_t'] - time_0))
                fn_full_i, fn_i, cam_i, time_, times_c, time_str_i = df_cami.iloc[i, :]
                if abs(int(time_str_i) - time_0) > th:
                    flg_same_time = False
                    break
                # subtract_lst.append([fn_full_i, fn_i, cam_i, time_str_i ,abs(int(time_str_i)-time_0)])
                subtract_lst.extend([time_])
            if flg_same_time:
                same_time_cam_lst.append(subtract_lst)
        same_time_cam_lst.sort(key=lambda x: x[0])
        return same_time_cam_lst

    def get_mask_by_segmentation_object_detection(self,fnpath):
        bboxes_ret, image = human_bbox_on_image_inferenced_result(fnpath, self.detector, self.category_index, th=0.5)
        disimg = cp.deepcopy(image)
        mask_img = np.zeros(image.shape)
        img_objs = get_img_bboxes(image, bboxes_ret)
        for imgbbox, bbox in zip(img_objs, bboxes_ret):
            xmin, xmax, ymin, ymax = tuple(bbox)
            image_for_prediction, cropped_image, img_org = load_image_small(imgbbox, self.input_size, t="image")
            seg_map = run_inference(self.interpreter, self.input_details, image_for_prediction, cropped_image)

            w, h = img_org.shape[:2]
            seg_map_mask = np.where(seg_map == 13, 1, 0)
            seg_map_expand = np.expand_dims(seg_map_mask, axis=2)
            seg_map_img = np.concatenate([seg_map_expand, seg_map_expand, seg_map_expand], axis=2)
            # disimg[ymin:ymax, xmin:xmax, :] = seg_map_img[0:w, 0:h, :]
            mask_img[ymin:ymax, xmin:xmax, :] = seg_map_img[0:w, 0:h, :]
        self.bboxes_ret=bboxes_ret
        return mask_img,disimg

    def load_model_segmentation_object_detection(self):
        self.detector, self.category_index = load_model_obj(self.mod_path_obj, self.path_category_index)
        self.interpreter, self.input_details, self.input_size = load_model_seg(self.mod_path_seg)
        # return detector, category_index, interpreter, input_details, input_size
def main_image_mask_ply(dt_id,fuchi_size,basepath,mob):
    pathi=f'{basepath}/image/{dt_id}'
    patho=f'{basepath}/out'
    path_ply=f'{basepath}/pointcloud_fuchi%02d/{dt_id}'%(fuchi_size)
    camlst=['cam0', 'cam1', 'cam2']
    os.makedirs(path_ply, exist_ok=True)
    # mob=moving_object_merge(mod_path_obj, path_category_index, mod_path_seg,flg=None)
    files_list, fn_list, dir_list = mob.get_absolute_file_paths(pathi, ext=".npy", fn='image.npy')
    #mob=moving_object_merge(mod_path_obj, path_category_index, mod_path_seg)
    files_list=sorted(files_list)
    cams=['cam0','cam1','cam2']
    wb=['w','b']
    #print(files_list)
    # "\\b\\" in files_list[3]
    same_time_cam_lst=[]
    for i in range(4):
        merge_cam0=files_list[i]
        merge_cam1=files_list[i+4]
        merge_cam2=files_list[i+8]
        same_time_cam_lst.append([merge_cam0,merge_cam1,merge_cam2])
        # time_str = os.path.basename(os.path.normpath(fd))
    # same_time_cam_lst=mob.get_same_time_image_from_cameras(pathi,camlst=camlst,th=200000)
    #print(same_time_cam_lst)
    # np.save(fn_same_time_cam_lst, same_time_cam_lst)
    cnt=len(same_time_cam_lst)
    same_time_cam_lst_new=[]
    for i, same_times in enumerate(same_time_cam_lst):
        merge_id = 'mergeid_%04d' % (i)
        for j, fn in enumerate(same_times):
            cam = camlst[j]
            # path_id = os.path.basename(os.path.normpath(fn))
            path_id = fn.replace('image.npy','')
            fn_img_moving = f'{path_id}image.npy'
            fn_img_average=f'{patho}/image_average_{cam}_s.npy'
            fn_pcd=f'{path_id}/pcd.npy'
            fd_ply=os.path.normpath(path_id).replace(os.path.normpath(pathi),os.path.normpath(path_ply))
            fn_ply=f'{fd_ply}/pcd_mask_{merge_id}.ply'
            os.makedirs(fd_ply, exist_ok=True)
            ply,img_msk,img=mob.get_moving_object_ply_by_mask(fn_img_moving,fn_img_average,fn_pcd)
            open3d.io.write_point_cloud(fn_ply, ply)
            print(i,cnt,fn_ply)
            image_diff_fp =f'{fd_ply}/moving_detection_mask_{merge_id}.npy'
            np.save(image_diff_fp, img_msk[:,:,0])
            
            image_diff_fp =f'{fd_ply}/moving_detection_mask_{merge_id}.png'
            img_msk=img_msk*255
            cv2.imwrite(image_diff_fp, img_msk)
            
            image_diff_fp =f'{fd_ply}/image_org_{merge_id}.png'
            img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_diff_fp, img_rgb)
        same_time_cam_lst_new.append([merge_id]+same_times)
    fn_same_time_cam_lst = f'{patho}/same_time_cam_lst.npy'
    np.save(fn_same_time_cam_lst, same_time_cam_lst_new)


def main_image_mask_ply_2021_0113(dt_id, fuchi_size, basepath, mob):
    # pathi=f'{basepath}/image/{dt_id}'
    pathi = f'{basepath}/{dt_id}'
    patho = f'{basepath}/out'
    path_ply = f'{basepath}/pointcloud_fuchi%02d/{dt_id}' % (fuchi_size)
    camlst = ['cam0', 'cam1', 'cam2']
    os.makedirs(path_ply, exist_ok=True)
    # mob=moving_object_merge(mod_path_obj, path_category_index, mod_path_seg,flg=None)
    # files_list, fn_list, dir_list = mob.get_absolute_file_paths(pathi, ext=".npy", fn='image.npy')
    # mob=moving_object_merge(mod_path_obj, path_category_index, mod_path_seg)
    # files_list = sorted(files_list)
    for j, cam in enumerate(camlst):
        # fn=f'{pathi}/{cam}/image.npy'
        merge_id = 'mergeid_%04d' % (j)
        path_id = f'{pathi}/{cam}/'
        fn_img_moving = f'{path_id}image.npy'
        fn_img_average = f'{patho}/image_average_{cam}_s.npy'
        fn_pcd = f'{path_id}/pcd.npy'
        fd_ply = os.path.normpath(path_id).replace(os.path.normpath(pathi), os.path.normpath(path_ply))
        fn_ply = f'{fd_ply}/pcd_mask_{merge_id}.ply'
        fn_bboxes= f'{path_id}bboxes.npy'
        os.makedirs(fd_ply, exist_ok=True)
        ply, img_msk, img = mob.get_moving_object_ply_by_mask(fn_img_moving, fn_img_average, fn_pcd)
        np.save(fn_bboxes,mob.bboxes_ret)
        open3d.io.write_point_cloud(fn_ply, ply)
        print(j, fn_ply)
        image_diff_fp = f'{fd_ply}/moving_detection_mask_{merge_id}.npy'
        np.save(image_diff_fp, img_msk[:, :, 0])

        image_diff_fp = f'{fd_ply}/moving_detection_mask_{merge_id}.png'
        img_msk = img_msk * 255
        cv2.imwrite(image_diff_fp, img_msk)

        image_diff_fp = f'{fd_ply}/image_org_{merge_id}.png'
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_diff_fp, img_rgb)
#from scipy.signal import medfilt2d as medianBlur
def remove_noise_mask(img,ksize):
    # fn='C:/00_work/05_src/data/20201124/202011161_sample/ply/cam0/20201116133925_495618/moving_detection_mask_mergeid_0047.png'
    # img=cv2.imread(fn)
    # ksize = 11
    # 中央値フィルタ
    img=np.float32(img)
    img_mask = cv2.medianBlur(img, ksize)
    return img


def check_fuchi(img, y, x,size):

    #point is black
    if np.any((img[y,x,0] == 0) & (img[y,x,1] == 0) & (img[y,x,2] == 0)):
        return False

    #search size
    # size = 15

    #image size
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    #print('height : ', h)
    #print('width  : ', w)
    #print('channel: ', c)

    #search area
    x0 = (x - size) if x > size else 0
    y0 = (y - size) if y > size else 0
    x1 = (x + size + 1) if (x + size + 1) < w else w
    y1 = (y + size + 1) if (y + size + 1) < h else h
    #print(x0, y0, x1, y1)

    img_check = img[y0:y1, x0:x1]
    #print(img_check)

    #search area including black
    return np.any((img_check[:,:,0] == 0) & (img_check[:,:,1] == 0) & (img_check[:,:,2] == 0))
def fuchi_deal_msk(msk_img,fuchi_size):
    if fuchi_size==0:
        return msk_img
    fuchi_msk=np.zeros(msk_img.shape)
    for x, y in itertools.product(
            [a for a in range(msk_img.shape[0])],
            [a for a in range(msk_img.shape[1])]):
        chk=check_fuchi(msk_img, x, y,fuchi_size)
        if chk:
            fuchi_msk[x,y,:]=0
        else:
            fuchi_msk[x,y,:]=msk_img[x,y,0]

    return fuchi_msk
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
if __name__ == "__main__":
    basepath='D:/02_AIPJ/004_ISB/20210113/20210113/data'
    # basepath='/home/user1/yu_develop/20201202_shiyoko_6f'
    #basepath='C:/00_work/05_src/data/20201202_shiyoko_6f'
    
    path_category_index = 'dt/category_index.npy'
    mod_path_seg = 'dt/lite-model_deeplabv3-xception65-ade20k_1_default_2.tflite'
    mod_path_obj = "https://tfhub.dev/tensorflow/efficientdet/d4/1"    
    mob=moving_object_merge(mod_path_obj, path_category_index, mod_path_seg,flg='AI')
    fuchi_sizes =[0]
    
    #'d2_a15_h1','d2_a30_h1','d2_a45_h1',
    dt_ids=['d5_c','d5_l','d5_r',
            'd8_c','d8_l','d8_r'
    ]
    for dt_id in dt_ids:
        for fuchi_size in fuchi_sizes:
           mob.fuchi_size=fuchi_size
           main_image_mask_ply_2021_0113(dt_id,fuchi_size,basepath,mob)
