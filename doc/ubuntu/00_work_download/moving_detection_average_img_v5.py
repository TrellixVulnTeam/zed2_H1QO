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
from object_detection_draw_v5 import object_detection_from_npy_all_file,load_image_from_npy
from move_detection_v2 import detect_frame
import os
from skimage.measure import compare_ssim

#image1,image2,difference=compute_difference_cv2(image1,image2,color=[255, 0, 0])
def compute_difference_cv2(image1,image2,color=[0, 0, 255]):
    # compute difference
    difference = cv2.subtract(image1, image2)
    # color the mask red
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    difference[mask != 255] = color

    # add the red mask to the images to make the differences obvious
    image1[mask != 255] = color
    image2[mask != 255] = color
    return image2,difference
def compute_difference_ssim(before,after,color=[0, 0, 255]):
    # Convert images to grayscale
    before = cv2.GaussianBlur(before, (3, 3), 0)
    after = cv2.GaussianBlur(after, (3, 3), 0)
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = compare_ssim(before_gray, after_gray, full=True)
    print("Image similarity", score)

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
    path_category_index = 'dt/category_index.npy'
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

        image_average = cv2.GaussianBlur(image_average, (5, 5), 0)
        image_average_gray = cv2.cvtColor(image_average, cv2.COLOR_BGR2GRAY)
        cnt = len(df_human.values.tolist())
        for i, dt in enumerate(df_human.values.tolist()):
            path_img, flg = dt
            print(i, i / cnt, path_img)
            path_img = path_img.replace('/media/user1/Data/data/20201116/',
                                        '/home/user1/yu_develop/202011161_sample/image/')
            pathoful = path_img.replace(path, patho)
            os.makedirs(pathoful, exist_ok=True)
            img, color_org = load_image_from_npy(path_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # image_diff, image_diff2 = get_diff_of_two_image_2(img_gray, image_average_gray)
            # image1, image_diff2, image_diff = compute_difference_cv2(image_average_gray, img, color=[255, 0, 0])
            image1, image_diff2, image_diff = func(image_average_gray, img, color=[255, 0, 0])
            image_diff_fp = pathoful.replace('image.npy', f'{ext}_moving_detection_diff.png')
            cv2.imwrite(image_diff_fp, image_diff)

            image_diff_fp = pathoful.replace('image.npy', f'{ext}_moving_detection_diff_2.png')
            cv2.imwrite(image_diff_fp, image_diff2)

            gray = cv2.cvtColor(image_average, cv2.COLOR_BGR2GRAY)
            avg = gray.copy().astype("float")
            img_detect, img_detect_concat, avg = detect_frame(img, avg)
            pil_img = Image.fromarray(img_detect.astype(np.uint8))

            image_diff_fp = pathoful.replace('image.npy', f'{ext}_moving_detection.png')
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

if __name__ == "__main__":
    # get_human_lst_each_cam()
    # get_human_lst_each_diff()
    #get_human_lst_each_cam_2()
    # get_human_lst_each_diff_2()
    path = '/home/user1/yu_develop/202011161_sample/image/'
    patho = '/home/user1/yu_develop/202011161_sample/out/'
    get_human_lst_each_diff_fun(path, patho, func=compute_difference_cv2, ext='cv2')
    get_human_lst_each_diff_fun(path, patho, func=compute_difference_ssim, ext='ssim')
    pass
