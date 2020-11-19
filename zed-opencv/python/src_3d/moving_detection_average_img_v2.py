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
import cv2,numpy as np
import pandas as pd
from PIL import Image
from object_detection_draw_v5 import object_detection_from_npy_all_file,load_image_from_npy
from move_detection_v2 import detect_frame
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
def get_human_lst_each_cam():
    path = '/media/user1/Data/data/20201116/'
    patho = '/media/user1/Data/data/Data_out_20201116/'
    camlst=['cam0','cam1','cam2']
    df_detection = pd.read_csv(f'{patho}/human_detect.csv', header=None)
    cnt=len(camlst)
    for i,cam in enumerate(camlst):
        df=df_detection[df_detection[0].str.contains(cam)]
        image_average=get_average_image(df)
        image_average_fp = f'{patho}/image_average_{cam}.npy'
        np.save(image_average_fp, image_average)
        df.columns = ['fn', 'flg']
        df = df.sort_values('fn')
        df.to_csv(f'{patho}/human_detect_{cam}.csv', header=None, index=None)

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
    get_human_lst_each_cam()
    get_human_lst_each_diff()
    pass
