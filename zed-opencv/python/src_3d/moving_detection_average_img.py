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


'''
import cv2,numpy as np
import pandas as pd
from PIL import Image
from object_detection_draw_v5 import object_detection_from_npy_all_file,load_image_from_npy
from move_detection_v2 import detect_frame
if __name__ == "__main__":
  path = 'C:/00_work/05_src/data/fromito/data/fromWATA'
  patho = 'C:/00_work2/05_src/data/fromito/data/fromWATA'
  path_category_index = 'dt/category_index.npy'
  # "https://tfhub.dev/tensorflow/centernet/hourglass_1024x1024_kpts/1"
  # "https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1"

  # category_index = np.load(path_category_index, allow_pickle='TRUE').item()
  path_mod = "https://tfhub.dev/tensorflow/efficientdet/d4/1"
  human_dt_lst=object_detection_from_npy_all_file(path,patho,path_mod,path_category_index)
  pd.DataFrame(human_dt_lst).to_csv(f'{patho}/human_detect.csv',header=None,index=None)
  human_dt_lst=pd.read_csv(f'{patho}/human_detect.csv',header=None)
  image_sum=None
  for i,dt in enumerate(human_dt_lst.values.tolist()):
      path_img,flg=dt
      if i==0:
          img, color_org = load_image_from_npy(path_img)
          image_sum=np.zeros(img.shape)
      if flg==1:
          img, color_org = load_image_from_npy(path_img)
          image_sum=image_sum+img
  image_average=image_sum/(i+1)
  image_average = np.array(image_average, dtype=np.uint8)
  image_average_fp=f'{patho}/image_average.npy'
  np.save(image_average_fp, image_average)

  image_average = np.load(image_average_fp)
  for i,dt in enumerate(human_dt_lst.values.tolist()):
      path_img,flg=dt
      if flg==1:
          pathoful=path_img.replace(path.replace('/','\\'),patho)
          img, color_org = load_image_from_npy(path_img)
          image_diff=img-image_average
          image_diff_fp=pathoful.replace('image.npy','moving_detection_diff.png')
          # image_diff_fp=f'{pathoful}/moving_detection_diff.png'
          cv2.imwrite(image_diff_fp,image_diff)

          gray = cv2.cvtColor(image_average, cv2.COLOR_BGR2GRAY)
          avg = gray.copy().astype("float")
          img_detect, img_detect_concat, avg = detect_frame(img, avg)
          pil_img = Image.fromarray(img_detect.astype(np.uint8))

          image_diff_fp=pathoful.replace('image.npy','moving_detection.png')
          pil_img.save(image_diff_fp)