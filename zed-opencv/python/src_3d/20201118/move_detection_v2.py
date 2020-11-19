import cv2
import numpy as np
import os
from PIL import Image
def detect_frame(frame,avg):
    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 比較用のフレームを取得する
    if avg is None:
        avg = gray.copy().astype("float")
        detect_img=np.zeros(frame.shape)
        return detect_img,frame,avg
    # 現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # デルタ画像を閾値処理を行う
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    # 画像の閾値に輪郭線を入れる
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frameDelta[frameDelta >= 8] = 255
    frameDelta[frameDelta < 8] = 0

    frameDelta_o=np.expand_dims(frameDelta,axis=2)
    frameDelta_n=np.concatenate([frameDelta_o,frameDelta_o,frameDelta_o],axis=2)
    # frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    # frame=np.hstack([frameDelta_n,frame])

    avg = gray.copy().astype("float")
    return frameDelta_n,frame,avg
def get_absolute_file_paths(directory,ext=".npy",fn='image.npy'):
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

def convert_bgra2rgba(img):
  # ZED2 numpy color is BGRA.
  rgba = np.zeros(img.shape).astype(np.uint8)
  rgba[:,:,0] = img[:,:,2] # R
  rgba[:,:,1] = img[:,:,1] # G
  rgba[:,:,2] = img[:,:,0] # B
  rgba[:,:,3] = img[:,:,3] # A
  return rgba
def load_image_from_npy(p, convert=True):
  color = np.load(p)
  color = convert_bgra2rgba(color) if convert else color
  # pil_img = Image.fromarray(color.astype(np.uint8))
  # pil_img.save(f'{p}/image.png')
  return color[:,:,:3]
def moving_detection_from_images(path):
    avg = None
    fils_list,fn_list,dir_list=get_absolute_file_paths(path, ext=".npy", fn='image.npy')
    fpdir=np.concatenate([np.expand_dims(fils_list,axis=1),np.expand_dims(dir_list,axis=1)],axis=1)
    l = list(fpdir)
    l.sort(key=lambda x: x[0])
    fpdir = np.array(l)
    cnt=len(fils_list)
    i=0
    for fp,fdir in fpdir:
        img=load_image_from_npy(fp)
        img_detect,img_detect_concat, avg = detect_frame(img, avg)
        pil_img = Image.fromarray(img_detect.astype(np.uint8))
        pil_img.save(f'{fdir}/moving_detection.png')
        # cv2.imwrite(f'{fdir}/moving_detection2.png',img_detect)
        pil_img = Image.fromarray(img_detect_concat.astype(np.uint8))
        pil_img.save(f'{fdir}/image.png')
        print(i,cnt,fp)
        i=i+1
def main_video():

    cap = cv2.VideoCapture(0)
    avg = None
    while True:
        # 1フレームずつ取得する。
        ret, frame = cap.read()
        if not ret:
            break
        frameDelta_n,frame,avg=detect_frame(frame,avg)

        cv2.imshow("Frame", frameDelta_n)
        key = cv2.waitKey(30)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main_video()
    p='C:/00_work/05_src/data/fromito/data/fromWATA'
    moving_detection_from_images(p)
