# def npy_2_img(img_in):
import numpy as np
import cv2
fn='C:/00_work/05_src/data/20201015155835/image.npy'
fnimg='C:/00_work/05_src/data/20201015155835/image.jpg'
img = np.load(fn)
img_save=img[:,:,:3]
cv2.imwrite(fnimg,img_save)