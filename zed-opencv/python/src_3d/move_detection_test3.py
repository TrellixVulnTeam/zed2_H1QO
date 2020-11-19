import numpy as np
import cv2

def convert_bgra2rgba(img):
  # ZED2 numpy color is BGRA.
  rgba = np.zeros(img.shape).astype(np.uint8)
  rgba[:,:,0] = img[:,:,2] # R
  rgba[:,:,1] = img[:,:,1] # G
  rgba[:,:,2] = img[:,:,0] # B
  rgba[:,:,3] = img[:,:,3] # A
  return rgba
def load_image_from_npy(p, convert=True):
  color_org = np.load(p)
  color = convert_bgra2rgba(color_org) if convert else color_org
  # pil_img = Image.fromarray(color.astype(np.uint8))
  # pil_img.save(f'{p}/image.png')
  return color[:,:,:3],color_org
path='dt/'
average_cam_id='image_average_cam0'
image_avg=np.load(path+f'{average_cam_id}.npy')
image_fn='dt/20201116133240_723445/image.npy'
img,img4=load_image_from_npy(image_fn)
image_avg_mean=np.mean(image_avg)
r=0.05
i=1
image_out=img.copy()
th_h=(1+i*r)*image_avg_mean
th_l=(1-i*r)*image_avg_mean
image_out[(image_out>=th_l) & (image_out<=th_h)]=255
image_out[(image_out<th_l) | (image_out>th_h)]=0
print(np.max(image_out),th_l)
cv2.imwrite(path+f'test/20201116133240_723445_mean_%03d.png'%(i-1),image_out)