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
path='dt/20201116/image_average/'
# for id in ['image_average_cam0','image_average_cam1','image_average_cam2']:
#   img=np.load(path+f'{id}.npy')
#   cv2.imwrite(path+f'{id}.png',img)
id='image_average_cam0'
image_avg=np.load(path+f'{id}.npy')
image_fn='dt/20201116133240_723445/image.npy'
img,img4=load_image_from_npy(image_fn)
image_avg_n=image_avg*(np.mean(img)/np.mean(image_avg))
image_out=img-image_avg_n
th=np.mean(image_out)
image_out[image_out>th]=255
image_out[image_out<=th]=0
cv2.imwrite(path+f'20201116133240_723445_{id}.png',image_out)