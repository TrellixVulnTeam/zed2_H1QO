import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('img_out.jpg')
rows,cols,ch = img.shape
print (img.shape)

#pcd 変換
pts_src = np.float32([[0,0],[cols,0],[0,177],[379,166]])
pts_dst = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])

M = cv2.getPerspectiveTransform(pts_src,pts_dst)
print(M)
dst = cv2.warpPerspective(img,M,(cols,rows))
# cv2.imwrite("img_out.jpg", dst)

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()