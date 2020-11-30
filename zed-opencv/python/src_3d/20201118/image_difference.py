import cv2
import numpy as np
from object_detection_draw_v5 import object_detection_from_npy_all_file,load_image_from_npy
# load images
path='C:/00_work2/05_src/data/fromito/data/'
leftImage=f'{path}/image_average_cam0.png'
rightImage=f'{path}/image.npy'
image1 = cv2.imread(leftImage)
image2,_ =load_image_from_npy(rightImage)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
cv2.imwrite(f'{path}/image2.png', image2)
#https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
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
    return image1,image2,difference
# store images[
image1,image2,difference=compute_difference_cv2(image1,image2,color=[255, 0, 0])
cv2.imwrite(f'{path}/diffOverImage1.png', image1)
cv2.imwrite(f'{path}/diffOverImage2.png', image2)
cv2.imwrite(f'{path}/diff.png', difference)