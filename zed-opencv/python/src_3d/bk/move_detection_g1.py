import cv2
import numpy as np

from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
def fcntOpencv(img):
    blur = cv2.GaussianBlur(img, (3, 3), 2)

    # th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    # th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)

    # imgC, contours, hierarchy = cv2.findContours(th.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # imgC, contours, hierarchy = cv2.findContours(th.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # imgC, contours, hierarchy = cv2.findContours(th.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(th.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    imgC = cv2.drawContours(img.copy(), contours, -1, (255, 255, 255), 3)


    blur2 = cv2.GaussianBlur(imgC,(3,3),0)

    blur2 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    # imgC2, contours2, hierarchy2 = cv2.findContours(imgC.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours2, hierarchy2 = cv2.findContours(blur2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    imgC2 = cv2.drawContours(img.copy(), contours2, -1, (255,255,255), 3)

    return imgC, imgC2

def detect_edge2(image):
    # # convert to RGB
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # # convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # gray = cv2.medianBlur(gray, 3)
    # # create a binary thresholded image
    # _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # med = np.median(image)
    # image = cv2.Canny(image, med * 0.7, med * 1.3)

    med_val = np.median(image)
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    blurred_img = cv2.blur(image, ksize=(7, 7))
    image = cv2.Canny(image=blurred_img, threshold1=lower, threshold2=upper)

    frameDelta_o=np.expand_dims(image,axis=2)
    image=np.concatenate([frameDelta_o,frameDelta_o,frameDelta_o],axis=2)
    return image

def detect_edge(image):
    th_s = 70
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = img
    ret, thresh = cv2.threshold(gray, th_s, 255, cv2.THRESH_BINARY)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 7)
    # cv2.imwrite(out_path+"thresh.jpg", thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    eroded = cv2.erode(thresh, kernel)
    # cv2.imwrite(out_path+"eroded.jpg", eroded)
    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    return image
cap = cv2.VideoCapture(0)
avg = None
def find_edge(cnts,canvas):
    cnts = sorted(cnts, key=cv2.contourArea)
    print(len(cnts))
    if len(cnts)==0:
        return canvas,canvas
    cnt = cnts[-1]

    ## approx the contour, so the get the corner points
    arclen = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * arclen, True)
    frame1=cv2.drawContours(canvas, [cnt], -1, (255, 0, 0), 1, cv2.LINE_AA)
    frame2=cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 1, cv2.LINE_AA)
    return frame1,frame2
def detect_frame(frame):
    global avg
    th=8
    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 比較用のフレームを取得する
    if avg is None:
        avg = gray.copy().astype("float")
        return frame
    # 現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(gray, avg, 0.85)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # デルタ画像を閾値処理を行う
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    # 画像の閾値に輪郭線を入れる
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frameDelta[frameDelta >= th] = 255
    frameDelta[frameDelta < th] = 0

    def one2three(img):
        frameDelta_o=np.expand_dims(img,axis=2)
        frameDelta_n=np.concatenate([frameDelta_o,frameDelta_o,frameDelta_o],axis=2)
        return frameDelta_n

    frameDelta_n=one2three(frameDelta)
    frameDelta_n_c = frameDelta_n.copy()
    # frame1,frame2=find_edge(contours, canvas)
    frame1,frame2=fcntOpencv(frameDelta)
    frame1=one2three(frame1)
    frame2=one2three(frame2)

    frame_edge=detect_edge(frameDelta_n)



    frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    frame=np.hstack([frameDelta_n_c,frame])
    frame4=np.hstack([frame_edge,frame2])
    frame=np.vstack([frame,frame4])

    return frame


while True:
    # 1フレームずつ取得する。
    ret, frame = cap.read()
    if not ret:
        break
    frame=detect_frame(frame)
    # frame=detect_frame(frame)
    # 結果を出力
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()