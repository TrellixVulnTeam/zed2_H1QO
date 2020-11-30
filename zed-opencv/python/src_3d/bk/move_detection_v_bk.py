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
def frame_sub(img1, img2, img3, th):
    # フレームの絶対差分
    diff1 = cv2.absdiff(img1, img2)
    diff2 = cv2.absdiff(img2, img3)

    # 2つの差分画像の論理積
    diff = cv2.bitwise_and(diff1, diff2)

    # 二値化処理
    diff[diff < th] = 0
    diff[diff >= th] = 255

    # メディアンフィルタ処理（ゴマ塩ノイズ除去）
    mask = cv2.medianBlur(diff, 3)

    return  mask
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
    cv2.accumulateWeighted(gray, avg, 0.6)
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
    frameDelta_n = cv2.medianBlur(frameDelta_n, 3)
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

def main():
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


def main2():
    # カメラのキャプチャ
    cap = cv2.VideoCapture(0)
    kernel = np.ones((50, 50), np.float32) / 2500

    # フレームを3枚取得してグレースケール変換
    frame1 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame2 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)
    frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

    flg_detect = 0
    cx_ary = []
    cy_ary = []
    temp_text = ""
    dsp_pos = 120
    dsp_col_r = 0
    dsp_col_b = 0

    while (cap.isOpened()):
        frame0 = cap.read()[1]

        # フレーム間差分を計算
        mask = frame_sub(frame1, frame2, frame3, th=30)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        ret, thresh = cv2.threshold(closing, 127, 125, 0)

        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnt_contour_selected = 0
        idx_contour = -1

        for i in range(0, len(contours)):
            area_contours = cv2.contourArea(contours[i])
            if area_contours > 9000:
                cnt_contour_selected += 1
                idx_contour = i

        if cnt_contour_selected == 1:
            M = cv2.moments(contours[idx_contour])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            x, y, w, h = cv2.boundingRect(contours[idx_contour])
            cv2.rectangle(frame0, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cx_ary.append(cx)
            cy_ary.append(cy)

            flg_detect = 1

        else:

            if (flg_detect == 1) & (len(cx_ary) > 0):
                ck_dir = 0

                for i in range(0, len(cx_ary)):
                    if i > 0:
                        dif_cx = cx_ary[i] - cx_ary[i - 1]
                        if dif_cx > 0:
                            ck_dir += 1

                        if dif_cx < 0:
                            ck_dir -= 1

                if (len(cx_ary) - 1) == abs(ck_dir):
                    if ck_dir > 0:
                        temp_text = "To Right"
                        dsp_pos = 120
                        dsp_col_r = 255
                        dsp_col_b = 0
                    else:
                        temp_text = "To Left"
                        dsp_pos = 420
                        dsp_col_r = 0
                        dsp_col_b = 255

            cx_ary = []
            cy_ary = []

            flg_detect = 0

        cv2.putText(frame0, temp_text, (30, dsp_pos), cv2.FONT_HERSHEY_SIMPLEX, 4, (dsp_col_b, 0, dsp_col_r), 15,
                    cv2.LINE_AA)

        # 結果を表示
        # cv2.imshow("Frame0", frame0)
        cv2.imshow("Mask", closing)

        # 3枚のフレームを更新
        frame1 = frame2
        frame2 = frame3
        frame3 = cv2.cvtColor(cap.read()[1], cv2.COLOR_RGB2GRAY)

        # qキー, ESC が押されたら途中終了
        key = cv2.waitKey(1)
        if (key & 0xFF == ord('q')) or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


cv2.namedWindow("Frame0", cv2.WINDOW_NORMAL)

if __name__ == "__main__":
    main()