import cv2
import numpy as np
#https://qiita.com/K_M95/items/4eed79a7da6b3dafa96d
import matplotlib.pyplot as plt
# filepath = "vtest.avi"
# cap = cv2.VideoCapture(filepath)
# Webカメラを使うときはこちら
cap = cv2.VideoCapture(0)
avg = None
def detect_frame(frame):
    global avg
    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 比較用のフレームを取得する
    if avg is None:
        avg = gray.copy().astype("float")
        return frame
    # 現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(gray, avg, 0.8)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # デルタ画像を閾値処理を行う
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    # 画像の閾値に輪郭線を入れる
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    frameDelta[frameDelta >= 8] = 255
    frameDelta[frameDelta < 8] = 0

    frameDelta_o=np.expand_dims(frameDelta,axis=2)
    frameDelta_n=np.concatenate([frameDelta_o,frameDelta_o,frameDelta_o],axis=2)

    frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    frame=np.hstack([frameDelta_n,frame])

    return frame


while True:
    # 1フレームずつ取得する。
    ret, frame = cap.read()
    if not ret:
        break
    frame=detect_frame(frame)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()