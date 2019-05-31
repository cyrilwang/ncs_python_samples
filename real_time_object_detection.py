# 載入所需函式庫
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# 定義程式所需參數
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True, help="filename of caffe network configuration")
ap.add_argument("-m", "--model", required=True, help="filename of trained caffe model")
ap.add_argument("-v", "--video", help="filename of the video (optional)")
args = vars(ap.parse_args())

# 如果沒有指定 video 參數就將變數 use_camera 設定為真，表示使用攝影機
use_camera = False
if not args.get("video", False):
    use_camera = True

# 定義 MobileNet SSD 訓練過的物件種類
CLASSES = ("background",
           "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# 載入 DNN 模型
net = cv2.dnn.readNetFromCaffe(args["config"], args["model"])

# 定義使用 NCS2 裝置
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# 根據參數決定載入影片或開啟 USB 攝影機
# 如果使用樹莓派的攝影機模組請把 src=0 改為 usePiCamera=True
if use_camera:
    print("開啟攝影機...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    vs = cv2.VideoCapture(args["video"])

# 抓取 frame 並加以處理
while True:
    frame = vs.read()
    frame = frame if use_camera else frame[1]
    if frame is None:
        break
    frame = imutils.resize(frame, width=400)

    # 將圖片轉換成 4 維的 blob 陣列
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # 進行推論 (inference)
    net.setInput(blob)
    detections = net.forward()

    # 處理預測結果
    for detection in detections.reshape(-1, 7):
        index = int(detection[1])
        confidence = float(detection[2])

        # 如果可能性 > 0.3 則顯示預測結果
        if confidence > 0.3:
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])
            label = "{}: {:.2f}%".format(CLASSES[index], confidence * 100)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), COLORS[index], 2)
            y = ymin - 15 if ymin > 30 else ymin + 15
            cv2.putText(frame, label, (xmin, ymin -15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[index], 2)

    # 顯示 frame
    cv2.imshow("Frame", frame)

    # 如果按下 q 鍵就中斷迴圈
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 釋放資源
cv2.destroyAllWindows()
if use_camera:
    vs.stop()
else:
    vs.release()
