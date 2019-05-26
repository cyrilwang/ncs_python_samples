# 載入所需函式庫
import argparse
import cv2

# 定義程式所需參數
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True, help="filename of network configuration")
ap.add_argument("-m", "--model", required=True, help="filename of trained model")
ap.add_argument("-i", "--image", required=True, help="filename of input image")
args = vars(ap.parse_args())

# 載入 DNN 模型
net = cv2.dnn.readNet(args["config"], args["model"])

# 定義使用 NCS2 裝置
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# 載入圖片
frame = cv2.imread(args["image"])

# 將圖片進行轉換後推論
blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
net.setInput(blob)
out = net.forward()

# 標示出偵測到的臉孔
for detection in out.reshape(-1, 7):
    confidence = float(detection[2])
    xmin = int(detection[3] * frame.shape[1])
    ymin = int(detection[4] * frame.shape[0])
    xmax = int(detection[5] * frame.shape[1])
    ymax = int(detection[6] * frame.shape[0])
    if confidence > 0.5:
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))

# 顯示標示過後的圖片
cv2.imshow("Frame", frame)
cv2.waitKey(0)
