import argparse
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2

# 定義程式執行時的輸入參數
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to video file. If not give, camera will be used")
ap.add_argument("-f", "--face", help="face detection network, 'openvino' (default), 'opencv' or 'haarcascade'")
ap.add_argument("-t", "--target", help="which device should be used for inference")
ap.add_argument("-r", "--resize", type=int, help="resize the input to what width")
args = vars(ap.parse_args())

# 為了簡化呼叫的參數，我們直接將模型設定檔名設定為變數
# OpenVINO 臉孔辨識網路 IR 模型檔案
openvino_face_network = "face-detection-adas-0001.xml"
openvino_face_model = "face-detection-adas-0001.bin"
# OpenVINO 年紀性別辨識網路 IR 模型檔案
openvino_age_gender_network = "age-gender-recognition-retail-0013.xml"
openvino_age_gender_model = "age-gender-recognition-retail-0013.bin"
# OpenCV 內建臉孔辨識網路檔案
opencv_face_model = "res10_300x300_ssd_iter_140000.caffemodel"
opencv_face_network = "deploy.prototxt"
cascade_scale = 1.2
cascade_neighbors = 6
minFaceSize = (30, 30)
# 定義模型推論結果 (機率) 所代表的性別，女生 (Female) 在前，男生 (Male) 在後。
GENDERS_FOR_OPENVINO = ['Female', 'Male']

# 分別建立各個辨識網路
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_net = cv2.dnn.readNet(openvino_face_network, openvino_face_model)
age_gender_net = cv2.dnn.readNet(openvino_age_gender_network, openvino_age_gender_model)
opencv_face_net = cv2.dnn.readNetFromCaffe(opencv_face_network, opencv_face_model)

# 是否使用 NCS/NCS2 進行推論
if args['target'] == 'vpu':
    face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    age_gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    opencv_face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
else:
    face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    age_gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    opencv_face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# 利用 OpenVINO 內建網路找出圖片中的臉孔
def get_faces_from_openvino(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
    face_net.setInput(blob)
    # 進行推論
    out = face_net.forward()
    faces = []
    for detection in out.reshape(-1, 7):
        confidence = float(detection[2])
        # 當可靠度大於 0.5 (50%) 時就當作已經找到臉孔
        if confidence > 0.5:
            xmin = max(int(detection[3] * w), 0)
            ymin = max(int(detection[4] * h), 0)
            xmax = min(int(detection[5] * w), w)
            ymax = min(int(detection[6] * h), h)
            if (xmax > xmin) and (ymax > ymin):
                faces.append([xmin, ymin, xmax-xmin, ymax-ymin])
    # 回傳找到的臉孔
    return faces


# 利用 OpenCV 內建網路找出圖片中的臉孔
def get_faces_from_opencv(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    opencv_face_net.setInput(blob)
    # 進行推論
    detections = opencv_face_net.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        detection = detections[0, 0, i]
        confidence = float(detection[2])
        # 當可靠度大於 0.5 (50%) 時就當作已經找到臉孔
        if confidence > 0.5:
            xmin = max(int(detection[3] * w), 0)
            ymin = max(int(detection[4] * h), 0)
            xmax = min(int(detection[5] * w), w)
            ymax = min(int(detection[6] * h), h)
            if (xmax > xmin) and (ymax > ymin):
                faces.append([xmin, ymin, xmax-xmin, ymax-ymin])
    # 回傳找到的臉孔
    return faces


# 利用 Haar Cascade 網路找出臉孔
def get_faces_from_haar_cascade(frame):
    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor=cascade_scale,
        minNeighbors=cascade_neighbors,
        minSize=minFaceSize,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # 回傳找到的臉孔
    return faces


# 用來判斷性別與年紀並加以標示的函式
def detect(frame, faces):
    # 針對圖片中的臉孔逐一處理
    for (x, y, w, h) in faces:
        # 將臉孔用綠色框線加以標示
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # "切下" 臉孔
        face = frame[y:y+h, x:x+w]
        # 將圖片轉換為 np.array，並作為模型的推論輸入
        blob = cv2.dnn.blobFromImage(face, size=(62, 62), ddepth=cv2.CV_8U)
        age_gender_net.setInput(blob)
        # 進行推論
        detections = age_gender_net.forwardAndRetrieve(['prob', 'age_conv3'])
        # 取得機率較高推論所代表的性別
        gender = GENDERS_FOR_OPENVINO[detections[0][0][0].argmax()]
        # 計算推論的年紀
        age = detections[1][0][0][0][0][0] * 100
        # 準備顯示用的文字，包含性別與年紀
        text = "gender = {}, age = {:.0f}".format(gender, age)
        # 根據不同性別採用不同顏色的顯示文字
        if gender == 'Male':
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # 回傳標示過的圖片
    return frame


# 根據 face 參數決定使用哪一個臉孔辨識網路
if args['face'] == 'haarcascade':
    face_detection_function = get_faces_from_haar_cascade
    print('[INFO] using Haar Cascade for face detection')
elif args['face'] == 'opencv':
    face_detection_function = get_faces_from_opencv
    print('[INFO] using OpenCV for face detection')
else:
    face_detection_function = get_faces_from_openvino
    print('[INFO] using OpenVINO for face detection')

# 根據 video 參數決定開始 USB 攝影機或讀取影片檔
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    vs = cv2.VideoCapture(args["video"])

# 開始 fps 統計
fps = FPS().start()
# 開始讀取 USB 攝影機或影片畫面
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    # 檢查是否已到影片的盡頭或因為其他原因無法讀取畫面
    if frame is None:
        break
    if args["resize"]:
        frame = imutils.resize(frame, width=args["resize"])
    # 取得臉孔資訊
    faces = face_detection_function(frame)
    # 判別臉孔的年紀與性別並加以標示
    frame = detect(frame, faces)
    # 顯示標示過後的影戲
    cv2.imshow("Frame", frame)
    # 當使用者按下 q 鍵時跳離迴圈
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # 更新 fps 統計
    fps.update()

# 停止 fps 統計並印出結果
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# 關閉全部視窗並釋放資源
cv2.destroyAllWindows()
if not args.get("video", False):
    vs.stop()
else:
    vs.release()
