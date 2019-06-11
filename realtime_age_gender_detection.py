# 使用 OpenCV 內建的臉孔辨識網路抓取圖片中的臉孔，之後利用 OpenVINO IR 模型來判斷該臉孔的性別與年齡
# 載入使用到的套件
import argparse
from imutils.video import VideoStream
import time
import cv2

# 定義程式執行時的輸入參數
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to video file. If not give, camera will be used")
ap.add_argument("-t", "--target", help="which device should be used for inference")
args = vars(ap.parse_args())

# 為了簡化呼叫的參數，我們直接將模型設定檔名設定為變數
# OpenVINO 年紀性別辨識網路 IR 模型
openvino_age_gender_network = "age-gender-recognition-retail-0013.xml"
openvino_age_gender_model = "age-gender-recognition-retail-0013.bin"
# OpenCV 內建臉孔辨識網路
opencv_face_model = "res10_300x300_ssd_iter_140000.caffemodel"
opencv_face_network = "deploy.prototxt"
# 定義模型推論結果 (機率) 所代表的性別，女生 (Female) 在前，男生 (Male) 在後。
GENDERS_FOR_OPENVINO = ['Female', 'Male']

# 分別建立臉孔與年紀/性別辨識網路
age_gender_net = cv2.dnn.readNet(openvino_age_gender_network, openvino_age_gender_model)
opencv_face_net = cv2.dnn.readNetFromCaffe(opencv_face_network, opencv_face_model)

# 是否使用 NCS/NCS2 進行推論
if args['target'] == 'vpu':
    age_gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    opencv_face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
else:
    age_gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    opencv_face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# 找出圖片中的臉孔
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


# 根據 video 參數決定開始 USB 攝影機或讀取影片檔
if not args.get("video", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    vs = cv2.VideoCapture(args["video"])

# 開始讀取 USB 攝影機或影片畫面
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    # 檢查是否已到影片的盡頭或因為其他原因無法讀取畫面
    if frame is None:
        break
    # 取得臉孔資訊
    faces = get_faces_from_opencv(frame)
    # 判別臉孔的年紀與性別並加以標示
    frame = detect(frame, faces)
    # 顯示標示過後的影戲
    cv2.imshow("Frame", frame)
    # 當使用者按下 q 鍵時跳離迴圈
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 關閉全部視窗並釋放資源
cv2.destroyAllWindows()
if not args.get("video", False):
    vs.stop()
else:
    vs.release()
