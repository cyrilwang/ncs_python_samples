# 使用訓練好的 OpenVINO IR 模型來偵測目錄下所有圖片裡的臉孔，並進行年紀與性別的推論
# 載入所需函式庫
import argparse
import glob
import cv2

# 定義程式所需參數
ap = argparse.ArgumentParser()
ap.add_argument("-x1", "--xml1", required=True, help="filename of face network configuration")
ap.add_argument("-b1", "--bin1", required=True, help="filename of trained face model")
ap.add_argument("-x2", "--xml2", required=True, help="filename of age network configuration")
ap.add_argument("-b2", "--bin2", required=True, help="filename of trained age model")
ap.add_argument("-p", "--path", help="path to image files")
ap.add_argument("-t", "--target", help="which device should be used for inference")
args = vars(ap.parse_args())

## 根據訓練好的模型建立推論用的 CNN 網路
face_net = cv2.dnn.readNet(args["xml1"], args["bin1"])
# 根據訓練好的模型建立推論用的 CNN 網路
openvino_net = cv2.dnn.readNet(args["xml2"], args["bin2"])
# 定義模型推論結果 (機率) 所代表的性別，女生 (Female) 在前，男生 (Male) 在後。
GENDERS_FOR_OPENVINO = ['Female', 'Male']

# 是否使用 NCS/NCS2 進行推論
if args['target'] == 'vpu':
    face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    openvino_net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
else:
    face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    openvino_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# 用來判斷性別與年紀並加以標示的函式
def detect_age_and_gender_by_openvino(frame):
    # 找出圖片中的臉孔
    blob = cv2.dnn.blobFromImage(frame, size=(672, 384), ddepth=cv2.CV_8U)
    face_net.setInput(blob)
    out = face_net.forward()
    faces = []
    for detection in out.reshape(-1, 7):
        confidence = float(detection[2])
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])
        if confidence > 0.5:
            faces.append([xmin, ymin, xmax-xmin, ymax-ymin])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0))

    # 針對圖片中的臉孔逐一處理
    for (x, y, w, h) in faces:
        # 將臉孔用綠色框線加以標示
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # "切下" 臉孔
        face = frame[y:y+h, x:x+w]
        # 將圖片轉換為 np.array，並作為模型的推論輸入
        blob = cv2.dnn.blobFromImage(face, size=(62, 62), ddepth=cv2.CV_8U)
        openvino_net.setInput(blob)
        # 進行推論
        detections = openvino_net.forwardAndRetrieve(['prob', 'age_conv3'])
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
        cv2.putText(frame, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # 回傳標示過的圖片
    return frame


# 讀取目錄下的所有檔案 (圖片)，取得檔名後讀入圖片、呼叫判斷性別與年紀的函式，並將結果顯示出來
for file in glob.glob(args["path"] + '/*'):
    filename = str(file)
    image = cv2.imread(filename)
    openvino_image = detect_age_and_gender_by_openvino(image)
    cv2.imshow("OpenVINO", openvino_image)
    cv2.waitKey(0)
