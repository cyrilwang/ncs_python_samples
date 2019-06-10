# 使用訓練好的 OpenVINO IR 模型來判斷目錄下所有圖片裡人物的性別與年齡
# 載入使用到的套件
import argparse
import glob
import cv2

# 定義程式執行時的輸入參數
ap = argparse.ArgumentParser()
ap.add_argument("-x", "--xml", required=True,
                help="path to the IR xml file")
ap.add_argument("-b", "--bin", required=True,
                help="path to the IR bin file")
ap.add_argument("-p", "--path", help="path to image files")
ap.add_argument("-t", "--target", help="which device should be used for inference")
args = vars(ap.parse_args())

GENDERS_FOR_OPENVINO = ['Female', 'Male']

openvino_net = cv2.dnn.readNet(args["xml"], args["bin"])
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 是否使用 NCS/NCS2 進行推論
if args['target'] == 'vpu':
    openvino_net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
else:
    openvino_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def detect_age_and_gender_by_openvino(frame):
    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = frame[y:y+h, x:x+w]
        blob = cv2.dnn.blobFromImage(face, size=(62, 62), ddepth=cv2.CV_8U)
        openvino_net.setInput(blob)
        gender_detections = openvino_net.forwardAndRetrieve(['prob', 'age_conv3'])
        gender = GENDERS_FOR_OPENVINO[gender_detections[0][0][0].argmax()]
        age = gender_detections[1][0][0][0][0][0]*100
        text = "gender = {}, age = {:.0f}".format(gender, age)
        if gender == 'Male':
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


for file in glob.glob(args["path"] + '/*'):
    filename = str(file)
    image = cv2.imread(filename)
    openvino_image = detect_age_and_gender_by_openvino(image.copy())
    cv2.imshow("OpenVINO", openvino_image)
    cv2.waitKey(0)
