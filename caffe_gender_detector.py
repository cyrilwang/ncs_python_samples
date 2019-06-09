# 使用訓練好的 Caffe Model 來判斷目錄下所有圖片裡人物的性別
# Caffe Model 來自於 https://github.com/GilLevi/AgeGenderDeepLearning
# 載入使用到的套件
import argparse
import glob
import cv2

# 定義程式執行時的輸入參數
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to the prototxt file")
ap.add_argument("-m", "--model", required=True, help="help to the caffe model file")
ap.add_argument("--path", help="path to image file")
args = vars(ap.parse_args())

# 定義模型訓練圖片的平均值
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# 定義模型推論結果 (機率) 所代表的性別，男生 (Male) 在前，女生 (Female) 在後。
GENDERS_FOR_CAFFE = ['Male', 'Female']
# 根據訓練好的模型建立推論用的 CNN 網路
caffe_gender_net = cv2.dnn.readNet(args["model"], args["prototxt"])
if args['target'] == 'vpu':
    caffe_gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
else:
    caffe_gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# 用來判斷性別並加以標示的函式
def detect_gender_by_caffe(face):
    # 將圖片轉換為 np.array，並作為模型的推論輸入
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES)
    caffe_gender_net.setInput(blob)
    # 進行推論
    gender_detections = caffe_gender_net.forward()
    # 取得機率較高推論所代表的性別
    gender = GENDERS_FOR_CAFFE[gender_detections[0].argmax()]
    # 準備顯示用的文字，包含性別與該性別的機率
    text = "gender : {}, conf = {:.4f}".format(gender, gender_detections[0].max())
    # 根據不同性別採用不同顏色的顯示文字
    if gender == 'Male':
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)
    cv2.putText(face, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    # 回傳標示過的圖片
    return face


# 讀取目錄下的所有檔案 (圖片)，取得檔名後讀入影片、呼叫判斷性別的函式，並將結果顯示出來
for file in glob.glob(args["path"] + '/*'):
    filename = str(file)
    image = cv2.imread(filename)
    caffe_image = detect_gender_by_caffe(image.copy())
    cv2.imshow("Caffe", caffe_image)
    cv2.waitKey(0)
