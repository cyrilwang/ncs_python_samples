# 使用訓練好的 Caffe Model 來判斷目錄下所有圖片裡人物的年紀
# Caffe Model 來自於 https://github.com/GilLevi/AgeGenderDeepLearning
# 載入使用到的套件
import argparse
import glob
import cv2

# 定義程式執行時的輸入參數
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to the prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="help to the caffe model file")
ap.add_argument("--path", help="path to image file")
args = vars(ap.parse_args())

# 定義模型訓練圖片的平均值
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# 定義模型推論結果 (機率) 所代表的年紀區間。
AGES_FOR_CAFFE = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
# 根據訓練好的模型建立推論用的 CNN 網路
caffe_age_net = cv2.dnn.readNet(args["model"], args["prototxt"])


# 用來判斷年紀並加以標示的函式
def detect_age_by_caffe(face):
    # 將圖片轉換為 np.array，並作為模型的推論輸入
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES)
    caffe_age_net.setInput(blob)
    # 進行推論
    age_detections = caffe_age_net.forward()
    # 取得機率較高推論所代表的性別
    age = AGES_FOR_CAFFE [age_detections[0].argmax()]
    # 準備顯示用的文字，包含性別與該性別的機率
    text = "age : {}, conf = {:.4f}".format(age, age_detections[0].max())
    cv2.putText(face, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    # 回傳標示過的圖片
    return face


# 讀取目錄下的所有檔案 (圖片)，取得檔名後讀入影片、呼叫判斷性別的函式，並將結果顯示出來
for file in glob.glob(args["path"] + '/*'):
    filename = str(file)
    image = cv2.imread(filename)
    caffe_image = detect_age_by_caffe(image.copy())
    cv2.imshow("Caffe", caffe_image)
    cv2.waitKey(0)
