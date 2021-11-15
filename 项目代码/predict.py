
import sys
from keras.utils.generic_utils import CustomObjectScope
from keras.layers import DepthwiseConv2D
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5 import uic
import PyQt5.QtGui
import numpy as np
import cv2
from keras import backend as K
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.applications.mobilenet import preprocess_input
from PIL import Image
import tensorflow as tf2
tf = tf2.compat.v1# 通过TensorFlow2.x调用1.x中的api



# 自定义激活函数relu6
def relu6(x):
    return K.relu(x, max_value=6)


# 读取CNN模型
def get_model():
    # 声明model为全局变量，因为后面模型装配还会用到
    global model
    with CustomObjectScope({'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D}):
        # 读取训练好的网络模型
        model = load_model('SLD_CNN.h5')
    print(" ---> Model loaded!")


# 数据预处理
def preprocess_image(image, target_size):
    # 修改图片格式
    if image.mode != "L":
        image = image.convert("L")
    # 格式化尺寸
    image = image.resize(target_size)
    # numpy维度处理
    image_arr = img_to_array(image)
    image_arr = np.expand_dims(image_arr, axis=0)
    image_arr = preprocess_input(image_arr)
    return image_arr, image


# 制作标签
def get_label(index):
    correct_label = [9, 0, 7, 6, 1, 8, 4, 3, 2, 5]
    return correct_label[index]


# 预测函数
def predict(path):
    image = Image.open(path)
    processed_image, gray_resized_image = preprocess_image(image, target_size=(128, 128))
    # print(processed_image)
    # print(processed_image.shape)
    # 输出预测结果
    prediction = model.predict(processed_image).tolist()
    result = {}
    # 寻找可能性最大的标签
    for i in range(0, 10):
        result[get_label(i)] = prediction[0][i]
    # 字典最大值
    ret = max(result, key=lambda x: result[x])
    return ret


# 重载predict，用来预处理摄像头捕获的内容
def predict_frame(frame):
    img_Image = Image.fromarray(np.uint8(frame))
    processed_image, gray_resized_image = preprocess_image(img_Image, target_size=(128, 128))
    prediction = model.predict(processed_image).tolist()
    result = {}
    for i in range(0, 10):
        result[get_label(i)] = prediction[0][i]
    ret = max(result, key=lambda x: result[x])
    return ret


class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()
        self.imgName = ''
        # 导入界面
        self.ui = uic.loadUi('./gestureforcnn.ui')
        self.ui.pushButton.clicked.connect(self.openImage)
        self.ui.pushButton_3.clicked.connect(self.openLen)

        self.ui.pushButton_2.clicked.connect(self.getResult)

        self.ui.label_2.setStyleSheet("QLabel{background:white;}"
                                      "QLabel{color:rgb(300,300,300,120);font-size:10px;font-weight:bold;font-family:宋体;}"
                                      )


    def openImage(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = PyQt5.QtGui.QPixmap(imgName).scaled(self.ui.label_2.width(), self.ui.label_2.height())
        # 获取文件名
        # print(imgName)
        self.imgName = imgName
        self.ui.label_2.setPixmap(jpg)


    def openLen(self):
        # 调用摄像头进行识别
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow('capture', frame)
            c = cv2.waitKey(1)
            if c & 0xff == 27:
                break
            elif c & 0xff == 13:
                # 得到的frame为图像资源
                # frame = np.array(frame)
                # print(frame.shape)
                # 对这个图片进行预处理
                caplabel = predict_frame(frame)
                self.ui.textEdit.setPlainText(str(caplabel))
        cap.release()
        cv2.destroyAllWindows()


    def getResult(self):
        # 按照路径可以正确预测
        self.imgName = './' + '/'.join(self.imgName.split('/')[6:])
        # 两种不同的图像处理方式
        # frame = cv2.imread(self.imgName)  # OpenCV预处理
        # ret = predict_frame(frame)
        ret = predict(self.imgName)
        print(ret)
        self.ui.textEdit.setPlainText(str(ret))


if __name__ == "__main__":
    get_model()
    app = QApplication([])
    win = picture()
    win.ui.show()
    sys.exit(app.exec_())
