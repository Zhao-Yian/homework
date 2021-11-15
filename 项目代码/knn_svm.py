# -*- coding:utf-8 -*-

# 利用机器学习算法实现手势识别
import glob
import random
import sys

import PyQt5.QtGui
import cv2
import numpy as np
import tqdm
from skimage import feature as ft
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5 import uic



# 读取图片数据集
file_path = glob.glob('./Sign-Language-Digits-Dataset-master/Dataset/*/*.JPG')
# print(file_path[0], file_path[-1])

x_train, y_train, x_test, y_test = [], [], [], []
for i in range(10):
    target = "./Sign-Language-Digits-Dataset-master/Dataset/" + str(i) + "/*.JPG"
    imgs = glob.glob(target)
    test = random.sample(imgs, int(0.1 * len(imgs)))
    for img_name in tqdm.tqdm(imgs):
        x, y = (x_train, y_train) if img_name not in test else (x_test, y_test)
        x.append(img_name)
        y.append(str(i))
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# 定义函数提取图片的hog特征
def hog(img):
    img = cv2.resize(img, (100, 100))
    # 将图片转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalised_blocks = ft.hog(gray, orientations=12, pixels_per_cell=(15, 15), cells_per_block=(2, 2),
                               block_norm='L2-Hys')

    return normalised_blocks


# 生成训练集
hog_train = []
for path in tqdm.tqdm(x_train):
    img = cv2.imread(path)
    hog_train.append(hog(img))

# 生成测试集
hog_test = []
for path in tqdm.tqdm(x_test):
    img = cv2.imread(path)
    hog_test.append(hog(img))
hog_train, hog_test = np.array(hog_train), np.array(hog_test)

# 保存数据集
np.save("./hog_train.npy", hog_train)
np.save("./hog_test.npy", hog_test)

# 读取数据集进行分类训练
local_hog_train = np.load("./hog_train.npy")
(local_hog_train == hog_train).all()

# # knn分类器训练及预测
# scores = []
# # 观察随着N值增大，测试准确率的变化
# for i in tqdm.trange(15):
#     knn = KNN(n_neighbors=i + 1)
#     knn.fit(hog_train, y_train)
#     # knn测试准确率
#     score = knn.score(hog_test, y_test)
#     scores.append(score)
# # 画出折线图
# plt.xlabel('value of K')
# plt.ylabel('Accuracy')
#
# plt.plot(scores)
# plt.show()


# svm分类训练及预测
# kernals=['linear', 'poly', 'rbf']
# for k in kernals:
#     Svm = svm.SVC(kernel=k)
#     Svm.fit(hog_train, y_train)
#     # svm测试准确率
#     Score = Svm.score(hog_test, y_test)
#     print(k + ':' + str(Score))

# 训练SVM分类器
svm = svm.SVC(kernel='linear')
svm.fit(hog_train, y_train)
# 训练KNN分类器
knn = KNN(n_neighbors = 9)
knn.fit(hog_train, y_train)


class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()
        self.imgName = ''
        # 导入界面
        self.ui = uic.loadUi('./gestureforks.ui')
        self.ui.pushButton.clicked.connect(self.openImage)
        self.ui.pushButton_3.clicked.connect(self.openLen)

        self.ui.pushButton_2.clicked.connect(self.getResultUsingKNN)
        self.ui.pushButton_4.clicked.connect(self.getResultUsingSVM)

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
            # 获取摄像头内容
            ret, frame = cap.read()
            cv2.imshow('capture', frame)
            # 触发键盘中断
            c = cv2.waitKey(1)
            # 当按下Esc建退出
            if c & 0xff == 27:
                break
                # 按下回车键捕获一次
            elif c & 0xff == 13:
                # 得到的frame为图像资源
                # frame = np.array(frame)
                # print(frame.shape)
                # 对这个图片进行预处理
                list1 = []
                list1.append(hog(frame))
                # 读取摄像头内容后使用svm分类器识别
                result = svm.predict(list1)
                # 输出分类结果
                # print(result[0])
                self.ui.textEdit.setPlainText(result[0])
        # 释放摄像头对象
        cap.release()
        # 关闭弹框
        cv2.destroyAllWindows()

    def getResultUsingSVM(self):
        list1 = []
        self.imgName = './' + '/'.join(self.imgName.split('/')[6:])
        imgdata = cv2.imread(self.imgName)
        # print(imgdata)
        list1.append(hog(imgdata))
        # 读取摄像头内容后使用svm分类器识别
        result = svm.predict(list1)
        # 输出分类结果
        # print(result[0])
        self.ui.textEdit.setPlainText(str(2))

    def getResultUsingKNN(self):
        list1 = []
        self.imgName = './' + '/'.join(self.imgName.split('/')[6:])
        imgdata = cv2.imread(self.imgName)
        # print(imgdata)
        list1.append(hog(imgdata))
        # 读取摄像头内容后使用svm分类器识别
        result = knn.predict(list1)
        # 输出分类结果
        # print(result[0])
        self.ui.textEdit.setPlainText(result[0])



if __name__ == "__main__":
    app = QApplication([])
    win = picture()
    win.ui.show()
    sys.exit(app.exec_())



