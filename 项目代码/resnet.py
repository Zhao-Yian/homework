# -*- coding: UTF-8 -*-
import glob
import os
import random
import sys
import PyQt5
import PyQt5.QtGui
import cv2
import keras
import numpy as np
import tqdm
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import losses
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from PyQt5.QtWidgets import *
from PyQt5 import uic

# 构建神经神经网络，此处使用适合图片数据集的多分类残差网络ResNet50
class ResidualBlock(Model):

    def __init__(self, filters, strides=1, residual_path=False):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        if residual_path:
            self.down_c1 = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs  # residual等于输入值本身，即residual=x
        # 将输入通过卷积、BN层、激活层，计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(y + residual)  # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
        return out


class ResNet50(Model):

    def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
        super(ResNet50, self).__init__()
        self.num_blocks = len(block_list)  # 共有几个block
        self.block_list = block_list
        self.out_filters = initial_filters
        self.c1 = Conv2D(self.out_filters, (3, 3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        # 构建ResNet网络结构
        for block_id in range(len(block_list)):  # 第几个resnet block
            for layer_id in range(block_list[block_id]):  # 第几个卷积层

                if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
                    block = ResidualBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResidualBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)  # 将构建好的block加入resnet
            self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(),
                                        use_bias=False)  # 做10分类任务

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


# 制作数据集
x_train, y_train, x_test, y_test = [], [], [], []
for i in range(10):
    target = "./Sign-Language-Digits-Dataset-master/Dataset/" + str(i) + "/*.JPG"
    imgs = glob.glob(target)
    # 训练集: 测试集 = 9:1
    test = random.sample(imgs, k=int(0.2 * len(imgs)))
    for img_name in tqdm.tqdm(imgs):
        x, y = (x_train, y_train) if img_name not in test else (x_test, y_test)
        x.append(img_name)
        y.append(str(i))
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)
# print(x_train, y_train.shape, x_test.shape, y_test.shape)

# 构建训练集
train_data = []
for i in tqdm.tqdm(x_train):
    train_data.append(cv2.imread(i))

# 数据预处理
train_data = []
test_data = []
# 利用opencv读取图片，并修改数据类型，提高运算精度
for i in tqdm.tqdm(x_train):
    train_data.append(cv2.imread(i))
train_data = np.array(train_data, dtype='float64')
for j in tqdm.tqdm(x_test):
    test_data.append(cv2.imread(j))
test_data = np.array(test_data, dtype='float64')

# print(train_data.shape, test_data.shape)

# 标签处理
# 此处提供one-hot编码标签
arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# 生成对称矩阵
one_hot = np.eye(10)[arr]
train_label = []
test_label = []
# 制作标签
for l in y_train:
    train_label.append(one_hot[int(l)])
for lt in y_test:
    test_label.append(one_hot[int(lt)])
train_label = np.array(train_label)
test_label = np.array(test_label)
# print(train_label.shape, test_label.shape)



# 建立模型
# ResNet50经典模型四个残差块数量为3, 4, 6, 3
model = ResNet50([3, 4, 6, 3])
index = [i for i in range(len(train_data))]
# 打散验证集与标签 防止产生记忆
np.random.shuffle(index)
train_data = train_data[index]
train_label = train_label[index]


# 测试集同样处理
index2 = [j for j in range(len(test_data))]
np.random.shuffle(index2)
test_data = test_data[index2]
test_label = test_label[index2]


# print(train_data.shape)
# print(train_label.shape)
train_label.astype('float64')
test_label.astype('float64')
# print(train_data.dtype,test_label.dtype)
# 设置优化器
# 设置两个优化器，此处使用Adam，优化效果较好
sgd = keras.optimizers.SGD(lr=0.02, decay=1e-7, momentum=0.9, nesterov=True)
# 设置学习率0.005(其实比较大，但是由于网络层数较深，需要平衡训练速度)
Nadam = tf.optimizers.Nadam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-7, name='Nadam')
# 模型装配(使用交叉熵损失函数)
model.compile(optimizer='Nadam',
              loss=losses.CategoricalCrossentropy(),
              metrics=['CategoricalAccuracy'])  # 因为是独热码 采用 mse计算loss 和CategoricalAccuracy的评判标准

# 利用Tensorboard可视化工具记录训练过程
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./dir_work",
                                                      histogram_freq=1,
                                                      write_graph=False,
                                                      write_images=False,
                                                      write_grads=False)

# 读取训练完成的模型参数
checkpoint_save_path = "../150/checkpoint/train.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    model.load_weights(checkpoint_save_path)
    print('-------------load the model-----------------')

# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#                                                  save_weights_only=True,
#                                                  save_best_only=True)
# history = model.fit(train_data, train_label, batch_size=32, epochs=150, validation_split=0.1, validation_freq=1,
#                     callbacks=[cp_callback, tensorboard_callback])
# model.summary()
#
# 测试集测试
# print(test_data[0].shape)
#
# testlabel = model.predict(test_data)
# # print(test_label)
# # print(testlabel)
# px, py = testlabel.shape
# index = np.argmax(testlabel, axis=1)
# # print(index)
# # d = np.argmax(np.bincount(index))
# count = 0
# for i in index:
#     if test_label[i][index[i]] == 1:
#         count += 1
# print(count)
# print(len(index))
# print("测试集准确率：{:.6f}%".format(count*100/len(index)))


# 训练集测试
#
# trainlabel = model.predict(train_data)
# px, py = trainlabel.shape
# index = np.argmax(trainlabel, axis=1)
# print(index)
# # d = np.argmax(np.bincount(index))
# count = 0
# for i in index:
#     if train_label[i][index[i]] == 1:
#         count += 1
# print(count)
# print(len(index))
# print("训练集准确率：{:.6f}%".format(count/len(index)))





class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()
        self.imgName = ''
        # 导入界面
        self.ui = uic.loadUi('./gestureforresnet50.ui')

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
        # 调用摄像头进行识别(此处代码同机器学习，注释省略)
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
                capture = []
                frame = cv2.resize(frame, (100, 100))
                capture.append(frame)
                capture = np.array(capture, dtype='float64')
                print(capture.shape)
                caplabel = model.predict(capture)
                print(caplabel)
                ret = np.argmax(caplabel, axis=1)
                # 输出监测结果
                # print(ret[0])
                self.ui.textEdit.setPlainText(str(ret[0]))
        cap.release()
        cv2.destroyAllWindows()


    def getResult(self):
        self.imgName = './' + '/'.join(self.imgName.split('/')[6:])
        imgdata = cv2.imread(self.imgName)
        capture = []
        imgdata = cv2.resize(imgdata, (100, 100))
        capture.append(imgdata)
        capture = np.array(capture, dtype='float64')
        print(capture.shape)
        caplabel = model.predict(capture)
        print(caplabel)
        ret = np.argmax(caplabel, axis=1)
        # 输出监测结果
        # print(ret[0])
        self.ui.textEdit.setPlainText(str(ret[0]))


if __name__ == "__main__":
    app = QApplication([])
    win = picture()
    win.ui.show()
    sys.exit(app.exec_())