# -*- coding: UTF-8 -*-
import cv2 as cv
import numpy as np
import os
from os import path

train_data = []
labels = []
types = None
# 获取图像


def get_img(_path):
    img_path = path.join(_path)
    types = os.listdir(img_path)
    # 将类别记录
    with open('svm_labels.dat', 'w') as f:
        f.write("#".join(types))

    for index, file in enumerate(types):
        label_path = path.join(img_path, file)
        get_data(label_path, index)
    train_SVM()
# start


def train_SVM():
    sample = np.array(train_data, dtype=np.float32)
    response = np.array(labels, dtype=np.int32)
    svm = create_SVM()
    svm.train(sample, cv.ml.ROW_SAMPLE, response)
    svm.save("svm_data.dat")
    print('save data done!')


def get_data(filename, label_type):
    images = os.listdir(filename)
    for file in images:
        img = cv.imread(path.join(filename, file), 1)
        # 默认HOG的描述子窗口为64x128， 窗口移动步长为 8x8
        img = cv.resize(img, (64, 128))
        hog = cv.HOGDescriptor()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        hist = hog.compute(gray)
        data = np.zeros([len(hist)], dtype=np.float32)
        for i in range(len(hist)):
            data[i] = hist[i][0]
        # train_SVM(data, 1)
        train_data.append(data)
        labels.append(label_type)


# SVM create


def create_SVM():
    svm = cv.ml.SVM_create()
    # set params
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(0.1)  # 惩罚因子
    svm.setGamma(1)  # 核函数
    return svm

# test
# cat 和 horse 测试


def test():
    cat = path.join('img', 'cat')
    truck = path.join('img', 'truck')
    get_data(cat,  3)  # cat label为3
    get_data(truck,  9)  # truck label为9
    sample = np.array(train_data, dtype=np.float32)
    response = np.array(labels, dtype=np.int32)
    svm = create_SVM()
    svm.train(sample, cv.ml.ROW_SAMPLE, response)
    svm.save("svm_data.dat")
    print('save data done!')

# start


def start():
    get_img('img')
    print("svm train success!")


if __name__ == '__main__':
    # test()
    start()
