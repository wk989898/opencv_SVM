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


def train_SVM():
    sample = np.array(train_data, dtype=np.float32)
    response = np.array(labels, dtype=np.int32)
    svm = create_SVM()
    print('train start...')
    svm.train(sample, cv.ml.ROW_SAMPLE, response)
    svm.save("svm_data.dat")
    print('train end\nsave data done!')


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
        train_data.append(data)
        labels.append(label_type)


# SVM create

def create_SVM():
    """
    -degree – Parameter degree of a kernel function (POLY).
    -gamma – Parameter \gamma of a kernel function (POLY / RBF / SIGMOID).
    -coef0 – Parameter coef0 of a kernel function (POLY / SIGMOID).
    -Cvalue – Parameter C of a SVM optimization problem (C_SVC / EPS_SVR / NU_SVR).
    -nu – Parameter nu of a SVM optimization problem (NU_SVC / ONE_CLASS / NU_SVR).
    -p – Parameter \epsilon of a SVM optimization problem (EPS_SVR).
    -class_weights – Optional weights in the C_SVC problem , assigned to particular classes. 
        They are multiplied by C so the parameter C of class #i becomes class\_weights_i * C. 
        Thus these weights affect the misclassification penalty for different classes. 
        The larger weight, the larger penalty on misclassification of data from the corresponding class.
    -term_crit – Termination criteria of the iterative SVM training procedure 
        which solves a partial case of constrained quadratic optimization problem. 
        You can specify tolerance and/or the maximum number of iterations.
    """
    svm = cv.ml.SVM_create()
    # set params
    svm.setKernel(cv.ml.SVM_RBF)
    svm.setType(cv.ml.SVM_NU_SVC)
    svm.setC(0.1)  
    svm.setNu(0.7)  
    svm.setGamma(0.05)  
    svm.setCoef0(0)
    svm.setDegree (3)
    return svm

# test
# cat 和 dog 测试
def test():
    print('dog and cat train!')
    cat = path.join('img', 'cat')
    dog = path.join('img', 'dog')
    get_data(cat,  3)  # cat label为3
    get_data(dog,  5)  # dog label为5
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
    test()
    # start()
