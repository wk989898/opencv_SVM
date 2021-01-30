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
    /* Default values to train SVM */
    svm->setCoef0( 0.0 );
    svm->setDegree( 3 );
    svm->setTermCriteria( TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 1e-3 ) );
    svm->setGamma( 0 ); 核函数
    svm->setKernel( SVM::LINEAR );
    svm->setNu( 0.5 );
    svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?
    svm->setC( 0.01 ); // From paper, soft classifier 惩罚因子
    svm->setType( SVM::EPS_SVR ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
    """
    svm = cv.ml.SVM_create()
    # set params
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setC(0.5)  
    # svm.setGamma(0.0001)  
    # svm.setCoef0()
    # svm.setDegree ()
    return svm

# test
# cat 和 dog 测试
def test():
    print('dog and cat !')
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
