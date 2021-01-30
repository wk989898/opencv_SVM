# -*- coding: UTF-8 -*-
import cv2 as cv
import numpy as np
import os
from os import path
from matplotlib import pyplot as plt


def get_label():
    with open('svm_labels.dat') as f:
        return f.read().split('#')


def get_hog(img):
    # 图像大小设置为64*128
    img = cv.resize(img, (64, 128))
    hog = cv.HOGDescriptor()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist = hog.compute(gray)
    data = np.zeros([len(hist)], dtype=np.float32)
    for i in range(len(hist)):
        data[i] = hist[i][0]
    return data

# 预测准确率
def predict_all():
    data_set = []
    labels=[]
    # labels = get_label()
    images = os.listdir(path.join('./test_img'))
    for image in images:
        #cat and dog 预测
        if image.startswith('cat'):
            img = cv.imread('./test_img/'+image, 1)
            data_set.append(get_hog(img))
            labels.append(3)
        if image.startswith('dog'):
            img = cv.imread('./test_img/'+image, 1)
            data_set.append(get_hog(img))
            labels.append(5)

        # img = cv.imread('./test_img/'+image, 1)
        # data_set.append(get_hog(img))

    data_set = np.array(data_set, dtype=np.float32)
    result = svm.predict(data_set)[1]
    acc = 0
    #cat and dog
    for i in range(len(result)):
        a=labels[i]
        b=int(result[i][0])
        if(a == b):
            acc += 1

    # with open('test_labels.dat', 'r') as f:
    #     res = f.read().split(' ')
    #     for i in range(len(result)):
    #         a=int(res[i])
    #         b=int(result[i][0])
    #         if(a == b):
    #             # print('match:',labels[a])
    #             acc += 1
    #         else:
    #             # print(labels[a], labels[b])
    #             continue

    print('\nthere have %d images' % (len(result)))
    print('accuracy is {:.2%}'.format(acc/len(result)))


# 预测单幅图像
def predict_single():
    data_set = []
    dog = cv.imread('./p_dog.jpg', 1)
    cat = cv.imread('./p_cat.jpg', 1)
    data_set.append(get_hog(dog))
    data_set.append(get_hog(cat))
    data_set = np.array(data_set, dtype=np.float32)
    result = svm.predict(data_set)[1]
    labels = get_label()
    for i in range(len(result)):
        result[i] = result[i][0]
    plt.figure('dog and cat')
    plt.subplot(1, 2, 1)
    plt.imshow(dog)

    plt.title(labels[int(result[0])])
    plt.subplot(1, 2, 2)
    plt.imshow(cat)
    plt.title(labels[int(result[1])])
    plt.show()
    plt.pause(0)


if __name__ == '__main__':
    svm = cv.ml.SVM_load('./svm_data.dat')
    print('predict start...')
    # predict_single()
    predict_all()
