# coding:utf-8
# from libsvm.svmutil import *
import cv2 as cv
import numpy as np
import os
from os import path


def get_label():
    with open('labels.dat') as f:
        return f.read().split(',')

# print(a[3])
# dir = os.listdir('./test_img')
# for image in dir:
#     img = cv.imread('./test_img/'+image, 1)
#     print(img.shape)
# # 读取图片
# img=cv.imread('./people.jpg',1)
# hog=cv.HOGDescriptor()
# print(img.shape)
# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# hist=hog.compute(gray)
# print(hist)

# # hog.compute()
# hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
# [rects, weights]= hog.detectMultiScale(img, winStride=(4, 4),padding=(8, 8),scale=1.25,useMeanshiftGrouping=False)
# for (x, y, w, h) in rects:
#     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
# print(weights)
# cv.imshow("people", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# svm=cv.ml.SVM_create()
# test
# src=cv.imread('./cat.jpg')
# hog = cv.HOGDescriptor()
# gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
# hist = hog.compute(gray, winStride=(8, 8), padding=(0, 0))
# print(len(hist))
# print(hist[0][0])
# cv.namedWindow('hog', cv.WINDOW_NORMAL)
# cv.imshow("hog", src)
# cv.waitKey(0)
# cv.destroyAllWindows()
# print(cv.ml.SVM_create)
