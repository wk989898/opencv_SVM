import pickle
import numpy as np
import os
import cv2 as cv


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def get_data(img_type):
    label_names = load_file('./data/batches.meta')['label_names']
    data_set = np.zeros([10])  # default 10 categary

    for i in range(1, 6):
        path = './data/data_batch_'+str(i)
        data = load_file(path)
        label = data['labels']
        for j in range(data['data'].shape[0]):
            image_name = label_names[label[j]]
            index = data_set[label[j]]
            data_set[label[j]] = index+1
            img = np.reshape(data['data'][j], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            mkdir('img/'+image_name)
            cv.imwrite('./img/'+image_name+'/' +
                       str(int(index))+'.'+img_type, img)




def get_testImage(img_type):
    mkdir('test_img')
    label_names = load_file('./data/batches.meta')['label_names']
    data_set = np.zeros([10])  # default 10 categary
    data = load_file('./data/test_batch')
    label = data['labels']
    with open('test_labels.dat','w') as f:
        f.write(" ".join(map(str,label)))
        
    for i in range(data['data'].shape[0]):
        image_name = label_names[label[i]]
        index = data_set[label[i]]
        data_set[label[i]] = index+1
        img = np.reshape(data['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        cv.imwrite('./test_img/'+image_name+
                   str(int(index))+'.'+img_type, img)
    print('test image done!')


def start():
    mkdir('img')
    get_data('jpg')
    print('get image done!')


if __name__ == '__main__':
    # start()
    get_testImage('jpg')
