#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-02-22 17:11:16
@LastEditTime: 2019-03-05 21:02:55
'''
import matplotlib.pyplot as plt
from skimage.io import imread, imshow

import os
import cv2
import numpy as np
import pandas as pd
from skimage.io import imsave

if not os.path.exists('anno_image/'):
    os.makedirs('anno_image/')


def draw_rectangle(boxes, image, image_name):
    new_box = boxes
    for box in new_box:
        left = np.rint(box[0])
        right = np.rint(box[1])
        top = np.rint(box[0] + box[2])
        bottom = np.rint(box[1] + box[3])

        # print('left, right, top, bottom:', left, right, top, bottom)
        cv2.rectangle(image,
                      (int(left), int(right)),
                      (int(top), int(bottom)),
                      (0, 255, 0),
                      1)
    imsave('anno_image/'+image_name, image)


if __name__ == '__main__':
    img_path = '../../raw_data/jinnan2_round1_train_20190222/restricted/'
    img_list = os.listdir(img_path)
    # print(img_list)
    train = pd.read_pickle('anno.pkl')
    # # for i in range(len(img_list)):
    # # print(train['img_name'])
    # for i in range(1450):
    #     image_name = img_list[i]
    #     # print(image_name)
    #     if image_name == '190122_182215_00169927.jpg':
    #         print(2)
    #         img_raw = cv2.imread(os.path.join(img_path, image_name))
    #         boxes_list = train[train.img_name == image_name]['bbox'].tolist()[
    #             0]
    #         draw_rectangle(boxes_list, img_raw, image_name)
    #     break
    image_name = '190122_182215_00169927.jpg'
    img_raw = cv2.imread(os.path.join(img_path, image_name))
    print(img_raw.shape)
    boxes_list = train[train.img_name == image_name]['bbox'].tolist()[
        0]
    print(boxes_list)
    draw_rectangle(boxes_list, img_raw, image_name)
