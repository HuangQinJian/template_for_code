#!/usr/bin/env python
# coding=UTF-8
'''
@Description:
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-02-22 17:11:16
@LastEditTime: 2019-03-10 19:12:07
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

font = cv2.FONT_HERSHEY_SIMPLEX


def draw_rectangle(boxes, image, image_name, category_id_list):
    new_box = boxes
    box_num = len(new_box)
    for i in range(box_num):
        left = np.rint(new_box[i][0])
        right = np.rint(new_box[i][1])
        top = np.rint(new_box[i][0] + new_box[i][2])
        bottom = np.rint(new_box[i][1] + new_box[i][3])
        category_id = str(int(category_id_list[i]))
        # print('left, right, top, bottom:', left, right, top, bottom)
        x_aixs = int(np.rint(new_box[i][0] + new_box[i][2]/2))
        y_aixs = int(np.rint(new_box[i][1] + new_box[i][3]/2))
        cv2.rectangle(image,
                      (int(left), int(right)),
                      (int(top), int(bottom)),
                      (0, 255, 0),
                      2)
        cv2.putText(image, category_id,
                    (x_aixs, y_aixs), font, 1, (255, 0, 0), 2, bottomLeftOrigin=True)
    imsave('anno_image/'+image_name, image)


if __name__ == '__main__':
    img_path = '../../keras-retinanet-master/keras_retinanet/CSV/data/jinnan2_round1_train_20190305/restricted/'
    img_list = os.listdir(img_path)
    train = pd.read_pickle('anno.pkl')
    # for i in range(len(img_list)):
    for i in range(150):
        image_name = img_list[i]
        # print(image_name)
        img_raw = cv2.imread(os.path.join(img_path, image_name))
        boxes = train[train.img_name == image_name]['bbox'].tolist()
        category_id_list = train[train.img_name ==
                                 image_name]['category_id'].tolist()
        draw_rectangle(boxes, img_raw, image_name, category_id_list)
