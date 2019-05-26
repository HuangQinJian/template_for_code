#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-05-01 12:56:08
@LastEditTime: 2019-05-01 13:11:38
'''
import pandas as pd
import random
import os
import shutil

if not os.path.exists('trained/'):
    os.mkdir('trained/')

if not os.path.exists('val/'):
    os.mkdir('val/')

val_rate = 0.15

img_path = 'train/'
img_list = os.listdir(img_path)
train = pd.read_csv('train_label_fix.csv')
# print(img_list)
random.shuffle(img_list)

total_num = len(img_list)
val_num = int(total_num*val_rate)
train_num = total_num-val_num

for i in range(train_num):
    img_name = img_list[i]
    shutil.copy('train/' + img_name, 'trained/' + img_name)
for j in range(val_num):
    img_name = img_list[j+train_num]
    shutil.copy('train/' + img_name, 'val/' + img_name)
