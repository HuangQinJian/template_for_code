#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: HuangQinJian
@LastEditors: HuangQinJian
@Date: 2019-04-23 11:28:23
@LastEditTime: 2019-05-01 13:15:57
'''
import sys
import os
import json
import cv2
import pandas as pd

START_BOUNDING_BOX_ID = 1
PRE_DEFINE_CATEGORIES = {}


def convert(csv_path, img_path, json_file):
    """
    csv_path : csv文件的路径
    img_path : 存放图片的文件夹
    json_file : 保存生成的json文件路径
    """
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    bnd_id = START_BOUNDING_BOX_ID
    categories = PRE_DEFINE_CATEGORIES
    csv = pd.read_csv(csv_path)
    img_nameList = os.listdir(img_path)
    img_num = len(img_nameList)
    print("图片总数为{0}".format(img_num))
    for i in range(img_num):
        # for i in range(30):
        image_id = i+1
        img_name = img_nameList[i]
        if img_name == '60f3ea2534804c9b806e7d5ae1e229cf.jpg' or img_name == '6b292bacb2024d9b9f2d0620f489b1e4.jpg':
            continue
        # 可能需要根据具体格式修改的地方
        lines = csv[csv.filename == img_name]
        img = cv2.imread(os.path.join(img_path, img_name))
        height, width, _ = img.shape
        image = {'file_name': img_name, 'height': height, 'width': width,
                 'id': image_id}
        print(image)
        json_dict['images'].append(image)
        for j in range(len(lines)):
            # 可能需要根据具体格式修改的地方
            category = str(lines.iloc[j]['type'])
            if category not in categories:
                new_id = len(categories)+1
                categories[category] = new_id
            category_id = categories[category]
            # 可能需要根据具体格式修改的地方
            xmin = int(lines.iloc[j]['X1'])
            ymin = int(lines.iloc[j]['Y1'])
            xmax = int(lines.iloc[j]['X3'])
            ymax = int(lines.iloc[j]['Y3'])
            # print(xmin, ymin, xmax, ymax)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width*o_height, 'iscrowd': 0, 'image_id':
                   image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict, indent=4)
    json_fp.write(json_str)
    json_fp.close()


if __name__ == '__main__':
    # csv_path = 'data/train_label_fix.csv'
    # img_path = 'data/train/'
    # json_file = 'train.json'
    csv_path = 'train_label_fix.csv'
    img_path = 'trained/'
    json_file = 'trained.json'
    convert(csv_path, img_path, json_file)
    csv_path = 'train_label_fix.csv'
    img_path = 'val/'
    json_file = 'val.json'
    convert(csv_path, img_path, json_file)
