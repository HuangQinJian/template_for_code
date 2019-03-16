'''
@Description:
@Author: HuangQinJian
@Date: 2019-02-17 15:39:19
@LastEditTime: 2019-02-24 15:52:29
@LastEditors: HuangQinJian
'''
import json

import pandas as pd

import cv2

label_path = '../../raw_data/jinnan2_round1_train_20190222/train_no_poly.json'


def read_image_info(label_path):
    with open(label_path, 'r') as load_f:
        load_dict = json.load(load_f)
        image_collect = load_dict['images']
        image_num = len(image_collect)
        anno_collect = load_dict['annotations']
        anno_num = len(anno_collect)

        img_name_list = []
        img_id_list = []
        img_height_list = []
        img_width_list = []
        anno_count_list = []
        bbox_list = []

        for i in range(image_num):
            img = image_collect[i]
            img_name = img['file_name']
            img_id = img['id']
            img_height = img['height']
            img_width = img['width']
            anno_count_per_image = 0
            bbox = []
            for j in range(anno_num):
                if anno_collect[j]['image_id'] == img_id:
                    anno_count_per_image += 1
                    bbox.append(anno_collect[j]['bbox'])
            img_name_list.append(img_name)
            img_id_list.append(img_id)
            img_height_list.append(img_height)
            img_width_list.append(img_width)
            anno_count_list.append(anno_count_per_image)
            bbox_list.append(bbox)
            # print(img_name, img_id, img_height, img_width,
            #       anno_count_per_image, len(bbox))

        anno = pd.DataFrame()
        anno['img_name'] = img_name_list
        anno['img_id'] = img_id_list
        anno['img_height'] = img_height_list
        anno['img_width'] = img_width_list
        anno['anno_count'] = anno_count_list
        anno['bbox'] = bbox_list

        anno.to_csv('anno.csv', index=None)
        anno.to_pickle('anno.pkl')


if __name__ == "__main__":
    read_image_info(label_path)
