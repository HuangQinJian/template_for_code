'''
@Description:
@Author: HuangQinJian
@Date: 2019-02-17 15:39:19
@LastEditTime: 2019-03-09 14:24:34
@LastEditors: HuangQinJian
'''
import json

import pandas as pd

import cv2

label_path = '../../keras-retinanet-master/keras_retinanet/CSV/data/jinnan2_round1_train_20190305/train_no_poly.json'


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
        category_id_list = []
        bbox_list = []

        for i in range(image_num):
            img = image_collect[i]
            img_name = img['file_name']
            img_id = img['id']
            img_height = img['height']
            img_width = img['width']
            # anno_count_per_image = 0
            for j in range(anno_num):
                if anno_collect[j]['image_id'] == img_id:
                    # anno_count_per_image += 1
                    img_name_list.append(img_name)
                    img_id_list.append(img_id)
                    img_height_list.append(img_height)
                    img_width_list.append(img_width)
                    # anno_count_list.append(anno_count_per_image)
                    category_id_list.append(anno_collect[j]['category_id'])
                    bbox_list.append(anno_collect[j]['bbox'])

        anno = pd.DataFrame()
        anno['img_name'] = img_name_list
        anno['img_id'] = img_id_list
        anno['img_height'] = img_height_list
        anno['img_width'] = img_width_list
        # anno['anno_count'] = anno_count_list
        anno['category_id'] = category_id_list
        anno['bbox'] = bbox_list

        anno.to_csv('anno.csv', index=None)
        anno.to_pickle('anno.pkl')


if __name__ == "__main__":
    read_image_info(label_path)
