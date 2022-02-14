from argparse import ArgumentParser

import mmcv
import numpy as np
import os.path as osp

import xml.etree.ElementTree as ET

CLASSES = ('n02691156', 'n02419796', 'n02131653', 'n02834778',
           'n01503061', 'n02924116', 'n02958343', 'n02402425',
           'n02084071', 'n02121808', 'n02503517', 'n02118333',
           'n02510455', 'n02342885', 'n02374451', 'n02129165',
           'n01674464', 'n02484322', 'n03790512', 'n02324045',
           'n02509815', 'n02411705', 'n01726692', 'n02355227',
           'n02129604', 'n04468005', 'n01662784', 'n04530566',
           'n02062744', 'n02391049')
class_name = ('airplane', 'antelope', 'bear', 'bicycle',
              'bird', 'bus', 'car', 'cattle',
              'dog', 'domestic_cat', 'elephant', 'fox',
              'giant_panda', 'hamster', 'horse', 'lion',
              'lizard', 'monkey', 'motorcycle', 'rabbit',
              'red_panda', 'sheep', 'snake', 'squirrel',
              'tiger', 'train', 'turtle', 'watercraft',
              'whale', 'zebra')


def load_annotations(ann_file, img_prefix):
    img_infos = dict()
    cls_name_map = {CLASSES[i]: class_name[i] for i in range(len(class_name))}
    cls_video_map = {class_name[i]: set() for i in range(len(class_name))}
    img_ids = mmcv.list_from_file(ann_file)
    for img_id in img_ids:
        img_id = img_id.strip().split(' ')[0]
        video_id = img_id.strip().split('/')[1]
        xml_path = osp.join(img_prefix, 'Annotations',
                            '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        if len(root.findall('object')) == 0:
            continue
        for obj in root.findall('object'):
            name = cls_name_map[obj.find('name').text]
            cls_video_map[name].add(video_id)
    return cls_video_map


def main():
    anno_file = './data/VID/ImageSets/VID_val_frames.txt'
    img_prefix = './data/VID'
    annos = load_annotations(anno_file, img_prefix)
    import pprint
    pprint.pprint(annos)


if __name__ == '__main__':
    main()
