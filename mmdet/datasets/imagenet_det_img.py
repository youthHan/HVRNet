import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class DETIMGDataset(XMLDataset):

    CLASSES = ('n02691156', 'n02419796', 'n02131653', 'n02834778',
						'n01503061', 'n02924116', 'n02958343', 'n02402425',
						'n02084071', 'n02121808', 'n02503517', 'n02118333',
						'n02510455', 'n02342885', 'n02374451', 'n02129165',
						'n01674464', 'n02484322', 'n03790512', 'n02324045',
						'n02509815', 'n02411705', 'n01726692', 'n02355227',
						'n02129604', 'n04468005', 'n01662784', 'n04530566',
						'n02062744', 'n02391049')

    def __init__(self, **kwargs):
        super(DETIMGDataset, self).__init__(**kwargs)
        # if 'VOC2007' in self.img_prefix:
        #     self.year = 2007
        # elif 'VOC2012' in self.img_prefix:
        #     self.year = 2012
        # else:
        #     raise ValueError('Cannot infer dataset year from img_prefix')

    def __getitem__(self, idx):
        # print("DET!")
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def load_annotations(self, ann_file):
        img_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            filename = 'JPEGImages/{}.JPEG'.format(img_id)
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(img_id))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=img_id, filename=filename, 
                     width=width, height=height, 
                     num_annos=len(root.findall('object'))))
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations',
                            '{}.xml'.format(img_id))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            try:
                label = self.cat2label[name]
            except KeyError:
                continue
            difficult = False # difficult = int(obj.find('difficult').text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann