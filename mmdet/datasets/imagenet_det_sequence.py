import random
import copy
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
import numpy as np

from .pipelines import Compose
from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class DETSeqDataset(XMLDataset):

    CLASSES = ('n02691156', 'n02419796', 'n02131653', 'n02834778',
						'n01503061', 'n02924116', 'n02958343', 'n02402425',
						'n02084071', 'n02121808', 'n02503517', 'n02118333',
						'n02510455', 'n02342885', 'n02374451', 'n02129165',
						'n01674464', 'n02484322', 'n03790512', 'n02324045',
						'n02509815', 'n02411705', 'n01726692', 'n02355227',
						'n02129604', 'n04468005', 'n01662784', 'n04530566',
						'n02062744', 'n02391049')

    def __init__(self, hnl=False, **kwargs):
        self.class_map = ('__background__', # always index 0
						'n02691156', 'n02419796', 'n02131653', 'n02834778',
						'n01503061', 'n02924116', 'n02958343', 'n02402425',
						'n02084071', 'n02121808', 'n02503517', 'n02118333',
						'n02510455', 'n02342885', 'n02374451', 'n02129165',
						'n01674464', 'n02484322', 'n03790512', 'n02324045',
						'n02509815', 'n02411705', 'n01726692', 'n02355227',
						'n02129604', 'n04468005', 'n01662784', 'n04530566',
						'n02062744', 'n02391049')
        super(DETSeqDataset, self).__init__(**kwargs)
        # if 'VOC2007' in self.img_prefix:
        #     self.year = 2007
        # elif 'VOC2012' in self.img_prefix:
        #     self.year = 2012
        # else:
        #     raise ValueError('Cannot infer dataset year from img_prefix')
        self.classes = ('airplane', 'antelope', 'bear', 'bicycle',
                        'bird', 'bus', 'car', 'cattle',
                        'dog', 'domestic_cat', 'elephant', 'fox',
                        'giant_panda', 'hamster', 'horse', 'lion',
                        'lizard', 'monkey', 'motorcycle', 'rabbit',
                        'red_panda', 'sheep', 'snake', 'squirrel',
                        'tiger', 'train', 'turtle', 'watercraft',
                        'whale', 'zebra')
        self.class_to_index = dict(zip(self.class_map, range(len(self.classes)+1)))
        if hnl:
            self.extra_cls = 2
            self.video_per_cls = 3
        self.get_video2idx()
        self.get_cls2video()

    '''
    Sample different `num` videos from idx, for training Hiera-Non-Local Network

    '''
    def sample_videos(self, idx, extra_cls_num=0, video_per_cls=1):
        sampled_ids = [idx]

        if extra_cls_num != 0:
            cur_video_id = self.idx_2_video[idx]
            cur_video_cls = self.video_2_cls[cur_video_id]
            c_cls_video_list = copy.deepcopy(self.cls_2_video[cur_video_cls])
            c_cls_video_list.remove(cur_video_id)
            for v_id in random.sample(c_cls_video_list, video_per_cls-1):
                sampled_ids.extend(random.sample(self.video_2_idx[v_id], 1))


            cls_set = list(range(len(self.class_map[1:])))
            cls_set.remove(cur_video_cls)
            extra_cls = random.sample(cls_set, extra_cls_num)
            for cls_id in extra_cls:
                video_ids = random.sample(self.cls_2_video[cls_id], video_per_cls)
                for v_id in video_ids:
                    sampled_ids.extend(random.sample(self.video_2_idx[v_id], 1))

        return sampled_ids


    '''
    To generate mapping from img id to video id, for training with frame list only!
    '''
    def get_video2idx(self):
        video2idx=dict()
        idx2video=list()
        for idx, img_info in enumerate(self.img_infos):
            video_id = img_info['id'].strip().split('/')[-1]
            video2idx.setdefault(video_id, []).append(idx)
            idx2video.append(video_id)
        self.video_2_idx = video2idx
        self.idx_2_video = idx2video


    def get_cls2video(self):
        vid2det_path = '/home/mfhan/ILSVRC/vid_2_det_id.txt'
        vid2det = dict()
        vid2det_lines = mmcv.list_from_file(vid2det_path)
        vid2det = {line.strip().split(' ')[0]:line.strip().split(' ')[1] 
                                                        for line in vid2det_lines}
        ext_videos = list(self.video_2_idx.keys())
        video2cls = dict()
        cls2video = list()
        for i in range(len(self.class_map[1:])):
            cls2video.append([])
            video_set_path = '/home/mfhan/ILSVRC/ImageSets/DET/train_{}.txt'.format(vid2det[str(i+1)])
            video_set = mmcv.list_from_file(video_set_path)
            for v_line in video_set:
                flag = v_line.strip().split(' ')[-1]
                if flag != '1':
                    continue
                # if 'n02492035' in v_line:
                #     print('hi')
                video_id = v_line.strip().split(' ')[0].strip().split('/')[-1]
                if video_id not in ext_videos:
                    continue
                video2cls[video_id] = i
                cls2video[i].append(video_id)
                
        self.video_2_cls = video2cls
        self.cls_2_video = cls2video

    '''
    Called when key frame pipeline is finished, 
    to generate a dynamic condition pipeline according to key_flipped
    '''
    def get_condition_pipeline(self, key_flipped):
        img_norm_cfg = dict(
            mean=[103.06, 115.90, 123.15], std=[1.0, 1.0, 1.0], to_rgb=False)
            
        self.condition_pipeline = [
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=self.selsa_with_aug, with_label=self.selsa_with_aug),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='Expand',
                mean=img_norm_cfg['mean'],
                to_rgb=img_norm_cfg['to_rgb'],
                ratio_range=(1, 2)),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3),
            dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5 if self.condition_random_flip else int(key_flipped)),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'] \
                                        if self.selsa_with_aug else ['img'])
        ]

    def __getitem__(self, idx):
        # print("DET!")
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            if not hasattr(self, 'extra_cls') or self.extra_cls == 0 :
                data = self.prepare_train_img(idx)
            else:
                data = self.prepare_train_img(idx, self.extra_cls, self.video_per_cls)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    #TODO: overwrite to get im, bef_im and aft_im
    def prepare_train_img(self, idx, extra_cls=0, video_per_cls=1):
        res_list = []
        idx_list = self.sample_videos(idx, extra_cls, video_per_cls)

        for idx in idx_list:
            img_info = self.img_infos[idx]
            ann_info = self.get_ann_info(idx)
            results = dict(img_info=img_info, ann_info=ann_info)
            if self.proposals is not None:
                results['proposals'] = self.proposals[idx]
            self.pre_pipeline(results)
            key_res=self.pipeline(results)
            key_flipped=key_res['img_meta'].data['flip']

            #Operations for condition frames
            bef_res=copy.deepcopy(key_res)
            aft_res=copy.deepcopy(key_res)
            # n_res = collate([key_res, bef_res, aft_res],3)
            assert bef_res is not None, "!!!DETSeq: bef_res can't be Nonetype!!!"
            assert aft_res is not None, "!!!DETSeq: aft_res can't be Nonetype!!!"
            res_list.extend([key_res, bef_res, aft_res])
        return res_list

    def load_annotations(self, ann_file):
        img_infos = []
        lines = [x.strip().split(' ') for x in mmcv.list_from_file(ann_file)]
        for line in lines:
            image_set_index = line[0]
            frame_id = int(line[1]) 
            filename = 'JPEGImages/{}.JPEG'.format(image_set_index)
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(image_set_index))
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=image_set_index, filename=filename, 
                     width=width, height=height, 
                     num_annos=len(root.findall('object'))))

        return img_infos

    '''
    Different from `get_anno_info`: read anno file by xml_path and return with [anno,[width, height]]
    '''
    def get_anno_info_by_path(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            label = self.class_to_index[name]
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
        return ann, [width, height]

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