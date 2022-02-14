import os.path as osp
import random
import copy
from collections import deque
import xml.etree.ElementTree as ET

import mmcv
import numpy as np
from mmcv.parallel import collate

from .pipelines import Compose
from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class VIDSeqDataset(XMLDataset):
    CLASSES = ('n02691156', 'n02419796', 'n02131653', 'n02834778',
						'n01503061', 'n02924116', 'n02958343', 'n02402425',
						'n02084071', 'n02121808', 'n02503517', 'n02118333',
						'n02510455', 'n02342885', 'n02374451', 'n02129165',
						'n01674464', 'n02484322', 'n03790512', 'n02324045',
						'n02509815', 'n02411705', 'n01726692', 'n02355227',
						'n02129604', 'n04468005', 'n01662784', 'n04530566',
						'n02062744', 'n02391049')
    def __init__(self, hnl=False, shuffle=False, video_shuffle=True, has_rpn=True, frame_interval=0, **kwargs):
        self.class_map = ('__background__', # always index 0
						'n02691156', 'n02419796', 'n02131653', 'n02834778',
						'n01503061', 'n02924116', 'n02958343', 'n02402425',
						'n02084071', 'n02121808', 'n02503517', 'n02118333',
						'n02510455', 'n02342885', 'n02374451', 'n02129165',
						'n01674464', 'n02484322', 'n03790512', 'n02324045',
						'n02509815', 'n02411705', 'n01726692', 'n02355227',
						'n02129604', 'n04468005', 'n01662784', 'n04530566',
						'n02062744', 'n02391049')
        self.MIN_OFFSET=-1000
        self.MAX_OFFSET=1000
        super(VIDSeqDataset, self).__init__(**kwargs)
        self.classes = ('airplane', 'antelope', 'bear', 'bicycle',
                        'bird', 'bus', 'car', 'cattle',
                        'dog', 'domestic_cat', 'elephant', 'fox',
                        'giant_panda', 'hamster', 'horse', 'lion',
                        'lizard', 'monkey', 'motorcycle', 'rabbit',
                        'red_panda', 'sheep', 'snake', 'squirrel',
                        'tiger', 'train', 'turtle', 'watercraft',
                        'whale', 'zebra')
        self.slices_set = False
        self.class_to_index = dict(zip(self.class_map, range(len(self.classes)+1)))
        if hnl:
            self.extra_cls = 2 #4
            self.video_per_cls = 3
        self.get_video2idx()
        self.get_cls2video()

        if self.test_mode:
            self.slices_set = True
            self.indices_list = None
            self.shuffle = shuffle
            self.video_shuffle = video_shuffle
            self.has_rpn = has_rpn

            # infer properties from roidb
            self.size = np.sum([x['frame_seg_len'] if 'frame_seg_len' in x else x['video_len'] \
                                for x in self.img_infos])
            self.index = np.arange(self.size)

            # decide data and label names (only for training)
            self.data_name = ['data', 'im_info', 'data_cache', 'feat_cache']
            self.label_name = None

            #
            self.cur_tid = 0
            self.cur_seg_len = 0
            self.key_frame_flag = -1

            # self.min_size = img_scale[1]
            # self.max_size = img_scale[0]
            self.frame_interval = frame_interval
            self.que_len = 2 * self.frame_interval + 1
            # self.data_batch = deque(maxlen=self.que_len)
            # self.feat_batch = deque(maxlen=self.que_len)
            # self.init_test_batch()
            self.init_seg_len = self.img_infos[0]['frame_seg_len']
            self.indices_list = self.get_indices(world_size=self.world_size)
    '''
    To generate mapping from img id to video id, for training with frame list only!
    '''
    def get_video2idx(self):
        video2idx=dict()
        idx2video=list()
        for idx, img_info in enumerate(self.img_infos):
            if self.test_mode:
                video_id = img_info['id'].strip().split('/')[-1]
            else:
                video_id = img_info['id'].strip().split('/')[-2]
            video2idx.setdefault(video_id, []).append(idx)
            idx2video.append(video_id)
        self.video_2_idx = video2idx
        self.idx_2_video = idx2video


    def get_cls2video(self):
        video2cls = dict()
        cls2video = list()
        for i in range(len(self.class_map[1:])):
            cls2video.append([])
            video_set_path = '/home/mfhan/ILSVRC/ImageSets/VID/train_{}.txt'.format(i+1)
            video_set = mmcv.list_from_file(video_set_path)
            for v_line in video_set:
                video_id = v_line.strip().split(' ')[0].strip().split('/')[-1]
                video2cls[video_id] = i
                cls2video[i].append(video_id)
        self.video_2_cls = video2cls
        self.cls_2_video = cls2video


    def get_indices(self, world_size):
        print("starting VIDSeqDataset Slicing")
        assert self.test_mode, "This function is only provided for test use."
        import math
        avg_size = math.ceil(self.size / world_size)
        indices_list = [[] for _ in range(world_size)]
        local_video_list = [[] for _ in range(world_size)]
        tmp_len = 0
        tmp_rank = 0
        pos_pointer = 0
        local_video_id = 0
        # list of all global video id <-> global image idx mapping
        self.global_video_list = []
        for i,video_info in enumerate(self.img_infos):
            self.global_video_list.extend([i]*video_info['frame_seg_len'])
            if tmp_len + video_info['frame_seg_len'] <= avg_size:
                tmp_len += video_info['frame_seg_len']
                self.img_infos[i]['frame_id'] -= 0 if tmp_rank == 0 else sum(len(l) for l in local_video_list[:tmp_rank])
                indices_list[tmp_rank].extend(list(np.arange(video_info['frame_seg_len'])+pos_pointer))
                local_video_list[tmp_rank].extend([local_video_id]*video_info['frame_seg_len'])
                local_video_id += 1
            else:
                if tmp_rank != world_size - 1:
                    tmp_rank += 1
                    local_video_id = 0
                    tmp_len = 0
                # video_info['frame_id'] -= pos_pointer)
                self.img_infos[i]['frame_id'] -= 0 if tmp_rank == 0 else sum(len(l) for l in local_video_list[:tmp_rank])
                indices_list[tmp_rank].extend(list(np.arange(video_info['frame_seg_len'])+pos_pointer))
                local_video_list[tmp_rank].extend([local_video_id]*video_info['frame_seg_len'])
                local_video_id += 1
                tmp_len += video_info['frame_seg_len']
                
            pos_pointer += video_info['frame_seg_len']
            # pos_pointer += video_info['frame_seg_len']
        self.indices_list = indices_list
        # list of local video id <-> local image idx mapping
        self.local_video_list = local_video_list
        # print(indices_list)
        self.local_frame_size_list = [len(indices) for indices in indices_list]
        self.global_video_size_list = [len(np.unique(l)) for l in self.local_video_list]
        return indices_list
        
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


    def __len__(self):
        if self.test_mode:
            return self.size
        else:
            return len(self.img_infos)

    def prepare_test_img(self, idx):
        self.cur_video_index = self.global_video_list[idx]
        if self.cur_tid == 0:  # new video
            self.key_frame_flag = 0
            self.cur_video = self.img_infos[self.cur_video_index].copy()
            if 'frame_seg_len' in self.cur_video:
                self.cur_seg_len = self.cur_video['frame_seg_len']
            elif 'video_len' in self.cur_video:
                self.cur_seg_len = self.cur_video['video_len']
            else:
                assert False, 'unknown video length type'
            self.video_index = np.arange(self.cur_seg_len).tolist()
            if self.video_shuffle:
                np.random.shuffle(self.video_index)
        else:  # normal frame
            self.key_frame_flag = 2

        if self.video_shuffle:
            cur_frame_offset = self.video_index[self.cur_tid]
        else:
            cur_frame_offset = self.cur_tid

        frame_img_infos, _, _ = self.make_img_info_anno_info(self.cur_video, [cur_frame_offset])
        results = dict(img_info=frame_img_infos[0])
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        try:
            cur_res=self.pipeline(results)
        except FileNotFoundError:
            print(results)
            print(cur_frame_offset)
            print(idx)
            print(self.cur_tid)
            print(self.video_index)
        #TODO: Add frame_offsets and video status to `image_meta` 
        if isinstance(cur_res['img_meta'], list):
            for i in range(len(cur_res['img_meta'])):
                cur_res['img_meta'][i].data.update(
                            dict(frame_offset=cur_frame_offset,
                                key_frame_flag=self.key_frame_flag,
                                seg_len = self.cur_video['frame_seg_len']
                            ))
        else:
            cur_res['img_meta'].data.update(
                            dict(frame_offset=cur_frame_offset,
                                key_frame_flag=self.key_frame_flag,
                                seg_len = self.cur_video['frame_seg_len']
                            ))
        # self.data_batch = deque(maxlen=self.que_len)
        # self.feat_batch = deque(maxlen=self.que_len)
        return cur_res

    '''
    Called when key frame pipeline is finished, 
    to generate a dynamic condition pipeline acc. to key_flipped
    '''
    def get_condition_pipeline(self, key_flipped):
        img_norm_cfg = dict(
            mean=[103.06, 115.90, 123.15], std=[1.0, 1.0, 1.0], to_rgb=False)
        self.condition_pipeline = [
            dict(type='LoadImageFromFile', to_float32=True),
            # dict(type='LoadAnnotations', with_bbox=self.selsa_with_aug, with_label=self.selsa_with_aug),
            dict(type='LoadAnnotations', with_bbox=True, with_label=True),
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
            dict(type='Pad', size_divisor=16),#32
            dict(type='DefaultFormatBundle'),
            # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'] \
            #                             if self.selsa_with_aug else ['img'])
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]

    #TODO: overwrite __getitem__ to tack the three images and return
    def __getitem__(self, idx):
        # print("VID!")
        if self.test_mode:
            frame_res = self.prepare_test_img(idx)
            # frame_offset = self.video_index[self.cur_tid] # offset to the first preifined frame in current video(roidb)
            # self.cur += self.batch_size # the parameter `idx` means the same
            self.cur_tid += 1
            if self.cur_tid == self.cur_seg_len:
                self.cur_video_index += 1 # id of video if current mini-batch
                self.cur_tid = 0 # currend frame id (defined by reading order) within video cur_roidb_index
                self.key_frame_flag = 1
            return frame_res
        while True:
            if not hasattr(self, 'extra_cls') or self.extra_cls == 0 :
                data = self.prepare_train_img(idx)
            else:
                data = self.prepare_train_img(idx, self.extra_cls, self.video_per_cls)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def pre_con_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []

    #TODO: overwrite to get im, bef_im and aft_im
    def prepare_train_img(self, idx, extra_cls=0, video_per_cls=1):
        res_list = []
        idx_list = self.sample_videos(idx, extra_cls, video_per_cls)

        # print(idx_list)
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
            assert 'pattern' in img_info, "There should be a pattern for VIDSeq"
            offsets = np.random.choice(self.MAX_OFFSET - self.MIN_OFFSET + 1, 2, replace=False) + self.MIN_OFFSET
            bef_id = min(max(img_info['frame_seg_id'] + offsets[0], 0), img_info['frame_seg_len'] - 1)
            aft_id = min(max(img_info['frame_seg_id'] + offsets[1], 0), img_info['frame_seg_len'] - 1)
            con_img_infos, con_ann_infos, discard_flags = self.make_img_info_anno_info(img_info, [bef_id, aft_id])

            # assert bef_id!=aft_id, "{}".format(img_info['frame_seg_id'])
            if bef_id == aft_id and self.selsa_with_aug:
                discard_flags[np.random.randint(0,2)]=True
            for i, discard in enumerate(discard_flags):
                still_discard=not discard
                while discard:
                    offsets = np.random.choice(self.MAX_OFFSET - self.MIN_OFFSET + 1, 2, replace=False) + self.MIN_OFFSET
                    new_id = min(max(img_info['frame_seg_id'] + offsets[i], 0), img_info['frame_seg_len'] - 1)
                    new_img_infos, new_ann_infos, new_discard_flags = self.make_img_info_anno_info(img_info, [new_id])
                    still_discard = discard = new_discard_flags[0]
                if not still_discard:
                    # print("!!!Discard occur!!! i {} from id {} to new_id {} in video {} with curren id {}".format(
                    #                                                         i, [bef_id,aft_id][i], new_id, 
                    #                                                         img_info['pattern'], img_info['frame_seg_id']))
                    con_img_infos[i] = new_img_infos[0]
                    con_ann_infos[i] = new_ann_infos[0]

            [bef_img_info, aft_img_info] = con_img_infos
            [bef_anno_info, aft_anno_info] = con_ann_infos
            if self.selsa_with_aug:
                bef_results = dict(img_info=bef_img_info, ann_info=bef_anno_info)
                aft_results = dict(img_info=aft_img_info, ann_info=aft_anno_info)
            else:
                bef_results = dict(img_info=bef_img_info, ann_info=ann_info)
                aft_results = dict(img_info=aft_img_info, ann_info=ann_info)
            self.get_condition_pipeline(key_flipped)
            self.supportive_pipeline = Compose(self.condition_pipeline)
            #TODO: readin bef_im and aft_im and do pipeline seperately
            self.pre_con_pipeline(bef_results)
            bef_res=self.supportive_pipeline(bef_results)
            self.pre_con_pipeline(aft_results)
            aft_res=self.supportive_pipeline(aft_results)
            # n_res = collate([key_res, bef_res, aft_res],3)
            assert bef_res is not None, "!!!VIDSeq: bef_res can't be Nonetype!!!"
            assert aft_res is not None, "!!!VIDSeq: aft_res can't be Nonetype!!!"
            # assert img_info['filename'].strip().split[-1] == key_res['img_meta'].data['filename'].strip().split[-1]
            res_list.extend([key_res, bef_res, aft_res])
        return res_list

    '''
    Make im_info and anno_info for bef_im and aft_im while training
    And also make frame info by video_info fed in
    '''
    def make_img_info_anno_info(self, img_info, conditon_seg_ids):
        con_img_infos=[]
        con_anno_infos=[]
        discard_flags=[]
        for con_seg_id in conditon_seg_ids:
            con_image_set_index = img_info['pattern'] % con_seg_id
            if not self.test_mode:
                con_xml_path = osp.join(self.img_prefix, 'Annotations',
                                        '{}.xml'.format(con_image_set_index)) 
                con_anno_info, [width, height], len_bboxes = self.get_anno_info_by_path(con_xml_path)
                con_anno_infos.append(con_anno_info)
            con_img_info = img_info.copy()
            con_img_info.update(dict(id=con_image_set_index))
            con_img_info.update(dict(filename='JPEGImages/{}.JPEG'.format(con_image_set_index)))
            con_img_info.update(dict(frame_seg_id=con_seg_id))
            if not self.test_mode:
                con_img_info.update(dict(width=width))
                con_img_info.update(dict(height=height))
                discard_flags.append(False if not self.selsa_with_aug else True if len_bboxes==0 else False)
            else:
                discard_flags = None
            con_img_infos.append(con_img_info)

        return con_img_infos, con_anno_infos, discard_flags

    def load_annotations(self, ann_file):
        img_infos = []
        lines = [x.strip().split(' ') for x in mmcv.list_from_file(ann_file)]
        assert len(lines[0])==4, \
            "It should contain segment information in the imageset file for VIDSeq"
        for line in lines:
            image_set_index = '%s/%06d' % (line[0], int(line[2]))
            filename = 'JPEGImages/{}.JPEG'.format(image_set_index)
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(image_set_index))
            pattern = line[0]+'/%06d' 
            frame_id = int(line[1])
            frame_seg_id = int(line[2])
            frame_seg_len = int(line[3])
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            img_infos.append(
                dict(id=image_set_index, filename=filename, width=width, height=height, num_annos=len(root.findall('object')),
                     pattern=pattern, frame_id=frame_id, frame_seg_id=frame_seg_id, frame_seg_len=frame_seg_len))

        return img_infos

    

    def get_ann_info(self, idx):
        if not self.test_mode:
            img_id = self.img_infos[idx]['id']
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(img_id))
            ann, _, _ = self.get_anno_info_by_path(xml_path)
        else:
            self.cur_video_index = self.global_video_list[idx]
            self.cur_video = self.img_infos[self.cur_video_index]
            if 'frame_seg_len' in self.cur_video:
                self.cur_seg_len = self.cur_video['frame_seg_len']
            elif 'video_len' in self.cur_video:
                self.cur_seg_len = self.cur_video['video_len']
            else:
                assert False, 'unknown video length type'
            frame_offset = self.cur_tid
            image_set_index = self.img_infos[self.cur_video_index]['pattern']%frame_offset
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                '{}.xml'.format(image_set_index))
            ann, _, _ = self.get_anno_info_by_path(xml_path)
            # self.cur += self.batch_size # the parameter `idx` means the same
            self.cur_tid += 1
            if self.cur_tid == self.cur_seg_len:
                self.cur_video_index += 1 # id of video if current mini-batch
                self.cur_tid = 0 # currend frame id (defined by reading order) within video cur_roidb_index
        return ann