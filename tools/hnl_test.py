import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile
import time
import random
import numpy as np
from collections import deque
import numpy as np

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, collate
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet import datasets
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def single_selsa_gpu_test(model, data_loader, all_frame_interval=21, show=False, rank=0, world_size=1):
    model.eval()
    results = []
    dataset = data_loader.dataset
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    num_images = dataset.size
    video_frames_strts = []
    for x in dataset.img_infos:
        if 'unique_ids' in x:
            video_frames_strts.append(x['unique_ids'])
        else:
            video_frames_strts.append(x['frame_id'])
    video_seg_lens = [x['frame_seg_len'] if 'frame_seg_len' in x else x['video_len'] \
                        for x in dataset.img_infos]
    batch_size = False
    video_idx = -1
    frame_idx = 0
    frame_ids = np.zeros(num_images, dtype=np.int)
    all_bboxes = [None for i in range(num_images)]

    t = time.time()
    for i, data in enumerate(data_loader):
        batch_size = batch_size if batch_size else len(data['img_meta'].data[0])
        img_meta = data['img_meta'].data[0][0]
        frame_offset = img_meta['frame_offset']
        key_frame_flag = dataset.key_frame_flag
        seg_len = img_meta['seg_len']

        t_data = time.time() - t
        t = time.time()
        
        #TODO:X An implementation in `selsa_rcnn` assembling `base.forward()` to extract backbone features
        #TODO:X An api feeding collated backbone features into selsa_bboxhead
        #TODO:X Collect the detection results as the original function does
        if key_frame_flag == 0:
            feat_list = deque(maxlen=all_frame_interval)
            frame_offset_list = deque(maxlen=all_frame_interval)
            img_meta_list = deque(maxlen=all_frame_interval)
            video_idx += 1
            with torch.no_grad():
                cur_feat = model(backbone_feat=True, **data)
            while len(feat_list) < int(all_frame_interval+1)/2:
                feat_list.append(cur_feat[0])
                frame_offset_list.append(frame_offset)
                img_meta_list.append(img_meta)

        elif key_frame_flag == 2:
            if len(feat_list) < all_frame_interval - 1:
                with torch.no_grad():
                    feat = model(backbone_feat=True, **data)
                feat_list.append(feat[0])
                frame_offset_list.append(frame_offset)
                img_meta_list.append(img_meta)
            else:
                with torch.no_grad():
                    feat = model(backbone_feat=True, **data)
                feat_list.append(feat[0])
                frame_offset_list.append(frame_offset)
                img_meta_list.append(img_meta)
                with torch.no_grad():
                    # c_img_meta = collate(img_meta_list, all_frame_interval)
                    result = model(x=feat_list, img=None, img_meta=img_meta_list, 
                                   forward_feat=True, return_loss=False, rescale=not show)
                    # print(result)
                if dataset.video_shuffle:
                    if not isinstance(video_frames_strts[video_idx], int):
                        frame_ids[frame_idx] = video_frames_strts[video_idx][
                                frame_offset_list[int((all_frame_interval-1)/2)]]
                    else:
                        frame_ids[frame_idx] = video_frames_strts[video_idx] + \
                                frame_offset_list[int((all_frame_interval-1)/2)]
                else:
                    assert "Unshuffled video validation not implemented"
                all_bboxes[frame_ids[frame_idx]-1] = result
                frame_idx += batch_size
                t_net = time.time() - t
                if rank == 0:
                    for _ in range(batch_size * world_size):
                        prog_bar.update()
        elif key_frame_flag == 1:
            end_counter = 0
            with torch.no_grad():
                feat = model(backbone_feat=True, **data)

            while len(feat_list) < all_frame_interval - 1:
                feat_list.append(feat[0])
                frame_offset_list.append(frame_offset)
                img_meta_list.append(img_meta)
            while end_counter < min(seg_len, int(all_frame_interval+1)/2):
                feat_list.append(feat[0])
                frame_offset_list.append(frame_offset)
                img_meta_list.append(img_meta)
                end_counter += 1
                with torch.no_grad():
                    # c_img_meta = collate(img_meta_list, all_frame_interval)
                    result = model(x=feat_list, img=None, img_meta=img_meta_list, 
                                   forward_feat=True, return_loss=False, rescale=not show)
                
                if dataset.video_shuffle:
                    if not isinstance(video_frames_strts[video_idx], int):
                        frame_ids[frame_idx] = video_frames_strts[video_idx][
                                frame_offset_list[int((all_frame_interval-1)/2)]]
                    else:
                        frame_ids[frame_idx] = video_frames_strts[video_idx] + \
                                frame_offset_list[int((all_frame_interval-1)/2)]
                else:
                    assert "Unshuffled video validation not implemented"
                all_bboxes[frame_ids[frame_idx]-1] = result
                frame_idx += batch_size
                t_net = time.time() - t
                if rank == 0:
                    for _ in range(batch_size * world_size):
                        prog_bar.update()

    return all_bboxes

def multi_selsa_gpu_test(model, data_loader, all_frame_interval, tmpdir=None, gpu_collect=False, show=False):
    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        dataset.is_dataset_global=['I am going to be mad!!!']

    # print("rank: {} message: {}".format(rank, hasattr(dataset, 'is_dataset_global')))

    num_images = dataset.local_frame_size_list[rank]
    # print(num_images)
    # print(num_images)
    pos_pointer=[np.max(t)+1 for t in dataset.local_video_list]
    video_frames_strts = []
    if rank == 0:
        rank_video_infos = dataset.img_infos[:dataset.global_video_size_list[rank]]
    else:
        rank_video_infos = dataset.img_infos[
            sum(dataset.global_video_size_list[:rank]):sum(dataset.global_video_size_list[:rank+1])]
    for i,x in enumerate(rank_video_infos):
        if i == 0:
            assert x['frame_id'] == 1, "Wrong frame_id of first video in local rank {}".format(rank)
        if 'unique_ids' in x:
            video_frames_strts.append(x['unique_ids'])
        else:
            video_frames_strts.append(x['frame_id'])
    video_seg_lens = [x['frame_seg_len'] if 'frame_seg_len' in x else x['video_len'] \
                        for x in rank_video_infos]
    batch_size = False
    video_idx = -1
    frame_idx = 0
    frame_ids = np.zeros(num_images, dtype=np.int)
    all_bboxes = [None for i in range(num_images)]

    t = time.time()
    print("rank: {} len: {} size: {}".format(rank, len(data_loader), num_images))
    for i, data in enumerate(data_loader):
        batch_size = batch_size if batch_size else len(data['img_meta'].data[0])
        img_meta = data['img_meta'].data[0][0]
        frame_offset = img_meta['frame_offset']
        key_frame_flag = dataset.key_frame_flag
        seg_len = img_meta['seg_len']

        t_data = time.time() - t
        t = time.time()
        # print("i at rank {}: {}".format(rank, i))
        #TODO:X An implementation in `selsa_rcnn` assembling `base.forward()` to extract backbone features
        #TODO:X An api feeding collated backbone features into selsa_bboxhead
        #TODO:X Collect the detection results as the original function does
        if key_frame_flag == 0:
            feat_list = deque(maxlen=all_frame_interval)
            frame_offset_list = deque(maxlen=all_frame_interval)
            img_meta_list = deque(maxlen=all_frame_interval)
            video_idx += 1
            with torch.no_grad():
                cur_feat = model(backbone_feat=True, **data)
            while len(feat_list) < int(all_frame_interval+1)/2:
                feat_list.append(cur_feat[0])
                frame_offset_list.append(frame_offset)
                img_meta_list.append(img_meta)

        elif key_frame_flag == 2:
            if len(feat_list) < all_frame_interval - 1:
                with torch.no_grad():
                    feat = model(backbone_feat=True, **data)
                feat_list.append(feat[0])
                frame_offset_list.append(frame_offset)
                img_meta_list.append(img_meta)
            else:
                with torch.no_grad():
                    feat = model(backbone_feat=True, **data)
                feat_list.append(feat[0])
                frame_offset_list.append(frame_offset)
                img_meta_list.append(img_meta)
                with torch.no_grad():
                    # c_img_meta = collate(img_meta_list, all_frame_interval)
                    result = model(x=feat_list, img=None, img_meta=img_meta_list, 
                                   forward_feat=True, return_loss=False, rescale=not show)
                    # print(result)
                if dataset.video_shuffle:
                    # print("video_indx: {}, len_video_frames_strts: {}".format(video_idx, len(video_frames_strts)))
                    if not isinstance(video_frames_strts[video_idx], int):
                        frame_ids[frame_idx] = video_frames_strts[video_idx][
                                frame_offset_list[int((all_frame_interval-1)/2)]]
                    else:
                        frame_ids[frame_idx] = video_frames_strts[video_idx] + \
                                frame_offset_list[int((all_frame_interval-1)/2)]
                else:
                    assert "Unshuffled video validation not implemented"
                try:
                    all_bboxes[frame_ids[frame_idx]-1] = result
                except:
                    print("rank: {}, frame_idx: {}, frame_ids[frame_idx]:{}".format(rank, frame_idx, frame_ids[frame_idx]))
                frame_idx += batch_size
                t_net = time.time() - t
                if rank == 0:
                    for _ in range(batch_size * world_size):
                        prog_bar.update()
        elif key_frame_flag == 1:
            end_counter = 0
            with torch.no_grad():
                feat = model(backbone_feat=True, **data)

            while len(feat_list) < all_frame_interval - 1:
                feat_list.append(feat[0])
                frame_offset_list.append(frame_offset)
                img_meta_list.append(img_meta)
            while end_counter < min(seg_len, int(all_frame_interval+1)/2):
                feat_list.append(feat[0])
                frame_offset_list.append(frame_offset)
                img_meta_list.append(img_meta)
                end_counter += 1
                with torch.no_grad():
                    # c_img_meta = collate(img_meta_list, all_frame_interval)
                    result = model(x=feat_list, img=None, img_meta=img_meta_list, 
                                   forward_feat=True, return_loss=False, rescale=not show)
                
                if dataset.video_shuffle:
                    if not isinstance(video_frames_strts[video_idx], int):
                        frame_ids[frame_idx] = video_frames_strts[video_idx][
                                frame_offset_list[int((all_frame_interval-1)/2)]]
                    else:
                        frame_ids[frame_idx] = video_frames_strts[video_idx] + \
                                frame_offset_list[int((all_frame_interval-1)/2)]
                else:
                    assert "Unshuffled video validation not implemented"
                try:
                    all_bboxes[frame_ids[frame_idx]-1] = result
                except:
                    print("rank: {}, frame_idx: {}, frame_ids[frame_idx]:{}, end_counter")
                frame_idx += batch_size
                t_net = time.time() - t
                if rank == 0:
                    for _ in range(batch_size * world_size):
                        prog_bar.update()
        # if rank == 3:
        #     print(i)


    # print("results at rank {}: {}".format(rank, len(all_bboxes)))
    # print(all_bboxes)
    if gpu_collect:       
        assert "gpu_collect Not implemented yet!"
    else:
        # print(all_bboxes)
        results = collect_selsa_results_cpu(all_bboxes, len(dataset), tmpdir)
    return results

def pre_padding_imgs(dataset, seg_len, num):
    video_index = np.arange(seg_len).tolist()
    np.random.shuffle(video_index)
    chosen_idx = np.random.choice(video_index, num, replace=num>seg_len).tolist()
    # chosen_idx = random.sample(video_index, num)
    frame_img_infos, _, _ = dataset.make_img_info_anno_info(dataset.cur_video, chosen_idx)

    res = []
    for img_info in frame_img_infos:
        results = dict(img_info=img_info)
        dataset.pre_pipeline(results)
        cur_res=dataset.pipeline(results)
        res.append(cur_res)
    c_res = collate(res, num)
    return c_res

def multi_hnl_gpu_test(model, data_loader, all_frame_interval, tmpdir=None, gpu_collect=False, show=False):
    model.eval()
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
        dataset.is_dataset_global=['I am going to be mad!!!']

    # print("rank: {} message: {}".format(rank, hasattr(dataset, 'is_dataset_global')))

    num_images = dataset.local_frame_size_list[rank]
    # print(num_images)
    # print(num_images)
    pos_pointer=[np.max(t)+1 for t in dataset.local_video_list]
    video_frames_strts = []
    if rank == 0:
        rank_video_infos = dataset.img_infos[:dataset.global_video_size_list[rank]]
    else:
        rank_video_infos = dataset.img_infos[
            sum(dataset.global_video_size_list[:rank]):sum(dataset.global_video_size_list[:rank+1])]
    for i,x in enumerate(rank_video_infos):
        if i == 0:
            assert x['frame_id'] == 1, "Wrong frame_id of first video in local rank {}".format(rank)
        if 'unique_ids' in x:
            video_frames_strts.append(x['unique_ids'])
        else:
            video_frames_strts.append(x['frame_id'])
    video_seg_lens = [x['frame_seg_len'] if 'frame_seg_len' in x else x['video_len'] \
                        for x in rank_video_infos]
    batch_size = False
    video_idx = -1
    frame_idx = 0
    frame_ids = np.zeros(num_images, dtype=np.int)
    all_bboxes = [None for i in range(num_images)]

    t = time.time()
    print("rank: {} len: {} size: {}".format(rank, len(data_loader), num_images))
    for i, data in enumerate(data_loader):
        batch_size = batch_size if batch_size else len(data['img_meta'].data[0])
        img_meta = data['img_meta'].data[0][0]
        frame_offset = img_meta['frame_offset']
        key_frame_flag = dataset.key_frame_flag
        seg_len = img_meta['seg_len']

        t_data = time.time() - t
        t = time.time()
        # print("i at rank {}: {}".format(rank, i))
        #TODO:X An implementation in `selsa_rcnn` assembling `base.forward()` to extract backbone features
        #TODO:X An api feeding collated backbone features into selsa_bboxhead
        #TODO:X Collect the detection results as the original function does
        if key_frame_flag == 0:
            feat_list = deque(maxlen=all_frame_interval)
            frame_offset_list = deque(maxlen=all_frame_interval)
            img_meta_list = deque(maxlen=all_frame_interval)
            video_idx += 1
            with torch.no_grad():
                cur_feat = model(backbone_feat=True, **data)
            # This is the original fill-up method
            # while len(feat_list) < int((all_frame_interval+1)/2):
            #     feat_list.append(cur_feat[0])
            #     frame_offset_list.append(frame_offset)
            #     img_meta_list.append(img_meta)
            # This is modified mothod that pre-fetch `int(all_frame_interval-1)/2 ` frames
            pre_frames = pre_padding_imgs(dataset, seg_len, int((all_frame_interval-1)/2))
            with torch.no_grad():
                pre_feat = model(backbone_feat=True, **pre_frames)
            feat_list.extend(pre_feat[0].split(1, dim=0))
            feat_list.append(cur_feat[0])
            frame_offset_list.extend([-1]*pre_feat[0].shape[0])
            frame_offset_list.append(frame_offset)
            img_meta_list.extend(pre_frames['img_meta'].data[0])
            img_meta_list.append(img_meta)
        elif key_frame_flag == 2:
            if len(feat_list) < all_frame_interval - 1:
                with torch.no_grad():
                    feat = model(backbone_feat=True, **data)
                feat_list.append(feat[0])
                frame_offset_list.append(frame_offset)
                img_meta_list.append(img_meta)
            else:
                with torch.no_grad():
                    feat = model(backbone_feat=True, **data)
                feat_list.append(feat[0])
                frame_offset_list.append(frame_offset)
                img_meta_list.append(img_meta)
                with torch.no_grad():
                    # c_img_meta = collate(img_meta_list, all_frame_interval)
                    result = model(x=feat_list, img=None, img_meta=img_meta_list, 
                                   forward_feat=True, return_loss=False, rescale=not show)
                    # print(result)
                if dataset.video_shuffle:
                    # print("video_indx: {}, len_video_frames_strts: {}".format(video_idx, len(video_frames_strts)))
                    if not isinstance(video_frames_strts[video_idx], int):
                        frame_ids[frame_idx] = video_frames_strts[video_idx][
                                frame_offset_list[int((all_frame_interval-1)/2)]]
                    else:
                        frame_ids[frame_idx] = video_frames_strts[video_idx] + \
                                frame_offset_list[int((all_frame_interval-1)/2)]
                else:
                    assert "Unshuffled video validation not implemented"
                try:
                    all_bboxes[frame_ids[frame_idx]-1] = result
                except:
                    print("rank: {}, frame_idx: {}, frame_ids[frame_idx]:{}".format(rank, frame_idx, frame_ids[frame_idx]))
                frame_idx += batch_size
                t_net = time.time() - t
                if rank == 0:
                    for _ in range(batch_size * world_size):
                        prog_bar.update()
        elif key_frame_flag == 1:
            end_counter = 0
            with torch.no_grad():
                feat = model(backbone_feat=True, **data)

            # The original method
            # while len(feat_list) < all_frame_interval - 1:
            #     feat_list.append(feat[0])
            #     frame_offset_list.append(frame_offset)
            #     img_meta_list.append(img_meta)
            # The new method with frames pre-fetching
            while end_counter < min(seg_len, int((all_frame_interval+1)/2)):
                feat_list.append(feat[0])
                frame_offset_list.append(frame_offset)
                img_meta_list.append(img_meta)
                end_counter += 1
                if len(feat_list) < all_frame_interval - 1:
                    pre_frames = pre_padding_imgs(dataset, seg_len, all_frame_interval-len(feat_list))
                    with torch.no_grad():
                        pre_feat = model(backbone_feat=True, **pre_frames)
                    feat_list.extend(pre_feat[0].split(1, dim=0))
                    frame_offset_list.extend([-1]*pre_feat[0].shape[0])
                    img_meta_list.extend(pre_frames['img_meta'].data[0])
                with torch.no_grad():
                    # c_img_meta = collate(img_meta_list, all_frame_interval)
                    result = model(x=feat_list, img=None, img_meta=img_meta_list, 
                                   forward_feat=True, return_loss=False, rescale=not show)
                
                if dataset.video_shuffle:
                    if not isinstance(video_frames_strts[video_idx], int):
                        frame_ids[frame_idx] = video_frames_strts[video_idx][
                                frame_offset_list[int((all_frame_interval-1)/2)]]
                    else:
                        frame_ids[frame_idx] = video_frames_strts[video_idx] + \
                                frame_offset_list[int((all_frame_interval-1)/2)]
                else:
                    assert "Unshuffled video validation not implemented"
                try:
                    all_bboxes[frame_ids[frame_idx]-1] = result
                except:
                    print("rank: {}, frame_idx: {}, frame_ids[frame_idx]:{}, end_counter")
                frame_idx += batch_size
                t_net = time.time() - t
                if rank == 0:
                    for _ in range(batch_size * world_size):
                        prog_bar.update()
        # if rank == 3:
        #     print(i)


    # print("results at rank {}: {}".format(rank, len(all_bboxes)))
    # print(all_bboxes)
    if gpu_collect:       
        assert "gpu_collectNot implemented yet!"
    else:
        # print(all_bboxes)
        results = collect_selsa_results_cpu(all_bboxes, len(dataset), tmpdir)
    return results

def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_selsa_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    print(tmpdir)
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            tmp_res = mmcv.load(part_file)
            print("rank {}: {}".format(i, len(tmp_res)))
            part_list.extend(tmp_res)
        # sort the results
        ordered_results = part_list
        # for res in zip(*part_list):
        #     print(len(list(res)))
        #     ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    print(tmpdir)
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            tmp_res = mmcv.load(part_file)
            part_list.append(tmp_res)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            print(len(list(res)))
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    _, world_size = get_dist_info()
    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)

    num_data_loader = 3
    data_loaders = []
    for i in range(num_data_loader):
        dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets, dict(test_mode=True, world_size=world_size))
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            selsa_imgs=cfg.data.selsa_imgs,
            shuffle=False)
        data_loaders.append(data_loader)
    # del data_loader

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show)
        outputs = multi_hnl_gpu_test(model, data_loader, 63, args.tmpdir,
                                 args.gpu_collect)
    else:
        model = MMDistributedDataParallel(model.cuda())
        # outputs = multi_gpu_test(model, data_loader, args.tmpdir,
        #                          args.gpu_collect)
        # outputs = multi_selsa_gpu_test(model, data_loader, 21, args.tmpdir,
        #                          args.gpu_collect)
        outputs = multi_hnl_gpu_test(model, data_loader, 63, args.tmpdir,
                                 args.gpu_collect)
        print("start_eval")

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json(dataset, outputs, args.out)
                    coco_eval(result_files, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file)
                        coco_eval(result_files, eval_types, dataset.coco)

    # Save predictions in the COCO json format
    if args.json_out and rank == 0:
        if not isinstance(outputs[0], dict):
            results2json(dataset, outputs, args.json_out)
        else:
            for name in outputs[0]:
                outputs_ = [out[name] for out in outputs]
                result_file = args.json_out + '.{}'.format(name)
                results2json(dataset, outputs_, result_file)


if __name__ == '__main__':
    main()
