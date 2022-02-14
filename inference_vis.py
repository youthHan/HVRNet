from os import path as osp
import os

from mmcv.video import frames2video
from mmdet.apis import init_detector, inference_detector, show_result
import mmcv

classes_names = ['airplane', 'antelope', 'bear', 'bicycle',
					'bird', 'bus', 'car', 'cattle',
					'dog', 'domestic_cat', 'elephant', 'fox',
					'giant_panda', 'hamster', 'horse', 'lion',
					'lizard', 'monkey', 'motorcycle', 'rabbit',
					'red_panda', 'sheep', 'snake', 'squirrel',
					'tiger', 'train', 'turtle', 'watercraft',
					'whale', 'zebra']
classes_map = ['n02691156', 'n02419796', 'n02131653', 'n02834778',
						'n01503061', 'n02924116', 'n02958343', 'n02402425',
						'n02084071', 'n02121808', 'n02503517', 'n02118333',
						'n02510455', 'n02342885', 'n02374451', 'n02129165',
						'n01674464', 'n02484322', 'n03790512', 'n02324045',
						'n02509815', 'n02411705', 'n01726692', 'n02355227',
						'n02129604', 'n04468005', 'n01662784', 'n04530566',
						'n02062744', 'n02391049']
name_to_class = {classes_map[i]: classes_names[i] for i in range(len(classes_names))}
class_to_name = {classes_names[i]: classes_map[i] for i in range(len(classes_map))}

config_file = './configs/mask_rcnn_r101_fpn_1x_vid_finetune.py'
checkpoint_file = './work_dirs/mask_rcnn_r101_fpn_1x_vid_det/epoch_8.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# data_root = './data/VID/JPEGImages/'
# out_dir = './output/VID/thresh_05/'
# # test a video and show the results
# # with open('./data/VID/Imagesets/VID_val_videos.txt') as h:
# #     test_videos_frame_lists= h.readlines()
with open('/home/mfhan/sda2/Sequence-Level-Semantics-Aggregation/data/ILSVRC2015/ImageSets/VID_val_sampled_videos_2.txt','r') as h:
    test_videos_frame_lists=h.readlines()
# for video_frames_line in test_videos_frame_lists:
#     frames_path, start_from, _, frames_num = video_frames_line.strip().split()
#     print("test: {}".format(frames_path))
#     frames_output_dir = osp.join(out_dir, frames_path)
#     if not osp.isdir(frames_output_dir):
#         os.makedirs(frames_output_dir)
#     else:
#         print("exists: {}".format(frames_path))
#         continue
#     for frame_id in range(int(frames_num)):
#         frame_name = '{:06}.JPEG'.format(frame_id)
#         frame = osp.join(data_root, frames_path, frame_name)
#         result = inference_detector(model, frame)
#         show_result(frame, result, classes_names, 
#                     score_thr=0.5,
#                     wait_time=0,
#                     thickness=2,
#                     font_scale=1.1,
#                     show=False, 
#                     out_file=osp.join(frames_output_dir, frame_name))

# video_dir = './output/VID/videos/'
video_dir = '/home/mfhan/sda2/Sequence-Level-Semantics-Aggregation/output/selsa_rcnn/imagenet_vid/resnet_v1_101_rcnn_selsa_aug/VID_val_sampled_videos_2/videos/'
out_dir='/home/mfhan/sda2/Sequence-Level-Semantics-Aggregation/output/selsa_rcnn/imagenet_vid/resnet_v1_101_rcnn_selsa_aug/VID_val_sampled_videos_2/'
frame_dir = out_dir
if not osp.isdir(video_dir):
    os.makedirs(video_dir)
for video_frames_line in test_videos_frame_lists:
    frames_path, start_from, _, frames_num = video_frames_line.strip().split()
    frames_name = frames_path.split()[0].strip()[4:]
    video_path = osp.join(video_dir, '{}.mp4'.format(frames_name))
    frames2video(osp.join(frame_dir + frames_path), video_path, filename_tmpl='{:06d}.JPEG')