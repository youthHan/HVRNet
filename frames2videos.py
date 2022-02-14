import os

import os.path as osp
from mmcv.video import frames2video

frames_dir='/home/mfhan/mmdetection/data/VID/vis/val/'
video_dir='/home/mfhan/mmdetection/data/VID/vis_video/'

for f_vid in os.listdir(frames_dir):
    video_name="{}.mp4".format(f_vid)
    video_path=osp.join(video_dir, video_name)
    frames2video(osp.join(frames_dir + f_vid), video_path, filename_tmpl='{:06d}.JPEG')