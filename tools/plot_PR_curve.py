import mmcv
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from mmdet.core import average_precision
from mmdet.core import imagenet_vid_classes
from torch.utils.tensorboard import SummaryWriter

class_names = imagenet_vid_classes()

# meta_analysis_res_1 = mmcv.load('/home/mfhan/mmdetection/hnl_epoch4_2018_meta_analysis.pkl')
# meta_analysis_res_2 = mmcv.load('/home/mfhan/mmdetection/selsa_epoch_12_meta_analysis.pkl')
meta_analysis_res_1 = mmcv.load('/home/mfhan/mmdetection/hnmb_branch_meta_analysis.pkl')
meta_analysis_res_2 = mmcv.load('/home/mfhan/mmdetection/hnmb_mining_meta_analysis.pkl')

eval_results = []
name = ['hnmb_branch','hnmb_mining_meta']

writers = []
for n in name:
    writer = SummaryWriter(log_dir='/home/mfhan/mmdetection/work_dirs/comparison/{}'.format(n))
    writers.append(writer)

for cls_id in range(len(class_names)):
    # i=14
    meta=meta_analysis_res_1[cls_id]
    tp = meta['tp']
    fp = meta['fp']
    num_gts = meta['num_gts']
    det_scores = meta['det_scores']

    meta2 = meta_analysis_res_2[cls_id]
    tp2 = meta2['tp']
    fp2 = meta2['fp']
    num_gts2 = meta2['num_gts']
    det_scores2 = meta2['det_scores']

    x = np.arange(len(det_scores))
    # h = open("D:/Projects/mmdetection/horse.csv", 'w')
    for ind, [tp, fp, num_gts, det_scores] in enumerate([[tp, fp, num_gts, det_scores], [tp2, fp2, num_gts2, det_scores2]]):
        # # calculate recall and precision with tp and fp
        # tp = np.cumsum(tp, axis=1)
        # fp = np.cumsum(fp, axis=1)
        # eps = np.finfo(np.float32).eps
        # recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        # precisions = tp / np.maximum((tp + fp), eps)
        # # calculate AP
        # recalls = recalls[0, :]
        # precisions = precisions[0, :]
        # num_gts = num_gts.item()
        # mode = 'area'
        # ap = average_precision(recalls, precisions, mode)

        # no_scale = False
        # if recalls.ndim == 1:
        #     no_scale = True
        #     recalls = recalls[np.newaxis, :]
        #     precisions = precisions[np.newaxis, :]
        # assert recalls.shape == precisions.shape and recalls.ndim == 2
        # num_scales = recalls.shape[0]
        # ap = np.zeros(num_scales, dtype=np.float32)
        # if mode == 'area':
        #     zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        #     ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        #     mrec = np.hstack((zeros, recalls, ones))
        #     mpre = np.hstack((zeros, precisions, zeros))
        #     for i in range(mpre.shape[1] - 1, 0, -1):
        #         mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        #     for i in range(num_scales):
        #         ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
        #         ap[i] = np.sum(
        #             (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])

        # eval_results.append({
        #     'num_gts': num_gts,
        #     'recall': recalls,
        #     'precision': precisions,
        #     'ap': ap
        # })

        # sns.set_color_codes()
        weight_by_tf = tp[0]*1 + fp[0]*(-1)
        y = weight_by_tf*det_scores
        # sns.barplot(x, y, palette="Blues", ax=axes[ind])
        # # plt.bar(x,y)
        #
        # plt.show()
        # print("")
        # line = ','.join(list(map(str, y)))
        # h.writelines(line + '\n')

        writer = writers[ind]
        for i in range(15000):
            writer.add_scalar('{}/15k'.format(class_names[cls_id]), y[i], i)
        for i in range(len(y)-1):
            writer.add_scalar('{}/all'.format(class_names[cls_id]), y[i], i)
        # plt.savefig('./horse.pdf', format='pdf')
    # h.close()

for writer in writers:
    writer.close()