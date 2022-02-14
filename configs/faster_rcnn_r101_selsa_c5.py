# model settings
monash = True
norm_cfg = dict(type='BN', requires_grad=False)
rcnn_sampler_num = 128
nms_pos = 300
frame_interval = 10
net_type = 'SelsaRCNN'
if net_type in ['SelsaRCNN', 'HNLRCNN']:
    selsa_imgs = 3
    # ! for test
    # selsa_imgs = 1
    imgs_per_gpu = 1
else:
    selsa_imgs = imgs_per_gpu = 1
    
model = dict(
    type=net_type,
    # pretrained='open-mmlab://resnet101_caffe',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=3,
        strides=(1,2,2),
        dilations=(1, 1, 1),
        out_indices=(2, ),
        frozen_stages=1,
        style='caffe',
        norm_eval=True,
        norm_cfg=norm_cfg),
    shared_head=dict(
        type='ResLayer',
        depth=101,
        stage=3,
        stride=1,
        dilation=2,
        style='caffe',
        norm_eval=True,
        norm_cfg=norm_cfg,
        external_conv=True),
    rpn_head=dict(
        type='RPNHead',
        in_channels=1024,
        feat_channels=512,
        anchor_scales=[4, 8, 16, 32],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[16],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=1024,
        featmap_strides=[16],
        feat_from_shared_head=True),
    bbox_head=dict(
        type='SelsaBBoxHead',
        sampler_num=rcnn_sampler_num,
        t_dim=3,
        with_avg_pool=False,
        in_channels=256,
        fc_feat_dim=1024,
        roi_feat_size=7,
        num_classes=31,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=True,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=6000,
        nms_post=nms_pos,
        max_num=nms_pos,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=[dict(
            type='RandomSampler',
            num=nms_pos,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
            dict(
            type='OHEMHNLSampler',
            num=rcnn_sampler_num,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            # add_gt_as_proposals=False
            )
            ],
        key_dim=0,
        pos_weight=-1,
        debug=False)
    )
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=6000,
        nms_post=nms_pos,
        max_num=nms_pos,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.001, nms=dict(type='nms', iou_thr=0.3), max_per_img=300,
        key_dim=10),
    bbox_head=dict(
        sampler_num=nms_pos,
        t_dim=frame_interval*2+1,
        key_dim=10),
    relation_setup=dict(
        shuffle=False,
        video_shuffle=True,
        has_rpn=True,
        frame_interval=frame_interval,
        frame_stride=1
    ))
# dataset settings
dataset_type1 = 'VIDSeqDataset'
dataset_type2 = 'DETSeqDataset'
data_root1 = './data/VID/'
data_root2 = './data/DET/'
img_norm_cfg = dict(
    mean=[103.06, 115.90, 123.15], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
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
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=16),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=16),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    imgs_per_gpu=imgs_per_gpu,
    workers_per_gpu=1,
    # ! For debug
    # workers_per_gpu=0,
    selsa_imgs=selsa_imgs,
    train=[dict(
        type=dataset_type1,
        ann_file=data_root1 + 'ImageSets/VID_train_15frames.txt',
        img_prefix=data_root1,
        pipeline=train_pipeline,
        selsa_with_aug=True),
        dict(
        type=dataset_type2,
        ann_file=data_root2 + 'ImageSets/DET_train_30classes.txt',
        img_prefix=data_root2,
        pipeline=train_pipeline,
        selsa_with_aug=True)
        ],
    val=dict(
        type=dataset_type1,
        ann_file=data_root1 + 'ImageSets/VID_val_videos.txt',
        img_prefix=data_root1,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type1,
        ann_file=data_root1 + 'ImageSets/VID_val_videos.txt',
        img_prefix=data_root1,
        pipeline=test_pipeline))
# optimizer
if monash:
    # monash server
    optimizer = dict(type='SGD', lr=0.00025, momentum=0.9, weight_decay=0.0001)
else:
    # siat server
    optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[12])
if monash:
    # monash server
    checkpoint_config = dict(interval=1, iter_interval=6000)
else:
    # siat server
    checkpoint_config = dict(interval=1, iter_interval=3000)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 16
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/mfhan/mmdetection/work_dirs/faster_rcnn_r101_selsa_vid/selsa_c5_rcnn_ohem_agn_512_aug/resume_from_8/'
load_from = None
resume_from = work_dir+'latest.pth'
workflow = [('train', 1)]
