_base_ = './htc_r50_fpn_1x_nuim.py'
model = dict(
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))

# learning policy
lr_config = dict(step=[16, 19])
runner = dict(max_epochs=20)

optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
load_from = 'http://download.openmmlab.com/mmdetection/v2.0/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco_20200312-946fd751.pth'  # noqa

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 900),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=[
                    'filename', 'ori_filename', 'ori_shape', 'img_shape',
                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                    'img_norm_cfg', 'img_info'
                ]),
        ])
]

data_root = 'data/nuscenes__/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=3,
    train=dict(
        ann_file=data_root + 'annotations/nuimages_v1.0-train.json',
        img_prefix=data_root,
    ),
    val=dict(
        ann_file=data_root + 'nuscenes_infos_train_mono3d.coco.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        ann_file=data_root + 'nuscenes_infos_train_mono3d.coco.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
