#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import pickle as pkl
import numpy as np

import fire
import torch
import mmcv
from mmdet.datasets import PIPELINES
from nuscenes.nuscenes import NuScenes
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile
from mmdet3d.models.fusion_layers.point_fusion import projection
from mmdet3d.core.bbox import box_np_ops


def img(pkl_path, data_root, data_version):
    """
    >>> python ./tools/script/dbinfo_converter.py img ./data/nuscenes/nuscenes_dbinfos_train.pkl \
                ./data/nuscenes/ v1.0-trainval
    """
    cam_list = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
    nusc = NuScenes(version=data_version, dataroot=data_root)

    def make_img_meta(dbinfo):
        img_meta = {}
        img_meta['sample_idx'] = dbinfo['image_idx']
        samp_data = nusc.get('sample', dbinfo['image_idx'])
        filenames = [
            nusc.get('sample_data', samp_data['data'][c_n])['filename']
            for c_n in cam_list
        ]
        img_meta['filename'] = filenames
        img_meta['ori_shape'] = [900, 1600]
        return img_meta

    info = pkl.load(open(pkl_path, 'rb'))
    points_loader = LoadPointsFromFile(
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=dict(backend='disk'))

    for info_cls in info:
        for idx in range(len(info[info_cls])):
            print('%s: %d / %d' % (info_cls, idx, len(info[info_cls])))
            db = info[info_cls][idx]
            file_path = os.path.join(data_root, db['path'])
            results = dict(pts_filename=file_path)
            s_points = points_loader(results)['points']
            s_points.translate(db['box3d_lidar'][:3])
            s_gt_bbox = np.expand_dims(db['box3d_lidar'], 0)
            bbox_corners = box_np_ops.center_to_corner_box3d(
                s_gt_bbox[:, :3],
                s_gt_bbox[:, 3:6],
                s_gt_bbox[:, 6],
                origin=(0.5, 0.5, 0),
                axis=2)[0]
            bbox_corners = np.vstack(
                [s_points.tensor[:, :3].numpy(), bbox_corners])
            img_meta = make_img_meta(db)
            img_list = []
            for c_i, c_name in enumerate(cam_list):
                img_list.append('')
                pts_2d, point_idx = projection(
                    torch.tensor(bbox_corners),
                    nusc,
                    img_meta,
                    c_i,
                    data_root=data_root)
                if len(point_idx) == 0:
                    continue
                p_min, p_max = pts_2d.min(0)[0].to(
                    torch.int), pts_2d.max(0)[0].to(torch.int)
                if (p_max[1] - p_min[1]) < 10 or (p_max[0] - p_min[0]) < 10:
                    continue
                img = mmcv.imread(nusc.dataroot + img_meta['filename'][c_i],
                                  'unchanged')
                img_crop = img[p_min[1]:p_max[1], p_min[0]:p_max[0]]
                img_path = '%s/nuscenes_img_gt_database/%s_%s_%s_%s.png' % (
                    data_root, db['image_idx'], info_cls, db['gt_idx'], c_name)
                cv2.imwrite(img_path, img_crop)
                img_list[c_i] = img_path
            info[info_cls][idx] = img_list
    pkl.dump(info, open('nuscenes_imgdbinfos_train.pkl', 'wb'))


if __name__ == '__main__':
    fire.Fire()
