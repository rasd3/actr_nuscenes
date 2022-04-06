# Copyright (c) OpenMMLab. All rights reserved.
import torch
import time
import numpy as np
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.core.bbox.structures import (get_proj_mat_by_coord_type,
                                          points_cam2img)
from mmdet3d.models.model_utils.actr import build as build_actr
from ..builder import FUSION_LAYERS
from . import apply_3d_transformation

from nuscenes.utils.geometry_utils import view_points
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


def point_sample(img_meta,
                 img_features,
                 points,
                 proj_mat,
                 coord_type,
                 img_scale_factor,
                 img_crop_offset,
                 img_flip,
                 img_pad_shape,
                 img_shape,
                 aligned=True,
                 padding_mode='zeros',
                 align_corners=True):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of \
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop \
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """

    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)

    # project points to camera coordinate
    pts_2d = points_cam2img(points, proj_mat)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        orig_h, orig_w = img_shape
        coor_x = orig_w - coor_x

    h, w = img_pad_shape
    coor_y = coor_y / h * 2 - 1
    coor_x = coor_x / w * 2 - 1
    grid = torch.cat([coor_x, coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCx1xN feats

    return point_features.squeeze().t()


@FUSION_LAYERS.register_module()
class PointFusion(BaseModule):
    """Fuse image features from multi-scale features.

    Args:
        img_channels (list[int] | int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels (int): Channels of point features
        mid_channels (int): Channels of middle layers
        out_channels (int): Channels of output fused features
        img_levels (int, optional): Number of image levels. Defaults to 3.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
            Defaults to 'LIDAR'.
        conv_cfg (dict, optional): Dict config of conv layers of middle
            layers. Defaults to None.
        norm_cfg (dict, optional): Dict config of norm layers of middle
            layers. Defaults to None.
        act_cfg (dict, optional): Dict config of activatation layers.
            Defaults to None.
        activate_out (bool, optional): Whether to apply relu activation
            to output features. Defaults to True.
        fuse_out (bool, optional): Whether apply conv layer to the fused
            features. Defaults to False.
        dropout_ratio (int, float, optional): Dropout ratio of image
            features to prevent overfitting. Defaults to 0.
        aligned (bool, optional): Whether apply aligned feature fusion.
            Defaults to True.
        align_corners (bool, optional): Whether to align corner when
            sampling features according to points. Defaults to True.
        padding_mode (str, optional): Mode used to pad the features of
            points that do not have corresponding image features.
            Defaults to 'zeros'.
        lateral_conv (bool, optional): Whether to apply lateral convs
            to image features. Defaults to True.
    """

    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 img_levels=3,
                 coord_type='LIDAR',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None,
                 activate_out=True,
                 fuse_out=False,
                 dropout_ratio=0,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 lateral_conv=True):
        super(PointFusion, self).__init__(init_cfg=init_cfg)
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.img_levels = img_levels
        self.coord_type = coord_type
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode

        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = ConvModule(
                    img_channels[i],
                    mid_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                self.lateral_convs.append(l_conv)
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels * len(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.img_transform = nn.Sequential(
                nn.Linear(sum(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        )

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                # For pts the BN is initialized differently by default
                # TODO: check whether this is necessary
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False))

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Xavier', layer='Linear', distribution='uniform')
            ]

    def forward(self, img_feats, pts, pts_feats, img_metas):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.

        Returns:
            torch.Tensor: Fused features of each point.
        """
        img_pts = self.obtain_mlvl_feats(img_feats, pts, img_metas)
        img_pre_fuse = self.img_transform(img_pts)
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(pts_feats)

        fuse_out = img_pre_fuse + pts_pre_fuse
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    def obtain_mlvl_feats(self, img_feats, pts, img_metas):
        """Obtain multi-level features for each point.

        Args:
            img_feats (list(torch.Tensor)): Multi-scale image features produced
                by image backbone in shape (N, C, H, W).
            pts (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Meta information for each sample.

        Returns:
            torch.Tensor: Corresponding image features of each point.
        """
        if self.lateral_convs is not None:
            img_ins = [
                lateral_conv(img_feats[i])
                for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
            ]
        else:
            img_ins = img_feats
        img_feats_per_point = []
        # Sample multi-level features
        for i in range(len(img_metas)):
            mlvl_img_feats = []
            for level in range(len(self.img_levels)):
                mlvl_img_feats.append(
                    self.sample_single(img_ins[level][i:i + 1], pts[i][:, :3],
                                       img_metas[i]))
            mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1)
            img_feats_per_point.append(mlvl_img_feats)

        img_pts = torch.cat(img_feats_per_point, dim=0)
        return img_pts

    def sample_single(self, img_feats, pts, img_meta):
        """Sample features from single level image feature map.

        Args:
            img_feats (torch.Tensor): Image feature map in shape
                (1, C, H, W).
            pts (torch.Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.

        Returns:
            torch.Tensor: Single level image features of each point.
        """
        # TODO: image transformation also extracted
        img_scale_factor = (
            pts.new_tensor(img_meta['scale_factor'][:2])
            if 'scale_factor' in img_meta.keys() else 1)
        img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
        img_crop_offset = (
            pts.new_tensor(img_meta['img_crop_offset'])
            if 'img_crop_offset' in img_meta.keys() else 0)
        proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
        img_pts = point_sample(
            img_meta=img_meta,
            img_features=img_feats,
            points=pts,
            proj_mat=pts.new_tensor(proj_mat),
            coord_type=self.coord_type,
            img_scale_factor=img_scale_factor,
            img_crop_offset=img_crop_offset,
            img_flip=img_flip,
            img_pad_shape=img_meta['input_shape'][:2],
            img_shape=img_meta['img_shape'][:2],
            aligned=self.aligned,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return img_pts


def projection(points, nusc, img_meta, idx, img_features=None):

    def translate(points, x):
        """
        Applies a translation to the point cloud.
        :param x: <np.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            points[i, :] = points[i, :] + x[i]
        return points

    def rotate(points, rot_matrix: np.ndarray):
        """
        Applies a rotation.
        :param rot_matrix: <np.float: 3, 3>. Rotation matrix.
        """
        points[:3, :] = np.dot(rot_matrix, points[:3, :])
        return points

    ##################################################################
    # projection
    # https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py
    # def map_pointcloud_to_image
    ##################################################################

    sample_token = nusc.get('sample', img_meta['sample_idx'])
    cam_token = sample_token['data'][img_meta['filename'][idx].split('__')[-2]]
    cam = nusc.get('sample_data', cam_token)
    pointsensor_token = sample_token['data']['LIDAR_TOP']
    min_dist = 1.0

    # pc = LidarPointCloud.from_file(img_meta['pts_filename'])
    img_shape = img_meta['ori_shape'][:2]
    pointsensor = nusc.get('sample_data', pointsensor_token)
    points_ = torch.transpose(points, 1, 0).cpu().numpy()

    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor',
                         pointsensor['calibrated_sensor_token'])
    points_ = rotate(points_,
                     Quaternion(cs_record['rotation']).rotation_matrix)
    points_ = translate(points_, np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    points_ = rotate(points_,
                     Quaternion(poserecord['rotation']).rotation_matrix)
    points_ = translate(points_, np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    points_ = translate(points_, -np.array(poserecord['translation']))
    points_ = rotate(points_,
                     Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    points_ = translate(points_, -np.array(cs_record['translation']))
    points_ = rotate(points_,
                     Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    depths = points_[2, :]
    pts_2d = view_points(
        points_[:3, :],
        np.array(cs_record['camera_intrinsic']),
        normalize=True)
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, pts_2d[0, :] > 1)
    mask = np.logical_and(mask, pts_2d[0, :] < img_shape[1] - 1)
    mask = np.logical_and(mask, pts_2d[1, :] > 1)
    mask = np.logical_and(mask, pts_2d[1, :] < img_shape[0] - 1)
    pts_2d = pts_2d[:, mask]
    pts_2d = points.new_tensor(np.transpose(pts_2d[:2, :], (1, 0)))
    point_idx = (mask != 0).nonzero()[0]

    # visualize
    if False:
        import cv2
        image = cv2.imread(img_meta['filename'][idx], cv2.COLOR_BGR2RGB)
        for i in pts_2d:
            x = i[0].item()
            y = i[1].item()
            image = cv2.circle(
                image, (int(x), int(y)),
                radius=1,
                color=(0, 0, 255),
                thickness=-1)
        cv2.imwrite(f"demo2.jpg", image)

    return pts_2d, point_idx


def point_multi_sample(img_meta,
                       img_features,
                       points,
                       proj_mat,
                       coord_type,
                       img_scale_factor,
                       img_crop_offset,
                       img_flip,
                       img_pad_shape,
                       img_shape,
                       nusc,
                       aligned=True,
                       padding_mode='zeros',
                       align_corners=True):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (torch.Tensor): 1 x C x H x W image features.
        points (torch.Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (torch.Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (torch.Tensor): Scale factor with shape of \
            (w_scale, h_scale).
        img_crop_offset (torch.Tensor): Crop offset used to crop \
            image during data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (tuple[int]): int tuple indicates the h & w after
            padding, this is necessary to obtain features in feature map.
        img_shape (tuple[int]): int tuple indicates the h & w before padding
            after scaling, this is necessary for flipping coordinates.
        aligned (bool, optional): Whether use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str, optional): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool, optional): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        torch.Tensor: NxC image features sampled by point coordinates.
    """
    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)

    point_feature = torch.zeros(points.shape[0],
                                img_features.shape[2]).to(device=points.device)
    for idx in range(len(img_meta['filename'])):
        # project points to camera coordinate
        pts_2d, point_idx = projection(points, nusc, img_meta, idx,
                                       img_features)

        # img transformation: scale -> crop -> flip
        # the image is resized by img_scale_factor
        img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
        img_coors -= img_crop_offset

        # grid sample, the valid grid range should be in [-1,1]
        coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

        if img_flip:
            # by default we take it as horizontal flip
            # use img_shape before padding for flip
            orig_h, orig_w = img_shape
            coor_x = orig_w - coor_x

        h, w = img_pad_shape
        coor_y = coor_y / h * 2 - 1
        coor_x = coor_x / w * 2 - 1
        grid = torch.cat([coor_x, coor_y],
                         dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

        # align_corner=True provides higher performance
        mode = 'bilinear' if aligned else 'nearest'
        point_features = F.grid_sample(
            img_features[:, idx, :, :, :],
            grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners)  # 1xCx1xN feats

        point_features = point_features.squeeze().t()
        point_feature[point_idx] = point_features

    return point_feature


def get_2d_coor_multi(img_meta, points, proj_mat, coord_type, img_scale_factor,
                      img_crop_offset, img_flip, img_pad_shape, img_shape,
                      nusc):
    """SEX
    """
    # apply transformation based on info in img_meta
    points = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True)

    coor_2d = torch.zeros(points.shape[0], 3).to(device=points.device)
    for idx in range(len(img_meta['filename'])):
        # project points to camera coordinate
        pts_2d, point_idx = projection(points, nusc, img_meta, idx)

        # img transformation: scale -> crop -> flip
        # the image is resized by img_scale_factor
        img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
        img_coors -= img_crop_offset

        # grid sample, the valid grid range should be in [-1,1]
        coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

        if img_flip:
            # by default we take it as horizontal flip
            # use img_shape before padding for flip
            orig_h, orig_w = img_shape
            coor_x = orig_w - coor_x

        h, w = img_pad_shape
        coor_y = coor_y / h * 2 - 1
        coor_x = coor_x / w * 2 - 1
        grid = torch.cat([coor_x, coor_y],
                         dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

        coor_2d[point_idx, 0] = idx
        coor_2d[point_idx, 1:3] = grid

    return coor_2d


@FUSION_LAYERS.register_module()
class PointMultiFusion(PointFusion):

    def __init__(self,
                 img_channels,
                 pts_channels,
                 mid_channels,
                 out_channels,
                 img_levels=3,
                 data_type='v1.0-mini',
                 coord_type='LIDAR',
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=None,
                 activate_out=True,
                 fuse_out=False,
                 dropout_ratio=0,
                 aligned=True,
                 align_corners=True,
                 padding_mode='zeros',
                 lateral_conv=True):
        super(PointMultiFusion, self).__init__(
            img_channels,
            pts_channels=pts_channels,
            mid_channels=mid_channels,
            out_channels=out_channels,
            img_levels=img_levels,
            coord_type=coord_type,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg,
            activate_out=activate_out,
            fuse_out=fuse_out,
            dropout_ratio=dropout_ratio,
            aligned=aligned,
            align_corners=align_corners,
            padding_mode=padding_mode,
            lateral_conv=lateral_conv)
        self.mid_channels = mid_channels
        self.data_type = data_type

        self.nusc = NuScenes(
            version=self.data_type, dataroot='./data/nuscenes', verbose=True)

    def obtain_mlvl_feats(self, img_feats, pts, img_metas):
        """Obtain multi-level features for each point.

        Args:
            img_feats (list(torch.Tensor)): Multi-scale image features produced
                by image backbone in shape (N, C, H, W).
            pts (list[torch.Tensor]): Points of each sample.
            img_metas (list[dict]): Meta information for each sample.

        Returns:
            torch.Tensor: Corresponding image features of each point.
        """
        if self.lateral_convs is not None:
            img_ins = []
            for i, lateral_conv in zip(self.img_levels, self.lateral_convs):
                B, N, C, H, W = img_feats[i].size()
                img_ins.append(
                    lateral_conv(img_feats[i].view(B * N, C, H, W)).view(
                        B, int(B * N / B), self.mid_channels, H, W))
        else:
            img_ins = img_feats
        img_feats_per_point = []
        # Sample multi-level features
        for i in range(len(img_metas)):
            mlvl_img_feats = []
            for level in range(len(self.img_levels)):
                mlvl_img_feats.append(
                    self.sample_single(img_ins[level][i:i + 1], pts[i][:, :3],
                                       img_metas[i]))
            mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1)
            img_feats_per_point.append(mlvl_img_feats)

        img_pts = torch.cat(img_feats_per_point, dim=0)
        return img_pts

    def sample_single(self, img_feats, pts, img_meta):
        """Sample features from single level image feature map.

        Args:
            img_feats (torch.Tensor): Image feature map in shape
                (1, C, H, W).
            pts (torch.Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.

        Returns:
            torch.Tensor: Single level image features of each point.
        """
        # TODO: image transformation also extracted
        img_scale_factor = (
            pts.new_tensor(img_meta['scale_factor'][:2])
            if 'scale_factor' in img_meta.keys() else 1)
        img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
        img_crop_offset = (
            pts.new_tensor(img_meta['img_crop_offset'])
            if 'img_crop_offset' in img_meta.keys() else 0)
        proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
        img_pts = point_multi_sample(
            img_meta=img_meta,
            img_features=img_feats,
            points=pts,
            proj_mat=pts.new_tensor(proj_mat),
            coord_type=self.coord_type,
            img_scale_factor=img_scale_factor,
            img_crop_offset=img_crop_offset,
            img_flip=img_flip,
            img_pad_shape=img_meta['input_shape'][:2],
            img_shape=img_meta['img_shape'][0][:2],
            nusc=self.nusc,
            aligned=self.aligned,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners)
        return img_pts


class BasicGate(nn.Module):
    # mod code from 3D-CVF
    def __init__(self, in_channel, convf='Conv1d'):
        super(BasicGate, self).__init__()
        self.in_channel = in_channel
        if convf == 'Conv1d':
            conv_func = nn.Conv1d
        if convf == 'Conv2d':
            conv_func = nn.Conv2d
        self.gating_conv = conv_func(
            self.in_channel,
            1,
            kernel_size=1,
            stride=1,
        )

    def forward(self, src, trg):
        g_map = torch.sigmoid(self.gating_conv(src))
        return trg * g_map


@FUSION_LAYERS.register_module()
class ACTR(BaseModule):

    def __init__(self,
                 actr_cfg,
                 init_cfg=None,
                 coord_type='LIDAR',
                 activate_out=False,
                 data_version='v1.0-trainval',
                 data_root='./data/nuscenes'):
        super(ACTR, self).__init__(init_cfg=init_cfg)
        self.fusion_method = actr_cfg['fusion_method']
        self.actr = build_actr(actr_cfg)
        self.coord_type = coord_type
        self.activate_out = activate_out
        if self.fusion_method == 'gating_v1':
            n_channel = actr_cfg['query_num_feat']
            self.trg_gating = BasicGate(n_channel)
            self.trg_channel_reduce = nn.Conv1d(
                n_channel * 2, n_channel, kernel_size=1, stride=1)

        self.nusc = NuScenes(
            version=data_version, dataroot=data_root, verbose=True)

    def split_param(self, pts_feats, coor_2d, img_feats, pts, num_points):
        """nuscene dataset have 6 imgae in each sample
        1. convert img_feats [B, 6, C, H, W] -> [B*6, C, H, W]
        2. convert else [B, P, C] -> [B*6, P, C] base coor_2d[:, 0]
        """
        N = 6  # number of images
        for b in range(len(img_feats)):
            B, _, C, H, W = img_feats[b].shape
            img_feats[b] = img_feats[b].view(-1, C, H, W)

        B, P, C = pts_feats.shape
        pts_feats_n = torch.zeros((B * N, P // 3, C), device=pts.device)
        coor_2d_n = torch.zeros((B * N, P // 3, 2), device=pts.device)
        pts_n = torch.zeros((B * N, P // 3, 3), device=pts.device)
        num_points_n = []
        for b in range(B):
            b_mod = coor_2d[b][:, 0][:num_points[b]].to(torch.long)
            for n in range(N):
                mask = (b_mod == n)
                mask_n = mask.sum()
                pts_feats_n[b, :mask_n] = pts_feats[b, :num_points[b]][mask]
                coor_2d_n[b, :mask_n] = coor_2d[b, :num_points[b], 1:3][mask]
                pts_n[b, :mask_n] = pts[b, :num_points[b]][mask]
                num_points_n.append(mask_n)

        return pts_feats_n, coor_2d_n, pts_n, num_points_n

    def agg_param(self, pts_feat, num_points):
        N = 6
        B, C = pts_feat.shape[0] // N, pts_feat.shape[2]
        pts_feat_n = torch.zeros((B, self.actr.max_num_ne_voxel, C),
                                 device=pts_feat.device)
        for b in range(B):
            st = 0
            for n in range(N):
                pts_feat_n[b, st:st + num_points[b * N +n]] = \
                    pts_feat[b * N +n, :num_points[b * N + n]]
                st += num_points[b * N + n]
        return pts_feat_n

    def forward(self, img_feats, pts, pts_feats, img_metas):
        """Forward function.

        Args:
            img_feats (list[torch.Tensor]): Image features.
            pts: [list[torch.Tensor]]: A batch of points with shape N x 3.
            pts_feats (torch.Tensor): A tensor consist of point features of the
                total batch.
            img_metas (list[dict]): Meta information of images.
        Returns:
            torch.Tensor: Fused features of each point.
        """
        batch_size = len(pts)
        img_feats = img_feats[:self.actr.num_backbone_outs]
        num_points = [i.shape[0] for i in pts]
        pts_feats_b = torch.zeros(
            (batch_size, self.actr.max_num_ne_voxel, pts_feats.shape[1]),
            device=pts_feats.device)
        coor_2d_b = torch.zeros((batch_size, self.actr.max_num_ne_voxel, 3),
                                device=pts_feats.device)
        pts_b = torch.zeros((batch_size, self.actr.max_num_ne_voxel, 3),
                            device=pts_feats.device)

        for b in range(batch_size):
            img_meta = img_metas[b]
            img_scale_factor = (
                pts[b].new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (
                pts[b].new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
            coor_2d = get_2d_coor_multi(
                img_meta=img_meta,
                points=pts[b],
                proj_mat=pts[b].new_tensor(proj_mat),
                coord_type=self.coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_meta['input_shape'][:2],
                img_shape=img_meta['img_shape'][0][:2],
                nusc=self.nusc)
            pts_b[b, :pts[b].shape[0]] = pts[b][:, :3]
            coor_2d_b[b, :pts[b].shape[0]] = coor_2d
            pts_feats_b[b, :pts[b].shape[0]] = pts_feats[b]

        pts_feats_n, coor_2d_n, pts_n, num_points_n = self.split_param(
            pts_feats_b, coor_2d_b, img_feats, pts_b, num_points)
        enh_feat_n = self.actr(
            v_feat=pts_feats_n,
            grid=coor_2d_n,
            i_feats=img_feats,
            lidar_grid=pts_n,
        )
        enh_feat = self.agg_param(enh_feat_n, num_points_n)
        enh_feat_cat = torch.cat(
            [f[:np] for f, np in zip(enh_feat, num_points)])

        if self.fusion_method == 'replace':
            fuse_out = enh_feat_cat
        elif self.fusion_method == 'concat':
            fuse_out = torch.cat((pts_feats, enh_feat_cat), dim=1)
        elif self.fusion_method == 'sum':
            fuse_out = pts_feats + enh_feat_cat
        elif self.fusion_method == 'gating_v1':
            pts_feats_u = pts_feats.unsqueeze(0).permute(0, 2, 1)
            enh_feat_cat_u = enh_feat_cat.unsqueeze(0).permute(0, 2, 1)
            gated_fuse_feat_u = self.trg_gating(pts_feats_u, enh_feat_cat_u)
            fuse_out = pts_feats_u + enh_feat_cat_u
            fuse_out = fuse_out.squeeze().permute(1, 0)
            #  fuse_out = torch.cat((pts_feats_u, gated_fuse_feat_u), dim=1)
            #  fuse_out = self.trg_channel_reduce(fuse_out)
            #  fuse_out = fuse_out.squeeze().permute(1, 0)
        else:
            NotImplementedError('Invalid ACTR fusion method')

        if self.activate_out:
            fuse_out = F.relu(fuse_out)

        return fuse_out
