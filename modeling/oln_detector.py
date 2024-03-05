#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：oln
@Product_name ：PyCharm
@File ：oln_detector.py
@Author ：RockJim
@Date ：2023/11/14 10:48
@Description ：模型的总框架
@Version ：1.0
"""
import os.path
from collections import OrderedDict

import numpy as np
import torch.distributed as dist

import torch
from torch import nn

from tools.config import set_random_seed


class OlnDetector(nn.Module):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone=None,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 load_pretrain_weights=None,
                 seed=2023):
        super().__init__()
        assert backbone is not None, "backbone参数是不存在"
        assert neck is not None, "neck参数是不存在"
        assert rpn_head is not None, "rpn_head参数是不存在"
        assert roi_head is not None, "roi_head参数是不存在"
        self.load_pretrain_weights = load_pretrain_weights
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        # 进行模型的初始化
        self.init_weights(seed=seed)

    def init_weights(self, seed=None):
        # set_random_seed(seed=seed)  # 设置随机数种子
        # --------------------------- backbone -----------------------------
        if self.load_pretrain_weights is not None:  # 采用预训练权重对backbone进行初始化
            assert isinstance(self.load_pretrain_weights, str) and os.path.exists(
                self.load_pretrain_weights), '预训练权重路径存在问题'
            checkpoint = torch.load(self.load_pretrain_weights, map_location='cpu')
            missing_keys, unexpected_keys = self.backbone.load_state_dict(checkpoint, strict=False)
            print(missing_keys, unexpected_keys)
        else:
            self.backbone.init_weights(seed=seed)
        # --------------------------- fpn -----------------------------
        self.neck.init_weights(seed=seed)
        # --------------------------- rpn -----------------------------
        self.rpn_head.init_weights(seed=seed)
        # --------------------------- roi -----------------------------
        self.roi_head.init_weights(seed=seed)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward(self, img, img_metas=None, gt_bboxes=None, gt_labels=None, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img[0], img_metas[0], gt_bboxes[0], gt_labels[0], **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # 提取特征
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        # proposal_cfg = self.train_cfg.get('rpn_proposal',
        #                                   self.test_cfg.rpn)
        # rpn_losses, proposal_list = self.rpn_head.forward_train(
        #     x,
        #     img_metas,
        #     gt_bboxes,
        #     gt_labels=None,
        #     gt_bboxes_ignore=gt_bboxes_ignore,
        #     proposal_cfg=proposal_cfg)

        # 生成proposal
        rpn_losses, proposal_list = self.rpn_head.forward_train(x, img_metas, gt_bboxes, None,
                                                                proposal_cfg=dict(nms_across_levels=False,
                                                                                  nms_pre=2000,
                                                                                  nms_post=2000,
                                                                                  max_num=2000,
                                                                                  nms_thr=0.7,
                                                                                  min_bbox_size=0))

        losses.update(rpn_losses)

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return self._parse_losses(losses)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        # for loss_name, loss_value in log_vars.items():
        #     # reduce loss when distributed training
        #     if dist.is_available() and dist.is_initialized():
        #         loss_value = loss_value.data.clone()
        #         dist.all_reduce(loss_value.div_(dist.get_world_size()))
        #     log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    async def async_simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Async test without augmentation."""
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        # from fnmatch import fnmatch
        # temp = torch.load('/home/jiang/桌面/a.pt')
        # new_state_dict = self.neck.state_dict()
        # for key in temp.keys():
        #     if fnmatch(key, "*_*.*.*.*"):
        #         parts = key.split('.')
        #         if parts[1].isdigit():
        #             new_string = '.'.join([parts[0], parts[1], parts[3]])
        #             new_state_dict[new_string] = temp[key]

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas, proposal_cfg=dict(nms_across_levels=False,
                                                                                          nms_pre=2000,
                                                                                          nms_post=2000,
                                                                                          max_num=2000,
                                                                                          nms_thr=0.7,
                                                                                          min_bbox_size=0))
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale, test_cfg=dict(score_thr=0.0,
                                                                                                     nms=dict(type='nms', iou_threshold=0.7),
                                                                                                     max_per_img=1500,))

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(imgs, img_metas, **kwargs)