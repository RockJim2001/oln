#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：oln
@Product_name ：PyCharm
@File ：train.py
@Author ：RockJim
@Date ：2023/11/14 10:51
@Description ：模型的训练入口
@Version ：1.0
"""
import datetime
import math
import os
import sys
from functools import partial
import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modeling.backbone.fpn import FPN
from modeling.backbone.resnet import ResNet
from modeling.dataset import transforms
from modeling.dataset.classes import COCO_BASE_CLASSES, COCO_NOVEL_CLASSES
from modeling.dataset.coco import COCODataset
from modeling.dataset.coco_oln_dataset import CocoOlnDetection
from modeling.dataset.dataset_path_config import COCO_JSON_ANNOTATIONS_DIR, COCO_IMAGE_ROOT_DIR
from modeling.dataset.group_sampler import GroupSampler
from modeling.dataset.transforms import LoadImageFromFile, MultiScaleFlipAug, Resize, RandomFlip, Normalize, Pad, \
    ImageToTensor, Collect, LoadAnnotations, DefaultFormatBundle
from modeling.dataset.utils import collate, worker_init_fn
from modeling.oln_detector import OlnDetector
from modeling.roi_head.bbox_head import Shared2FCBBoxScoreHead
from modeling.roi_head.oln_roi_head import OlnRoIHead
from modeling.roi_head.utils import SingleRoIExtractor
from modeling.rpn_head.oln_rpn_head import OlnRPNHead
from modeling.utils.bbox_coder import TBLRBBoxCoder, DeltaXYWHBBoxCoder
from modeling.utils.losses import CrossEntropyLoss, IoULoss, L1Loss
from modeling.utils.util import AnchorGenerator, MaxIoUAssigner, RandomSampler
from tools.config import set_output_path
from tools.distributed_utils import MetricLogger, SmoothedValue, warmup_lr_scheduler, reduce_dict
from tools.utils import parse_args


def create_model(args=None, training=True):
    """
        构建模型结构
    :param args:
    :param training:  是否为train模式
    :return: 返回模型的结构
    """
    # 构建用于训练的模型
    # ----------------------------- backbone -------------------------
    print("构建ResNet50……")
    resnet = ResNet(depth=50,  # 构建模型结构
                    num_stages=4,
                    out_indices=(0, 1, 2, 3),
                    frozen_stages=1,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    norm_eval=True,
                    style='pytorch',
                    )
    # print(resnet)
    # ----------------------------- FPN ---------------------------
    fpn = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5)

    # ----------------------------- rpn --------------------------
    # AnchorGenerator 的配置
    anchor_generator = AnchorGenerator(scales=[8], ratios=[1.0], strides=[4, 8, 16, 32, 64])
    # bbox_coder 配置
    bbox_coder = TBLRBBoxCoder(normalizer=1.0)
    # loss_cls
    loss_cls = CrossEntropyLoss(use_sigmoid=True, loss_weight=0.0)
    # loss_bbox
    loss_bbox = IoULoss(linear=True, loss_weight=10.0)
    # loss_objectness
    loss_objectness = L1Loss(loss_weight=1.0)

    # cls 和 bbox
    assigner = MaxIoUAssigner(pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, ignore_iof_thr=-1)
    sampler = RandomSampler(num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False)
    # objectness
    objectness_assigner = MaxIoUAssigner(pos_iou_thr=0.3, neg_iou_thr=0.1, min_pos_iou=0.3, ignore_iof_thr=-1)
    objectness_sampler = RandomSampler(num=256, pos_fraction=1., neg_pos_ub=-1, add_gt_as_proposals=False)

    if training:
        nms_across_levels = False,
        nms_pre = 2000,
        nms_post = 2000,
        max_num = 2000,
        nms_thr = 0.7,
        min_bbox_size = 0
    else:
        nms_across_levels = False,
        nms_pre = 2000,
        nms_post = 2000,
        max_num = 2000,
        nms_thr = 0.0,
        min_bbox_size = 0
    num_classes = args.num_classes  # 检测的类别数量
    rpn = OlnRPNHead(
        # rpn 的配置信息
        in_channels=256, feat_channels=256, num_classes=num_classes,
        anchor_generator=anchor_generator,
        bbox_coder=bbox_coder,
        objectness_type='Centerness',
        reg_decoded_bbox=True,
        allowed_border=0,
        pos_weight=-1,
        debug=False,

        assigner=assigner,
        sampler=sampler,
        objectness_assigner=objectness_assigner,
        objectness_sampler=objectness_sampler,

        loss_cls=loss_cls,
        loss_bbox=loss_bbox,
        loss_objectness=loss_objectness,
        nms_across_levels=nms_across_levels,
        nms_pre=nms_pre,
        nms_post=nms_post,
        max_num=max_num,
        nms_thr=nms_thr,
        min_bbox_size=min_bbox_size
    )

    # ----------------------------- roi --------------------------
    # -------------------- RoiExtractor ---------------------
    # 构建 bbox_roi_extractor
    bbox_roi_extractor = SingleRoIExtractor(out_channels=256, featmap_strides=[4, 8, 16, 32])

    # 构架bbox_head
    bbox_coder = DeltaXYWHBBoxCoder(target_means=[0., 0., 0., 0.], target_stds=[0.1, 0.1, 0.2, 0.2])
    loss_cls = CrossEntropyLoss(use_sigmoid=False, loss_weight=0.0)
    loss_bbox = L1Loss(loss_weight=1.0)
    loss_bbox_score = L1Loss(loss_weight=1.0)
    pos_weight = -1
    bbox_head = Shared2FCBBoxScoreHead(in_channels=256,
                                       fc_out_channels=1024,
                                       roi_feat_size=7,
                                       num_classes=1,
                                       bbox_coder=bbox_coder,
                                       reg_class_agnostic=False,
                                       loss_cls=loss_cls,
                                       loss_bbox=loss_bbox,
                                       bbox_score_type='BoxIoU',
                                       loss_bbox_score=loss_bbox_score,
                                       pos_weight=pos_weight)

    bbox_assigner = MaxIoUAssigner(pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5, ignore_iof_thr=-1,
                                   match_low_quality=False)
    bbox_sampler = RandomSampler(num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True)

    # 构建 roi_head
    roi_head = OlnRoIHead(
        bbox_roi_extractor=bbox_roi_extractor,
        bbox_head=bbox_head,
        bbox_assigner=bbox_assigner,
        bbox_sampler=bbox_sampler,
    )
    # ----------------------------- 构建总的模型 --------------------------
    model = OlnDetector(backbone=resnet, neck=fpn, rpn_head=rpn, roi_head=roi_head, load_pretrain_weights=args.pretrain,
                        seed=args.seed)

    if training:
        model.train()
    else:
        model.eval()
    return model


def create_dataloader(args=None, train_classes=None, eval_classes=None, mode='train'):
    """
            构建dataloader
    :param args: 命令行参数
    :param train_classes: 进行数据加载的训练类的ids
    :param eval_classes: 进行数据加载的测试类的ids
    :param mode: 是否为train模式
    :return: dataloader对象
    """

    # --------------------------- 加载dataset ---------------------------------
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    )

    transform = None
    train_class = train_classes
    eval_class = eval_classes
    test_mode = False
    seed = None
    data_cfg = dict()
    if mode == 'train':
        seed = 2023
        # train_class = classes
        transform = transforms.Compose([LoadImageFromFile(),
                                        LoadAnnotations(with_bbox=True),
                                        Resize(img_scale=(1333, 800), keep_ratio=True),
                                        RandomFlip(flip_ratio=0.5),
                                        Normalize(**img_norm_cfg),
                                        Pad(size_divisor=32),
                                        DefaultFormatBundle(),
                                        Collect(keys=['img', 'gt_bboxes', 'gt_labels'])
                                        ])
        data_cfg = {
            'ann_file': os.path.join(COCO_JSON_ANNOTATIONS_DIR, 'instances_train2017.json'),
            'pipeline': transform,
            'classes_ids': COCO_NOVEL_CLASSES,
            'data_root': os.path.join(COCO_IMAGE_ROOT_DIR, 'train2017'),
        }
    elif mode == 'val':
        test_mode = True
        transform = transforms.Compose([
            LoadImageFromFile(),
            MultiScaleFlipAug(
                img_scale=(1333, 800),
                flip=False,
                transforms=transforms.Compose([
                    Resize(keep_ratio=True),
                    RandomFlip(),
                    Normalize(**img_norm_cfg),
                    Pad(size_divisor=32),
                    ImageToTensor(keys=['img']),
                    Collect(keys=['img'])
                ])
            ),
        ])
        data_cfg = {
            'ann_file': os.path.join(COCO_JSON_ANNOTATIONS_DIR, 'instances_val2017.json'),
            'pipeline': transform,
            'classes_ids': COCO_BASE_CLASSES,
            'data_root': os.path.join(COCO_IMAGE_ROOT_DIR, 'val2017'),
        }

    dataset = CocoOlnDetection(is_class_agnostic=True, train_class=train_class, eval_class=eval_class, mode=mode,
                               transform=transform, test_mode=test_mode)

    # dataset = COCODataset(is_class_agnostic=True, filter_image_type='mix', **data_cfg)

    print(f"一共加载到了{len(dataset)}张图片")
    torch.manual_seed(2023)  # 设置PyTorch的随机种子
    # 构建dataloader
    # dataloader_ = DataLoader(dataset=dataset,
    #                          batch_size=args.batch_size,
    #                          worker_init_fn=worker_init_fn,
    #                          collate_fn=collate)

    samples_per_gpu = args.batch_size
    num_workers = args.batch_size
    rank = 0
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    sampler = GroupSampler(dataset, samples_per_gpu) if mode == 'train' else None

    dataloader_ = DataLoader(dataset=dataset,
                             batch_size=args.batch_size,
                             sampler=sampler,
                             num_workers=num_workers,
                             worker_init_fn=init_fn,
                             collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
                             pin_memory=False)

    return dataloader_


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None, writer=None, iterate=0):
    model.train()
    metric_logger = MetricLogger(delimiter="  ", writer=writer)

    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6e}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 1 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(500, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    for i, obj in enumerate(metric_logger.log_every(data_loader, print_freq, header, epoch)):
        img_metas = obj['img_metas'].data
        img = [item.to(device) for item in obj['img'].data]
        gt_bboxes = [item.to(device) for item in obj['gt_bboxes'].data[0]]
        gt_labels = [item.to(device) for item in obj['gt_labels'].data[0]]

        # images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # loss_dict = model(img, img_metas, [gt_bboxes], [gt_labels])
            losses, loss_dict = model(img, img_metas, [gt_bboxes], [gt_labels])
            # for item, value in loss_dict.items():
            #     # writer.add_scalar('train/' + item, value, i * (epoch + 1))
            #     writer.add_scalar('train/' + item, value, iterate)
            # losses = sum(loss for loss in loss_dict.values())
            # writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], i * (epoch + 1))
            # writer.add_scalar('momentum', optimizer.param_groups[0]["momentum"], i * (epoch + 1))
            if writer is not None:
                writer.add_scalar('learning_rate', optimizer.param_groups[0]["lr"], iterate)
                writer.add_scalar('momentum', optimizer.param_groups[0]["momentum"], iterate)
            loss_dict.pop('loss')
        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = reduce_dict(loss_dict)
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        losses_reduced = losses

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

        # flag += 1
        # if flag > 1000:
        #     break

        iterate += 1

    return mloss, now_lr, writer, iterate


if __name__ == '__main__':

    args = parse_args()
    # pth = r"/home/jiang/model/oln/modeling/backbone/resnet50-19c8e357.pth"

    # 存储路径不存在的时候，对其进行创建
    output = os.path.join(os.getcwd(), args.output)
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    # 设置全局的存储路径
    set_output_path(args.output)
    from tools.config import OUTPUT_PATH

    writer = SummaryWriter(log_dir=OUTPUT_PATH)
    # writer = None

    # 获取可用的设备信息
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("在设备{}上进行训练.".format(device.type))

    # 创建模型结构
    model = create_model(args)
    model.to(device)
    print(model)

    train_loss = []
    learning_rate = []
    val_map = []

    # 构建dataloader
    train_data_loader = create_dataloader(args, train_classes=COCO_BASE_CLASSES, eval_classes=COCO_NOVEL_CLASSES)

    # val_data_loader = create_dataloader(args, classes=COCO_NOVEL_CLASSES)

    # 进行训练
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_steps,
                                                        gamma=args.lr_gamma)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                     milestones=args.lr_steps,
    #                                                     gamma=args.lr_gamma)
    iterate = 0
    for epoch in range(args.start_epoch, args.epochs + 1):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr, writer_now, iterate_now = train_one_epoch(model, optimizer, train_data_loader,
                                                                 device, epoch, print_freq=10,
                                                                 warmup=True, scaler=scaler, writer=writer,
                                                                 iterate=iterate)
        iterate = iterate_now
        writer = writer_now
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        # coco_info = utils.evaluate(model, val_data_loader, device=device)

        # write into txt
        # with open(results_file, "a") as f:
        #     # 写入的数据包括coco指标还有loss和learning rate
        #     result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
        #     txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        #     f.write(txt + "\n")
        #
        # val_map.append(coco_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        save_path = os.path.join(args.output, 'save_weights/model_{}.pth'.format(epoch))
        # 路径不存在的话，进行创建
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        torch.save(save_files, save_path)
