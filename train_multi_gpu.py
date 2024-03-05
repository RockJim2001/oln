import datetime
import os
import time
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from modeling.backbone.fpn import FPN
from modeling.backbone.resnet import ResNet
from modeling.dataset import transforms
from modeling.dataset.classes import COCO_BASE_CLASSES, COCO_NOVEL_CLASSES
from modeling.dataset.coco_oln_dataset import CocoOlnDetection
from modeling.dataset.group_sampler import GroupSampler
from modeling.dataset.transforms import LoadImageFromFile, Resize, MultiScaleFlipAug, RandomFlip, LoadAnnotations, \
    Normalize, Pad, DefaultFormatBundle, Collect, ImageToTensor
from modeling.dataset.utils import worker_init_fn, collate
from modeling.oln_detector import OlnDetector
from modeling.roi_head.bbox_head import Shared2FCBBoxScoreHead
from modeling.roi_head.oln_roi_head import OlnRoIHead
from modeling.roi_head.utils import SingleRoIExtractor
from modeling.rpn_head.oln_rpn_head import OlnRPNHead
from modeling.utils.bbox_coder import TBLRBBoxCoder, DeltaXYWHBBoxCoder
from modeling.utils.losses import CrossEntropyLoss, IoULoss, L1Loss
from modeling.utils.util import AnchorGenerator, MaxIoUAssigner, RandomSampler
from tools.config import set_output_path
from tools.distributed_utils import mkdir, init_distributed_mode, save_on_master
from tools.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from train import train_one_epoch


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


# def create_dataloader(args=None, train_classes=None, eval_classes=None, mode='train'):
#     """
#             构建dataloader
#     :param args: 命令行参数
#     :param train_classes: 进行数据加载的训练类的ids
#     :param eval_classes: 进行数据加载的测试类的ids
#     :param mode: 是否为train模式
#     :return: dataloader对象
#     """
#
#
#     torch.manual_seed(2023)  # 设置PyTorch的随机种子
#     # 构建dataloader
#     # dataloader_ = DataLoader(dataset=dataset,
#     #                          batch_size=args.batch_size,
#     #                          worker_init_fn=worker_init_fn,
#     #                          collate_fn=collate)
#
#     samples_per_gpu = args.batch_size
#     num_workers = args.batch_size
#     rank = 0
#     init_fn = partial(
#         worker_init_fn, num_workers=num_workers, rank=rank,
#         seed=seed) if seed is not None else None
#
#     sampler = GroupSampler(dataset, samples_per_gpu) if mode == 'train' else None
#
#     dataloader_ = DataLoader(dataset=dataset,
#                              batch_size=args.batch_size,
#                              sampler=sampler,
#                              num_workers=num_workers,
#                              worker_init_fn=init_fn,
#                              collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
#                              pin_memory=False)
#
#     return dataloader_


def create_dataset(args=None, train_classes=None, eval_classes=None, mode='train'):
    """
            构建dataset
    :param args: 命令行参数
    :param train_classes: 进行数据加载的训练类的ids
    :param eval_classes: 进行数据加载的测试类的ids
    :param mode: 是否为train模式
    :return: dataset对象
    """
    # --------------------------- 加载dataset ---------------------------------
    img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
    )

    transform = None
    train_class = train_classes
    eval_class = eval_classes
    test_mode = False
    seed = args.seed
    if mode == 'train':
        # seed = 2023
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
    elif mode == 'val':
        # eval_class = classes
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

    dataset = CocoOlnDetection(is_class_agnostic=True, train_class=train_class, eval_class=eval_class, mode=mode,
                               transform=transform, test_mode=test_mode)
    print(f"一共加载到了{len(dataset)}张图片")

    return dataset


def main(args):
    init_distributed_mode(args)
    print(args)

    # 存储路径不存在的时候，对其进行创建
    output = os.path.join(os.getcwd(), args.output_dir)
    if not os.path.exists(output):
        os.makedirs(output, exist_ok=True)

    # 设置全局的存储路径
    set_output_path(args.output_dir)
    from tools.config import OUTPUT_PATH


    # 获取可用的设备信息
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("在设备{}上进行训练.".format(device.type))

    # 用来保存coco_info的文件
    # results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # 构建dataloader
    train_dataset = create_dataset(args, train_classes=COCO_BASE_CLASSES, eval_classes=COCO_NOVEL_CLASSES)

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # test_sampler = torch.utils.data.SequentialSampler(val_dataset)

    if args.aspect_ratio_group_factor >= 0:
        # 统计所有图像比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    # data_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
    #     collate_fn=train_dataset.collate_fn)

    # 构建dataloader
    seed = args.seed
    samples_per_gpu = args.batch_size
    num_workers = args.workers
    rank = torch.distributed.get_rank()
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_sampler=train_batch_sampler,
                                  # batch_size=args.batch_size,
                                  worker_init_fn=init_fn,
                                  collate_fn=partial(collate, samples_per_gpu=samples_per_gpu))

    # 创建模型结构
    print("创建模型结构")
    model = create_model(args)
    model.to(device)
    print(model)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    train_loss = []
    learning_rate = []
    val_map = []

    print("Start training")
    start_time = time.time()
    iterate = 0
    if args.rank in [-1, 0]:
        writer = SummaryWriter(log_dir=OUTPUT_PATH)
    for epoch in range(args.start_epoch, args.epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        mean_loss, lr, writer_now, iterate_now = train_one_epoch(model, optimizer, train_dataloader,
                                                                 device, epoch, args.print_freq,
                                                                 warmup=True, scaler=scaler, iterate=iterate,
                                                                 writer=writer if args.rank in [-1, 0] else None)
        iterate = iterate_now
        writer = writer_now
        # update learning rate
        lr_scheduler.step()

        # evaluate after every epoch
        # coco_info = utils.evaluate(model, data_loader_test, device=device)

        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:
            train_loss.append(mean_loss.item())
            learning_rate.append(lr)

        #     val_map.append(coco_info[1])  # pascal mAP
        #
        #     # write into txt
        #     with open(results_file, "a") as f:
        #         # 写入的数据包括coco指标还有loss和learning rate
        #         result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
        #         txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
        #         f.write(txt + "\n")

        if args.output_dir:
            # 只在主节点上执行保存权重操作
            save_files = {'model': model_without_ddp.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'lr_scheduler': lr_scheduler.state_dict(),
                          'args': args,
                          'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            save_on_master(save_files,
                           os.path.join(args.output_dir, f'model_{epoch}.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # if args.rank in [-1, 0]:
    #     # plot loss and lr curve
    #     if len(train_loss) != 0 and len(learning_rate) != 0:
    #         from plot_curve import plot_loss_and_lr
    #         plot_loss_and_lr(train_loss, learning_rate)
    #
    #     # plot mAP curve
    #     if len(val_map) != 0:
    #         from plot_curve import plot_map
    #         plot_map(val_map)

    print("哈哈哈哈")


if __name__ == '__main__':
    # args = parse_args()
    #
    # # 存储路径不存在的时候，对其进行创建
    # output = os.path.join(os.getcwd(), args.output)
    # if not os.path.exists(output):
    #     os.makedirs(output, exist_ok=True)

    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    # 预训练权重
    parser.add_argument('--pretrain', type=str, default='modeling/backbone/resnet50-19c8e357.pth',
                        help='backbone的预训练权重路径')
    # 训练文件的根目录(coco2017)
    parser.add_argument('--data-path', default='/home/jiang/model/datasets/coco', help='dataset')
    # 训练设备类型
    parser.add_argument('--device', default='cuda', help='device')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num-classes', default=1, type=int, help='num_classes')
    # 每块GPU上的batch_size
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=1, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=8, type=int, metavar='N',
                        help='number of total epochs to run')
    # 数据加载以及预处理的线程数
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # 学习率，这个需要根据gpu的数量以及batch_size进行设置0.02 / 8 * num_GPU
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 针对torch.optim.lr_scheduler.StepLR的参数
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[6, 7], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 训练过程打印信息的频率
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./tools/multi_train', help='path where to save')
    # 基于上次的训练结果接着训练
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)

    # 随机数种子
    parser.add_argument('--seed', type=int, default=2023, help='设置随机数种子')

    # 开启的进程数(注意不是线程)
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", type=bool, default=False)
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()

    # 如果指定了保存文件地址，检查文件夹是否存在，若不存在，则创建
    if args.output_dir:
        mkdir(args.output_dir)

    # 设置全局的存储路径
    # set_output_path(args.output)

    main(args)
