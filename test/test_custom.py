import unittest
from fnmatch import fnmatch
import torch
from modeling.backbone.fpn import FPN
from modeling.backbone.resnet import ResNet
from modeling.dataset import transforms
from modeling.dataset.classes import COCO_BASE_CLASSES, COCO_NOVEL_CLASSES
from modeling.dataset.coco_oln_dataset import CocoOlnDetection
from modeling.dataset.transforms import Collect, LoadImageFromFile, LoadAnnotations, Resize, RandomFlip, Normalize, Pad, \
    DefaultFormatBundle, MultiScaleFlipAug, ImageToTensor
from modeling.roi_head.bbox_head import Shared2FCBBoxScoreHead
from modeling.roi_head.oln_roi_head import OlnRoIHead
from modeling.roi_head.utils import SingleRoIExtractor
from modeling.rpn_head.oln_rpn_head import OlnRPNHead
from modeling.utils.bbox_coder import TBLRBBoxCoder, DeltaXYWHBBoxCoder
from modeling.utils.losses import CrossEntropyLoss, IoULoss, L1Loss
from modeling.utils.util import AnchorGenerator, MaxIoUAssigner, RandomSampler
from tools.config import set_random_seed


# 对backbone部分进行测试
class TestBackbone(unittest.TestCase):
    def test_resnet_50(self):
        print("对resnet50网络进行测试")
        resnet = ResNet(depth=50,  # 构建模型结构
                        num_stages=4,
                        out_indices=(0, 1, 2, 3),
                        frozen_stages=1,
                        norm_cfg=dict(type='BN', requires_grad=True),
                        norm_eval=True,
                        style='pytorch',
                        )
        # print(resnet)
        checkpoint = torch.load(r"/home/jiang/model/oln/modeling/backbone/resnet50-19c8e357.pth", map_location='cpu')
        missing_keys, unexpected_keys = resnet.load_state_dict(checkpoint, strict=False)
        print(missing_keys, unexpected_keys)
        resnet.train()

        return resnet

        seed = 2023
        torch.manual_seed(seed)
        inputs = torch.rand(1, 3, 32, 32)
        output = resnet.forward(inputs)

        # 添加fpn进行测试

        set_random_seed(seed, deterministic=False)

        fpn = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5)
        fpn.init_weights()
        input_true = torch.load("/home/jiang/桌面/input_true.pt")
        new_state_dict = fpn.state_dict()
        for key in input_true.keys():
            if fnmatch(key, "*_*.*.*.*"):
                parts = key.split('.')
                if parts[1].isdigit():
                    new_string = '.'.join([parts[0], parts[1], parts[3]])
                    new_state_dict[new_string] = input_true[key]
        missing_keys, unexpected_keys = fpn.load_state_dict(new_state_dict)
        print(missing_keys, unexpected_keys)
        output = fpn(output)

        torch.save(inputs, r"/home/jiang/model/oln/test/inputs.pt")
        torch.save(output, r"/home/jiang/model/oln/test/output.pt")

    def test_dataloader(self, train_class=None, eval_class=None, mode='train'):
        # 对dataloader进行测试
        seed = 2023
        set_random_seed(seed, deterministic=False)
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
        )
        transform = None
        train_class = train_class
        eval_class = eval_class
        test_mode = False
        if mode == 'train':
            # train_class = classes
            transform = transforms.Compose([LoadImageFromFile(),
                                            LoadAnnotations(with_bbox=True),
                                            Resize(img_scale=(1333, 800), keep_ratio=True),
                                            RandomFlip(flip_ratio=1.0),
                                            Normalize(**img_norm_cfg),
                                            Pad(size_divisor=32),
                                            DefaultFormatBundle(),
                                            Collect(keys=['img', 'gt_bboxes', 'gt_labels'])
                                            ])
        elif mode == 'val':
            test_mode = True
            # eval_class = classes
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

        # if transform is None:

        train = CocoOlnDetection(is_class_agnostic=True, train_class=train_class, eval_class=eval_class, mode=mode,
                                 transform=transform, test_mode=test_mode)
        print(len(train))
        return train
        t = train[0]
        # print(t)
        return t

    def test_neck(self):
        seed = 2023
        set_random_seed(seed, deterministic=False)

        fpn = FPN(in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5)
        # input_true = torch.load("/home/jiang/桌面/input_true.pt")
        # new_state_dict = fpn.state_dict()
        # for key in input_true.keys():
        #     if fnmatch(key, "*_*.*.*.*"):
        #         parts = key.split('.')
        #         if parts[1].isdigit():
        #             new_string = '.'.join([parts[0], parts[1], parts[3]])
        #             new_state_dict[new_string] = input_true[key]
        # missing_keys, unexpected_keys = fpn.load_state_dict(new_state_dict)
        # print(missing_keys, unexpected_keys)
        # print(fpn)
        return fpn

    def test_rpn(self):
        # 对rpn进行测试
        seed = 2023
        set_random_seed(seed)
        training = True  # 训练模式
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
        num_classes = 1
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

        # print(rpn)
        return rpn

    def test_roi(self):
        # 对 roi_head 进行测试
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
        print(roi_head)
        return roi_head

    def test_model_train(self):
        # 对模型的结构进行测试 ---- train
        # --------------------------  先加载数据并构建模型 ----------------------------
        results = self.test_dataloader(train_class=COCO_BASE_CLASSES, eval_class=COCO_NOVEL_CLASSES)[0]
        resnet50 = self.test_resnet_50()
        fpn = self.test_neck()
        set_random_seed(seed=2023)
        fpn.init_weights()
        rpn = self.test_rpn()
        roi = self.test_roi()

        # 提取特征
        output = resnet50(results['img'].data.unsqueeze(0))
        output = fpn(output)

        # 生成proposal
        rpn_loss, proposal_list = rpn.forward_train(output, [results['img_metas'].data], [results['gt_bboxes'].data],
                                                    None, proposal_cfg=dict(nms_across_levels=False,
                                                                            nms_pre=2000,
                                                                            nms_post=2000,
                                                                            max_num=2000,
                                                                            nms_thr=0.7,
                                                                            min_bbox_size=0))

        print(f"rpn_loss: {rpn_loss}, proposales: {len(proposal_list)}")

        roi_loss = roi.forward_train(output, [results['img_metas'].data], proposal_list,
                                     [results['gt_bboxes'].data], [results['gt_labels'].data],
                                     gt_bboxes_ignore=None, gt_masks=None)
        print(f"roi_loss: {roi_loss}")

    def test_model_test(self):
        results = self.test_dataloader(train_class=COCO_BASE_CLASSES, eval_class=COCO_NOVEL_CLASSES, mode='val')[0]
        resnet50 = self.test_resnet_50()
        resnet50.eval()
        fpn = self.test_neck()
        set_random_seed(seed=2023)
        fpn.init_weights()
        fpn.eval()
        rpn = self.test_rpn()
        rpn.eval()
        roi = self.test_roi()
        roi.eval()
        output = resnet50(results['img'][0].unsqueeze(0))
        output = fpn(output)
        print(len(output))

        proposal_list = rpn.simple_test_rpn(output, [item.data for item in results['img_metas']],
                                            proposal_cfg=dict(nms_across_levels=False,
                                                              nms_pre=2000,
                                                              nms_post=2000,
                                                              max_num=2000,
                                                              nms_thr=0.0,
                                                              min_bbox_size=0))
        print(len(proposal_list))

        roi_list = roi.simple_test(output, proposal_list, [item.data for item in results['img_metas']], rescale=True,
                                   test_cfg=dict(
                                       score_thr=0.0,
                                       nms=dict(type='nms', iou_threshold=0.7),
                                       max_per_img=1500,
                                   ))
        print(len(roi_list))

    def test_evaluation(self):
        # 对evaluation进行测试
        dataset = self.test_dataloader(classes=COCO_NOVEL_CLASSES, mode='val')
        import torch
        outputs = torch.load('/home/jiang/桌面/outputs.pt')
        print(dataset.evaluate(outputs, metric=['bbox']))


if __name__ == '__main__':
    print("对backbone进行测试，看功能是否实现")
    unittest.main()
