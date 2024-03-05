import itertools
import os
import time
from collections import defaultdict, OrderedDict

import numpy as np
from terminaltables import AsciiTable

from modeling.dataset.base import BaseDataset, BaseFewShotDataset
from modeling.evaluation.cocoeval_wrappers import COCOEvalXclassWrapper


class COCODataset(BaseDataset):
    def __init__(self, is_class_agnostic=False, filter_image_type='mix', **data_cfg_):
        self.is_class_agnostic = is_class_agnostic
        assert filter_image_type in ['mix', 'pure']
        self.filter_image_type = filter_image_type
        super(COCODataset, self).__init__(**data_cfg_)

    def load_annotations(self):
        from pycocotools.coco import COCO
        self.coco = COCO(self.ann_file)

        self.cat_ids = self.coco.getCatIds(catIds=self.classes_ids)
        # self.train_cat_ids = self.coco.getCatIds(catIds=self.train_class_ids)
        # self.eval_cat_ids = self.coco.getCatIds(catIds=self.eval_class_ids)
        if self.is_class_agnostic:
            self.cat2label = {cat_id: 0 for cat_id in self.cat_ids}
        else:
            self.cat2label = {
                cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        self.img_ids = self.coco.getImgIds()

        data_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            # if '000000200365.jpg' in info['file_name']:
            #     print("哈哈哈哈")
            annotation_ids = self.coco.getAnnIds(imgIds=[info['id']])
            annotations = self.coco.load_anns(ids=annotation_ids)
            # TODO： 进行各种筛选
            if self._filter(info, annotations):
                info['ann'] = self._parse_ann_info(img_info=info, ann_info=annotations)
                data_infos.append(info)
        return data_infos

    def _filter(self, info, annotations):
        def valid_at_least_one(anns):
            """
            当一张图片中有一个实例类别在指定类别{self.classes_ids}范围内的话，就采用这张图片,
            :param anns:
            :return:
            """
            for ann in anns:
                if ann.get("category_id") in self.cat_ids:
                    return True
            return False

        def valid_only_one(anns):
            """
            当一张图片中所有实例类别都在指定类别{self.classes_ids}范围内的话，就采用这张图片
            :param anns:
            :return:
            """
            for ann in anns:
                if ann.get("category_id") in self.cat_ids:
                    continue
                else:
                    return False
            return True

        def _filter_imgs(img_info, min_size=32):
            if min(img_info['width'], img_info['height']) >= min_size:
                return True
            return False

        if self.filter_image_type == 'mix':
            return valid_at_least_one(annotations) and _filter_imgs(info)
        else:
            return valid_only_one(annotations) and _filter_imgs(info)

    # def get_ann_info(self, idx):
    #     """Get COCO annotation by index.
    #
    #     Args:
    #         idx (int): Index of data.
    #
    #     Returns:
    #         dict: Annotation info of specified index.
    #     """
    #
    #     img_id = self.data_infos[idx]['id']
    #     # ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
    #     ann_ids = self.coco.getAnnIds(imgIds=[img_id])
    #     ann_info = self.coco.loadAnns(ann_ids)
    #     return self._parse_ann_info(self.data_infos[idx], ann_info)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            # if inter_w * inter_h == 0:
            #     continue
            # if ann['area'] <= 0 or w < 1 or h < 1:
            #     continue
            # if ann['category_id'] not in self.train_cat_ids:
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            # if ann.get('iscrowd', False):
            #     gt_bboxes_ignore.append(bbox)
            # else:
            gt_bboxes.append(bbox)
            gt_labels.append(self.cat2label[ann['category_id']])
            gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(10, 20, 30, 50, 100, 300, 500, 1000, 1500),
                 iou_thrs=None,
                 metric_items=None):
        """Evaluation in COCO-Split protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]
        start_time = time.time()
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        end_time = time.time()
        print(f"一共花费了{end_time - start_time}s")
        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            # print_log(msg, logger=logger)
            print(msg)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                # print_log(log_msg, logger=logger)
                print(log_msg)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                # print_log(
                #     'The testing results of the whole dataset is empty.',
                #     logger=logger,
                #     level=logging.ERROR)
                print('The testing results of the whole dataset is empty.')
                break

            iou_type = 'bbox' if metric == 'proposal' else metric

            # Class manipulation.
            for idx, ann in enumerate(cocoGt.dataset['annotations']):
                if ann['category_id'] in self.eval_cat_ids:
                    cocoGt.dataset['annotations'][idx]['ignored_split'] = 0
                else:
                    cocoGt.dataset['annotations'][idx]['ignored_split'] = 1

            # Cross-category evaluation wrapper.
            cocoEval = COCOEvalXclassWrapper(cocoGt, cocoDt, iou_type)

            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@10': 6,
                'AR@20': 7,
                'AR@50': 8,
                'AR@100': 9,
                'AR@300': 10,
                'AR@500': 11,
                'AR@1000': 12,
                'AR@1500': 13,
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            cocoEval.params.useCats = 0  # treat all FG classes as single class.
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval['precision']
                # precision: (iou, recall, cls, area range, max dets)
                assert len(self.cat_ids) == precisions.shape[2]

                results_per_category = []
                for idx, catId in enumerate(self.cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = self.coco.loadCats(catId)[0]
                    precision = precisions[:, :, idx, 0, -1]
                    precision = precision[precision > -1]
                    if precision.size:
                        ap = np.mean(precision)
                    else:
                        ap = float('nan')
                    results_per_category.append(
                        (f'{nm["name"]}', f'{float(ap):0.3f}'))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(
                    itertools.chain(*results_per_category))
                headers = ['category', 'AP'] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[
                    results_flatten[i::num_columns]
                    for i in range(num_columns)
                ])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                # print_log('\n' + table.table, logger=logger)
                print('\n' + table.table)

            if metric_items is None:
                metric_items = [
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                ]

            for metric_item in metric_items:
                key = f'{metric}_{metric_item}'
                val = float(
                    f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                )
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f'{metric}_mAP_copypaste'] = (
                f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


class FewShotCOCODataset(COCODataset):

    def __init__(self, pretrain=True, filter_image_type='mix', **data_cfg):
        self.pretrain = pretrain
        super(FewShotCOCODataset, self).__init__(filter_image_type=filter_image_type, **data_cfg)

    def load_annotations(self):

        temp_data_infos = super().load_annotations()

        if self.pretrain:  # 加载pretraining的数据集
            return temp_data_infos
        # 加载metatraining的数据集，按照classes进行划分
        new_data_infos = defaultdict(list)
        for data_info in temp_data_infos:
            record = {}
            record['file_name'] = os.path.join(self.data_root, data_info['file_name'])
            record['height'] = data_info['height']
            record['width'] = data_info['width']
            # if '000000200365.jpg' in data_info['file_name']:
            #     print("哈哈哈哈")
            record['id'] = data_info['id']
            record['license'] = data_info['license']
            temp_ann = data_info['ann']
            for index in range(len(set(temp_ann['labels']))):
                cat_id = list(set(temp_ann['labels']))[index]
                select_item = np.where(temp_ann['labels'] == cat_id)[0]
                new_data_infos[cat_id].append(
                    {**record, **({'ann':
                                       {'bboxes': temp_ann['bboxes'][select_item],
                                        'labels': temp_ann['labels'][select_item],
                                        'bboxes_ignore': temp_ann['bboxes_ignore'],
                                        'masks': [temp_ann['masks'][temp_item] for temp_item in select_item],
                                        'seg_map': temp_ann['seg_map']
                                        }
                                   })}
                )
        return new_data_infos

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        return None

    def prepare_train_img(self, idx):
        pass



# -------------------------------------------------- 对CoCoDataset进行测试 -----------------------------------------------------

# from modeling.dataset.transforms import LoadImageFromFile, MultiScaleFlipAug, Resize, RandomFlip, Normalize, Pad, \
#     Collect, LoadAnnotations, DefaultFormatBundle
# from modeling.dataset import transforms
# from modeling.dataset.classes import COCO_BASE_CLASSES, COCO_NOVEL_CLASSES
# from modeling.dataset.dataset_path_config import COCO_JSON_ANNOTATIONS_DIR, COCO_IMAGE_ROOT_DIR
# import os
#
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
# )
# transform = transforms.Compose([LoadImageFromFile(),
#                                 LoadAnnotations(with_bbox=True),
#                                 Resize(img_scale=(1333, 800), keep_ratio=True),
#                                 RandomFlip(flip_ratio=0.5),
#                                 Normalize(**img_norm_cfg),
#                                 Pad(size_divisor=32),
#                                 DefaultFormatBundle(),
#                                 Collect(keys=['img', 'gt_bboxes', 'gt_labels'])
#                                 ])
#
# data_cfg = {
#     'ann_file': os.path.join(COCO_JSON_ANNOTATIONS_DIR, 'instances_train2017.json'),
#     'pipeline': transform,
#     'classes_ids': COCO_NOVEL_CLASSES,
#     'data_root': os.path.join(COCO_IMAGE_ROOT_DIR, 'train2017'),
# }

# dataset = COCODataset(filter_image_type='pure', **data_cfg)
# print(f"数据集长度为：{len(dataset)}")
# print(dataset[0])

# -------------------------------------------------- 对FewShotCOCODataset进行测试 -----------------------------------------------------


# data_cfg['instance_wise'] = True
# data_cfg['image_wise'] = False

# dataset_fewshot_coco = FewShotCOCODataset(pretrain=False, filter_image_type='mix', **data_cfg)
# print(f"数据集长度为：{len(dataset_fewshot_coco)}")
# print(dataset_fewshot_coco[0])
