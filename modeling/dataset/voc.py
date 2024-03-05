import copy
from typing import Optional, List, Dict, Union
import os.path as osp
import numpy as np
import xml.etree.ElementTree as ET

from modeling.dataset.base import BaseDataset
from modeling.dataset.utils import list_from_file, imread


class VocDataset(BaseDataset):
    # 维护一个类编码表
    label2cat = {
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tvmonitor'
    }

    def __init__(self,
                 is_class_agnostic: bool=True,
                 use_difficult: bool = False,
                 coordinate_offset: List[int] = [-1, -1, 0, 0],
                 min_bbox_area: Optional[Union[int, float]] = None,
                 **data_cfg_):
        self.is_class_agnostic = is_class_agnostic
        self.use_difficult = use_difficult
        self.coordinate_offset = coordinate_offset
        self.min_bbox_area = min_bbox_area
        self.cat2label = {cat: label for label, cat in self.label2cat.items()}
        super(VocDataset, self).__init__(**data_cfg_)

    def load_annotations(self):
        f"""
        通过{self.ann_file}文件来加载{self.classes_ids}中指定类别的标注信息
        self.ann_file: 标注信息的存储路径（绝对路径）
        self.classes_ids: 指定类别的类别id
        :return: 
        """
        # 判断self.classes_ids是否全部都符合要求
        assert set(self.classes_ids).issubset(
            self.label2cat.keys()), f'所指定的类别{str(self.classes_ids)}必须是{str(self.label2cat.keys())}的子集'
        self.CLASSES = [self.label2cat[cat_id] for cat_id in self.classes_ids]
        data_infos = self.load_annotations_xml(self.ann_file, self.CLASSES)

        return data_infos

    def load_annotations_xml(
            self,
            ann_file: str,
            classes: Optional[List[str]] = None) -> List[Dict]:
        """Load annotation from XML style ann_file.

        It supports using image id or image path as image names
        to load the annotation file.

        Args:
            ann_file (str): Path of annotation file.
            classes (list[str] | None): Specific classes to load form xml file.
                If set to None, it will use classes of whole dataset.
                Default: None.

        Returns:
            list[dict]: Annotation info from XML file.
        """
        data_infos = []
        img_names = list_from_file(ann_file)
        for img_name in img_names:
            # ann file in image path format
            if 'VOC2007' in img_name:
                dataset_year = 'VOC2007'
                img_id = img_name.split('/')[-1].split('.')[0]
                filename = img_name
            # ann file in image path format
            elif 'VOC2012' in img_name:
                dataset_year = 'VOC2012'
                img_id = img_name.split('/')[-1].split('.')[0]
                filename = img_name
            # ann file in image id format
            elif 'VOC2007' in ann_file:
                dataset_year = 'VOC2007'
                img_id = img_name
                filename = f'VOC2007/JPEGImages/{img_name}.jpg'
            # ann file in image id format
            elif 'VOC2012' in ann_file:
                dataset_year = 'VOC2012'
                img_id = img_name
                filename = f'VOC2012/JPEGImages/{img_name}.jpg'
            else:
                raise ValueError('Cannot infer dataset year from img_prefix')

            xml_path = osp.join(self.img_prefix, dataset_year, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            if size is not None:
                width = int(size.find('width').text)
                height = int(size.find('height').text)
            else:
                img_path = osp.join(self.img_prefix, dataset_year,
                                    'JPEGImages', f'{img_id}.jpg')
                img = imread(img_path)
                width, height = img.size
            ann_info = self._get_xml_ann_info(dataset_year, img_id, classes)
            data_infos.append(
                dict(
                    id=img_id,
                    file_name=filename,
                    width=width,
                    height=height,
                    ann=ann_info))
        return data_infos

    def _get_xml_ann_info(self,
                          dataset_year: str,
                          img_id: str,
                          classes: Optional[List[str]] = None) -> Dict:
        """Get annotation from XML file by img_id.

        Args:
            dataset_year (str): Year of voc dataset. Options are
                'VOC2007', 'VOC2012'
            img_id (str): Id of image.
            classes (list[str] | None): Specific classes to load form
                xml file. If set to None, it will use classes of whole
                dataset. Default: None.

        Returns:
            dict: Annotation info of specified id with specified class.
        """
        if classes is None:
            classes = self.CLASSES
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []

        xml_path = osp.join(self.img_prefix, dataset_year, 'Annotations',
                            f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in classes:
                continue
            label = self.cat2label[name]
            if self.use_difficult:
                difficult = 0
            else:
                difficult = obj.find('difficult')
                difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')

            # It should be noted that in the original mmdet implementation,
            # the four coordinates are reduced by 1 when the annotation
            # is parsed. Here we following detectron2, only xmin and ymin
            # will be reduced by 1 during training. The groundtruth used for
            # evaluation or testing keep consistent with original xml
            # annotation file and the xmin and ymin of prediction results
            # will add 1 for inverse of data loading logic.
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            if not self.test_mode:
                bbox = [
                    i + offset
                    for i, offset in zip(bbox, self.coordinate_offset)
                ]
            ignore = False
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
        ann_info = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann_info

    def _filter_imgs(self,
                     min_size: int = 32,
                     min_bbox_area: Optional[int] = None) -> List[int]:
        """Filter images not meet the demand.

        Args:
            min_size (int): Filter images with length or width
                smaller than `min_size`. Default: 32.
            min_bbox_area (int | None): Filter images with bbox whose
                area smaller `min_bbox_area`. If set to None, skip
                this filter. Default: None.

        Returns:
            list[int]: valid indices of `data_infos`.
        """
        valid_inds = []
        if min_bbox_area is None:
            min_bbox_area = self.min_bbox_area
        for i, img_info in enumerate(self.data_infos):
            # filter empty image
            if self.filter_empty_gt:
                cat_ids = img_info['ann']['labels'].astype(np.int64).tolist()
                if len(cat_ids) == 0:
                    continue
            # filter images smaller than `min_size`
            if min(img_info['width'], img_info['height']) < min_size:
                continue
            # filter image with bbox smaller than min_bbox_area
            # it is usually used in Attention RPN
            if min_bbox_area is not None:
                skip_flag = False
                for bbox in img_info['ann']['bboxes']:
                    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if bbox_area < min_bbox_area:
                        skip_flag = True
                if skip_flag:
                    continue
            valid_inds.append(i)
        return valid_inds



# -------------------------------------------------- 对VocDataset进行测试 -----------------------------------------------------

from modeling.dataset.transforms import LoadImageFromFile, MultiScaleFlipAug, Resize, RandomFlip, Normalize, Pad, \
    Collect, LoadAnnotations, DefaultFormatBundle
from modeling.dataset import transforms
from modeling.dataset.classes import COCO_BASE_CLASSES, COCO_NOVEL_CLASSES
from modeling.dataset.dataset_path_config import COCO_JSON_ANNOTATIONS_DIR, COCO_IMAGE_ROOT_DIR
import os

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
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
    'ann_file': '/home/jiang/model/fsdet/mmfewshot-main/data/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt',
    'pipeline': transform,
    'classes_ids': [1, 2, 3, 4, 5],
    'data_root': '/home/jiang/model/fsdet/mmfewshot-main/data/VOCdevkit',
}

dataset = VocDataset(**data_cfg)
print(f"数据集长度为：{len(dataset)}")
print(dataset[0])
