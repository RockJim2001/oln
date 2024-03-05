import copy
import json
import tempfile
import warnings
from typing import Optional, Dict, List, Union, Sequence

import numpy as np
from torch.utils.data import Dataset
import os.path as osp


class BaseDataset(Dataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 classes_ids=[],
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True):
        """
            通用的数据集加载方式，适用于目标检测
        :param ann_file: 数据集的标注文件路径，其中包含有关每个样本的注释信息。这可能是一个 JSON 文件、XML 文件、COCO 格式文件等，具体格式取决于你的数据集。
        :param pipeline: 数据预处理的管道，它描述了在加载图像和标注之后对数据进行的一系列处理步骤，例如调整大小、标准化等。pipeline 通常是一个由一系列数据处理操作组成的列表。
        :param classes_ids: 可选参数，用于指定数据集中的类别。如果不提供，则默认使用所有类别的id。
        :param data_root: 可选参数，指定数据集的根目录。如果提供了 data_root，ann_file 中的路径将相对于 data_root。
        :param img_prefix: 可选参数，图像文件的前缀路径。如果提供了 img_prefix，则 img_filename 将会是 img_prefix 和图像文件名的组合。
        :param seg_prefix:  可选参数，分割标注文件的前缀路径。如果提供了 seg_prefix，则 seg_map_filename 将会是 seg_prefix 和分割标注文件名的组合。
        :param proposal_file:  可选参数，用于指定包含候选框提议的文件。这在一些目标检测任务中很有用，例如Faster R-CNN等。
        :param test_mode:  可选参数，如果设置为 True，则表示当前是在测试模式下加载数据。在测试模式下，一些特定的数据处理步骤可能会被省略或者进行不同的处理。
        :param filter_empty_gt: 可选参数，如果设置为 True，则在加载数据时会过滤掉没有 ground truth 的样本。
        """
        # 初始化一些参数
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.classes_ids = classes_ids

        self.connection_path_and_load_data()

        # 过滤掉小的或者没有标注的数据， 训练阶段有用
        if not test_mode:
            # valid_inds = self._filter_imgs()
            # self.data_infos = [self.data_infos[i] for i in valid_inds]
            # if self.proposals is not None:
                # self.proposals = [self.proposals[i] for i in valid_inds]
            # set group flag for the sampler
            self._set_group_flag()

        self.pipeline = pipeline

    def connection_path_and_load_data(self):
        # 设置或者更新标注文件的路径
        if self.data_root is not None:
            if not osp.isabs(self.ann_file):
                self.ann_file = osp.join(self.data_root, self.ann_file)
            if not (self.img_prefix is None or osp.isabs(self.img_prefix)):
                self.img_prefix = osp.join(self.data_root, self.img_prefix)
            if not (self.seg_prefix is None or osp.isabs(self.seg_prefix)):
                self.seg_prefix = osp.join(self.data_root, self.seg_prefix)
            if not (self.proposal_file is None or osp.isabs(self.proposal_file)):
                self.proposal_file = osp.join(self.data_root, self.proposal_file)
        # 通过路径加载标注数据
        self.data_infos = self.load_annotations()

        if self.proposal_file is not None:
            self.proposals = self.load_proposals()
        else:
            self.proposals = None

    def __getitem__(self, index):
        if self.test_mode:
            return self.prepare_test_img(index)
        while True:
            data = self.prepare_train_img(index)
            if data is None:
                index = self._rand_another(index)
                continue
            return data

    def __len__(self):
        return len(self.data_infos)

    def load_annotations(self):
        """Load annotation from annotation file."""
        # return mmcv.load(ann_file)
        # return 'mmcv.load(ann_file)'
        pass

    def load_proposals(self):
        """Load proposal from proposal file."""
        # return mmcv.load(proposal_file)
        # return 'mmcv.load(proposal_file)'
        pass

    def get_ann_info(self, idx):
        """Get annotation by index.

        When override this function please make sure same annotations are used
        during the whole training.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return copy.deepcopy(self.data_infos[idx]['ann'])

    def get_cat_ids(self, idx):
        """Get category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        return self.data_infos[idx]['ann']['labels'].astype(np.int).tolist()

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.data_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data  after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys intorduced by \
                piepline.
        """

        img_info = self.data_infos[idx]
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    # def format_results(self, results, **kwargs):
    #     """Place holder to format result to dataset specific output."""
    #     pass

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        # assert len(results) == len(self), (
        #     'The length of results is not equal to the dataset len: {} != {}'.
        #     format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            # mmcv.dump(json_results, result_files['bbox'])
            with open(result_files['bbox'], 'w') as f:
                json.dump(json_results, f)
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            # mmcv.dump(json_results[0], result_files['bbox'])
            with open(result_files['bbox'], 'w') as f:
                json.dump(json_results[0], f)
            # mmcv.dump(json_results[1], result_files['segm'])
            with open(result_files['segm'], 'w') as f:
                json.dump(json_results[1], f)
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            # mmcv.dump(json_results, result_files['proposal'])
            with open(result_files['proposal'], 'w') as f:
                json.dump(json_results, f)
        else:
            raise TypeError('invalid type of results')
        return result_files

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]


    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn(
                'CustomDataset does not support filtering empty gt images.')
        valid_inds = []
        # TODO： 需要进行合并，以减少不必要的循环
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds


class BaseFewShotDataset(BaseDataset):

    def __init__(self,
                 ann_file,
                 is_class_agnostic: bool = False,
                 class_wise: bool = False,
                 image_wise: bool = True,
                 instance_wise: bool = False,  # 是否一张图片仅包含一个实例
                 # multi_pipelines: Optional[Dict[str, List[Dict]]] = None,
                 classes_ids=[],
                 pipeline: Optional[List[Dict]] = None,
                 data_root: Optional[str] = None,
                 img_prefix: str = '',
                 seg_prefix: Optional[str] = None,
                 proposal_file: Optional[str] = None,
                 test_mode: bool = False,
                 filter_empty_gt: bool = True,
                 min_bbox_size: Optional[Union[int, float]] = None,
                 min_bbox_area: Optional[Union[int, float]] = None,
                 ann_shot_filter: Optional[Dict] = None,
                 dataset_name: Optional[str] = None):
        """
        通用的few shot数据加载，适用于voc 和coco数据集
        :param is_class_agnostic: 是否跟类别无关
        :param class_wise: data_infos 以类别作为单位，一张图片中当前类别的所有instances作为一条数据
            data_infos = [{
                   30: [{ 'file_name': ………,
                          'height': xxx,
                          'weight': xxx,
                          'image_id': x,
                          'annotations':[{
                                'bbox': [1, 2, 3, 4],
                                'iscrowd': 0/1,
                                'category_id': 30,
                                },
                                {}]
                   }]
            }]
        :param image_wise: data_infos 以图片作为单位， 一张图片中所有类别的instances作为一条数据（普通的形式）
            data_infos = [{
                       0001: [{ 'file_name': ………,
                              'height': xxx,
                              'weight': xxx,
                              'image_id': 0001,
                              'annotations':[{
                                    'bbox': [1, 2, 3, 4],
                                    'iscrowd': 0/1,
                                    'category_id': 30,
                                    },
                                    {
                                    'bbox': [1, 2, 3, 4],
                                    'iscrowd': 0/1,
                                    'category_id': 31,
                                    }]
                       }]
                }]
        :param instance_wise: data_infos 以实例作为单位， 一个实例作为一条数据
            data_infos = [{
                   30: [{ 'file_name': ………,
                          'height': xxx,
                          'weight': xxx,
                          'image_id': x,
                          'annotations':[{
                                'bbox': [1, 2, 3, 4],
                                'iscrowd': 0/1,
                                'category_id': 30,
                                }]
                   }]
            }]
        :param multi_pipelines: support_set和 query_set分开定义数据处理流程
        :param classes: 通过类别来筛选数据
        :param pipeline: 对support_set和query_set使用相同的数据处理流程
        :param data_root: 数据的存储路径
        :param img_prefix: 前缀
        :param seg_prefix: 前缀
        :param proposal_file: proposal标注文件路径
        :param test_mode: 是train还是val
        :param filter_empty_gt: 过滤没有标注的数据
        :param min_bbox_size: 最小的bbox的大小，w < min_bbox_size and h < min_bbox_size
        :param ann_shot_filter: 将某些类的shot设置在一定的范围之内
            'ann_shot_filter': {
                'cat': 10,
                'dog': 10,
                'person': 2,
                'car': 2,
            },
        :param dataset_name: 数据集的名称
        """

        self.ann_file = ann_file
        self.is_class_agnostic = is_class_agnostic
        self.class_wise = class_wise
        self.image_wise = image_wise
        self.instance_wise = instance_wise
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.proposal_file = proposal_file
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.classes_ids = classes_ids
        self.ann_shot_filter = ann_shot_filter
        self.dataset_name = dataset_name
        self.min_bbox_area = min_bbox_area
        self.connection_path_and_load_data()  # 设置路径并加载数据

        # filter images too small and containing no annotations
        if not test_mode:
            # filter bbox smaller than the min_bbox_size
            if min_bbox_size:
                self.data_infos = self._filter_bboxs(min_bbox_size)
            # filter images
            valid_inds = self._filter_imgs()
            self.data_infos = [self.data_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]
            # filter annotations by ann_shot_filter
            if ann_shot_filter is not None:
                if isinstance(ann_shot_filter, dict):
                    for class_name in list(ann_shot_filter.keys()):
                        assert class_name in self.CLASSES, \
                            f'{self.dataset_name} : class ' \
                            f'{class_name} in ann_shot_filter not in CLASSES.'
                else:
                    raise TypeError('ann_shot_filter only support dict')
                self.ann_shot_filter = ann_shot_filter
                self.data_infos = self._filter_annotations(
                    self.data_infos, self.ann_shot_filter)

        # 数据处理
        if self.instance_wise:  # 实例级别的
            instance_wise_data_infos = []
            for data_info in self.data_infos:
                num_instance = data_info['ann']['labels'].size
                # split annotations
                if num_instance > 1:
                    for i in range(data_info['ann']['labels'].size):
                        tmp_data_info = copy.deepcopy(data_info)
                        tmp_data_info['ann']['labels'] = np.expand_dims(
                            data_info['ann']['labels'][i], axis=0)
                        tmp_data_info['ann']['bboxes'] = np.expand_dims(
                            data_info['ann']['bboxes'][i, :], axis=0)
                        instance_wise_data_infos.append(tmp_data_info)
                else:
                    instance_wise_data_infos.append(data_info)
            self.data_infos = instance_wise_data_infos
        elif self.class_wise:
            print()
        # merge different annotations with the same image
        elif self.image_wise:   # 图片级别的
            merge_data_dict = {}
            for i, data_info in enumerate(self.data_infos):
                # merge data_info with the same image id
                if merge_data_dict.get(data_info['id'], None) is None:
                    merge_data_dict[data_info['id']] = data_info
                else:
                    ann_a = merge_data_dict[data_info['id']]['ann']
                    ann_b = data_info['ann']
                    merge_dat_info = {
                        'bboxes':
                            np.concatenate((ann_a['bboxes'], ann_b['bboxes'])),
                        'labels':
                            np.concatenate((ann_a['labels'], ann_b['labels'])),
                    }
                    # merge `bboxes_ignore`
                    if ann_a.get('bboxes_ignore', None) is not None:
                        if not (ann_a['bboxes_ignore']
                                == ann_b['bboxes_ignore']).all():
                            merge_dat_info['bboxes_ignore'] = \
                                np.concatenate((ann_a['bboxes_ignore'],
                                                ann_b['bboxes_ignore']))
                            merge_dat_info['labels_ignore'] = \
                                np.concatenate((ann_a['labels_ignore'],
                                                ann_b['labels_ignore']))
                    merge_data_dict[
                        data_info['id']]['ann'] = merge_dat_info
            self.data_infos = [
                merge_data_dict[key] for key in merge_data_dict.keys()
            ]
        else:
            raise ValueError(f"{self.image_wise}, {self.class_wise}, {self.instance_wise}三者中必须且只能有一个是True")

        # if multi_pipelines is not None:
        #     assert isinstance(multi_pipelines, dict), \
        #         f'{self.dataset_name} : multi_pipelines is type of dict'
        #     self.multi_pipelines = {}
        #     for key in multi_pipelines.keys():
        #         self.multi_pipelines[key] = multi_pipelines[key]
        # elif pipeline is not None:
        #     assert isinstance(pipeline, list), \
        #         f'{self.dataset_name} : pipeline is type of list'
        #     self.pipeline = pipeline
        # else:
        #     raise ValueError('missing pipeline or multi_pipelines')

        # # 打印数据统计
        # print(self.__repr__())
        self.pipeline = pipeline

    def _filter_bboxs(self, min_bbox_size: int) -> List[Dict]:
        new_data_infos = []
        for data_info in self.data_infos:
            ann = data_info['ann']
            keep_idx, ignore_idx = [], []
            for i in range(ann['bboxes'].shape[0]):
                bbox = ann['bboxes'][i]
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                # check bbox size
                if w < min_bbox_size or h < min_bbox_size:
                    ignore_idx.append(i)
                else:
                    keep_idx.append(i)
            # remove undesired bbox
            if len(ignore_idx) > 0:
                bboxes_ignore = ann.get('bboxes_ignore', np.zeros((0, 4)))
                labels_ignore = ann.get('labels_ignore', np.zeros((0, )))
                new_bboxes_ignore = ann['bboxes'][ignore_idx]
                new_labels_ignore = ann['labels'][ignore_idx]
                bboxes_ignore = np.concatenate(
                    (bboxes_ignore, new_bboxes_ignore))
                labels_ignore = np.concatenate(
                    (labels_ignore, new_labels_ignore))
                data_info.update(
                    ann=dict(
                        bboxes=ann['bboxes'][keep_idx],
                        labels=ann['labels'][keep_idx],
                        bboxes_ignore=bboxes_ignore,
                        labels_ignore=labels_ignore))
            new_data_infos.append(data_info)
        return new_data_infos

    def _filter_annotations(self, data_infos: List[Dict],
                            ann_shot_filter: Dict) -> List[Dict]:
        """Filter out extra annotations of specific class, while annotations of
        classes not in filter remain unchanged and the ignored annotations will
        be removed.

        Args:
            data_infos (list[dict]): Annotation infos.
            ann_shot_filter (dict): Specific which class and how many
                instances of each class to load from annotation file.
                For example: {'dog': 10, 'cat': 10, 'person': 5}

        Returns:
            list[dict]: Annotation infos where number of specified class
                shots less than or equal to predefined number.
        """
        if ann_shot_filter is None:
            return data_infos
        # build instance indices of (img_id, gt_idx)
        filter_instances = {key: [] for key in ann_shot_filter.keys()}
        keep_instances_indices = []
        for idx, data_info in enumerate(data_infos):
            ann = data_info['ann']
            for i in range(ann['labels'].shape[0]):
                instance_class_name = self.CLASSES[ann['labels'][i]]
                # only filter instance from the filter class
                if instance_class_name in ann_shot_filter.keys():
                    filter_instances[instance_class_name].append((idx, i))
                # skip the class not in the filter
                else:
                    keep_instances_indices.append((idx, i))
        # filter extra shots
        for class_name in ann_shot_filter.keys():
            num_shots = ann_shot_filter[class_name]
            instance_indices = filter_instances[class_name]
            if num_shots == 0:
                continue
            # random sample from all instances
            if len(instance_indices) > num_shots:
                random_select = np.random.choice(
                    len(instance_indices), num_shots, replace=False)
                keep_instances_indices += \
                    [instance_indices[i] for i in random_select]
            # number of available shots less than the predefined number,
            # which may cause the performance degradation
            else:
                # check the number of instance
                if len(instance_indices) < num_shots:
                    warnings.warn(f'number of {class_name} instance is '
                                  f'{len(instance_indices)} which is '
                                  f'less than predefined shots {num_shots}.')
                keep_instances_indices += instance_indices

        # keep the selected annotations and remove the undesired annotations
        new_data_infos = []
        for idx, data_info in enumerate(data_infos):
            selected_instance_indices = \
                sorted(instance[1] for instance in keep_instances_indices
                       if instance[0] == idx)
            if len(selected_instance_indices) == 0:
                continue
            ann = data_info['ann']
            selected_ann = dict(
                bboxes=ann['bboxes'][selected_instance_indices],
                labels=ann['labels'][selected_instance_indices],
            )
            new_data_infos.append(
                dict(
                    id=data_info['id'],
                    filename=data_info['filename'],
                    width=data_info['width'],
                    height=data_info['height'],
                    ann=selected_ann))
        return new_data_infos

    def prepare_train_img(self,
                          idx: int,
                          pipeline_key: Optional[str] = None,
                          gt_idx: Optional[List[int]] = None) -> Dict:
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.
            pipeline_key (str): Name of pipeline
            gt_idx (list[int]): Index of used annotation.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)

        # select annotation in `gt_idx`
        if gt_idx is not None:
            selected_ann_info = {
                'bboxes': ann_info['bboxes'][gt_idx],
                'labels': ann_info['labels'][gt_idx]
            }
            # keep pace with new annotations
            new_img_info = copy.deepcopy(img_info)
            new_img_info['ann'] = selected_ann_info
            results = dict(img_info=new_img_info, ann_info=selected_ann_info)
        # use all annotations
        else:
            results = dict(img_info=copy.deepcopy(img_info), ann_info=ann_info)

        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]

        self.pre_pipeline(results)
        if pipeline_key is None:
            return self.pipeline(results)
