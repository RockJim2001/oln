#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：train_coco_dataset
@Product_name ：PyCharm
@File ：config.py
@Author ：RockJim
@Date ：2023/11/8 15:29
@Description ：全局配置文件
@Version ：1.0
"""
import os.path
import random

import numpy as np
import torch

# 获取当前Python脚本的目录
current_directory = os.path.dirname(__file__)
PROJECT_ROOT_PATH = os.path.dirname(current_directory)  # 项目的根目录

# 设置输出的保存路径
OUTPUT_PATH = "./multi_train"  # 默认的地址


# 通过参数来修改全局变量的路径
def set_output_path(new_output_path: str):
    global OUTPUT_PATH
    if not os.path.exists(new_output_path):
        os.mkdir(new_output_path)
    OUTPUT_PATH = new_output_path


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


