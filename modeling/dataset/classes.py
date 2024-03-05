#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project ：train_coco_dataset
@Product_name ：PyCharm
@File ：classes.py
@Author ：RockJim
@Date ：2023/11/2 16:09
@Description ：设置各个子数据集的类别信息
@Version ：1.0
"""
#  ------------------------------- coco数据集的类别划分 ------------------------------------------
# 将 pascal voc 的20个类划分为novel类
COCO_BASE_CLASSES = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
# 将 coco数据集中除了Pascal voc 的20个类之外的60个类划分为若干个session
SESSION_NUMBER = 4  # 共有多少个session, 将novel的20个类划分成4个session
COCO_NOVEL_CLASSES = [
    8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 84, 85, 86, 87, 88, 89, 90,
]