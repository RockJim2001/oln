# 修改tensorboard生成的events事件的tag
import os

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter


def test():
    input_path = '/home/jiang/model/oln/tools/outputs/train_test_1000/tf_log/oln/events.out.tfevents.1702992848.jiang.61505.0'  # 输入需要指定event文件
    output_path = '/home/jiang/model/oln/tools/outputs/train_test_1000/tf_log/oln'  # 输出只需要指定文件夹即可

    # 读取需要修改的event文件
    ea = event_accumulator.EventAccumulator(input_path)
    ea.Reload()
    tags = ea.scalars.Keys()  # 获取所有scalar中的keys

    # 写入新的文件
    writer = SummaryWriter(output_path)  # 创建一个SummaryWriter对象
    for tag in tags:
        scalar_list = ea.scalars.Items(tag)

        # if tag == 'val_Acc_':  # 修改一下对应的tag即可
        #     tag = tag[:-1]
        new_tag = 'train/' + tag

        for scalar in scalar_list:
            writer.add_scalar(new_tag, scalar.value, scalar.step, scalar.wall_time)  # 添加修改后的值到新的event文件中
    writer.close()  # 关闭SummaryWriter对象


def statisc_index():
    # 绘制oln中各个指标的信息
    result = {
        1: {
            'AR10': 0.132,
            'AR20': 0.181,
            'AR30': 0.211,
            'AR50': 0.250,
            'AR100': 0.297,
            'AR300': 0.359,
            'AR500': 0.383,
            'AR1000': 0.413,
            'AR1500': 0.427,
        },
        2: {
            'AR10': 0.125,
            'AR20': 0.177,
            'AR30': 0.211,
            'AR50': 0.252,
            'AR100': 0.306,
            'AR300': 0.373,
            'AR500': 0.398,
            'AR1000': 0.427,
            'AR1500': 0.437,
        },
        3: {
            'AR10': 0.135,
            'AR20': 0.188,
            'AR30': 0.221,
            'AR50': 0.258,
            'AR100': 0.304,
            'AR300': 0.366,
            'AR500': 0.388,
            'AR1000': 0.418,
            'AR1500': 0.426,
        },
        4: {
            'AR10': 0.124,
            'AR20': 0.178,
            'AR30': 0.214,
            'AR50': 0.257,
            'AR100': 0.311,
            'AR300': 0.378,
            'AR500': 0.402,
            'AR1000': 0.430,
            'AR1500': 0.437,
        },
        5: {
            'AR10': 0.138,
            'AR20': 0.196,
            'AR30': 0.228,
            'AR50': 0.269,
            'AR100': 0.315,
            'AR300': 0.375,
            'AR500': 0.399,
            'AR1000': 0.430,
            'AR1500': 0.442,
        },
        6: {
            'AR10': 0.143,
            'AR20': 0.200,
            'AR30': 0.232,
            'AR50': 0.274,
            'AR100': 0.324,
            'AR300': 0.384,
            'AR500': 0.407,
            'AR1000': 0.438,
            'AR1500': 0.449,
        },
        7: {
            'AR10': 0.167,
            'AR20': 0.227,
            'AR30': 0.259,
            'AR50': 0.296,
            'AR100': 0.338,
            'AR300': 0.396,
            'AR500': 0.421,
            'AR1000': 0.453,
            'AR1500': 0.461,
        },
        8: {
            'AR10': 0.171,
            'AR20': 0.228,
            'AR30': 0.261,
            'AR50': 0.295,
            'AR100': 0.336,
            'AR300': 0.393,
            'AR500': 0.418,
            'AR1000': 0.451,
            'AR1500': 0.459,
        }
    }
    output_path = '/home/jiang/model/obj_oln/object_localization_network-main/work_dirs/train/test/object_localization_network'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    writer = SummaryWriter(output_path)
    for key, value in result.items():
        for indic, val in value.items():
            writer.add_scalar(indic, val, key)
    writer.close()


def statisc_index_():
    # 绘制oln中各个指标的信息
    result = {
        1: {
            'AR10': 0.120,
            'AR20': 0.165,
            'AR30': 0.195,
            'AR50': 0.231,
            'AR100': 0.276,
            'AR300': 0.339,
            'AR500': 0.364,
            'AR1000': 0.395,
            'AR1500': 0.405,
        },
        2: {
            'AR10': 0.140,
            'AR20': 0.191,
            'AR30': 0.219,
            'AR50': 0.254,
            'AR100': 0.295,
            'AR300': 0.354,
            'AR500': 0.380,
            'AR1000': 0.413,
            'AR1500': 0.420,
        },
        3: {
            'AR10': 0.141,
            'AR20': 0.187,
            'AR30': 0.215,
            'AR50': 0.246,
            'AR100': 0.286,
            'AR300': 0.340,
            'AR500': 0.365,
            'AR1000': 0.401,
            'AR1500': 0.410,
        },
        4: {
            'AR10': 0.134,
            'AR20': 0.184,
            'AR30': 0.212,
            'AR50': 0.248,
            'AR100': 0.293,
            'AR300': 0.354,
            'AR500': 0.380,
            'AR1000': 0.414,
            'AR1500': 0.423,
        },
        5: {
            'AR10': 0.136,
            'AR20': 0.190,
            'AR30': 0.219,
            'AR50': 0.256,
            'AR100': 0.302,
            'AR300': 0.364,
            'AR500': 0.389,
            'AR1000': 0.423,
            'AR1500': 0.437,
        },
        6: {
            'AR10': 0.135,
            'AR20': 0.186,
            'AR30': 0.217,
            'AR50': 0.255,
            'AR100': 0.305,
            'AR300': 0.371,
            'AR500': 0.398,
            'AR1000': 0.428,
            'AR1500': 0.438,
        },
        7: {
            'AR10': 0.172,
            'AR20': 0.229,
            'AR30': 0.262,
            'AR50': 0.302,
            'AR100': 0.346,
            'AR300': 0.406,
            'AR500': 0.431,
            'AR1000': 0.462,
            'AR1500': 0.468,
        },
        8: {
            'AR10': 0.174,
            'AR20': 0.232,
            'AR30': 0.263,
            'AR50': 0.300,
            'AR100': 0.342,
            'AR300': 0.399,
            'AR500': 0.425,
            'AR1000': 0.456,
            'AR1500': 0.462,
        }
    }
    output_path = '/home/jiang/model/obj_oln/object_localization_network-main/work_dirs/train/test/object_localization_network_original'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    writer = SummaryWriter(output_path)
    for key, value in result.items():
        for indic, val in value.items():
            writer.add_scalar(indic, val, key)
    writer.close()


def statisc_index_oln():
    # 绘制oln中各个指标的信息
    result = {
        1: {
            'AR10': 0.125,
            'AR20': 0.171,
            'AR30': 0.201,
            'AR50': 0.236,
            'AR100': 0.282,
            'AR300': 0.344,
            'AR500': 0.367,
            'AR1000': 0.398,
            'AR1500': 0.408,
        },
        2: {
            'AR10': 0.130,
            'AR20': 0.181,
            'AR30': 0.213,
            'AR50': 0.252,
            'AR100': 0.299,
            'AR300': 0.364,
            'AR500': 0.384,
            'AR1000': 0.412,
            'AR1500': 0.422,
        },
        3: {
            'AR10': 0.139,
            'AR20': 0.193,
            'AR30': 0.227,
            'AR50': 0.270,
            'AR100': 0.323,
            'AR300': 0.388,
            'AR500': 0.412,
            'AR1000': 0.440,
            'AR1500': 0.446,
        },
        4: {
            'AR10': 0.130,
            'AR20': 0.180,
            'AR30': 0.210,
            'AR50': 0.245,
            'AR100': 0.291,
            'AR300': 0.352,
            'AR500': 0.375,
            'AR1000': 0.405,
            'AR1500': 0.418,
        },
        5: {
            'AR10': 0.145,
            'AR20': 0.199,
            'AR30': 0.231,
            'AR50': 0.270,
            'AR100': 0.317,
            'AR300': 0.383,
            'AR500': 0.408,
            'AR1000': 0.437,
            'AR1500': 0.447,
        },
        6: {
            'AR10': 0.140,
            'AR20': 0.192,
            'AR30': 0.224,
            'AR50': 0.264,
            'AR100': 0.310,
            'AR300': 0.371,
            'AR500': 0.395,
            'AR1000': 0.425,
            'AR1500': 0.435,
        },
        7: {
            'AR10': 0.169,
            'AR20': 0.229,
            'AR30': 0.263,
            'AR50': 0.300,
            'AR100': 0.344,
            'AR300': 0.402,
            'AR500': 0.426,
            'AR1000': 0.456,
            'AR1500': 0.463,
        },
        8: {
            'AR10': 0.170,
            'AR20': 0.227,
            'AR30': 0.262,
            'AR50': 0.299,
            'AR100': 0.341,
            'AR300': 0.398,
            'AR500': 0.422,
            'AR1000': 0.455,
            'AR1500': 0.463,
        }
    }
    output_path = '/home/jiang/model/obj_oln/object_localization_network-main/work_dirs/train/test/oln'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    writer = SummaryWriter(output_path)
    for key, value in result.items():
        for indic, val in value.items():
            writer.add_scalar(indic, val, key)
    writer.close()


if __name__ == '__main__':
    statisc_index()
    statisc_index_()
    statisc_index_oln()
