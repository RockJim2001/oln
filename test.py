# 模型的测试
import os
import os.path as osp
import time
import torch
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from modeling.dataset.classes import COCO_NOVEL_CLASSES, COCO_BASE_CLASSES
from modeling.dataset.geometric import imresize
from modeling.utils.util import tensor2imgs, encode_mask_results, load_checkpoint
from modeling.utils.visualation import show_result
from tools.config import set_output_path
from tools.utils import parse_args, ProgressBar
from train import create_model, create_dataloader


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    device='cpu',
                    out_dir=None,
                    show_score_thr=0.3):
    # TODO: 可视化预测结果，将show和out_dir进行设置
    # show = True
    # out_dir = r'/home/jiang/model/oln/tools/multi_train/test/image_show'
    show_score_thr = 0.70

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    flag = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            img_metas = data['img_metas'][0].data
            img = [item.to(device) for item in data['img']]
            result = model(img, img_metas, None, None, return_loss=False, rescale=True)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    win_name=img_meta['ori_filename'],
                    fig_size=(ori_w, ori_h),
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
        # flag += 1
        # if flag > 100:
        #     break
    return results


def main():
    start_time = time.time()
    args = parse_args()
    # 存储路径不存在的时候，对其进行创建
    output_path = os.path.join(os.getcwd(), args.output)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    # 设置全局的存储路径
    set_output_path(args.output)
    from tools.config import OUTPUT_PATH
    # 创建可视化对象
    writer = SummaryWriter(log_dir=os.path.join(OUTPUT_PATH, 'test/runs'))

    # 获取可用的设备信息
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("在设备{}上进行训练.".format(device.type))

    # 构建dataloader
    val_data_loader = create_dataloader(args, train_classes=COCO_BASE_CLASSES, eval_classes=COCO_NOVEL_CLASSES,
                                        mode='val')

    # 创建模型结构
    model = create_model(args, training=False)
    model.to(device)
    print(model)

    # 可视化模型结构
    test_image_data = val_data_loader.dataset[0]
    img_metas = test_image_data['img_metas'][0].data
    img = [item.to(device) for item in test_image_data['img']]
    # writer.add_graph(model=model, input_to_model=(img[0], img_metas, None, None, False), verbose=True)

    # 如果传入resume参数，即上次训练的权重地址，则接着上次的参数训练
    if args.resume:
        # If map_location is missing, torch.load will first load the module to CPU
        # and then copy each parameter to where it was saved,
        # which would result in all processes on the same machine using the same set of devices.
        checkpoint = torch.load(args.resume, map_location='cpu')  # 读取之前保存的权重文件(包括优化器以及学习率策略)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")

    outputs = single_gpu_test(model, val_data_loader, device=device)
    eval_kwargs = {}
    print(val_data_loader.dataset.evaluate(outputs, **eval_kwargs))

    end_time = time.time()
    print(f"总共花费:{end_time - start_time}秒")


if __name__ == '__main__':
    main()
