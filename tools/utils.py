import argparse
import sys
from shutil import get_terminal_size

from tools.timer import Timer


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Model Training')

    # 添加模型相关的命令行参数
    parser.add_argument('--pretrain', type=str, default='modeling/backbone/resnet50-19c8e357.pth',
                        help='backbone的预训练权重路径')
    parser.add_argument('--num_classes', type=int, default=10, help='目标检测中类别的数量')
    parser.add_argument('--batch_size', type=int, default=4, help='训练时的批次大小')
    parser.add_argument('--output', type=str, default='../tools/output/test/', help='模型训练时的输出保存路径')
    parser.add_argument('--seed', type=int, default=2023, help='设置随机数种子')
    parser.add_argument('--device', type=str, default='cuda:0', help='设置gpu设备编号')
    parser.add_argument('--lr', type=float, default=0.02, help='训练的学习率')
    parser.add_argument('--resume', default='', type=str, help='若需要接着上次训练，则指定上次训练保存权重文件地址')
    # parser.add_argument('--seed', type=int, default=2023, help='训练时的随机数种子')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=1, type=int, help='从哪个epoch数开始训练')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=8, type=int, metavar='N',
                        help='训练的总epoch数')

    # 添加其他训练参数...
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='SGD的momentum参数')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='SGD的weight_decay参数 (默认为: 1e-4)',
                        dest='weight_decay')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[6, 7], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    return args


class ProgressBar:
    """A progress bar which can print the progress."""

    def __init__(self, task_num=0, bar_width=50, start=True, file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(f'[{" " * self.bar_width}] 0/{self.task_num}, '
                            'elapsed: 0s, ETA:')
        else:
            self.file.write('completed: 0, elapsed: 0s')
        self.file.flush()
        self.timer = Timer()

    def update(self, num_tasks=1):
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = f'\r[{{}}] {self.completed}/{self.task_num}, ' \
                  f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, ' \
                  f'ETA: {eta:5}s'

            bar_width = min(self.bar_width,
                            int(self.terminal_width - len(msg)) + 2,
                            int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                f' {fps:.1f} tasks/s')
        self.file.flush()


