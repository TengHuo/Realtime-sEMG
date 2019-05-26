# -*- coding: UTF-8 -*-
# cnn.py
# @Time     : 24/May/2019
# @Author   : TENG HUO
# @Email    : teng_huo@outlook.com
# @Version  : 1.0.0
# @License  : MIT
#
#


import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from emg.models.torch_model import start_train


hyperparameters = {
    'input_size': (16, 8),
    'seq_length': 1,
    'seq_result': False,
    'frame_input': True
}


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size ** 2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out


class CNN(nn.Module):
    def __init__(self, gesture_num):
        super(CNN, self).__init__()
        self.cov = nn.Sequential(
            nn.BatchNorm2d(1, momentum=0.9),
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(True),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(True),

            LocallyConnected2d(64, 64, (16, 8), 1, 1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(True),

            LocallyConnected2d(64, 64, (16, 8), 1, 1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )
        self.flat = nn.Sequential(
            nn.Linear(8192, 512),
            nn.BatchNorm1d(512, momentum=0.9),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512, momentum=0.9),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128, momentum=0.9),
            nn.ReLU(True)
        )
        self.fc = nn.Linear(128, gesture_num)

    def forward(self, x):
        x = self.cov(x)
        x = x.view(x.size(0), -1)
        x = self.flat(x)
        return self.fc(x)


def main(train_args):
    # 1. 设置好optimizer
    # 2. 定义好model
    args = {**train_args, **hyperparameters}
    model = CNN(args['gesture_num'])
    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'])

    start_train(args, model, optimizer)


if __name__ == "__main__":
    test_args = {
        'model': 'cnn',
        'gesture_num': 8,
        'lr': 0.01,
        'epoch': 10,
        'train_batch_size': 64,
        'val_batch_size': 256,
        'stop_patience': 5,
        'load_model': False
    }

    main(test_args)
