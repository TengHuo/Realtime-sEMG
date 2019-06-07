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

from emg.models.base import EMGClassifier
from emg.data_loader.capg_data import CapgDataset


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


def main(train_args, TEST_MODE=False):
    args = train_args

    model = CNN(args['gesture_num'])
    name = args['model'] + '-compare_gesture'  # + str(args['gesture_num'])
    sub_folder = 'gesture-{}'.format(args['gesture_num'])

    from emg.utils import config_tensorboard
    tensorboard_cb = config_tensorboard(name, sub_folder)  # , model, (1, 1, 16, 8))

    # from emg.utils.lr_scheduler import DecayLR
    # lr_callback = DecayLR(start_lr=0.001, gamma=0.1, step_size=20)

    train_set = CapgDataset(gesture=args['gesture_num'],
                            sequence_len=1,
                            frame_x=True,
                            test_mode=TEST_MODE,
                            train=True)

    net = EMGClassifier(module=model,
                        model_name=name,
                        sub_folder=sub_folder,
                        hyperparamters=args,
                        optimizer=torch.optim.Adam,
                        max_epochs=args['epoch'],
                        lr=args['lr'],
                        dataset=train_set,
                        callbacks=[tensorboard_cb])

    net.fit_with_dataset()

    test_set = CapgDataset(gesture=args['gesture_num'],
                           sequence_len=1,
                           frame_x=True,
                           test_mode=TEST_MODE,
                           train=False)

    avg_score = net.test_model(test_set)
    print('test accuracy: {:.4f}'.format(avg_score))


if __name__ == "__main__":
    test_args = {
        'model': 'cnn',
        'gesture_num': 8,
        'lr': 0.001,
        'lr_step': 5,
        'epoch': 1,
        'train_batch_size': 128,
        'valid_batch_size': 1024,
        'stop_patience': 5,
        'log_interval': 100
    }

    main(test_args, TEST_MODE=False)
